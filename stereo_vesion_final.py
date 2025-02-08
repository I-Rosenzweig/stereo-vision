import cv2
import socket
import pickle
import struct
import threading
import numpy as np
from ultralytics import YOLO  # Install YOLOv8 using 'pip install ultralytics'
from datetime import datetime  # Import datetime for timestamp


# Global variables
left_frame = None
right_frame = None
frame_lock = threading.Lock()

# Load YOLOv8n model
model = YOLO("yolov8n.pt")  # Replace with your YOLOv8 model path if needed

# Video Writer setup (for recording)
#fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for video
#out = cv2.VideoWriter('recorded_stream.avi', fourcc, 20.0, (720, 640))  # Change resolution to match your stream

def receive_stream(host, port, frame_name):
    """Receive video stream from a given host and port."""
    global left_frame, right_frame

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, port))

    data = b""
    payload_size = struct.calcsize("Q")

    while True:
        try:
            while len(data) < payload_size:
                packet = client_socket.recv(4096)
                if not packet:
                    break
                data += packet

            packed_msg_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = struct.unpack("Q", packed_msg_size)[0]

            while len(data) < msg_size:
                data += client_socket.recv(4096)

            frame_data = data[:msg_size]
            data = data[msg_size:]

            frame = pickle.loads(frame_data)
            frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

            with frame_lock:
                if frame_name == "left":
                    left_frame = frame
                elif frame_name == "right":
                    right_frame = frame
        except Exception as e:
            print(f"Error in stream {frame_name}: {e}")
            break

    client_socket.close()
def calculate_average_disparity(depth_map: np.ndarray, center_y: int, center_x: int, n: int) -> float:
    """
    Calculate average disparity using nxn neighborhood around the center point using loops.
    """
    if (center_y < (n-1) or center_y >= depth_map.shape[0] - n or
        center_x < (n-1) or center_x >= depth_map.shape[1] - n):
        return depth_map[center_y, center_x]

    sum_disparity = 0.0
    count = 0

    for i in range(-(n-1), n):
        for j in range(-(n-1), n):
            disparity = depth_map[center_y + i, center_x + j]
            if disparity > 0:
                sum_disparity += disparity
                count += 1

    return sum_disparity / count if count > 0 else depth_map[center_y, center_x]

def compute_depth_map_and_detect():
    """Compute the disparity map, detect objects, and estimate distances."""
    global left_frame, right_frame

    while left_frame is None:
        import time
        time.sleep(0.1)  # Wait for the first frame to be received

    frame_height, frame_width = left_frame.shape[:2]

    current_time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    video_filename = f"recorded_stream_{current_time}.mp4"  # Video filename with timestamp
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # More widely supported codec
    out = cv2.VideoWriter(video_filename, fourcc, 20.0, (frame_width, frame_height))

    # Stereo matcher configuration
    left_matcher = cv2.StereoBM_create(numDisparities=16 * 8, blockSize=5)
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(8000.0)
    wls_filter.setSigmaColor(1.5)

    baseline = 0.45  # Distance between cameras in meters
    focal_length = 650  # Camera focal length in mm
    fx = 600  # Focal length in pixels (example value)

    # Define the YOLO class labels (COCO example)
    class_names = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

    while True:
        with frame_lock:
            if left_frame is not None and right_frame is not None:
                # Convert frames to grayscale for disparity computation
                left_gray = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
                right_gray = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

                # Compute disparity maps
                left_disparity = left_matcher.compute(left_gray, right_gray).astype(np.float32) / 16.0
                right_disparity = right_matcher.compute(right_gray, left_gray).astype(np.float32) / 16.0

                # Apply WLS filter to refine the disparity map
                filtered_disparity = wls_filter.filter(left_disparity, left_gray, disparity_map_right=right_disparity)

                # Normalize the disparity for visualization
                disp_vis = cv2.normalize(filtered_disparity, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

                # Detect objects using YOLOv8 on the left frame
                results = model(left_frame)

                # Draw bounding boxes and estimate distances
                for result in results[0].boxes:
                    # Get confidence score and check threshold
                    confidence = float(result.conf)
                    if confidence < 0.65:  # 80% confidence threshold
                        continue

                    # Bounding box and label
                    x1, y1, x2, y2 = map(int, result.xyxy[0].tolist())
                    class_id = int(result.cls)  # Get the class ID
                    label = class_names[class_id]  # Get the class label by ID

                    # Calculate the center of the bounding box
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    # Extract disparity value at the center of the object
                    disparity = calculate_average_disparity(disp_vis, cy, cx, 9)
                    if disparity > 0:  # Avoid division by zero
                        distance = (baseline * fx) / disparity  # Distance in meters

                        # Display the label, distance, and confidence on the frame
                        label_text = f"{label} {distance:.2f}m {confidence:.2%}"
                        cv2.putText(left_frame, label_text,
                                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        cv2.rectangle(left_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Display the left frame with detections
                cv2.imshow("Left Frame with Detections", left_frame)

                # Display the depth map
                cv2.imshow("Depth Map", cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET))

                # Write the annotated frame to the video file
                out.write(left_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    out.release()

def radar_client():
    radar_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    radar_socket.connect((host, 12345))  # Replace with Raspberry Pi's IP
    while True:
        try:
            data = radar_socket.recv(1024)
            if not data:
                break

            # Update the radar display based on received data
            distance, angle = struct.unpack("ff", data[:8])
            print(f"{distance},{angle}")

        except Exception as e:
            print(f"Error: {e}")
            break

    radar_socket.close()

if __name__ == "__main__":
    #host = "192.168.1.213"  # Replace with the server IP
    host = "192.168.137.94"  # Replace with the server IP


    # Start threads to receive streams from left and right cameras
    threading.Thread(target=receive_stream, args=(host, 10050, "left"), daemon=True).start()
    threading.Thread(target=receive_stream, args=(host, 10051, "right"), daemon=True).start()
    threading.Thread(target=radar_client, daemon=True).start()
    # Start the disparity computation, detection, and visualization loop

    compute_depth_map_and_detect()

    # Release the video writer and close all windows
    #out.release()
    cv2.destroyAllWindows()
