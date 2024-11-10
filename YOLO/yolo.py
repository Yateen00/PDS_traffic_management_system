import cv2
import numpy as np
import time
from collections import deque

# Add these constants at the top after imports
DISTANCE_THRESHOLD = 50  # Maximum distance to consider as the same vehicle
MOVEMENT_THRESHOLD = 10  # Minimum movement to consider vehicle as "moving"
FPS = 15  # Assuming 30 fps, adjust based on your video

# Initialize variables for tracking
vehicle_count = 0
vehicle_positions = {}  # Store previous positions
vehicle_waiting_times = {}
waiting_time_history = {}
trackers = []

# Initialize video capture
cap = cv2.VideoCapture(
    "vid.mp4"
)  # Replace with your video file path or use 0 for webcam
output_width = 1280  # Set your desired output width
output_height = 720  # Set your desired output height

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_writer = cv2.VideoWriter("output.mp4", fourcc, FPS, (output_width, output_height))

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]


def calculate_center(box):
    x, y, w, h = box
    return (x + w // 2, y + h // 2)


def get_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame to increase resolution
    frame = cv2.resize(frame, (output_width, output_height))
    height, width, channels = frame.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(
        frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False
    )
    net.setInput(blob)
    outs = net.forward(output_layers)

    boxes = []
    confidences = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Store current detections for matching
    current_detections = []
    for i in range(len(boxes)):
        if i in indexes:
            box = boxes[i]
            current_detections.append(box)

    # Match current detections with existing vehicles
    matched_vehicles = set()
    new_positions = {}

    for box in current_detections:
        current_center = calculate_center(box)

        # Try to match with existing vehicles
        best_match = None
        min_distance = float("inf")

        for vehicle_id, prev_center in vehicle_positions.items():
            distance = get_distance(current_center, prev_center)
            if distance < DISTANCE_THRESHOLD and distance < min_distance:
                min_distance = distance
                best_match = vehicle_id

        if best_match is not None:
            # Update existing vehicle
            matched_vehicles.add(best_match)
            new_positions[best_match] = current_center

            # Calculate if vehicle is moving
            prev_center = vehicle_positions[best_match]
            movement = get_distance(current_center, prev_center)

            # Update waiting time based on movement
            if movement < MOVEMENT_THRESHOLD:
                vehicle_waiting_times[best_match] = vehicle_waiting_times.get(
                    best_match, 0
                ) + (1.0 / FPS)  # Convert frames to seconds
            else:
                vehicle_waiting_times[best_match] = 0

            # Draw bounding box and waiting time
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            wait_time = int(vehicle_waiting_times[best_match])
            if wait_time > 0:
                cv2.putText(
                    frame,
                    f"Waiting: {wait_time}s",
                    (x, y + h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )
        else:
            # New vehicle detected
            new_id = f"vehicle_{len(vehicle_positions)}"
            new_positions[new_id] = current_center
            vehicle_waiting_times[new_id] = 0

            # Draw bounding box for new vehicle
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Remove vehicles that are no longer visible
    vehicle_positions = new_positions

    # Update vehicle count
    vehicle_count = len(vehicle_positions)
    cv2.putText(
        frame,
        f"Vehicles: {vehicle_count}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
    )

    video_writer.write(frame)
    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()
