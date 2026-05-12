"""
Crowd Detection — simple OpenCV window
Press Q to quit.
"""

import cv2
from ultralytics import YOLO

RTSP_URL = "rtsp://192.168.1.16:5543/live/channel0"
CONF     = 0.40

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Create window at actual frame size (no scaling)
cv2.namedWindow("Crowd Detection  |  Q = quit", cv2.WINDOW_AUTOSIZE)

while True:
    # drain buffer → always latest frame
    cap.grab()
    ret, frame = cap.retrieve()

    if not ret:
        print("Stream lost, reconnecting...")
        cap.release()
        cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        continue

    results = model.predict(frame, conf=CONF, classes=[0], verbose=False)[0]
    count   = len(results.boxes)

    # draw boxes
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # big count overlay
    cv2.rectangle(frame, (0, 0), (260, 60), (0, 0, 0), -1)
    cv2.putText(frame, f"Persons: {count}", (10, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 3)

    cv2.imshow("Crowd Detection  |  Q = quit", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
