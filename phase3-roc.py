import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# --- INITIALIZE MODELS ---
# Using yolov8n.pt instead of -seg unless you are explicitly extracting polygon masks.
# It is much faster for high-FPS dashcam processing.
model = YOLO('yolov8n.pt') 
tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0) 

cap = cv2.VideoCapture('Extra-Data\\00318.mp4')

# --- OPTICAL FLOW SETUP ---
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
feature_params = dict(maxCorners=50, qualityLevel=0.3, minDistance=7, blockSize=7)

# Read the first frame to initialize
ret, old_frame = cap.read()
if not ret:
    print("Failed to open video source.")
    cap.release()
    exit()

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# --- GLOBAL DATA PERSISTENCE (FIXED) ---
vehicle_kinematics = {}  # Keeps track of velocities across frames

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    annotated_frame = frame.copy()

    # 1. YOLOv8 Perception (Optimized classes, inference size, and confidence threshold)
    results = model(frame, verbose=False, conf=0.4, classes=[0, 1, 2, 3, 5, 7])[0]
    detections = []

    if results.boxes is not None:
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            class_id = int(box.cls[0].item())
            
            w, h = x2 - x1, y2 - y1
            detections.append(([x1, y1, w, h], conf, class_id))

    # 2. DeepSORT Tracking
    tracks = tracker.update_tracks(detections, frame=frame)

    # 3. OPTICAL FLOW KINEMATICS
    for track in tracks:
        if not track.is_confirmed():
            continue
            
        track_id = track.track_id
        ltrb = track.to_ltrb()
        
        # Ensure coordinates stay within frame boundaries
        h_img, w_img = frame.shape[:2]
        x1, y1 = max(0, int(ltrb[0])), max(0, int(ltrb[1]))
        x2, y2 = min(w_img, int(ltrb[2])), min(h_img, int(ltrb[3]))
        
        # Draw bounding boxes and ID
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"ID:{track_id}", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Extract corresponding ROIs from previous and current frames
        roi_old_gray = old_gray[y1:y2, x1:x2]
        roi_frame_gray = frame_gray[y1:y2, x1:x2]

        if roi_old_gray.size > 0 and roi_frame_gray.size > 0:
            p0 = cv2.goodFeaturesToTrack(roi_old_gray, mask=None, **feature_params)
            
            if p0 is not None:
                p1, st, err = cv2.calcOpticalFlowPyrLK(roi_old_gray, roi_frame_gray, p0, None, **lk_params)
                
                if p1 is not None and len(st) > 0:
                    good_new = p1[st == 1]
                    good_old = p0[st == 1]
                    
                    if len(good_new) > 0:
                        # Calculate average displacement vectors
                        avg_dx = np.mean(good_new[:, 0] - good_old[:, 0])
                        avg_dy = np.mean(good_new[:, 1] - good_old[:, 1])
                        
                        velocity_x = round(avg_dx, 2)
                        velocity_y = round(avg_dy, 2)
                        
                        acceleration_x = 0.0
                        acceleration_y = 0.0
                        
                        # Kinematics calculations now persist across frames
                        if track_id in vehicle_kinematics:
                            prev_vx, prev_vy = vehicle_kinematics[track_id]
                            acceleration_x = round(velocity_x - prev_vx, 2)
                            acceleration_y = round(velocity_y - prev_vy, 2)
                            
                        vehicle_kinematics[track_id] = (velocity_x, velocity_y)

                        # Output data metrics
                        print(f"Vehicle ID: {track_id} | Vel: ({velocity_x}, {velocity_y}) | Accel: ({acceleration_x}, {acceleration_y})")
                        
                        # Overlay metrics onto the output display
                        cv2.putText(annotated_frame, f"Vx:{velocity_x} Ax:{acceleration_x}", 
                                    (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    cv2.imshow("A3PS Phase I & II: DeepSORT + Optical Flow", annotated_frame)
    old_gray = frame_gray.copy()  # Update the base reference frame
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()