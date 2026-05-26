import cv2
import numpy as np
import time  # For calculating real-time FPS
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# --- INITIALIZE MODELS ---
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
h_img, w_img = old_gray.shape[:2]

# --- GLOBAL DATA PERSISTENCE ---
vehicle_kinematics = {}  # Keeps track of velocities across frames

# --- SEPARATED CLASS-BASED DISPLAY ID MAPPING ---
deepsort_to_display_label = {}  # Maps raw DeepSORT ID string to labeled string
next_vehicle_id = 1             # Counter for vehicles
next_pedestrian_id = 1          # Counter for pedestrians

# --- FPS COUNTER VARIABLES ---
prev_time = 0  # Timestamp of the previous frame calculated

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    annotated_frame = frame.copy()

    # --- CALCULATE REAL-TIME FPS ---
    current_time = time.time()
    # Handle the very first frame boundary edge case cleanly
    if prev_time != 0:
        fps = 1 / (current_time - prev_time)
    else:
        fps = 0.0
    prev_time = current_time

    # 1. YOLOv8 Perception
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
    active_current_ids = set()

    # 3. OPTICAL FLOW KINEMATICS & TRACK FILTERING
    for track in tracks:
        if not track.is_confirmed():
            continue
            
        track_id = track.track_id
        active_current_ids.add(track_id)
        det_class = track.get_det_class()
        
        if det_class is None:
            continue

        ltrb = track.to_ltrb()
        raw_x1, raw_y1, raw_x2, raw_y2 = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
        track_w = raw_x2 - raw_x1
        track_h = raw_y2 - raw_y1

        # --- SMART BOUNDARY CULLING ---
        is_too_far_left  = (raw_x2 <= 15)
        is_too_far_right = (raw_x1 >= w_img - 15)
        is_too_far_top   = (raw_y2 <= 15)
        is_too_far_bot   = (raw_y1 >= h_img - 15)
        
        is_collapsed_at_edge = (
            (raw_x1 <= 2 and track_w < 20) or 
            (raw_x2 >= w_img - 2 and track_w < 20) or 
            (raw_y1 <= 2 and track_h < 20) or 
            (raw_y2 >= h_img - 2 and track_h < 20)
        )

        if is_too_far_left or is_too_far_right or is_too_far_top or is_too_far_bot or is_collapsed_at_edge:
            if track_id in vehicle_kinematics:
                del vehicle_kinematics[track_id]
            if track_id in deepsort_to_display_label:
                del deepsort_to_display_label[track_id]
            continue

        # --- DYNAMIC CLASS-SPECIFIC LABELLING ---
        if track_id not in deepsort_to_display_label:
            if det_class == 0:
                deepsort_to_display_label[track_id] = f"P{next_pedestrian_id}"
                next_pedestrian_id += 1
            else:
                deepsort_to_display_label[track_id] = f"V{next_vehicle_id}"
                next_vehicle_id += 1
            
        display_label = deepsort_to_display_label[track_id]

        x1, y1 = max(0, raw_x1), max(0, raw_y1)
        x2, y2 = min(w_img, raw_x2), min(h_img, raw_y2)
        
        if (x2 - x1) < 5 or (y2 - y1) < 5:
            continue
        
        box_color = (255, 0, 0) if "P" in display_label else (0, 255, 0)
        
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)
        cv2.putText(annotated_frame, f"ID:{display_label}", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

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
                        avg_dx = np.mean(good_new[:, 0] - good_old[:, 0])
                        avg_dy = np.mean(good_new[:, 1] - good_old[:, 1])
                        
                        velocity_x = float(avg_dx)
                        velocity_y = float(avg_dy)
                        
                        acceleration_x = 0.0
                        acceleration_y = 0.0
                        
                        if track_id in vehicle_kinematics:
                            prev_vx, prev_vy = vehicle_kinematics[track_id]
                            acceleration_x = velocity_x - prev_vx
                            acceleration_y = velocity_y - prev_vy
                            
                        vehicle_kinematics[track_id] = (velocity_x, velocity_y)

                        print(f"Target Label: {display_label} (Internal:{track_id}) | Vel: ({velocity_x:.2f}, {velocity_y:.2f}) | Accel: ({acceleration_x:.2f}, {acceleration_y:.2f})")
                        
                        metrics_text = f"V:({velocity_x:.2f}, {velocity_y:.2f}) A:({acceleration_x:.2f}, {acceleration_y:.2f})"
                        cv2.putText(annotated_frame, metrics_text, (x1, y2 + 15), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    # --- DRAW THE FPS MONITOR OVERLAY ---
    # Placed in the upper-left corner with a semi-transparent background plate for readability
    fps_text = f"System Performance: {fps:.1f} FPS"
    cv2.rectangle(annotated_frame, (10, 10), (280, 45), (0, 0, 0), -1)  # Dark background block
    cv2.putText(annotated_frame, fps_text, (20, 33), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)  # High-visibility Yellow/Cyan text

    cv2.imshow("A3PS Phase I & II: DeepSORT + Optical Flow", annotated_frame)
    old_gray = frame_gray.copy()
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()