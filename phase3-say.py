import cv2
import numpy as np
import torch # Added for Phase 3 Tensors
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# ---------------------------------------------------------
# PHASE 3: DATA BRIDGE FUNCTION
# ---------------------------------------------------------
def prepare_trajectory_tensors(trajectory_history, history_length=30):
    """Converts the Python dictionary into PyTorch tensors for the Transformer."""
    valid_tracks = []
    
    for track_id, history in trajectory_history.items():
        if len(history) >= history_length:
            recent_history = history[-history_length:]
            valid_tracks.append(recent_history)
            
    if not valid_tracks:
        return None 
        
    # Shape: [Number of Vehicles, History Length, 4 Features]
    input_tensor = torch.tensor(np.array(valid_tracks), dtype=torch.float32)
    return input_tensor

# ---------------------------------------------------------
# INITIALIZATION (PHASE 1 & 2)
# ---------------------------------------------------------
model = YOLO('yolov8n-seg.pt') 
tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0) 

cap = cv2.VideoCapture('Extra-Data\\00318.mp4')

lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# =========================================================
# CRITICAL FIX: Memory Dictionaries moved OUTSIDE the loop
# =========================================================
vehicle_kinematics = {} 
trajectory_history = {} # Stores the rolling 3-second window
MAX_HISTORY_FRAMES = 90 # 90 frames = 3 seconds (assuming 30fps)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 1. YOLOv8 Perception
    results = model(frame, verbose=False)[0]
    detections = []

    if results.boxes is not None:
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            class_id = int(box.cls[0].item())
            if class_id in [0, 1, 2, 3, 5, 7]: 
                w, h = x2 - x1, y2 - y1
                detections.append(([x1, y1, w, h], conf, class_id))

    # 2. DeepSORT Tracking
    tracks = tracker.update_tracks(detections, frame=frame)
    annotated_frame = frame.copy()

    # 3. OPTICAL FLOW KINEMATICS
    for track in tracks:
        if not track.is_confirmed():
            continue
            
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = int(max(0, ltrb[0])), int(max(0, ltrb[1])), int(max(0, ltrb[2])), int(max(0, ltrb[3]))
        
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"ID:{track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        roi_old_gray = old_gray[y1:y2, x1:x2]
        roi_frame_gray = frame_gray[y1:y2, x1:x2]

        if roi_old_gray.size > 0 and roi_frame_gray.size > 0:
            p0 = cv2.goodFeaturesToTrack(roi_old_gray, mask=None, **feature_params)
            
            if p0 is not None:
                p1, st, err = cv2.calcOpticalFlowPyrLK(roi_old_gray, roi_frame_gray, p0, None, **lk_params)
                
                if p1 is not None:
                    good_new = p1[st == 1]
                    good_old = p0[st == 1]
                    
                    if len(good_new) > 0:
                        avg_dx = np.mean(good_new[:, 0] - good_old[:, 0])
                        avg_dy = np.mean(good_new[:, 1] - good_old[:, 1])
                        
                        velocity_x = round(avg_dx, 2)
                        velocity_y = round(avg_dy, 2)
                        
                        acceleration_x = 0.0
                        acceleration_y = 0.0
                        
                        if track_id in vehicle_kinematics:
                            prev_vx, prev_vy = vehicle_kinematics[track_id]
                            acceleration_x = round(velocity_x - prev_vx, 2)
                            acceleration_y = round(velocity_y - prev_vy, 2)
                            
                        vehicle_kinematics[track_id] = (velocity_x, velocity_y)

                        # =========================================================
                        # PHASE 3 PREPARATION: Storing the Trajectory
                        # =========================================================
                        if track_id not in trajectory_history:
                            trajectory_history[track_id] = []
                            
                        # Store (x, y, vx, vy)
                        current_state = (x1, y1, velocity_x, velocity_y)
                        trajectory_history[track_id].append(current_state)
                        
                        # Keep only the last 90 frames
                        if len(trajectory_history[track_id]) > MAX_HISTORY_FRAMES:
                            trajectory_history[track_id].pop(0)

                        cv2.putText(annotated_frame, f"Vx: {velocity_x} Vy: {velocity_y}", 
                        (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # =========================================================
    # PHASE 3 EXECUTION: Bridging the data to PyTorch Tensors
    # =========================================================
    # We call this every frame. It checks if any car has been tracked for at least 30 frames.
    past_trajectories_tensor = prepare_trajectory_tensors(trajectory_history, history_length=30)
    
    if past_trajectories_tensor is not None:
        # This will print once a car has been on screen long enough!
        print(f"Phase 3 Data Ready! Tensor Shape: {past_trajectories_tensor.shape}")
        # FUTURE STEP: predictions = model_transformer(past_trajectories_tensor)

    cv2.imshow("A3PS Phase I, II & III Bridge", annotated_frame)
    old_gray = frame_gray.copy() 
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()