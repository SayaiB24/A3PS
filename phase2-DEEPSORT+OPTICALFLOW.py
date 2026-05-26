import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Initialize Models
model = YOLO('yolov8n-seg.pt') 
tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0) 

cap = cv2.VideoCapture('00133.mp4')

# --- OPTICAL FLOW SETUP ---
# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
# Parameters for Shi-Tomasi corner detection (finding good points to track on the car)
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

# Read the first frame to initialize Optical Flow
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 1. YOLOv8 Perception
    results = model(frame, verbose=False)[0]
    detections = []
    vehicle_kinematics = {}

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

    # 3. OPTICAL FLOW KINEMATICS (The missing piece!)
    # 3. OPTICAL FLOW KINEMATICS (Extracting the Numbers)
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
                    
                    # --- THE DATA EXTRACTION ---
                    if len(good_new) > 0:
                        # 1. Calculate the average shift in X and Y for all points on the car
                        avg_dx = np.mean(good_new[:, 0] - good_old[:, 0])
                        avg_dy = np.mean(good_new[:, 1] - good_old[:, 1])
                        
                        # This is your Instantaneous Velocity (in pixels per frame)
                        velocity_x = round(avg_dx, 2)
                        velocity_y = round(avg_dy, 2)
                        
                        # 2. Calculate Acceleration (Change in velocity)
                        acceleration_x = 0.0
                        acceleration_y = 0.0
                        
                        if track_id in vehicle_kinematics:
                            prev_vx, prev_vy = vehicle_kinematics[track_id]
                            acceleration_x = round(velocity_x - prev_vx, 2)
                            acceleration_y = round(velocity_y - prev_vy, 2)
                            
                        # Update the dictionary with current velocity for the next frame's math
                        vehicle_kinematics[track_id] = (velocity_x, velocity_y)

                        # --- PRINTING THE DATA ---
                        # You can print this to the console, or write it to a CSV file later!
                        print(f"Vehicle ID: {track_id} | Vel(x,y): {velocity_x}, {velocity_y} | Accel(x,y): {acceleration_x}, {acceleration_y}")
                        
                        # (Optional) Display the velocity numbers directly on the video
                        cv2.putText(annotated_frame, f"Vx: {velocity_x} Vy: {velocity_y}", 
                        (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow("A3PS Phase I & II: DeepSORT + Optical Flow", annotated_frame)
    old_gray = frame_gray.copy() # Update the previous frame for the next loop
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

'''Here is the math and the exact code snippet to extract 
those numbers.The Math (Pixels per Frame)Optical Flow gives us a list of points on 
the car in the previous frame ($P_{old}$) and where those exact points moved to in the current frame ($P_{new}$).
Instantaneous Velocity ($v$): This is simply the change in position.$v_x = x_{new} - x_{old}$$v_y = y_{new} - y_{old}$Instantaneous Acceleration ($a$):
 This is the change in velocity between the current frame and the last frame.$a = v_{current} - v_{previous}$How to Extract the Numbers in PythonTo get clean data, we will average the movement of all the little points on the car to get one solid "Vehicle Velocity" vector. We will also create a dictionary to remember the car's last speed so we 
                                                                                                                                                                                                                                                                                                                                                                                                             can calculate acceleration.Add this dictionary to the very top of your script (before the while cap.isOpened(): loop):
Important Note on "Real World" SpeedThe numbers printing in your console
will be in Pixels per Frame.For your Phase 3 trajectory prediction model, pixels per frame are perfectly fine
 because the model will learn the spatial relationships based on the camera view. However, if you ever want to display 
a real-world speed on your dashboard (e.g., "$40~km/h$"), you will eventually need to apply a mathematical transformation 
 to convert those pixel shifts into real-world meters, 
 usually by calibrating the camera angle and knowing the frame rate (FPS) of the video.'''