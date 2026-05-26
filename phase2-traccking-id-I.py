#ONLY DEEPSORT Tracking
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# ---------------------------------------------------------
# 1. INITIALIZE MODELS (PHASE I & PHASE II)
# ---------------------------------------------------------
# Load Phase 1: High-Fidelity Perception (YOLOv8-Seg)
model = YOLO('yolov8n-seg.pt') 

# Load Phase 2: Temporal Tracking (DeepSORT)
# max_age: How many frames to remember an object if it gets hidden (Occlusion fallback)
tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0) 

# Open your video file (Replace with your actual video path)
video_path = 'd1-fyp.mp4'
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # ---------------------------------------------------------
    # PHASE I: PERCEPTION (Get masks and boxes)
    # ---------------------------------------------------------
    # Run YOLOv8-Seg on the current frame
    results = model(frame, verbose=False)[0]
    
    # We need to format YOLO's output into the exact format DeepSORT expects:
    # DeepSORT Format: [ [ [left, top, width, height], confidence, class_id ], ... ]
    detections = []
    
    if results.boxes is not None:
        for box in results.boxes:
            # Extract coordinates, confidence, and class
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            class_id = int(box.cls[0].item())
            
            # Filter: We only care about dynamic actors (0: person, 2: car, 3: motorcycle, etc.)
            if class_id in [0, 1, 2, 3, 5, 7]: 
                # Convert (x1, y1, x2, y2) to (x, y, width, height)
                w = x2 - x1
                h = y2 - y1
                detections.append(([x1, y1, w, h], conf, class_id))

    # ---------------------------------------------------------
    # PHASE II: TRACKING (Assign persistent IDs)
    # ---------------------------------------------------------
    # Update the tracker with the new detections
    # DeepSORT extracts appearance features and updates the Kalman filter here
    tracks = tracker.update_tracks(detections, frame=frame)
    
    # ---------------------------------------------------------
    # VISUALIZATION (Draw results on screen)
    # ---------------------------------------------------------
    # First, let YOLO draw its pixel-perfect masks on the frame
    annotated_frame = results.plot(boxes=False) # boxes=False so DeepSORT can draw the tracked boxes
    
    # Next, draw the DeepSORT bounding boxes and permanent Track IDs
    for track in tracks:
        if not track.is_confirmed():
            continue
            
        track_id = track.track_id
        ltrb = track.to_ltrb() # left, top, right, bottom coordinates
        x1, y1, x2, y2 = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
        
        # Draw the tracking box and the ID label (e.g., "ID: 4")
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"ID: {track_id}", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show the final combined output
    cv2.imshow("A3PS Phase I & II: Segmentation + Tracking", annotated_frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()