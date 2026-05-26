from ultralytics import YOLO

# 1. Load the official pre-trained YOLOv8 segmentation model
# 'n' stands for nano (fastest). You can use 's' (small) or 'm' (medium) later for higher accuracy.
model = YOLO('yolov8n-seg.pt') 

# 2. Run inference on a sample video
# Replace 'sample_dashcam.mp4' with the path to any driving video you have
results = model.predict(source='d2-fypi.mp4', show=True, save=True)

# The model will automatically process the video frame-by-frame, 
# draw the pixel-perfect masks, and save the output.