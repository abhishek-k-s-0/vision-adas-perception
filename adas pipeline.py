import cv2
import numpy as np
from ultralytics import YOLO

class UnifiedAdasPipeline:
    def __init__(self, model_path="yolov8n.pt"):
        # Load lightweight pre-trained AI weights
        self.model = YOLO(model_path)
        
        # Camera Calibration constants for Monocular Distance Estimation
        self.KNOWN_CAR_WIDTH = 1.8  # Average car width in meters
        self.FOCAL_LENGTH = 700     # Calibrated focal length in pixels

    def process_lane_tracking(self, frame):
        """Isolates road surface and tracks lane markers using Hough Lines."""
        height, width = frame.shape[:2]
        
        # 1. Convert to grayscale and blur to remove image noise
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 2. Canny Edge Detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # 3. Mask Region of Interest (ROI) to isolate only the lower road triangle
        mask = np.zeros_like(edges)
        roi_vertices = np.array([[
            (int(width * 0.1), height),
            (int(width * 0.45), int(height * 0.6)),
            (int(width * 0.55), int(height * 0.6)),
            (int(width * 0.9), height)
        ]], dtype=np.int32)
        cv2.fillPoly(mask, roi_vertices, 255)
        masked_edges = cv2.bitwise_and(edges, mask)
        
        # 4. Hough Line Transform to convert edge pixels into real geometric lines
        lines = cv2.HoughLinesP(masked_edges, rho=1, theta=np.pi/180, threshold=20, minLineLength=20, maxLineGap=300)
        
        # Create a transparent overlay layer to render lane indicators cleanly
        lane_overlay = np.zeros_like(frame)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(lane_overlay, (x1, y1), (x2, y2), (0, 255, 255), 5) # Yellow lanes
                
        return cv2.addWeighted(frame, 1.0, lane_overlay, 0.6, 0)

    def run_pipeline(self, frame):
        """Processes AI object detection, spatial math tracking, and lane systems concurrently."""
        # Step A: Run Lane Tracking Processing first
        output_frame = self.process_lane_tracking(frame)
        
        # Step B: Run AI object detection tracking on the frame
        # Classes: 0=pedestrian, 2=car, 5=bus, 7=truck
        results = self.model(frame, classes=[0, 2, 5, 7], verbose=False)[0]
        
        for box in results.boxes:
            # Extract box boundary coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            
            # Calculate pixel metrics for geometry computation
            pixel_width = x2 - x1
            
            # Step C: Monocular Distance Calculation (D = (W * F) / w)
            distance_meters = (self.KNOWN_CAR_WIDTH * self.FOCAL_LENGTH) / max(1, pixel_width)
            
            # Determine proximity hazard severity (ADAS Warning Lock logic)
            if distance_meters < 4.0:
                box_color = (0, 0, 255)       # Red: Immediate collision threat
                label_color = (0, 0, 255)
                status_text = "BRAKE CRITICAL"
            elif distance_meters < 10.0:
                box_color = (0, 165, 255)     # Orange: Approaching warning zone
                label_color = (0, 165, 255)
                status_text = "TAILGATING"
            else:
                box_color = (0, 255, 0)       # Green: Safe tracking distance
                label_color = (0, 255, 0)
                status_text = "TRACKING"

            # Draw AI tracking UI overlays
            label = f"{results.names[cls_id].upper()} | {distance_meters:.1f}m | {status_text}"
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(output_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 2)
            
        return output_frame

# --- Execution Entrypoint ---
if __name__ == "__main__":
    pipeline = UnifiedAdasPipeline()
    
    # Target local video path
    video_target = "sample.mp4"
    cap = cv2.VideoCapture(video_target)
    
    # Fallback Mechanism: If your file isn't found, load a live video sample over the network
    if not cap.isOpened():
        print(f"\n[!] Notice: '{video_target}' not found in your directory.")
        print("Switching to online public sample video stream for testing...")
        # Live cloud-hosted public test file
        online_stream = "https://github.com/intel-iot-devkit/sample-videos/raw/master/driver-戲-action-detection.mp4"
        cap = cv2.VideoCapture(online_stream)
        
    if not cap.isOpened():
        print("[ERROR] Unstable network or missing video player drivers. Cannot start process stream.")
        exit()
        
    print("\nLaunching ADAS perception pipeline... Press 'Q' inside the pop-up window to exit.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Processing completed or end of video stream reached.")
            break
            
        # Downscale frame size safely so computer processor doesn't lag
        frame = cv2.resize(frame, (854, 480))
        
        # Process through full pipeline architecture
        processed_view = pipeline.run_pipeline(frame)
        
        # Render visual display box
        cv2.imshow("UNIFIED AI & COMPUTER VISION ADAS PASSTHROUGH", processed_view)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    print("Pipeline shut down successfully.")