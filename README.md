# Unified AI & Computer Vision ADAS Perception Pipeline 🚗🛡️

A hybrid, real-time Advanced Driver Assistance System (ADAS) written in Python. This framework concurrently runs convolutional neural networks for spatial object detection alongside traditional geometric image filters to trace driving lanes and calculate vehicular proximity hazards.

By combining **Deep Learning (AI)** with **Mathematical Geometry (Focal Length Proportions)**, the system achieves monocular depth estimation—calculating precise distances to obstacles in front of the vehicle using a single, standard camera feed without relying on expensive dual-lens (stereo) hardware or active LiDAR sensors.

---

## 🚀 System Architecture & Core Logic

The pipeline processes input frames through two independent computer vision paths before rendering a unified telemetry overlay:

### 1. Traditional Geometry Lane Tracking
*   **Noise Reduction:** Passes frames through a `5x5 Gaussian Blur` kernel to filter out high-frequency road noise, asphalt variations, and rain artifacts.
*   **Edge Isolation:** Executes a structural `Canny Edge Detection` algorithm to isolate stark pixel gradients.
*   **Spatial Masking:** Restricts pixel processing to a dynamic **Region of Interest (ROI)** triangular polygon mapping only the lower lane horizon.
*   **Vectorization:** Runs a `Probabilistic Hough Line Transform` ($cv2.HoughLinesP$) to compute raw edge segments into permanent lane boundaries.

### 2. Deep Learning Object Tracking & Telemetry
*   **Object Tracking:** Leverages a lightweight, pre-trained `YOLOv8n` detector isolating specific transport class filters: Pedestrians, Cars, Buses, and Trucks.
*   **Monocular Distance Calculation:** Computes true spatial distances dynamically using the camera lens intercept equation:
    $$D = \frac{W \times F}{w}$$
    Where $W$ represents the average global vehicle width ($1.8\text{ meters}$), $F$ is the calibrated camera focal length ($700\text{ pixels}$), and $w$ represents the dynamic bounding box pixel width detected by YOLO.
*   **Hazard Warning Lock:** Evaluates calculated proximity metrics to fire warning triggers:
    *   🔴 `BRAKE CRITICAL` (Distance less than 4.0 meters)
    *   🟠 `TAILGATING` (Distance between 4.0 and 10.0 meters)
    *   🟢 `TRACKING SAFE` (Distance greater than 10.0 meters)

---

## 📂 Repository Layout

```text
├── adas_pipeline.py       # Core tracking and processing script
├── requirements.txt       # System library dependencies
└── README.md              # Detailed documentation
