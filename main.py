# ================================
# Imports
# ================================
from fastapi import FastAPI, HTTPException, Header
from ultralytics import YOLO
import cv2
import yt_dlp
import os
import gdown

# ================================
# FastAPI App
# ================================
app = FastAPI()

# ================================
# API Key (can change later)
# ================================
API_KEY = os.getenv("API_KEY", "harish17varma632")

# ================================
# Model Setup (Google Drive)
# ================================
MODEL_PATH = "best.pt"

if not os.path.exists(MODEL_PATH):
    print("⬇️ Downloading model...")
    gdown.download(
        "https://drive.google.com/uc?id=1qYrTPvJyHg3Rd7zlN0L-xqV7TfB5Nv29",
        MODEL_PATH,
        quiet=False
    )

model = YOLO(MODEL_PATH)
print("✅ Model Loaded Successfully")

# ================================
# Home Route
# ================================
@app.get("/")
def home():
    return {"message": "YOLO API Running 🚀"}

# ================================
# Detection API
# ================================
@app.post("/detect/")
def detect(youtube_url: str, x_api_key: str = Header(None)):

    # 🔐 API Key Check
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")

    try:
        # ================================
        # Get Direct Video URL
        # ================================
        ydl_opts = {'format': 'best[ext=mp4]', 'quiet': True}

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            video_url = info['url']

        cap = cv2.VideoCapture(video_url)

        detections = []
        frame_count = 0

        # ================================
        # Process Video Frames
        # ================================
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # 🔥 Limit frames (Render safety)
            if frame_count > 300:
                break

            # 🔥 Skip frames for speed
            if frame_count % 5 != 0:
                continue

            results = model(frame)

            # Safe label extraction
            if results[0].boxes is not None:
                labels = [model.names[int(c)] for c in results[0].boxes.cls]
            else:
                labels = []

            alerts = []

            if "NO-Hardhat" in labels:
                alerts.append("No Helmet")

            if "NO-Mask" in labels:
                alerts.append("No Mask")

            if "NO-Safety Vest" in labels:
                alerts.append("No Vest")

            if alerts:
                detections.append({
                    "frame": frame_count,
                    "alerts": alerts
                })

        cap.release()

        return {
            "status": "success",
            "total_frames_checked": frame_count,
            "detections": detections[:20]
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}