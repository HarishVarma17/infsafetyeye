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
# API Key
# ================================
API_KEY = "harish17varma632"

# ================================
# Model Setup (Google Drive)
# ================================
MODEL_PATH = "best.pt"

if not os.path.exists(MODEL_PATH):
    gdown.download("https://drive.google.com/uc?id=1qYrTPvJyHg3Rd7zlN0L-xqV7TfB5Nv29", MODEL_PATH, quiet=False)

model = YOLO(MODEL_PATH)

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

    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")

    try:
        # Get video URL
        ydl_opts = {'format': 'best[ext=mp4]', 'quiet': True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            video_url = info['url']

        cap = cv2.VideoCapture(video_url)

        detections = []
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # 🔥 SPEED OPTIMIZATION (IMPORTANT)
            if frame_count % 5 != 0:
                continue

            results = model(frame)

            labels = [model.names[int(c)] for c in results[0].boxes.cls]

            # Store alerts
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
            "detections": detections[:20]   # limit output
        }

    except Exception as e:
        return {"error": str(e)}