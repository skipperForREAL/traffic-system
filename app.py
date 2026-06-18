import cv2
import threading
import time
from collections import deque
from flask import Flask, Response, jsonify, render_template, request
from ultralytics import YOLO

app = Flask(__name__)


MODEL_PATH = "yolo11s.pt"
VIDEO_PATH = "traffic.mp4"
JAM_THRESHOLD = 25
HEAVY_THRESHOLD = 18
SAMPLE_EVERY_N_FRAMES = 3
SMOOTHING_WINDOW = 20


VEHICLE_CLASSES = {
    0: "car",
    1: "daladala",
    2: "motorcycle",
    3: "person",
    4: "public-bus",
    5: "truck",
    6: "tuk tuk",
}


state_lock = threading.Lock()
state = {
    "vehicle_count": 0,
    "class_counts": {},
    "density": "LOW",
    "recommendation": "Route is clear — safe to proceed.",
    "confidence": 0.0,
    "fps": 0.0,
    "frame_number": 0,
    "total_frames": 0,
    "history": deque(maxlen=SMOOTHING_WINDOW),
}

model = None
video_thread = None

def classify_density(count):
    if count >= JAM_THRESHOLD:
        return "JAM", "🚨 Heavy traffic jam detected — take an alternate route!"
    elif count >= HEAVY_THRESHOLD:
        return "HEAVY", "⚠️ Traffic is heavy — consider an alternate route."
    elif count >= 10:
        return "MODERATE", "🟡 Moderate traffic — proceed with caution."
    else:
        return "LOW", "✅ Route is clear — safe to proceed."


def smooth_count(history):
    if not history:
        return 0
    return round(sum(history) / len(history))


def process_video(video_path):
    global state

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0
    prev_time = time.time()

    with state_lock:
        state["total_frames"] = total

    while True:
        ret, frame = cap.read()
        if not ret:
            # Loop video
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_idx = 0
            continue

        frame_idx += 1

        # Skip frames for speed
        if frame_idx % SAMPLE_EVERY_N_FRAMES != 0:
            continue

        # Run YOLO inference
        results = model(frame, verbose=False)[0]

        # Count vehicles by class
        class_counts = {}
        total_vehicles = 0

        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id in VEHICLE_CLASSES:
                label = VEHICLE_CLASSES[cls_id]
                class_counts[label] = class_counts.get(label, 0) + 1
                total_vehicles += 1

        # Smooth count
        with state_lock:
            state["history"].append(total_vehicles)
            smoothed = smooth_count(state["history"])
            density, recommendation = classify_density(smoothed)

            # FPS
            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            state.update({
                "vehicle_count": smoothed,
                "class_counts": class_counts,
                "density": density,
                "recommendation": recommendation,
                "fps": round(fps, 1),
                "frame_number": frame_idx,
            })

    cap.release()


# ─────────────────────────────────────────
#  ANNOTATED VIDEO STREAM (MJPEG)
# ─────────────────────────────────────────
def generate_frames():
    cap = cv2.VideoCapture(VIDEO_PATH)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame_idx += 1
        if frame_idx % SAMPLE_EVERY_N_FRAMES != 0:
            continue

        results = model(frame, verbose=False)[0]
        annotated = results.plot()


        with state_lock:
            density = state["density"].strip()
            count = state["vehicle_count"]

        colors = {"JAM": (0, 0, 220),
                  "HEAVY": (0, 120, 255),
                  "MODERATE": (0, 200, 255),
                  "LOW": (0, 200, 80)}
        color = colors.get(density, (200, 200, 200))

        cv2.rectangle(annotated, (10, 10), (280, 60), color, -1)
        cv2.putText(annotated, f"DENSITY: {density}  ({count} vehicles)",
                    (18, 43), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)

        _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
        frame_bytes = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

    cap.release()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/status")
def api_status():
    with state_lock:
        return jsonify({
            "vehicle_count": state["vehicle_count"],
            "class_counts": state["class_counts"],
            "density": state["density"],
            "recommendation": state["recommendation"],
            "fps": state["fps"],
            "frame_number": state["frame_number"],
            "total_frames": state["total_frames"],
        })


@app.route("/api/config", methods=["POST"])
def update_config():
    global JAM_THRESHOLD, HEAVY_THRESHOLD
    data = request.json
    if "jam_threshold" in data:
        JAM_THRESHOLD = int(data["jam_threshold"])
    if "heavy_threshold" in data:
        HEAVY_THRESHOLD = int(data["heavy_threshold"])
    return jsonify({"status": "updated", "jam": JAM_THRESHOLD, "heavy": HEAVY_THRESHOLD})


if __name__ == "__main__":
    print("[INFO] Loading YOLO11 model...")
    model = YOLO(MODEL_PATH)

    print(f"[INFO] Starting video processor on: {VIDEO_PATH}")
    t = threading.Thread(target=process_video, args=(VIDEO_PATH,), daemon=True)
    t.start()

    print("[INFO] Dashboard running at http://localhost:5511")
    app.run(host="0.0.0.0", port=5511, debug=False, threaded=True)