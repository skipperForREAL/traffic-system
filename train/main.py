from ultralytics import YOLO

model = YOLO("/my_model.pt")
results = model("'/Users/shakirasalim/Downloads/youtube-downloads/Screenshot 2026-06-04 at 10.19.18.png'")
results[0].show()
