from ultralytics import YOLO
from roboflow import Roboflow

"""
#установка дата сета
rf = Roboflow(api_key="7lPCpGpEFFZwytwZG6vt")
project = rf.workspace("vgtu-n8zmy").project("military-vehicle")
version = project.version(4)
dataset = version.download("yolov11")
"""
model = YOLO("yolo11m.pt")

model.train(data="military-vehicle-4/data.yaml", imgsz = 640, batch = 16, epochs = 10, device = "cpu")

# Evaluate model performance on the validation set
metrics = model.val()

# Perform object detection on an image
results = model("C:/Users/kukun/Desktop/диплом/military-vehicle-4/test/images/5_jpg.rf.2cb0f27a60274f89284f474f01932dc6.jpg")
results[0].show()

# Export the model to ONNX format
path = model.export(format="onnx")  # return path to exported model