from ultralytics import YOLO
from roboflow import Roboflow
import torch 
import numpy as np

torch.device(0)

model = YOLO("weights/best.pt")
model.to('cpu')
results = model("c:/Users/kukun/Downloads/HfapG-ZhjC8.jpg")

ruslt_np = results[0].numpy()

print(ruslt_np.orig_img)
"""
#установка дата сета
rf = Roboflow(api_key="7lPCpGpEFFZwytwZG6vt")
project = rf.workspace("vgtu-n8zmy").project("military-vehicle")
version = project.version(4)
dataset = version.download("yolov11")

model = YOLO("yolo11m.pt")

model.train(data="military-vehicle-4/data.yaml", imgsz = 640, batch = 16, epochs = 10, device = "cpu")

# Evaluate model performance on the validation set
metrics = model.val()

# Perform object detection on an image

results[0].show()

# Export the model to ONNX format
path = model.export(format="onnx")  # return path to exported model
"""