import supervision as sv
from autodistill.detection import DetectionBaseModel
from ultralytics import YOLO

class YOLOv6(DetectionBaseModel): 
    def __init__(self, model_name):
        self.yolo = YOLO(model_name)

    def predict(self, input: str, confidence: int = 0.5) -> sv.Detections:
        return self.yolo(input, conf=confidence)

    def train(self, dataset_yaml, epochs=300)
        self.yolo.train(dataset_yaml, epochs=epochs)
