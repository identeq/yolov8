"""
-- Created by Pravesh Budhathoki
-- Treeleaf Technologies Pvt. Ltd.
-- Created on 2023-01-24
"""
import random
from typing import List

import numpy as np
import torch
from ultralytics.yolo.utils.torch_utils import select_device

from ultralytics import YOLO


class BoundingBox:
    def __init__(self, class_id, label, confidence, bbox, image_width, image_height):
        self.class_id = class_id
        self.label = label
        self.confidence = confidence
        self.bbox = bbox  # t,l,b,r or x1,y1,x2,y2
        self.bbox_normalized = np.array(bbox) / (image_height, image_width, image_height, image_width)
        self.__x1 = bbox[0]
        self.__y1 = bbox[1]
        self.__x2 = bbox[2]
        self.__y2 = bbox[3]
        self.__u1 = self.bbox_normalized[0]
        self.__v1 = self.bbox_normalized[1]
        self.__u2 = self.bbox_normalized[2]
        self.__v2 = self.bbox_normalized[3]

    @property
    def width(self):
        return self.bbox[2] - self.__x1

    @property
    def height(self):
        return self.__y2 - self.__y1

    @property
    def center_absolute(self):
        return 0.5 * (self.__x1 + self.__x2), 0.5 * (self.__y1 + self.__y2)

    @property
    def center_normalized(self):
        return 0.5 * (self.__u1 + self.__u2), 0.5 * (self.__v1 + self.__v2)

    @property
    def size_absolute(self):
        return self.__x2 - self.__x1, self.__y2 - self.__y1

    @property
    def size_normalized(self):
        return self.__u2 - self.__u1, self.__v2 - self.__v1

    def __repr__(self) -> str:
        return f'BoundingBox(class_id: {self.class_id}, label: {self.label}, bbox: {self.bbox}, confidence: {self.confidence:.2f})'


def _postprocess(boxes, scores, classes, labels, img_w, img_h):
    if len(boxes) == 0:
        return boxes

    detected_objects = []
    for box, score, class_id, label in zip(boxes, scores, classes, labels):
        detected_objects.append(BoundingBox(class_id, label, score, box, img_w, img_h))
    return detected_objects


class YoloV8Detector:

    def __init__(self, model_name="yolov8n.pt", img_size=640, device=''):
        self.model = YOLO(model_name)
        self.device = select_device(device=device)
        self.img_size = img_size
        self._id2labels = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self._labels2ids = dict((_label, _id) for _id, _label in self._id2labels.items())
        self.labels = list(self._labels2ids.keys())
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.labels]

    @torch.no_grad()
    def detect(self, img, thresh=0.25, iou_thresh=0.45, classes=None, class_labels=None, agnostic=True):
        if img.shape[0] == 0 or img.shape[1] == 0:
            return []

        if not classes and class_labels:
            classes = self.labels2ids(class_labels)
        pred = self.model.predict(img, conf=thresh, iou=iou_thresh, classes=classes, agnostic_nms=agnostic,
                                  imgsz=self.img_size, device=self.device)[0]
        det = pred.boxes.boxes
        boxes = []
        confidences = []
        class_ids = []
        if len(pred) > 0:
            for *xyxy, conf, cls in reversed(det):
                t, l, b, r = np.array(xyxy).astype(int)
                boxes.append([t, l, b, r])
                confidences.append(float(conf))
                class_ids.append(int(cls))
        else:
            return []
        labels = [self._id2labels[class_id] for class_id in class_ids]
        detections = _postprocess(boxes, confidences, class_ids, labels, img.shape[1], img.shape[0])
        return detections

    def labels2ids(self, labels: List[str]):
        return [self._labels2ids[label] for label in labels]
