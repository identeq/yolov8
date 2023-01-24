"""
-- Created by Pravesh Budhathoki
-- Treeleaf Technologies Pvt. Ltd.
-- Created on 2023-01-24
"""
import random

import cv2

from yolov8.yolov8_detector import YoloV8Detector


def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def detect_and_plot(model, _image):
    # targets = None
    targets = ['car', 'motorcycle', 'truck', 'bus']
    detections = model.detect(_image, class_labels=targets)
    for detection in detections:
        label = f'{detection.label} {detection.confidence:.2f}'
        plot_one_box(detection.bbox, _image, label=label, color=model.colors[detection.class_id], line_thickness=3)
    return _image


def video_example():
    import time
    yoloV8 = YoloV8Detector(model_name="yolov8l.pt")
    url = "https://storage.googleapis.com/identeq/identeq_demo/VID_20220523_085255.mp4"
    cap = cv2.VideoCapture(url)
    assert cap.isOpened(), f'Failed to open {url}'
    fps = cap.get(cv2.CAP_PROP_FPS) % 100
    print("FPS:", fps)
    start_time = time.time()
    processed_frames = 0
    skip_num_frames = 3
    while cap.isOpened():
        if skip_num_frames > 0:
            for i in range(skip_num_frames):
                cap.grab()
        else:
            cap.grab()
        success, image = cap.retrieve()
        if not success:
            break
        img = detect_and_plot(yoloV8, image)
        processed_frames += 1
        end_time = time.time()
        processing_fps = processed_frames / (end_time - start_time)
        cv2.namedWindow("YoloV8 Image", cv2.WINDOW_NORMAL)
        cv2.imshow("YoloV8 Image", img)
        time.sleep(1 / fps)
        print(f"Processing FPS: {processing_fps:.2f}")
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    video_example()
