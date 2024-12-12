import time
import cv2
import torch
import platform
import pathlib
import numpy as np

class DetectionResult:
    def __init__(self, path, box, confidence, class_id):
        self.path = path
        self.box = box
        self.confidence = confidence
        self.class_id = class_id

    def box_center(self):
        x, y, w, h = self.box
        center_x = x + w / 2
        center_y = y + h / 2
        return center_x, center_y

    def box_bottom_center(self):
        x, y, w, h = self.box
        bottom_center_x = x + w / 2
        bottom_center_y = y + h
        return bottom_center_x, bottom_center_y
    
    def size(self):
        x, y, w, h = self.box
        return w, h
    
    def distance(self):
        return self.box
    
    def angle(self):
        return self.box
    
    def distance_other(self, other):
        delta_x = self.box[0] - other.box[0]
        delta_y = self.box[1] - other.box[1]
        distance = delta_x ** 2 + delta_y ** 2
        return distance

    def __repr__(self):
        return f"DetectionResult(path={self.path}, box={self.box}, confidence={self.confidence}, class_id={self.class_id})"

# Check the operating system and set the appropriate path type
if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath
else:
    pathlib.WindowsPath = pathlib.PosixPath

def load_yolo_v5_tiny_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5n')  # yolov5n: Tiny 모델
    return model

def load_trained_yolo_model():
    model = torch.hub.load("ultralytics/yolov5", "custom", path="wieght.pt")
    return model

def main():
    # YOLO 모델 로드 (커스텀 모델 또는 Tiny 모델)
    model = load_trained_yolo_model()
    # model = load_yolo_v5_tiny_model()

    # 동영상 파일 경로 지정
    video_path = "asset/example.mp4"  # 동영상 파일 경로
    cap = cv2.VideoCapture(video_path)

    # 동영상이 열리지 않으면 종료
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # 동영상 프레임을 순차적으로 읽기
    while cap.isOpened():
        ret, frame = cap.read()  # 프레임 읽기
        if not ret:
            print("Video has ended or cannot be read.")
            break

        # 추론 시간 측정 시작
        start_time = time.time()

        # YOLO 모델로 추론
        results = model(frame)

        # 추론 시간 측정 끝
        end_time = time.time()
        inference_time = end_time - start_time
        print(f"Inference time: {inference_time:.4f} seconds")

        # 탐지 결과 표시
        results.render()  # 탐지된 결과를 프레임에 그리기
        output_frame = results.ims[0]  # 렌더링된 프레임 가져오기

        detection_results = []
        for *box, confidence, class_id in results.xyxy[0].tolist():
            # xyxy: [x1, y1, x2, y2, confidence, class_id]
            x1, y1, x2, y2 = map(int, box)
            w, h = x2 - x1, y2 - y1
            x, y = x1, y1

            # DetectionResult 객체 생성
            detection = DetectionResult(
                path="path",
                box=[x, y, w, h],
                confidence=confidence,
                class_id=int(class_id),
            )
            print(detection)
            detection_results.append(detection)

        # OpenCV로 결과 보여주기1.
        cv2.imshow("YOLOv5 Detection", output_frame)

        # 'q'를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 자원 해제
    cap.release()
    cv2.destroyAllWindows()

# main 함수 호출
main()
