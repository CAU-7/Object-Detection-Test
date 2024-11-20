import cv2
import numpy as np

class DetectionResult:
    def __init__(self, path, box, confidence, class_id):
        self.path = path
        self.box = box
        self.confidence = confidence
        self.class_id = class_id

    def __repr__(self):
        return f"DetectionResult(path={self.path}, box={self.box}, confidence={self.confidence}, class_id={self.class_id})"

# YOLO 모델 로드
def load_yolo_model():
    net = cv2.dnn.readNet("yolo/yolov3.weights", "yolo/yolov3.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
    return net, output_layers

# 이미지에서 객체 탐지
def detect_objects(event_queue, event_ready):
    net, output_layers = load_yolo_model()
    classes = []
    with open("yolo/coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    paths = [f"asset/test{i}.jpg" for i in range(10)]

    for path in paths:
        image = cv2.imread(path)
        height, width, _ = image.shape
        blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outputs = net.forward(output_layers)

        detection_results = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:  # 신뢰도 기준
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # 바운딩 박스의 좌표
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    # DetectionResult 객체 생성
                    detection = DetectionResult(path, [x, y, w, h], confidence, class_id)
                    detection_results.append(detection)

        # 탐지 결과 배열을 큐에 추가
        event_queue.put(detection_results)
        event_ready.set()  # 이벤트 발생 신호
