import torch
import cv2
import numpy as np
import platform
import pathlib
import message

if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath
else:
    pathlib.WindowsPath = pathlib.PosixPath

class DetectionResult:
    def __init__(self, path, box, confidence, class_id):
        self.path = path
        self.box = box
        self.confidence = confidence
        self.class_id = class_id
        self.id = 0
        self.count = 30
        self.display = True

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
        return x, y

    def cal_distance(self, expected):
        focal_length = 200
        w_scale = (focal_length * expected.expected_w) / self.box[2]
        h_scale = (focal_length * expected.expected_h) / self.box[3]
        self.distance = (h_scale + w_scale) / 2
        return self.distance
    
    def cal_angle(self):
        fov = 90
        image_width = 640
        center = self.box_center()
        self.angle = (center[0] - image_width / 2) * (fov / image_width)
        return self.angle
    
    def distance_other(self, other):
        delta_x = self.box[0] - other.box[0]
        delta_y = self.box[1] - other.box[1]
        distance = delta_x ** 2 + delta_y ** 2
        return distance

    def __repr__(self):
        return f"res:(path={self.path}, box={self.box}, confidence={self.confidence}, class_id={self.class_id}), count={self.count}"

def load_yolov5_model():
    # 모델 다운로드 및 로드 (사전 학습된 weights)
    model = torch.hub.load("ultralytics/yolov5", "custom", path="model/weight.pt")
    return model

# 이미지에서 객체 탐지
def detect_objects(event_queue, event_ready):
    net, output_layers = load_yolov3_model()
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
                    detection = DetectionResult(path, [x, y, w, h], confidence, classes[class_id])
                    detection_results.append(detection)

        # 탐지 결과 배열을 큐에 추가
        event_queue.put(detection_results)
        event_ready.set()  # 이벤트 발생 신호

def detect_objects_from_video(event_queue, event_ready, video_path):
    model = load_yolov5_model()
    classes = []
    with open("model/coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    # 비디오 파일 열기
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # 비디오 끝

        results = model(frame)

        # height, width, _ = frame.shape
        # blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        # net.setInput(blob)
        # outputs = net.forward(output_layers)

        # detection_results = []

        detection_results = []
        for *box, confidence, class_id in results.xyxy[0].tolist():
            # xyxy: [x1, y1, x2, y2, confidence, class_id]
            x1, y1, x2, y2 = map(int, box)
            w, h = x2 - x1, y2 - y1
            x, y = x1, y1

            # print(f"detect class_id:{class_id}")

            class_id = int(class_id)

            # DetectionResult 객체 생성
            detection = DetectionResult(
                path="path",
                box=[x, y, w, h],
                confidence=confidence,
                class_id=classes[class_id],
            )
            # print(detection)
            detection_results.append(detection)

        # 탐지 결과 배열을 큐에 추가
        event_queue.put(detection_results)
        event_ready.set()  # 이벤트 발생 신호

    cap.release()

def detect_objects_from_video_test(video_path, expected_dict):
    model = load_yolov5_model()
    classes = []
    with open("model/coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    # 비디오 파일 열기
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    last_results = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # 비디오 끝
        
        results = model(frame)

        # height, width, _ = frame.shape
        # blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        # net.setInput(blob)
        # outputs = net.forward(output_layers)

        detection_results = []
        for *box, confidence, class_id in results.xyxy[0].tolist():
            # xyxy: [x1, y1, x2, y2, confidence, class_id]
            x1, y1, x2, y2 = map(int, box)
            w, h = x2 - x1, y2 - y1
            x, y = x1, y1

            # print(f"detect class_id:{class_id}")

            class_id = int(class_id)

            # DetectionResult 객체 생성
            detection = DetectionResult(
                path="path",
                box=[x, y, w, h],
                confidence=confidence,
                class_id=classes[class_id],
            )
            # print(detection)
            detection_results.append(detection)

        # print(last_results)

        results = message.test_msg(detection_results, expected_dict, last_results)

        # print(f"results len: {len(results)}")

        for result in results:
            # 바운딩 박스와 클래스 이름을 프레임에 그리기
            if result.display == False:
                continue
            x, y, w, h = result.box
            class_id = result.class_id
            confidence = result.confidence
            id = result.id

            np.random.seed(id)
            temp = tuple(np.random.randint(0, 256, 3))
            color = tuple(int(c) for c in temp)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            label = f"{class_id} {result.angle:.2f} {result.distance:.2f}"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        last_results = results
        
        # 프레임을 화면에 표시
        cv2.imshow("Object Detection", frame)

        # `q` 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
