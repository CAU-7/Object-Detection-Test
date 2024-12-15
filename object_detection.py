import torch
import cv2
import numpy as np
import platform
import pathlib
import message
import tensorflow as tf
import time as t
import math

if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath
else:
    pathlib.WindowsPath = pathlib.PosixPath

class DetectionResult:
    def __init__(self, box, confidence, class_id, class_num):
        self.box = box
        self.confidence = confidence
        self.class_id = class_id
        self.class_num = class_num
        self.id = 0
        self.count = 30
        self.display = True
        self.con = 0
        self.flag = False
        self.delta = 0

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
        self.distance = min(h_scale, w_scale)
        return self.distance
    
    def cal_angle(self):
        fov = 45
        image_width = 406
        center = self.box_center()
        self.angle = (center[0] - image_width / 2) * (fov / image_width)
        return self.angle

    def cal_delta(self, other, frame_diff):
        x = self.distance * math.cos(self.angle)
        y = self.distance * math.sin(self.angle)

        o_x = other.distance * math.cos(other.angle)
        o_y = other.distance * math.sin(other.angle)

        distance = math.sqrt((x - o_x) ** 2 + (y - o_y) ** 2)
        delta = distance / frame_diff
        self.delta = abs(delta - 12)
        return self.delta
    
    def distance_other(self, other):
        delta_x = self.box[0] - other.box[0]
        delta_y = self.box[1] - other.box[1]
        distance = delta_x ** 2 + delta_y ** 2
        return distance
    
    def scale_other(self, other):
        delta_x = self.box[2] / other.box[2]
        delta_y = self.box[3] / other.box[3]
        scale = (delta_x + delta_y) / 2
        return scale

    def __repr__(self):
        return f"res:(delta={self.delta}, box={self.box}, class_id={self.class_id}), count={self.count}"

def load_yolov5_model():
    # 모델 다운로드 및 로드 (사전 학습된 weights)
    model = torch.hub.load("ultralytics/yolov5", "custom", path="model/best.pt")
    return model

def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details

def apply_nms(boxes, confidences, score_threshold=0.5, nms_threshold=0.4):
    # OpenCV의 NMSBoxes를 사용하여 중복 박스를 제거
    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold, nms_threshold)
    return indices

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

    time = 0

    while True:
        start_time = t.perf_counter() 
        ret, frame = cap.read()
        if not ret:
            break  # 비디오 끝
        
        results = model(frame)

        # height, width, _ = frame.shape
        # blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        # net.setInput(blob)
        # outputs = net.forward(output_layers)

        detection_results = []
        boxes = []
        confidences = []

        height, width, _ = frame.shape

        for *box, confidence, class_id in results.xyxy[0].tolist():
            # xyxy: [x1, y1, x2, y2, confidence, class_id]
            x1, y1, x2, y2 = map(int, box)
            w, h = x2 - x1, y2 - y1
            x, y = x1, y1

            # print(f"detect class_id:{class_id}")

            class_id = int(class_id)

            # DetectionResult 객체 생성
            detection = DetectionResult(
                box=[x, y, w, h],
                confidence=confidence,
                class_id=classes[class_id],
                class_num=class_id
            )
            # print(detection)
            detection_results.append(detection)
            boxes.append([x, y, w, h])
            confidences.append(confidence)

        end_time = t.perf_counter()  # 루프 종료 시간 측정
        elapsed_time = end_time - start_time  # 경과 시간 계산
        
        # NMS 적용
        indices = apply_nms(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)

        # NMS 결과에 따라 detection_results를 갱신
        nms_results = []
        for i in indices:  # flatten하여 각 인덱스를 가져옴
            nms_results.append(detection_results[i])  # NMS 후 결과만 추가

        results = message.test_msg(nms_results, expected_dict, last_results, time)

        time += 1

        # print(f"results len: {len(results)}")

        for result in results:
            # 바운딩 박스와 클래스 이름을 프레임에 그리기
            if result.display == False and result.count < 28:
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
    
        # print(f"frame detect elapsed time: {elapsed_time:.6f} seconds")

    cap.release()
    cv2.destroyAllWindows()

def detect_objects_from_video_tflite(video_path, expected_dict):
    # Load TFLite model and allocate tensors
    interpreter, input_details, output_details = load_tflite_model("model/best-fp16.tflite")

    # Load labels
    with open("model/coco.names", 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    # Open video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    last_results = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        height, width, _ = frame.shape

        # Preprocess frame for TFLite model
        input_shape = input_details[0]['shape']
        input_height, input_width = input_shape[1], input_shape[2]
        input_frame = cv2.resize(frame, (input_width, input_height))
        input_frame = np.expand_dims(input_frame, axis=0).astype(np.float32) / 255.0

        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], input_frame)

        # Run inference
        interpreter.invoke()

        # Get output tensor
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]  # Output tensor (6300, 85)

        detection_results = []

        # Post-process results
        for i in range(output_data.shape[0]):  # Iterate over 6300 possible boxes
            box = output_data[i][:4]  # Get bounding box coordinates [ymin, xmin, ymax, xmax]
            confidence = output_data[i][4]  # Get confidence score
            class_probs = output_data[i][5:]  # Get class probabilities (size 80)

            # Find the most probable class and its score
            class_id = np.argmax(class_probs)
            class_prob = class_probs[class_id]

            # Filter detections based on confidence threshold
            if confidence * class_prob > 0.5:  # Confidence threshold
                ymin, xmin, ymax, xmax = box

                # Convert box coordinates to original frame size
                x = int(xmin * width)
                y = int(ymin * height)
                w = int((xmax - xmin) * width)
                h = int((ymax - ymin) * height)

                # Add detection to results
                detection_results.append(DetectionResult([x, y, w, h], confidence * class_prob, labels[class_id]))

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
