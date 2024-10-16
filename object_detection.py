import cv2
import numpy as np

img_path = "asset/test.jpg"

# YOLO 모델 로드
def load_yolo_model():
    net = cv2.dnn.readNet("yolo/yolov3.weights", "yolo/yolov3.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
    return net, output_layers

# 이미지에서 객체 탐지
def detect_objects(image, net, output_layers):
    height, width, _ = image.shape
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

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

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    return boxes, confidences, class_ids

# 바운딩 박스 그리기
def draw_labels(image, boxes, confidences, class_ids, classes):
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    if len(indexes) > 0:  # indexes가 비어 있지 않은 경우에만 처리
        for i in indexes.flatten():  # flatten() 메서드를 사용하여 1차원 배열로 변환
            box = boxes[i]
            x, y, w, h = box
            label = str(classes[class_ids[i]])
            color = (0, 255, 0)  # 초록색
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# 메인 함수
def main():
    net, output_layers = load_yolo_model()
    classes = []
    with open("yolo/coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    image = cv2.imread(img_path)  # 탐지할 이미지 경로
    boxes, confidences, class_ids = detect_objects(image, net, output_layers)
    draw_labels(image, boxes, confidences, class_ids, classes)

    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
