import time
import cv2
import torch
import platform
import pathlib

# Check the operating system and set the appropriate path type
if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath
else:
    pathlib.WindowsPath = pathlib.PosixPath

def load_yolo_v5_tiny_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5n')  # yolov5n: Tiny 모델
    return model

def load_trained_yolo_model():
    model = torch.hub.load("ultralytics/yolov5", "custom", path="best (1).pt")
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

        # OpenCV로 결과 보여주기
        cv2.imshow("YOLOv5 Detection", output_frame)

        # 'q'를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 자원 해제
    cap.release()
    cv2.destroyAllWindows()

# main 함수 호출
main()
