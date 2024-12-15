import cv2
import numpy as np
import random

# 이미지 로드
# image = cv2.imread('asset/block.png')

video_path = 'asset/demo.mp4'

# 비디오 파일 열기
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Cannot open video {video_path}")

i = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break  # 비디오 끝

    if i <= 5200:
        i += 1
        continue

    image = frame

    # BGR을 HSV로 변환
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 노란색 범위 정의 (Hue, Saturation, Value 범위 설정)
    lower_yellow = np.array([20, 100, 100])  # 노란색 하한값
    upper_yellow = np.array([35, 230, 230])  # 노란색 상한값

    # 범위 내 픽셀 추출
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # 흰색 픽셀 퍼뜨리기
    spread_mask = np.copy(mask)
    h, w = mask.shape

    y_threshold = 200
    
    # y 좌표 기준으로 필터링
    h, w = mask.shape
    for y in range(h):
        if y <= y_threshold:
            spread_mask[y, :] = 0  # y 좌표가 기준 이상인 모든 픽셀을 검은색으로 변경

    # 흰색 픽셀을 가진 부분을 원본 이미지에서 흰색으로 변경
    result = np.zeros_like(image)
    result[spread_mask > 0] = [255, 255, 255]  # 퍼진 영역 흰색
    result[spread_mask == 0] = [0, 0, 0]       # 나머지 영역 검정색

    # 결과 저장 및 표시
    cv2.imwrite('/mnt/data/spread_masked_image.png', result)
    cv2.imshow("Result", result)

    # `q` 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
