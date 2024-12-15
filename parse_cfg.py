class DetectionInfo:
    def __init__(self, class_id, expected_w, expected_h, move_speed):
        self.class_id = class_id  # 객체 이름
        self.expected_w = expected_w  # 예상 너비
        self.expected_h = expected_h  # 예상 높이
        self.move_speed = move_speed

    def __repr__(self):
        return f"DetectionInfo(class_id='{self.class_id}', expected_w={self.expected_w}, expected_h={self.expected_h}, speed={self.move_speed})"

def parse_detection_file(file_path):
    detection_dict = {}

    with open(file_path, "r") as file:
        for line in file:
            # 라인에서 공백 제거 및 데이터 분리
            parts = line.split()
            if len(parts) == 4:
                class_id = parts[0]
                expected_w = int(parts[1])
                expected_h = int(parts[2])
                move_speed = int(parts[3])
                
                # print(class_id)
                # print(expected_w)
                # print(expected_h)

                # DetectionInfo 객체를 id를 키로 딕셔너리에 추가
                detection_dict[class_id] = DetectionInfo(class_id, expected_w, expected_h, move_speed)
    
    return detection_dict

def parse():
	# 파일 경로 설정
	file_path = "guide.cfg"

	# 파일 파싱
	detection_info_dict = parse_detection_file(file_path)

	# 딕셔너리 전체 출력
	# print("\nAll detection data:")
	# for obj_id, info in detection_info_dict.items():
	# 	print(f"{obj_id}: {info}")
          
	return detection_info_dict
