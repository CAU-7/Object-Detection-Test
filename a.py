import json
import time

warnings = []

# 파일에서 데이터 읽기
with open('out.txt', 'r', encoding='utf-8') as file:
    for line in file:
        line = line.strip()  # 줄바꿈 제거
        if line:  # 빈 줄 무시
            line = line.replace("'", '"')  # 싱글 쿼트를 더블 쿼트로 변환
            try:
                warnings.append(json.loads(line))  # JSON 파싱
            except json.JSONDecodeError as e:
                print(f"JSONDecodeError: {e} at line: {line}")

# t = 0
# i = 0

# while t <= 18268:
#     warning = warnings[i]
#     time_value = warning['data']['time']
#     if t == time_value:
#         print(json.dumps(warning, indent=4, ensure_ascii=False))
#         i += 1
#     t += 1
#     time.sleep(0.005)

result_json = []

# warnings 데이터 처리
for msg in warnings:
    time_value = msg['data']['time']
    type_value = msg['type']
    res = None
    entry = {"time": time_value, "type": type_value}  # 기본 구조
    
    if type_value == 'warning':
        res = msg['data']['class_name']
        distance = msg['data']['dis']
        entry.update({"class_name": res, "distance": f"{distance / 100:.1f}"})
    else:
        res = type_value
        entry.update({"class_name": res})
    result_json.append(entry)

with open('output.json', 'w', encoding='utf-8') as f:
    json.dump(result_json, f, indent=4, ensure_ascii=False)


# # warnings 확인
# print(warnings)
