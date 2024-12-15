import message

frame_count = 15

# delta_distance에 화면상의 거리를 넣어야 함
# 1. 멀리 있으면 pixel 조금 움직여도 상대적으로 많이 움직임
# 2. 가까이 있으면 pixel 많이 움직여도 상대적으로 조금 움직임

def make_json(time, obj, priorty):
    dis = obj.distance / 100
    angle = abs(obj.angle)
    msg = 0

    if obj.angle >= 0:
        msg = f"{dis:.1f}m 앞 우측 {angle:.0f}도 방향에 {obj.class_id} 있습니다"
    else:
        msg = f"{dis:.1f}m 앞 좌측 {angle:.0f}도 방향에 {obj.class_id} 있습니다"

    data = {
        "type": "warning",
        "data": {
            "time": time,
            "class_name": obj.class_id,
            "angle": obj.angle,
            "dis": obj.distance,
            "priorty": priorty,
            "message": msg
        }
    }

    return data

def is_vehicle(class_num):
    if class_num <= 2 or class_num == 6 or class_num == 9 or class_num == 12:
        return True
    return False

def new_object_detect(new_result, existing_results, expected_dict, count=[0]):
    threshold = 1000
    min_score = 10000
    similar = 0
    
    for existing_result in existing_results:
        if new_result.class_id != existing_result.class_id and not is_vehicle(new_result.class_num):
            continue
        # print(f"new_result.class_id={new_result.class_id}")
        score = new_result.distance_other(existing_result)
        distance = new_result.cal_distance(expected_dict[new_result.class_id]) / 100
        score *= distance / 300
        if min_score > score:
            min_score = score
            similar = existing_result

    # print(f"min: {min_score}, existing: {similar}")

    # 새로운 객체가 인식 됨
    if min_score > threshold:
        new_result.id = count[0]
        new_result.count = frame_count
        new_result.display = True
        count[0] += 1
        new_result.cal_distance(expected_dict[new_result.class_id])
        new_result.cal_angle()
        return True
    # 원래 객체가 인식 됨
    if similar != 0:
        # print(new_result.scale_other(similar))
        new_result.cal_distance(expected_dict[new_result.class_id])
        new_result.cal_angle()
        if new_result.count != frame_count:
            new_result.cal_delta(similar, frame_count - similar.count + 1)
        new_result.id = similar.id
        new_result.flag = similar.flag
        new_result.count = frame_count
        new_result.display = True
        new_result.con = similar.con + 1
        existing_results.remove(similar)
        return True
    new_result.con = 0
    return False

def get_msg(obj, time):
    if obj.flag == True:
        return

    priority = 0
    max_angle = 0
    max_distance = 0

    dis = obj.distance
    angle = abs(obj.angle)

    # 고정형 물체일때
    if obj.class_num >= 12:
        max_angle = 5
        max_distance = 200
        priority = 7
    # 탈 것일때
    elif is_vehicle(obj.class_num) and obj.delta >= 200:
        max_angle = 30
        max_distance = 400
        priority = 0
    # 나머지 (사람, 개, 고양이, 휠체어 등)
    elif not is_vehicle(obj.class_num):
        max_angle = 15
        max_distance = 200
        priority = 4
    
    
    if dis < max_distance and angle < max_angle and obj.con >= 2:
        obj.flag = True
        priority += int(dis / 100) - 1
        msg = make_json(time, obj, priority)
        print(msg)
        # client.send_json(msg)
        # print(f"[{time}]: msg: {obj.class_id}, angle: {obj.angle}, dis: {dis}, delta: {obj.delta}")

    # return message.Message(3, f"new_result: {new_result.class_id}, box: {new_result.box}, id: {new_result}")

def test_func(existing_results, new_results, expected_dict, time):
    updated_results = []

    # print(expected_dict)
    # print(new_results)

    # 새로운 결과와 기존 결과를 비교
    # new_result와 existing_result를 돌며 동일한게 있으면 (함수에 넣어서 임계점 넘으면) 기존으로 추가, 추가 안내 없음
    # 새로운게 있으면 추가, 안내 우선순위를 설정하고 우선순위 큐에 추가, 메인에선 우선순위 큐를 쭉 설명한다.
    for new_result in new_results:
        # 비교 로직 (필요에 따라 박스나 신뢰도를 비교할 수 있음)
        
        if new_object_detect(new_result, existing_results, expected_dict):
            updated_results.append(new_result)  # 새로운 결과로 갱신
            new_result.cal_distance(expected_dict[new_result.class_id])
            new_result.cal_angle()
            # print(new_result.distance)

    for result in existing_results:
        # print(f"exist and non detected: {result}")
        result.display = False
        result.count -= 1

        if result.count >= 0:
            updated_results.append(result)
    
    for result in updated_results:
        if result.display == False:
            continue
        get_msg(result, time)
        # 고정형이고 거리가 d 이하, 각도가 a 이하면 안내 -> 우선순위는 낮게
        # 이동형 탈것 아니고 거리가 d 이하, 각도가 a 이하면 안내 -> 우선순위 중간
        # 이동형 탈것이고 거리가 d 이하, 각도가 a 이하면 안내 -> 우선순위 가장 높게

    return updated_results