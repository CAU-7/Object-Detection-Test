import message

# delta_distance에 화면상의 거리를 넣어야 함
# 1. 멀리 있으면 pixel 조금 움직여도 상대적으로 많이 움직임
# 2. 가까이 있으면 pixel 많이 움직여도 상대적으로 조금 움직임

def check_similarity(new_result, exist_result, expected):
    size_threshold = 2
    offset = 20
    distance_threshold = expected.move_speed * offset

    # delta_size = new_result.size() / exist_result.size()
    delta_distance = new_result.distance_other(exist_result)

    # if delta_size > size_threshold:
    #     return 0
    if delta_distance > distance_threshold:
        return 0

    return delta_distance

def new_object_detect(new_result, existing_results, expected_dict):
    theshold = 3
    min_distance = 1000
    similar = 0
    
    for existing_result in existing_results:
        if new_result.class_id != existing_result.class_id:
            continue
        score = check_similarity(new_result, existing_result, expected_dict[new_result.class_id])
        if min_distance > score:
            min_distance = score
            similar = existing_result

    # print(f"min: {min_distance}, existing: {similar}")

    if min_distance > theshold:
        if similar != 0:
            existing_results.remove(similar)
        return True
    return False

def get_msg(new_result, expeted_dict):
    return message.Message(3, f"new_result: {new_result.class_id}, box: {new_result.box}")

def test_func(existing_results, new_results, expected_dict, msg_queue, msg_event):
    updated_results = []

    # 새로운 결과와 기존 결과를 비교
    # new_result와 existing_result를 돌며 동일한게 있으면 (함수에 넣어서 임계점 넘으면) 기존으로 추가, 추가 안내 없음
    # 새로운게 있으면 추가, 안내 우선순위를 설정하고 우선순위 큐에 추가, 메인에선 우선순위 큐를 쭉 설명한다.
    for new_result in new_results:
        # 비교 로직 (필요에 따라 박스나 신뢰도를 비교할 수 있음)
        updated_results.append(new_result)  # 새로운 결과로 갱신
        
        if new_object_detect(new_result, existing_results, expected_dict):
            msg_queue.put(get_msg(new_result, expected_dict))
            msg_event.set()
            

    return updated_results