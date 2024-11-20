import message

def scoring_similarity(new_result, exist_result):
	# print(f"new_result: {new_result.class_id}, box: {new_result.box}")
	return 51

def new_object_detect(new_result, existing_results):
    threshold = 50
    
    for existing_result in existing_results:
        if new_result.class_id != existing_result.class_id:
            continue
        if scoring_similarity(new_result, existing_result) > threshold:
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
        
        if new_object_detect(new_result, existing_results):
            msg_queue.put(get_msg(new_result, expected_dict))
            msg_event.set()
            

    return updated_results