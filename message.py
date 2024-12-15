import guide
import parse_cfg
import cv2
import numpy as np

class Message:
    def __init__(self, priority, content):
        self.priority = priority
        self.content = content

    def __lt__(self, other):
        # priority에 따라 우선순위 결정
        return self.priority < other.priority

    def __repr__(self):
        return f"Message(priority={self.priority}, content='{self.content}')"
    
def handle_msg(img_queue, img_event, msg_queue, msg_event, expected_dict):
    existing_results = []

    while True:
        # 이벤트가 설정될 때까지 대기
        img_event.wait()

        # 이벤트 발생 시, 큐에서 데이터를 꺼내 처리
        while not img_queue.empty():
            new_results = img_queue.get()
            existing_results = guide.guide_func(existing_results, new_results, expected_dict, msg_queue, msg_event)

            # for result in existing_results:
            #     print(result)

        # 이벤트 초기화 (다음 이벤트를 기다리기 위해)
        img_event.clear()


def test_msg(new_results, expected_dict, existing_results, time):
    # 이벤트가 설정될 때까지 대기

    # 이벤트 발생 시, 큐에서 데이터를 꺼내 처리
    existing_results = guide.test_func(existing_results, new_results, expected_dict, time)

        # for result in existing_results:
        #     print(result)

    return existing_results