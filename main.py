import threading
import queue
import object_detection
import message

# 메인 함수
def main():
    img_queue = queue.Queue()
    msg_queue = queue.PriorityQueue()
    img_event = threading.Event()
    msg_event = threading.Event()

    msg_thread = threading.Thread(target=message.handle_msg, args=(img_queue, img_event, msg_queue, msg_event))
    msg_thread.start()

    # 워커 스레드 생성 및 시작
    img_thread = threading.Thread(target=object_detection.detect_objects, args=(img_queue, img_event))
    img_thread.start()

    while True:
        # 이벤트가 설정될 때까지 대기
        msg_event.wait()

        # 이벤트 발생 시s, 큐에서 데이터를 꺼내 처리
        if not msg_queue.empty():
            new_msg = msg_queue.get()
            print(new_msg.content)

        # 이벤트 초기화 (다음 이벤트를 기다리기 위해)
        msg_event.clear()

    img_thread.join()
    msg_thread.join()

if __name__ == "__main__":
    main()