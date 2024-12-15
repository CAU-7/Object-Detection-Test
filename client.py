import socket
import json

HOST = '127.0.0.1'
PORT = 65432

# JSON 데이터
data = {
    "message": "message example",
    "priority": 1
}

# 소켓 생성
def send_json(data):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
        client_socket.connect((HOST, PORT))  # 서버에 연결
        for i in range (30):
            message = json.dumps(data)  # 딕셔너리를 JSON 문자열로 변환
            client_socket.sendall(message.encode())  # 데이터를 전송
            data = client_socket.recv(1024)  # 서버의 응답 수신
            print(f"서버 응답: {data.decode()}")

send_json(data)
