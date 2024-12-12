import socket
import json

HOST = '127.0.0.1'  # 서버 주소
PORT = 65432        # 사용할 포트

# 소켓 생성
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
    server_socket.bind((HOST, PORT))  # 주소와 포트를 바인딩
    server_socket.listen()  # 연결 대기
    print(f"서버가 {HOST}:{PORT}에서 대기 중입니다...")
    
    conn, addr = server_socket.accept()  # 클라이언트 연결 허용
    with conn:
        print(f"{addr}에서 연결되었습니다.")
        while True:
            data = conn.recv(1024)  # 클라이언트로부터 데이터 수신
            if not data:
                break
            message = data.decode()
            print(f"받은 메시지: {message}")
            
            # JSON 파싱
            try:
                json_data = json.loads(message)  # JSON 형식으로 파싱
                print(f"메세지: {json_data['message']}, 우선순위: {json_data['priority']}")
            except json.JSONDecodeError:
                print("유효하지 않은 JSON 형식입니다.")
                
            conn.sendall(data)  # 받은 데이터를 다시 전송 (에코)
