import torch
import torch.onnx
import tensorflow as tf
import onnx
from onnx_tf.backend import prepare

# 1. PyTorch 모델을 ONNX로 변환
def convert_pytorch_to_onnx(pytorch_model, dummy_input, onnx_path):
    print(f"ONNX 모델 저장됨: {onnx_path}")
    torch.onnx.export(pytorch_model, dummy_input, onnx_path, verbose=True)
    print(f"ONNX 모델 저장됨: {onnx_path}")

# 2. ONNX 모델을 TensorFlow 모델로 변환
def convert_onnx_to_tf(onnx_path, tf_model_path):
    print(f"TensorFlow 모델 저장됨: {tf_model_path}")
    onnx_model = onnx.load(onnx_path)
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(tf_model_path)
    print(f"TensorFlow 모델 저장됨: {tf_model_path}")

# 3. TensorFlow 모델을 TFLite로 변환
def convert_tf_to_tflite(tf_model_path, tflite_model_path):
    print(f"TFLite 모델 저장됨: {tflite_model_path}")
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
    tflite_model = converter.convert()
    with open(tflite_model_path, "wb") as f:
        f.write(tflite_model)
    print(f"TFLite 모델 저장됨: {tflite_model_path}")

# 예시: ResNet 모델을 사용한 변환
def main():
    # PyTorch 모델 로드 (자신의 모델로 변경 가능)
    # model = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)
    model = torch.hub.load("ultralytics/yolov5", "custom", path="wieght.pt")
    model.eval()

    # 더미 입력 생성 (모델의 입력 크기에 맞게)
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # 경로 설정
    onnx_path = "model.onnx"
    tf_model_path = "model_tf"
    tflite_model_path = "model.tflite"

    # 1. PyTorch -> ONNX
    convert_pytorch_to_onnx(model, dummy_input, onnx_path)

    # 2. ONNX -> TensorFlow
    convert_onnx_to_tf(onnx_path, tf_model_path)

    # 3. TensorFlow -> TFLite
    convert_tf_to_tflite(tf_model_path, tflite_model_path)

if __name__ == "__main__":
    main()
