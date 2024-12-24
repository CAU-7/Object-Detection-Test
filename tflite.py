import cv2
import numpy as np
import tensorflow as tf

class DetectionResult:
    def __init__(self, box, confidence, class_id):
        self.box = box
        self.confidence = confidence
        self.class_id = class_id

    def __repr__(self):
        return f"DetectionResult(box={self.box}, confidence={self.confidence}, class_id={self.class_id})"

def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details

def detect_objects_from_video_tflite(video_path, model_path, label_path):
    # Load TFLite model and allocate tensors
    interpreter, input_details, output_details = load_tflite_model(model_path)

    # Load labels
    with open(label_path, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    # Open video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        height, width, _ = frame.shape

        # Preprocess frame for TFLite model
        input_shape = input_details[0]['shape']
        input_height, input_width = input_shape[1], input_shape[2]
        input_frame = cv2.resize(frame, (input_width, input_height))
        input_frame = np.expand_dims(input_frame, axis=0).astype(np.float32) / 255.0

        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], input_frame)

        # Run inference
        interpreter.invoke()

        # Get output tensor
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]  # Output tensor (6300, 85)

        detection_results = []

        # Post-process results
        for i in range(output_data.shape[0]):  # Iterate over 6300 possible boxes
            box = output_data[i][:4]  # Get bounding box coordinates [ymin, xmin, ymax, xmax]
            confidence = output_data[i][4]  # Get confidence score
            class_probs = output_data[i][5:]  # Get class probabilities (size 80)

            # Find the most probable class and its score
            class_id = np.argmax(class_probs)
            class_prob = class_probs[class_id]

            # Filter detections based on confidence threshold
            if confidence * class_prob > 0.5:  # Confidence threshold
                ymin, xmin, ymax, xmax = box

                # Convert box coordinates to original frame size
                x = int(xmin * width)
                y = int(ymin * height)
                w = int((xmax - xmin) * width)
                h = int((ymax - ymin) * height)

                # Add detection to results
                detection_results.append(DetectionResult([x, y, w, h], confidence * class_prob, class_id))

                # Draw bounding box and label
                label = f"{labels[class_id]}: {confidence * class_prob:.2f}"
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Show frame with detections
        cv2.imshow("Object Detection", frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Usage example
model_path = "model/yolov5.tflite"  # Path to your TFLite model
label_path = "model/coco.names"     # Path to your label file
video_path = "asset/example.mp4" # Path to your video file

detect_objects_from_video_tflite(video_path, model_path, label_path)
