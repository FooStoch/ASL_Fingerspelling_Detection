import cv2
import mediapipe as mp
import torch
import numpy as np
from tqdm import tqdm
import os
import time

# Initialize MediaPipe Holistic for landmark detection
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Determine the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))

# Build a portable path referencing best_model.pth in the same directory
model_path = os.path.join(current_dir, 'best_model.pth')

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}. Ensure the file exists and the path is correct.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the model class (same as original)
class SimplerSignLanguageLSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout_rate=0.5):
        super(SimplerSignLanguageLSTM, self).__init__()
        self.input_dropout = torch.nn.Dropout(0.3)
        self.input_bn = torch.nn.BatchNorm1d(input_size)
        self.lstm = torch.nn.LSTM(input_size, hidden_size, batch_first=True, dropout=dropout_rate, bidirectional=True)
        self.fc1 = torch.nn.Linear(hidden_size * 2, 64)
        self.fc2 = torch.nn.Linear(64, num_classes)
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, packed_input):
        padded_input, lengths = torch.nn.utils.rnn.pad_packed_sequence(packed_input, batch_first=True)
        padded_input = padded_input.transpose(1, 2)
        padded_input = self.input_bn(padded_input)
        padded_input = padded_input.transpose(1, 2)
        padded_input = self.input_dropout(padded_input)
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(padded_input, lengths, batch_first=True, enforce_sorted=False)
        packed_output, (h_n, c_n) = self.lstm(packed_input)
        output_forward = h_n[0, :, :]
        output_backward = h_n[1, :, :]
        output = torch.cat((output_forward, output_backward), dim=1)
        output = self.fc1(output)
        output = torch.nn.functional.relu(output)
        output = self.dropout(output)
        output = self.fc2(output)
        return output

# Load and initialize the model with new input size
num_classes = 4  # Assuming the same number of classes as before
model = SimplerSignLanguageLSTM(input_size=225, hidden_size=128, num_classes=num_classes, dropout_rate=0.5)
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict['model_state_dict'])  # Load the new model's state
model.to(device)
model.eval()

# Class names (ensure these match your new model's training labels)
class_names = ['eat', 'why', 'meat', 'you']

# Constants for landmark counts
NUM_POSE_LANDMARKS = 33
NUM_HAND_LANDMARKS = 21

# Color mapping (adjusted to exclude face landmarks)
COLOR_MAP = {
    'red': (0, 0, 255),    # Pose
    'green': (0, 255, 0),  # Left hand
    'purple': (128, 0, 128) # Right hand
}

# Modified function to extract only pose and hand landmarks
def extract_landmarks(frame):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(image_rgb)

    # Allocate array for pose and hand landmarks only (225 features)
    all_landmarks = np.zeros(3 * (NUM_POSE_LANDMARKS + 2 * NUM_HAND_LANDMARKS))

    offset = 0
    landmark_types = [
        (results.pose_landmarks, NUM_POSE_LANDMARKS),
        (results.left_hand_landmarks, NUM_HAND_LANDMARKS),
        (results.right_hand_landmarks, NUM_HAND_LANDMARKS)
    ]

    for landmarks, expected_count in landmark_types:
        if landmarks:
            for landmark in landmarks.landmark:
                all_landmarks[offset:offset + 3] = [landmark.x, landmark.y, landmark.z]
                offset += 3
        else:
            offset += expected_count * 3

    if not all_landmarks.any():
        print("Warning: No landmarks detected in this frame")
        return None
    return all_landmarks.tolist()

# Modified function to draw only pose and hand landmarks
def draw_landmarks(frame, landmarks):
    frame_height, frame_width = frame.shape[:2]
    
    offset = 0
    landmark_types = [
        (landmarks[offset:offset + NUM_POSE_LANDMARKS*3], 'red', mp_holistic.POSE_CONNECTIONS),
        (landmarks[offset + NUM_POSE_LANDMARKS*3:offset + (NUM_POSE_LANDMARKS + NUM_HAND_LANDMARKS)*3], 'green', mp_holistic.HAND_CONNECTIONS),
        (landmarks[offset + (NUM_POSE_LANDMARKS + NUM_HAND_LANDMARKS)*3:], 'purple', mp_holistic.HAND_CONNECTIONS)
    ]

    for points, color, connections in landmark_types:
        points = np.array(points).reshape(-1, 3)
        bgr_color = COLOR_MAP[color]
        for idx, point in enumerate(points):
            x, y, _ = point * [frame_width, frame_height, 1]
            cv2.circle(frame, (int(x), int(y)), 5, bgr_color, -1)
        
        for connection in connections:
            start_idx, end_idx = connection
            if start_idx < len(points) and end_idx < len(points):
                start = points[start_idx] * [frame_width, frame_height, 1]
                end = points[end_idx] * [frame_width, frame_height, 1]
                cv2.line(frame, (int(start[0]), int(start[1])), (int(end[0]), int(end[1])), bgr_color, 2)

    return frame

# Prediction function (unchanged)
def predict_sign(landmark_sequence, model, device):
    if not landmark_sequence or len(landmark_sequence) == 0:
        print("Warning: Empty landmark sequence for prediction")
        return "Unknown", 0.0
    
    sequence_tensor = torch.tensor(landmark_sequence, dtype=torch.float32).unsqueeze(0).to(device)
    lengths = torch.tensor([len(landmark_sequence)], dtype=torch.int32)
    packed_input = torch.nn.utils.rnn.pack_padded_sequence(sequence_tensor, lengths, batch_first=True, enforce_sorted=False)
    
    with torch.no_grad():
        output = model(packed_input)
        print(f"Model output before softmax: {output}")
        probabilities = torch.softmax(output, dim=1)
        print(f"Probabilities: {probabilities}")
        predicted_idx = torch.argmax(probabilities).item()
        confidence = probabilities[0, predicted_idx].item()
    
    return class_names[predicted_idx], confidence

# Main function (unchanged)
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    landmark_buffer = []
    frame_count = 0
    total_frames = 30
    target_fps = 10
    last_frame_time = time.time()
    current_prediction = None

    while True:
        current_time = time.time()
        if current_time - last_frame_time < 1.0 / target_fps:
            continue
        last_frame_time = current_time

        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        landmarks = extract_landmarks(frame)
        if landmarks is not None:
            landmark_buffer.append(landmarks)
            frame = draw_landmarks(frame, landmarks)

        frame_count += 1
        progress = min(frame_count, total_frames)
        
        display_text = f"Progress: {progress}/{total_frames}"
        if progress == total_frames and landmark_buffer:
            print(f"Landmark buffer before prediction: {len(landmark_buffer)} frames, sample: {landmark_buffer[0][:10]}")
            sign, confidence = predict_sign(landmark_buffer, model, device)
            current_prediction = f"Sign: {sign} ({confidence*100:.2f}%)"
            landmark_buffer = []
            frame_count = 0
            time.sleep(3)

        cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        if current_prediction:
            cv2.putText(frame, current_prediction, (frame.shape[1] - 300, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('Sign Language Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
