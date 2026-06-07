import os
import time

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn


# Config
CAMERA_FRAME_RATE = 10
CAMERA_INDEX = 0
CONFIDENCE_THRESHOLD = 0.0

NUM_POSE = 33
NUM_FACE = 468
NUM_HAND = 21
FULL_FEATURE_COUNT = 3 * (NUM_POSE + NUM_FACE + 2 * NUM_HAND)

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12

COLOR_POSE = (0, 0, 255)
COLOR_FACE = (255, 0, 0)
COLOR_LEFT_HAND = (0, 255, 0)
COLOR_RIGHT_HAND = (128, 0, 128)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mp_holistic = mp.solutions.holistic

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "best_model.pt")


# Model
class TemporalResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1, dropout=0.25):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
        )
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(x + self.net(x))


class SignTCNClassifier(nn.Module):
    def __init__(self, input_size, channels, num_classes, dropout=0.30, num_blocks=4):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_size)
        self.input_projection = nn.Conv1d(input_size, channels, kernel_size=1)
        dilations = [2 ** (idx % 3) for idx in range(num_blocks)]
        self.blocks = nn.Sequential(
            *[TemporalResidualBlock(channels, dilation=dilation, dropout=dropout) for dilation in dilations]
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(channels * 2),
            nn.Dropout(dropout),
            nn.Linear(channels * 2, channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels, num_classes),
        )

    def forward(self, x):
        x = self.input_norm(x).transpose(1, 2)   # [B, T, F] -> [B, F, T]
        x = self.input_projection(x)
        x = self.blocks(x).transpose(1, 2)       # [B, C, T] -> [B, T, C]
        pooled = torch.cat([x.mean(dim=1), x.max(dim=1).values], dim=1)
        return self.classifier(pooled)


# Loading
def load_checkpoint(path):
    checkpoint = torch.load(path, map_location=device)

    if checkpoint.get("model_type") != "tcn":
        raise ValueError(f"Expected a TCN checkpoint, got {checkpoint.get('model_type')!r}")

    label_to_id = checkpoint["label_to_id"]
    id_to_label = {idx: label for label, idx in label_to_id.items()}

    model_config = checkpoint["model_config"]
    input_size = int(checkpoint["input_size"])
    num_classes = int(model_config["num_classes"])

    model = SignTCNClassifier(
        input_size=input_size,
        channels=int(model_config["channels"]),
        num_classes=num_classes,
        dropout=float(model_config.get("dropout", 0.30)),
        num_blocks=int(model_config.get("num_blocks", 4)),
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    feature_mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
    feature_std = np.asarray(checkpoint["feature_std"], dtype=np.float32)
    sequence_length = int(checkpoint["sequence_length"])

    landmark_selection = checkpoint.get("landmark_selection")
    selected_feature_indices = None
    if landmark_selection is not None:
        selected_feature_indices = np.asarray(landmark_selection["selected_feature_indices"], dtype=np.int64)

    if len(feature_mean) != input_size or len(feature_std) != input_size:
        raise ValueError("feature_mean / feature_std do not match model input size")

    return {
        "model": model,
        "id_to_label": id_to_label,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "sequence_length": sequence_length,
        "model_input_size": input_size,
        "selected_feature_indices": selected_feature_indices,
    }


ckpt = load_checkpoint(model_path)
model = ckpt["model"]
id_to_label = ckpt["id_to_label"]
feature_mean = ckpt["feature_mean"]
feature_std = ckpt["feature_std"]
sequence_length = ckpt["sequence_length"]
model_input_size = ckpt["model_input_size"]
selected_feature_indices = ckpt["selected_feature_indices"]


# Landmark processing
def landmarks_to_array(landmarks, expected_count):
    arr = np.zeros((expected_count, 3), dtype=np.float32)
    if landmarks is None:
        return arr
    for i, landmark in enumerate(landmarks.landmark[:expected_count]):
        arr[i] = [landmark.x, landmark.y, landmark.z]
    return arr


def normalize_landmarks(pose, face, left_hand, right_hand):
    all_points = np.concatenate([pose, face, left_hand, right_hand], axis=0)
    detected_mask = np.any(all_points != 0, axis=1)

    left_shoulder = pose[LEFT_SHOULDER]
    right_shoulder = pose[RIGHT_SHOULDER]

    if np.any(left_shoulder) and np.any(right_shoulder):
        center = (left_shoulder + right_shoulder) / 2.0
        scale = np.linalg.norm(left_shoulder[:2] - right_shoulder[:2])
        if scale > 1e-6:
            all_points[detected_mask] = (all_points[detected_mask] - center) / scale

    return all_points.reshape(-1).astype(np.float32)


def standardize_sequence(sequence):
    sequence = sequence.copy().astype(np.float32)
    mask = sequence != 0
    sequence[mask] = ((sequence - feature_mean) / feature_std)[mask]
    return sequence


def select_landmark_features(sequence):
    sequence = np.asarray(sequence, dtype=np.float32)
    if sequence.ndim != 2:
        raise ValueError(f"Expected 2D sequence, got {sequence.shape}")

    if selected_feature_indices is None:
        if sequence.shape[1] != model_input_size:
            raise ValueError(
                f"Checkpoint expects {model_input_size} features, got {sequence.shape[1]}"
            )
        return sequence

    if sequence.shape[1] != FULL_FEATURE_COUNT:
        raise ValueError(
            f"Expected full {FULL_FEATURE_COUNT}-feature vectors before selection, got {sequence.shape[1]}"
        )

    return sequence[:, selected_feature_indices]


def extract_landmarks(frame, holistic):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = holistic.process(image_rgb)

    pose = landmarks_to_array(results.pose_landmarks, NUM_POSE)
    face = landmarks_to_array(results.face_landmarks, NUM_FACE)
    left_hand = landmarks_to_array(results.left_hand_landmarks, NUM_HAND)
    right_hand = landmarks_to_array(results.right_hand_landmarks, NUM_HAND)

    features = normalize_landmarks(pose, face, left_hand, right_hand)
    return features, results


# Drawing
def draw_landmark_group(frame, landmark_list, connections, color, radius=4, thickness=2):
    if landmark_list is None:
        return frame

    height, width = frame.shape[:2]
    points = []

    for landmark in landmark_list.landmark:
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        points.append((x, y))
        cv2.circle(frame, (x, y), radius, color, -1)

    for start_idx, end_idx in connections:
        if start_idx < len(points) and end_idx < len(points):
            cv2.line(frame, points[start_idx], points[end_idx], color, thickness)

    return frame


def draw_face_dots(frame, landmark_list):
    if landmark_list is None:
        return frame

    height, width = frame.shape[:2]
    for landmark in landmark_list.landmark:
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        cv2.circle(frame, (x, y), 1, COLOR_FACE, -1)
    return frame


def draw_results(frame, results):
    if results is None:
        return frame

    draw_face_dots(frame, results.face_landmarks)
    draw_landmark_group(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, COLOR_POSE)
    draw_landmark_group(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, COLOR_LEFT_HAND)
    draw_landmark_group(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, COLOR_RIGHT_HAND)
    return frame


# Prediction
def predict_sign(landmark_sequence):
    if len(landmark_sequence) != sequence_length:
        raise ValueError(f"Expected {sequence_length} frames, got {len(landmark_sequence)}")

    sequence = np.asarray(landmark_sequence, dtype=np.float32)
    sequence = select_landmark_features(sequence)
    sequence = standardize_sequence(sequence)

    sequence_tensor = torch.from_numpy(sequence).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(sequence_tensor)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = int(torch.argmax(probs).item())
        confidence = float(probs[pred_idx].item())

    if confidence < CONFIDENCE_THRESHOLD:
        return "Unknown", confidence

    return id_to_label[pred_idx], confidence


# Main loop
def main():
    capture_seconds = sequence_length / CAMERA_FRAME_RATE

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    buffer = []
    last_sample_time = 0.0
    current_prediction = "Sign: waiting..."
    last_results = None
    sign_start_time = time.time()

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            now = time.time()
            if now - last_sample_time >= 1.0 / CAMERA_FRAME_RATE:
                last_sample_time = now
                landmarks, results = extract_landmarks(frame, holistic)
                last_results = results
                buffer.append(landmarks)

                if len(buffer) >= sequence_length:
                    elapsed = now - sign_start_time
                    sign, confidence = predict_sign(buffer[:sequence_length])
                    current_prediction = f"Sign: {sign} ({confidence * 100:.1f}%)"
                    print(f"Prediction in {elapsed:.2f}s: {sign} ({confidence * 100:.1f}%)")
                    buffer = []
                    sign_start_time = time.time()

            frame = draw_results(frame, last_results)

            cv2.putText(
                frame,
                f"Frames: {len(buffer)}/{sequence_length} | {CAMERA_FRAME_RATE} fps | {capture_seconds:.1f}s",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                current_prediction,
                (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("ASL Detector", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
