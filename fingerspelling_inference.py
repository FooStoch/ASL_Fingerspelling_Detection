import json
import os
import pickle
from collections import OrderedDict

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_layers, num_classes, dropout=0.2):
        super().__init__()
        layers = []
        prev = input_dim
        for width in hidden_layers:
            layers.append(nn.Linear(prev, width))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = width
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def load_model(model_name="MLP_3.pt"):
    path = os.path.join(BASE_DIR, model_name)

    ckpt = torch.load(path, map_location="cpu")
    cfg = ckpt["config"]
    model = MLP(63, tuple(cfg["hidden_layers"]), 26, dropout=cfg["dropout"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    return (model, ckpt)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3,
)

def preprocess_hand(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if not results.multi_hand_landmarks:
        return None, None

    hand_landmarks = results.multi_hand_landmarks[0]
    handedness = "Right"
    if results.multi_handedness:
        handedness = results.multi_handedness[0].classification[0].label

    coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)
    if coords.shape != (21, 3):
        return None, None

    if handedness == "Left":
        coords[:, 0] = 1.0 - coords[:, 0]

    wrist = coords[0].copy()
    coords = coords - wrist

    scale = np.max(np.abs(coords))
    if scale < 1e-6:
        return None, None

    coords = coords / scale
    return coords.flatten().astype(np.float32), results

def predict(model_obj, sample):
    if isinstance(model_obj, tuple):
        model, _ = model_obj
        with torch.no_grad():
            x = torch.tensor(sample, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            logits = model(x)
            pred = int(torch.argmax(logits, dim=1).item())
        return pred

def main():
    label_map = {
        0: "A",
        1: "B",
        2: "K",
        3: "L",
        4: "M",
        5: "N",
        6: "O",
        7: "P",
        8: "Q",
        9: "R",
        10: "S",
        11: "T",
        12: "C",
        13: "U",
        14: "V",
        15: "W",
        16: "X",
        17: "Y",
        18: "Z",
        19: "D",
        20: "E",
        21: "F",
        22: "G",
        23: "H",
        24: "I",
        25: "J"
    }

    best_model = load_model()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    last_prediction = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        sample, results = preprocess_hand(frame)
        display = frame.copy()

        if sample is not None:
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        display, # Use display to show on screen
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )

            pred_idx = predict(best_model, sample)
            pred_label = label_map[pred_idx]
            last_prediction = f"{pred_label}"
            cv2.putText(display, f"{last_prediction}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)
        else:
            cv2.putText(display, "No hand detected", (35, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

        cv2.imshow("Fingerspelling", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()

if __name__ == "__main__":
    main()