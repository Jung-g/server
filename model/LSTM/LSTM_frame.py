import os
import json
import base64
import numpy as np
import cv2
import torch
import mediapipe as mp
from typing import Tuple, Optional
from model.LSTM.utils import Vector_Normalization

MODEL_DIR = "D:/Bit/server/model"

data_mean = np.load(os.path.join(MODEL_DIR, "data_mean.npy"))
data_std = np.load(os.path.join(MODEL_DIR, "data_std.npy"))

with open(os.path.join(MODEL_DIR, "label_map.json"), encoding="utf-8") as f:
    labels_map_raw = json.load(f)
labels_map = {v: k for k, v in labels_map_raw.items()}

model = torch.jit.load(os.path.join(MODEL_DIR, "lstm_sign_language_model_scripted.pt"))
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

SEQ_LEN = 45
FEATURE_DIM = 166

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
pose = mp_pose.Pose(model_complexity=0)
hands = mp_hands.Hands(max_num_hands=2)

def decode_base64_to_numpy(base64_string: str) -> np.ndarray:
    img_bytes = base64.b64decode(base64_string)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img_np

def extract_features_from_frame(frame_np: np.ndarray) -> Optional[np.ndarray]:
    frame_small = cv2.resize(frame_np, (480, 360))
    frame_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)

    results_pose = pose.process(frame_rgb)
    results_hands = hands.process(frame_rgb)

    if not results_pose.pose_landmarks:
        return None

    pose_lm = results_pose.pose_landmarks.landmark
    shoulder_center_x = (pose_lm[11].x + pose_lm[12].x) / 2
    shoulder_center_y = (pose_lm[11].y + pose_lm[12].y) / 2
    shoulder_width = np.linalg.norm([
        pose_lm[11].x - pose_lm[12].x,
        pose_lm[11].y - pose_lm[12].y
    ]) + 1e-6

    pose_indices = [11, 12, 13, 14, 15, 16, 23, 24]
    pose_features = []
    for idx in pose_indices:
        pose_features.append((pose_lm[idx].x - shoulder_center_x) / shoulder_width)
        pose_features.append((pose_lm[idx].y - shoulder_center_y) / shoulder_width)

    left_hand_features = np.zeros(75)
    right_hand_features = np.zeros(75)

    if results_hands.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results_hands.multi_hand_landmarks):
            handedness = results_hands.multi_handedness[i].classification[0].label
            joint = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
            v, angle_label = Vector_Normalization(joint)
            hand_features = np.concatenate([v.flatten(), angle_label.flatten()])
            if handedness == "Left":
                left_hand_features = hand_features
            elif handedness == "Right":
                right_hand_features = hand_features

    final_features = np.concatenate([pose_features, left_hand_features, right_hand_features])
    return final_features

def predict_with_model(sequence: np.ndarray) -> Tuple[str, float]:
    sequence_standardized = (sequence - data_mean) / (data_std + 1e-8)
    input_tensor = torch.tensor(sequence_standardized, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)

    confidence, predicted_idx = torch.max(probabilities, 1)
    predicted_label = labels_map.get(predicted_idx.item(), "Unknown")

    print(f"예측 결과: {predicted_label}, 확신도: {confidence.item() * 100:.2f}%")
    return predicted_label, confidence.item() * 100
