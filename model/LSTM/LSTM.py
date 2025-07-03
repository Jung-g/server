import json
import os
import numpy as np
import torch
import cv2
import mediapipe as mp
from model.LSTM.utils import Vector_Normalization

MODEL_DIR = "D:\Bit\server\model"
model_path = os.path.join(MODEL_DIR, "lstm_sign_language_model_scripted.pt")
mean_path = os.path.join(MODEL_DIR, "data_mean.npy")
std_path = os.path.join(MODEL_DIR, "data_std.npy")
label_map_path = os.path.join(MODEL_DIR, "label_map.json")

FEATURE_DIM = 166
SEQ_LEN = 90
FULL_FEATURE_DIM = 498

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델, 표준화 파라미터, 라벨맵 로드
model = torch.jit.load(model_path)
model.eval()
model.to(device)
data_mean = np.load(mean_path)
data_std = np.load(std_path)
with open(label_map_path, 'r', encoding='utf-8') as f:
    labels_map_raw = json.load(f)
labels_map = {v: k for k, v in labels_map_raw.items()}

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

pose_model = mp_pose.Pose(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands_model = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

def preprocess_frame(frame, pose, hands):
    final_features = np.zeros(FEATURE_DIM)

    frame_small = cv2.resize(frame, (480, 360), interpolation=cv2.INTER_AREA)
    # frame_small = resize_video(frame)
    frame_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)

    results_pose = pose.process(frame_rgb)
    results_hands = hands.process(frame_rgb)

    if results_pose.pose_landmarks:
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

def build_sequence_from_frames(frames):
    sequence = []
    prev = None
    prev_v = None
    fail_count = 0
    with mp_pose.Pose(static_image_mode=False) as pose, mp_hands.Hands(static_image_mode=False, max_num_hands=2) as hands:
        for idx, frame in enumerate(frames):
            f = preprocess_frame(frame, pose, hands)
            v = np.zeros_like(f) if prev is None else f - prev
            a = np.zeros_like(v) if prev_v is None else v - prev_v
            combined = np.concatenate([f, v, a])
            sequence.append(combined)
            prev = f
            prev_v = v
            if np.count_nonzero(f) == 0:
                print(f"포즈 인식 실패: frame[{idx}]")
                fail_count += 1
    print(f"전체 중 pose 인식 실패 프레임 수: {fail_count}/{len(frames)}")
    padded_sequence = np.zeros((SEQ_LEN, FULL_FEATURE_DIM), dtype=np.float32)
    length = min(SEQ_LEN, len(sequence))
    padded_sequence[:length] = sequence[:length]
    return padded_sequence

def predict_sign_language(sequence: np.ndarray) -> str:
    assert sequence.shape == (SEQ_LEN, FULL_FEATURE_DIM), f"Expected shape ({SEQ_LEN}, {FULL_FEATURE_DIM}), got {sequence.shape}"
    sequence_standardized = (sequence - data_mean) / (data_std + 1e-8)
    input_tensor = torch.tensor(sequence_standardized, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
    confidence, predicted_idx = torch.max(probabilities, 1)
    predicted_label = labels_map.get(predicted_idx.item(), "Unknown")
    print(f"예측: {predicted_label}, 확신도: {confidence.item():.4f}")
    if confidence.item() > 0.2:
        predicted_label = labels_map.get(predicted_idx.item(), "Unknown")
        return predicted_label
    else:
        return "알 수 없는 단어"

def result(frames, stride=5, threshold=3):
    prediction_history = []

    for i in range(0, len(frames) - SEQ_LEN + 1, stride):
        sub_frames = frames[i:i + SEQ_LEN]
        sequence = build_sequence_from_frames(sub_frames)
        label = predict_sign_language(sequence)
        prediction_history.append(label)

        if len(prediction_history) >= threshold:
            recent = prediction_history[-threshold:]
            if len(set(recent)) == 1:
                return recent[0]

    if prediction_history:
        return max(set(prediction_history), key=prediction_history.count)
    else:
        return "알 수 없는 단어"

# def resize_video(frame, target_width=480, target_height=360):
#     h, w = frame.shape[:2]
#     scale = min(target_width / w, target_height / h)
#     new_w, new_h = int(w * scale), int(h * scale)
#     resized = cv2.resize(frame, (new_w, new_h))

#     top = (target_height - new_h) // 2
#     bottom = target_height - new_h - top
#     left = (target_width - new_w) // 2
#     right = target_width - new_w - left

#     padded = cv2.copyMakeBorder(resized, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
#     return padded
