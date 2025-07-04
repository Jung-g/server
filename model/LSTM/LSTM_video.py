import cv2
import mediapipe as mp
import numpy as np
import torch
import time
import json
import os
from PIL import ImageFont, ImageDraw, Image
from model.LSTM.utils import Vector_Normalization

# VIDEO_FILE_PATH = "C:/Users/bit/Desktop/777.mp4" 
MODEL_DIR = "D:\Bit\server\model"

def run_model(VIDEO_FILE_PATH: str) -> str:
    # Mediapipe 초기화
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands
    pose = mp_pose.Pose(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(VIDEO_FILE_PATH)
    if not cap.isOpened():
        return "영상 파일 열기에 실패했습니다."

    # 설정 경로
    model_path = os.path.join(MODEL_DIR, "lstm_sign_language_model_scripted.pt")
    mean_path = os.path.join(MODEL_DIR, "data_mean.npy")
    std_path = os.path.join(MODEL_DIR, "data_std.npy")
    label_map_path = os.path.join(MODEL_DIR, "label_map.json")

    SEQ_LEN = 45
    OVERLAP_LEN = 40
    FEATURE_DIM = 166
    STABLE_THRESHOLD = 3
    IDLE_THRESHOLD = 0.5
    IDLE_TIME_THRESHOLD = 45

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델 로드
    try:
        model = torch.jit.load(model_path).to(device)
        model.eval()
    except Exception as e:
        return f"모델 로드 오류: {e}"

    # 표준화 파라미터
    try:
        data_mean = np.load(mean_path)
        data_std = np.load(std_path)
    except Exception as e:
        return f"표준화 파라미터 오류: {e}"

    # 라벨 맵
    try:
        with open(label_map_path, 'r', encoding='utf-8') as f:
            labels_map_raw = json.load(f)
        labels_map = {v: k for k, v in labels_map_raw.items()}
    except Exception as e:
        return f"라벨맵 로드 오류: {e}"

    # 초기화
    keypoints_buffer = []
    prediction_history = []
    stable_prediction = "Waiting..."
    stable_confidence = 0.0
    idle_counter = 0
    previous_features = None
    previous_velocity = None
    frame_process_counter = 0

    print("\n분석 시작")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        frame_process_counter += 1

        if frame_process_counter % 2 != 0:
            continue

        frame_small = cv2.resize(frame, (480, 360))
        frame_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)

        results_pose = pose.process(frame_rgb)
        results_hands = hands.process(frame_rgb)

        final_features = np.zeros(FEATURE_DIM)

        if results_pose.pose_landmarks:
            pose_lm = results_pose.pose_landmarks.landmark
            shoulder_center_x = (pose_lm[11].x + pose_lm[12].x) / 2
            shoulder_center_y = (pose_lm[11].y + pose_lm[12].y) / 2
            shoulder_width = np.linalg.norm([
                pose_lm[11].x - pose_lm[12].x, pose_lm[11].y - pose_lm[12].y
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

        velocity = np.zeros_like(final_features) if previous_features is None else final_features - previous_features
        acceleration = np.zeros_like(velocity) if previous_velocity is None else velocity - previous_velocity

        previous_features = final_features
        previous_velocity = velocity

        movement = np.sum(np.abs(velocity))
        idle_counter = idle_counter + 1 if movement < IDLE_THRESHOLD else 0

        if idle_counter >= IDLE_TIME_THRESHOLD:
            stable_prediction = "Waiting..."
            stable_confidence = 0.0
            prediction_history.clear()
            idle_counter = 0

        combined_features = np.concatenate([final_features, velocity, acceleration])
        keypoints_buffer.append(combined_features)

        if len(keypoints_buffer) >= SEQ_LEN:
            sequence_to_predict = np.array(keypoints_buffer[-SEQ_LEN:])
            sequence_standardized = (sequence_to_predict - data_mean) / (data_std + 1e-8)
            input_tensor = torch.tensor(sequence_standardized, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)

            confidence, predicted_idx = torch.max(probabilities, 1)
            predicted_label = labels_map.get(predicted_idx.item(), "Unknown")
            print(f"Frame {frame_process_counter}: Raw Prediction='{predicted_label}', Confidence={confidence.item():.4f}")

            if confidence.item() > 0.7:
                prediction_history.append(predicted_label)
                if len(prediction_history) > STABLE_THRESHOLD:
                    prediction_history.pop(0)
                if len(prediction_history) == STABLE_THRESHOLD and len(set(prediction_history)) == 1:
                    if stable_prediction != prediction_history[0]:
                        stable_prediction = prediction_history[0]
                        stable_confidence = confidence.item() * 100
                        print(f"New stable prediction: {stable_prediction} ({stable_confidence:.2f}%)")

            del keypoints_buffer[:SEQ_LEN - OVERLAP_LEN]

    cap.release()
    cv2.destroyAllWindows()

    if stable_prediction == "인식실패 다시 시도해주세요" or stable_confidence < 70:
        return "학습되지 않은 동작입니다"

    return stable_prediction