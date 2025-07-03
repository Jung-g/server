import cv2
import mediapipe as mp
import numpy as np
import torch
import time
import json
import os

from PIL import ImageFont, ImageDraw, Image
from utils import Vector_Normalization # ⭐ 1. Vector_Normalization 함수를 불러옵니다.

# Mediapipe 초기화
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
pose = mp_pose.Pose(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# 웹캠 또는 IP 카메라 스트림 열기
# cap = cv2.VideoCapture('http://10.101.112.153:4747/video')
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# --- 설정값 로드 ---
MODEL_DIR = "D:\Bit\server\model"
model_path = os.path.join(MODEL_DIR, "lstm_sign_language_model_scripted.pt")
mean_path = os.path.join(MODEL_DIR, "data_mean.npy")
std_path = os.path.join(MODEL_DIR, "data_std.npy")
label_map_path = os.path.join(MODEL_DIR, "label_map.json")

SEQ_LEN = 90
OVERLAP_LEN = 85

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device for GUI inference: {device}")

# 모델 로드
try:
    model = torch.jit.load(model_path)
    model.eval()
    model.to(device)
    print(f"Model loaded successfully from {model_path} and moved to {device}.")
except Exception as e:
    print(f"Error loading model from {model_path}: {e}")
    exit()

# 표준화 파라미터 로드
try:
    data_mean = np.load(mean_path)
    data_std = np.load(std_path)
    print("Standardization mean and std loaded successfully.")
except Exception as e:
    print(f"Error loading standardization parameters: {e}")
    exit()

# 라벨 맵 로드
try:
    with open(label_map_path, 'r', encoding='utf-8') as f:
        labels_map_raw = json.load(f)
    labels_map = {v: k for k, v in labels_map_raw.items()}
    print("Label map loaded successfully.")
except Exception as e:
    print(f"Error loading label map from {label_map_path}: {e}")
    exit()

# 버퍼 및 예측 관련 변수 초기화
keypoints_buffer = []
prediction_history = [] # 최근 예측 단어들을 저장할 리스트
STABLE_THRESHOLD = 3 # 몇 번 연속으로 일치해야 인정할지 결정 (3번)
stable_prediction = "Waiting..." # 안정화된 최종 예측 결과
stable_confidence = 0.0 
current_prediction = "Waiting..."
current_confidence = 0.0
current_probabilities = {label: 0.0 for label in labels_map.values()}

# --- 👇 유휴 상태 감지용 변수 추가 ---
IDLE_THRESHOLD = 0.5  # 움직임 감지를 위한 임계값. 이 값보다 작으면 정지 상태로 간주.
IDLE_TIME_THRESHOLD = 45 # 유휴 상태로 판단하기 위한 프레임 수 (약 1.5초, 30fps 기준)
idle_counter = 0      # 정지 상태가 지속된 프레임 수를 세는 카운터

# --- 💡 수정/추가된 부분 시작 ---
# 동적 특징 계산을 위한 변수 추가
previous_features = None
previous_velocity = None
FEATURE_DIM = 166 # 원본 특징(좌표)의 차원
# --- 💡 수정/추가된 부분 끝 ---



# 폰트 설정
try:
    font_path = "C:/Windows/Fonts/malgunbd.ttf"
    font = ImageFont.truetype(font_path, 30)
    font_small = ImageFont.truetype(font_path, 20)
except IOError:
    print("Error: Could not load font. Using default.")
    font = ImageFont.load_default()
    font_small = ImageFont.load_default()

# --- 프레임 처리 관련 변수 ---
frame_process_counter = 0
last_results_pose = None
last_results_hands = None
start_time = time.time()
fps_frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break


    frame_process_counter += 1
    frame_bgr = frame.copy()
    
    # --------------------------------------------------------------------------
    # 1. 특정 프레임마다 모델 입력 데이터 생성 및 예측 (핵심 처리 로직)
    # --------------------------------------------------------------------------
    if frame_process_counter % 2 == 0:
        frame_small = cv2.resize(frame, (480, 360), interpolation=cv2.INTER_AREA)
        frame_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)

        results_pose = pose.process(frame_rgb)
        results_hands = hands.process(frame_rgb)
        # 나중에 부드러운 시각화를 위해 결과 저장
        last_results_pose = results_pose
        last_results_hands = results_hands

        # ⭐ 2. 키포인트 추출 및 정규화 로직 전체 변경
        final_features = np.zeros(FEATURE_DIM) # 포즈(16) + 왼손(75) + 오른손(75) = 166


        if results_pose.pose_landmarks:# 포즈가 감지되면, 실제 랜드마크 값으로 final_features를 채움
            pose_lm = results_pose.pose_landmarks.landmark

            # 몸의 기준점(어깨 중심)과 크기(어깨너비) 계산
            shoulder_center_x = (pose_lm[11].x + pose_lm[12].x) / 2
            shoulder_center_y = (pose_lm[11].y + pose_lm[12].y) / 2
            shoulder_width = np.linalg.norm(
                [pose_lm[11].x - pose_lm[12].x, pose_lm[11].y - pose_lm[12].y]
            ) + 1e-6

            # 상체 포즈 특징 추출 및 정규화
            pose_indices = [11, 12, 13, 14, 15, 16, 23, 24] # 어깨, 팔꿈치, 손목, 골반
            pose_features = []
            for idx in pose_indices:
                pose_features.append((pose_lm[idx].x - shoulder_center_x) / shoulder_width)
                pose_features.append((pose_lm[idx].y - shoulder_center_y) / shoulder_width)
            
            # 양손 특징 추출 (벡터 정규화)
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

            # 모든 특징 결합
            final_features = np.concatenate([pose_features, left_hand_features, right_hand_features])
            
         # --- 동적 특징 및 유휴 상태 감지 (포즈 감지 여부와 상관없이 실행) ---
        velocity = np.zeros_like(final_features) if previous_features is None else final_features - previous_features
        acceleration = np.zeros_like(velocity) if previous_velocity is None else velocity - previous_velocity
            
        # 유휴 상태 감지
        movement = np.sum(np.abs(velocity))
        if movement < IDLE_THRESHOLD:
            idle_counter += 1
        else:
            idle_counter = 0

        if idle_counter >= IDLE_TIME_THRESHOLD:
            stable_prediction = "Waiting..."
            stable_confidence = 0.0
            prediction_history.clear()
            idle_counter = 0
            
        # 모든 특징 결합 후 버퍼에 추가
        combined_features = np.concatenate([final_features, velocity, acceleration])
        keypoints_buffer.append(combined_features)
        
        # 다음 프레임 계산을 위해 현재 값 업데이트
        previous_features = final_features
        previous_velocity = velocity

         # --- 모델 예측 (버퍼가 다 찼을 때만 실행) ---
        if len(keypoints_buffer) >= SEQ_LEN:
            sequence_to_predict = np.array(keypoints_buffer[-SEQ_LEN:])
            sequence_standardized = (sequence_to_predict - data_mean) / (data_std + 1e-8)
            input_tensor = torch.tensor(sequence_standardized, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
            
            confidence, predicted_idx = torch.max(probabilities, 1)
            
            # 안정화 필터 로직
            if confidence.item() > 0.7:
                predicted_label = labels_map.get(predicted_idx.item(), "Unknown")
                prediction_history.append(predicted_label)
                if len(prediction_history) > STABLE_THRESHOLD:
                    prediction_history.pop(0)
                if len(prediction_history) == STABLE_THRESHOLD and len(set(prediction_history)) == 1:
                    if stable_prediction != prediction_history[0]:
                        stable_prediction = prediction_history[0]
                        stable_confidence = confidence.item() * 100
            
            # 오래된 버퍼 제거 (슬라이딩 윈도우)
            del keypoints_buffer[:SEQ_LEN - OVERLAP_LEN]
    
    # --------------------------------------------------------------------------
    # 2. 시각화 로직 (매 프레임마다 부드럽게 보여주기)
    # --------------------------------------------------------------------------
    if last_results_pose and last_results_pose.pose_landmarks:
        # 랜드마크 그리기 (필요 시 주석 해제)
        # mp_drawing.draw_landmarks(...) 
        pass
    
    img_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    # 예측 결과 텍스트 표시
    text_prediction = f"Prediction: {stable_prediction} ({stable_confidence:.2f}%)"
    draw.text((10, 30), text_prediction, font=font, fill=(255, 255, 0))
    
    # 버퍼 상태 텍스트 표시
    text_frame_count = f"Frames in buffer: {len(keypoints_buffer)} / {SEQ_LEN}"
    draw.text((10, 65), text_frame_count, font=font_small, fill=(255, 165, 0))
    
    frame_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    cv2.imshow('Sign Language Recognition', frame_bgr)

    # --------------------------------------------------------------------------
    # 3. 키보드 입력 처리
    # --------------------------------------------------------------------------
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        print("Prediction state has been reset by user.")
        keypoints_buffer.clear()
        prediction_history.clear()
        stable_prediction = "Waiting..."
        stable_confidence = 0.0
        idle_counter = 0

cap.release()
cv2.destroyAllWindows()