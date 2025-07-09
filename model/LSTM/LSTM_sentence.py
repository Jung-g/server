import cv2
import mediapipe as mp
import numpy as np
import torch
import json
import os
import torch.nn.functional as F
from model.LSTM.utils import Vector_Normalization
from pyctcdecode import build_ctcdecoder

MODEL_DIR = "D:/Bit/server/model"

def run_sentence_model(video_path: str) -> str:
    # 경로 및 하이퍼파라미터    
    model_path = os.path.join(MODEL_DIR, "best_sentence_model.pth")
    vocab_path = os.path.join(MODEL_DIR, "vocab.json")
    INPUT_DIM = 498
    HIDDEN_DIM = 256
    NUM_LAYERS = 2
    SEQ_LEN = 90
    OVERLAP_LEN = 80
    FEATURE_DIM = 166

    # 클래스 정의
    class SentenceLSTM_CNN(torch.nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, output_size):
            super().__init__()
            self.conv1 = torch.nn.Conv1d(input_size, 256, 3, padding=1)
            self.conv2 = torch.nn.Conv1d(256, 256, 3, padding=1)
            self.lstm = torch.nn.LSTM(256, hidden_size, num_layers, batch_first=True, bidirectional=True)
            self.fc = torch.nn.Linear(hidden_size * 2, output_size)

        def forward(self, x):
            x = x.permute(0, 2, 1)  # (B, T, C) -> (B, C, T)
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = x.permute(0, 2, 1)  # (B, C, T) -> (B, T, C)
            x, _ = self.lstm(x)
            x = self.fc(x)
            x = F.log_softmax(x, dim=2)
            return x.permute(1, 0, 2)  # (T, B, C)

    # 장치 설정 및 모델 로드
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        with open(vocab_path, encoding='utf-8') as f:
            word_to_idx = json.load(f)
        idx_to_word = {i: w for w, i in word_to_idx.items()}
        vocab = [w for w in idx_to_word.values() if w not in ['<pad>', '<blank>']]
        decoder = build_ctcdecoder(vocab, kenlm_model_path=None, alpha=0, beta=0.6)
    except:
        return "어휘 사전 또는 디코더 생성 실패"

    model = SentenceLSTM_CNN(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, len(word_to_idx)).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    except:
        return "모델 로드 실패"

    # 비디오 로드
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "영상 파일 열기에 실패했습니다."

    # MediaPipe 초기화
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands
    pose = mp_pose.Pose(model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    buffer = []
    prev_feat = None
    prev_vel = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_res = pose.process(image)
        hand_res = hands.process(image)

        final_feat = np.zeros(FEATURE_DIM)
        if pose_res.pose_landmarks:
            lm = pose_res.pose_landmarks.landmark
            scx = (lm[11].x + lm[12].x) / 2
            scy = (lm[11].y + lm[12].y) / 2
            sw = np.linalg.norm([lm[11].x - lm[12].x, lm[11].y - lm[12].y]) + 1e-6
            pose_idx = [11, 12, 13, 14, 15, 16, 23, 24]
            pose_feat = [(lm[i].x - scx) / sw for i in pose_idx] + [(lm[i].y - scy) / sw for i in pose_idx]

            lh_feat = np.zeros(75)
            rh_feat = np.zeros(75)
            if hand_res.multi_hand_landmarks:
                for i, h in enumerate(hand_res.multi_hand_landmarks):
                    joint = np.array([[lm.x, lm.y, lm.z] for lm in h.landmark])
                    v, a = Vector_Normalization(joint)
                    feat = np.concatenate([v.flatten(), a.flatten()])
                    if hand_res.multi_handedness[i].classification[0].label == "Left":
                        lh_feat = feat
                    else:
                        rh_feat = feat

            final_feat = np.concatenate([pose_feat, lh_feat, rh_feat])

        velocity = np.zeros_like(final_feat) if prev_feat is None else final_feat - prev_feat
        accel = np.zeros_like(velocity) if prev_vel is None else velocity - prev_vel
        prev_feat = final_feat
        prev_vel = velocity

        buffer.append(np.concatenate([final_feat, velocity, accel]))

    cap.release()
    cv2.destroyAllWindows()

    # 예측
    if not buffer:
        return "영상에서 특징을 추출하지 못했습니다."

    x = torch.FloatTensor([buffer]).to(device)
    with torch.no_grad():
        log_probs = model(x)

    probs = F.softmax(log_probs.squeeze(1).cpu(), dim=-1)
    word_probs = probs[:, 2:]
    blank = probs[:, 1].unsqueeze(1)
    probs_input = torch.cat([word_probs, blank], dim=1)

    raw_pred = decoder.decode(probs_input.numpy())

    # 띄어쓰기 후처리
    def post_process(text, vocab):
        sorted_vocab = sorted(vocab, key=len, reverse=True)
        result = []
        while text:
            for word in sorted_vocab:
                if text.startswith(word):
                    result.append(word)
                    text = text[len(word):]
                    break
            else:
                result.append(text[0])
                text = text[1:]
        return " ".join(result)

    final = post_process(raw_pred, vocab)
    return final if final.strip() else "학습되지 않은 동작입니다"
