# model/SignLanguageRecognizer.py

from collections import deque
import cv2
import mediapipe as mp
import numpy as np
import torch
import tensorflow as tf
import json
import os

CONFIG = {
    "MODEL_DIR": "./model",
    "SEQ_LEN_WORD": 45,
    "STABLE_THRESHOLD_WORD": 1,
    "CONF_THRESHOLD_WORD": 0.85,
    "SEQ_LEN_ALPHABET": 10,
    "CONF_THRESHOLD_ALPHABET": 0.78,
    "IDLE_TIME_SECS": 1.8,
    "MOVEMENT_THRESHOLD": 0.6,
}

class FeatureExtractor:
    """
    Mediapipe를 사용하여 비디오 프레임에서 수어 인식을 위한 포즈 및 손 특징점을 추출합니다.
    단어 모델과 지문자 모델을 위한 특징을 모두 계산하고 움직임을 감지합니다.
    (기존 LSTM_video_OOP2B.py의 클래스를 기반으로 함)
    """
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.pose = self.mp_pose.Pose(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.8)
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.8)
        self.previous_features = None
        self.previous_velocity = None

    def _get_75d_hand_features(self, joint):
        v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :]
        v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :]
        v = v2 - v1
        v = v / (np.linalg.norm(v, axis=1)[:, np.newaxis] + 1e-6)
        angle_v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :2]
        angle_v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :2]
        angle_v = angle_v2 - angle_v1
        angle_v = angle_v / (np.linalg.norm(angle_v, axis=1)[:, np.newaxis] + 1e-6)
        angle = np.degrees(np.arccos(np.einsum('nt,nt->n',
            angle_v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:],
            angle_v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])))
        return np.concatenate([v.flatten(), angle])

    def _get_55d_hand_features(self, joint):
        v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :2]
        v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :2]
        v = v2 - v1
        v = v / (np.linalg.norm(v, axis=1)[:, np.newaxis] + 1e-6)
        angle = np.degrees(np.arccos(np.einsum('nt,nt->n',
            v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:],
            v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])))
        return np.concatenate([v.flatten(), angle])

    def extract(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_pose = self.pose.process(frame_rgb)
        results_hands = self.hands.process(frame_rgb)

        pose_features = np.zeros(16)
        if results_pose.pose_landmarks:
            pose_lm = results_pose.pose_landmarks.landmark
            shoulder_center_x = (pose_lm[11].x + pose_lm[12].x) / 2
            shoulder_center_y = (pose_lm[11].y + pose_lm[12].y) / 2
            shoulder_width = np.linalg.norm([pose_lm[11].x - pose_lm[12].x, pose_lm[11].y - pose_lm[12].y]) + 1e-6
            temp_pose_features = []
            for idx in [11, 12, 13, 14, 15, 16, 23, 24]:
                temp_pose_features.append((pose_lm[idx].x - shoulder_center_x) / shoulder_width)
                temp_pose_features.append((pose_lm[idx].y - shoulder_center_y) / shoulder_width)
            pose_features = np.array(temp_pose_features)

        left_hand_features, right_hand_features = np.zeros(75), np.zeros(75)
        alphabet_feature = None
        if results_hands.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(results_hands.multi_hand_landmarks):
                handedness = results_hands.multi_handedness[i].classification[0].label
                joint = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
                hand_features_75d = self._get_75d_hand_features(joint)
                if handedness == "Left":
                    left_hand_features = hand_features_75d
                elif handedness == "Right":
                    right_hand_features = hand_features_75d
                    alphabet_feature = self._get_55d_hand_features(joint)

        word_model_features = np.concatenate([pose_features, left_hand_features, right_hand_features])
        if self.previous_features is None: self.previous_features = np.zeros_like(word_model_features)
        velocity = word_model_features - self.previous_features
        if self.previous_velocity is None: self.previous_velocity = np.zeros_like(velocity)
        acceleration = velocity - self.previous_velocity
        movement = np.sum(np.abs(velocity))
        self.previous_features = word_model_features
        self.previous_velocity = velocity
        dynamic_word_features = np.concatenate([word_model_features, velocity, acceleration])
        return dynamic_word_features, alphabet_feature, movement


        #num_hands_detected = len(results_hands.multi_hand_landmarks) if results_hands.multi_hand_landmarks else 0
        #return dynamic_word_features, alphabet_feature, movement,num_hands_detected

    def close(self):
        self.pose.close()
        self.hands.close()

class Predictor:
    """
    단어(PyTorch) 및 지문자(TFLite) 모델을 로드하고 예측을 수행합니다.
    (기존 LSTM_video_OOP2B.py의 클래스를 기반으로 함)
    """
    def __init__(self, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config

        # Word Model (PyTorch)
        self.word_model = torch.jit.load(os.path.join(config['MODEL_DIR'], "lstm_sign_language_model_scripted.pt")).to(self.device)
        self.word_model.eval()
        self.data_mean = np.load(os.path.join(config['MODEL_DIR'], "data_mean.npy"))
        self.data_std = np.load(os.path.join(config['MODEL_DIR'], "data_std.npy"))
        with open(os.path.join(config['MODEL_DIR'], "label_map.json"), 'r', encoding='utf-8') as f:
            self.word_labels_map = {v: k for k, v in json.load(f).items()}

        # Fingerspelling Model (TFLite)
        tflite_path = os.path.join(config['MODEL_DIR'], "multi_hand_gesture_classifier.tflite")
        self.alphabet_interpreter = tf.lite.Interpreter(model_path=tflite_path)
        self.alphabet_interpreter.allocate_tensors()
        self.alphabet_input_details = self.alphabet_interpreter.get_input_details()
        self.alphabet_output_details = self.alphabet_interpreter.get_output_details()
        self.alphabet_actions = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ','ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ','ㅐ', 'ㅒ', 'ㅔ', 'ㅖ', 'ㅢ', 'ㅚ', 'ㅟ']

        # Buffers
        self.word_buffer = deque()
        self.word_history = []
        self.alphabet_buffer = deque(maxlen=config['SEQ_LEN_ALPHABET'])
        self.alphabet_confirm_buffer = deque(maxlen=5)
        self.last_confirmed_alphabet = None

    def predict_word(self, features):
        self.word_buffer.append(features)
        if len(self.word_buffer) > self.config['SEQ_LEN_WORD']: self.word_buffer.popleft()
        if len(self.word_buffer) < self.config['SEQ_LEN_WORD']: return None, 0.0

        sequence = np.array(list(self.word_buffer))
        normalized_sequence = (sequence - self.data_mean) / (self.data_std + 1e-8)
        input_tensor = torch.tensor(normalized_sequence, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            probabilities = torch.softmax(self.word_model(input_tensor), dim=1)
        confidence, idx = torch.max(probabilities, 1)
        confidence_item = confidence.item()
        
        # 디버깅용 출력
        label_for_debug = self.word_labels_map.get(idx.item(), "Unknown")
        print(f"    [Predictor|Word] Raw Predict: '{label_for_debug}' (Conf: {confidence_item:.4f})")

        if confidence_item < self.config['CONF_THRESHOLD_WORD']: return None, 0.0
        label = self.word_labels_map.get(idx.item(), "Unknown")
        self.word_history.append(label)
        if len(self.word_history) > self.config['STABLE_THRESHOLD_WORD']: self.word_history.pop(0)
        if len(self.word_history) == self.config['STABLE_THRESHOLD_WORD'] and len(set(self.word_history)) == 1:
            return self.word_history[0], confidence_item
        return None, 0.0

    def predict_fingerspelling(self, features):
        if features is None:
            self.alphabet_confirm_buffer.append(None)
            self.last_confirmed_alphabet = None
            return None, 0.0

        self.alphabet_buffer.append(features)
        if len(self.alphabet_buffer) < self.config['SEQ_LEN_ALPHABET']: return None, 0.0

        input_data = np.expand_dims(np.array(self.alphabet_buffer, dtype=np.float32), axis=0)
        self.alphabet_interpreter.set_tensor(self.alphabet_input_details[0]['index'], input_data)
        self.alphabet_interpreter.invoke()
        y_pred = self.alphabet_interpreter.get_tensor(self.alphabet_output_details[0]['index'])
        i_pred = int(np.argmax(y_pred[0]))
        confidence = y_pred[0][i_pred]
        
        # 디버깅용 출력
        char_for_debug = self.alphabet_actions[i_pred]
        print(f"    [🔍Predictor|Alphabet] Raw Predict: '{char_for_debug}' (Conf: {confidence:.4f})")

        if confidence > self.config['CONF_THRESHOLD_ALPHABET']:
            self.alphabet_confirm_buffer.append(self.alphabet_actions[i_pred])
        else:
            self.alphabet_confirm_buffer.append(None)
            self.last_confirmed_alphabet = None

        if (len(self.alphabet_confirm_buffer) == self.alphabet_confirm_buffer.maxlen and
                len(set(self.alphabet_confirm_buffer)) == 1 and self.alphabet_confirm_buffer[0] is not None):
            current_stable_prediction = self.alphabet_confirm_buffer[0]
            if current_stable_prediction != self.last_confirmed_alphabet:
                self.last_confirmed_alphabet = current_stable_prediction
                return current_stable_prediction, confidence
        return None, 0.0

    def reset_word_buffer(self):
        self.word_buffer.clear()
        self.word_history.clear()

    def reset(self):
        self.reset_word_buffer()
        self.alphabet_buffer.clear()
        self.alphabet_confirm_buffer.clear()
        self.last_confirmed_alphabet = None


class SignLanguageRecognizer:
    """
    통합된 수어 인식기 클래스.
    실시간 프레임 스트림 처리와 단일 비디오 파일 처리를 모두 담당합니다.
    """
    def __init__(self, config):
        self.config = config
        self.feature_extractor = FeatureExtractor()
        self.predictor = Predictor(config)
        self.sentence_words = []
        self.idle_counter = 0
        fps = 30  # Assume 30 fps
        self.IDLE_TIME_THRESHOLD_FRAMES = int(self.config['IDLE_TIME_SECS'] * fps)
        
        
        #self.HAND_HISTORY_LENGTH = 15 
        #self.hand_presence_history = deque(maxlen=self.HAND_HISTORY_LENGTH)
        
        self.frame_process_counter = 0

    def reset(self):
        """인식기 상태(누적 단어, 카운터, 예측 버퍼)를 완전히 초기화합니다."""
        self.sentence_words.clear()
        self.idle_counter = 0
        self.predictor.reset()
        self.frame_process_counter = 0
        print("--- Recognizer state has been reset. ---")

    def process_frame(self, frame: np.ndarray) -> str | None:
        """
        단일 프레임을 처리하여 단어/지문자를 인식하고 상태를 업데이트합니다.
        새로운 토큰(단어/지문자)이 안정적으로 인식되면 해당 토큰을 반환합니다.
        (실시간 스트리밍용 메소드)
        """
        
        self.frame_process_counter += 1
        if self.frame_process_counter % 2 != 0:
            print(f"    [SKIPPING FRAME #{self.frame_process_counter}]")
            return None
        
        
        word_feats, alphabet_feats, movement = self.feature_extractor.extract(frame)

        if movement < self.config['MOVEMENT_THRESHOLD']:
            self.idle_counter += 1
        else:
            self.idle_counter = 0

        if self.idle_counter >= self.IDLE_TIME_THRESHOLD_FRAMES:
            if self.sentence_words:
                print(f"IDLE - Resetting sentence: {' '.join(self.sentence_words)}")
            self.reset()
            return None

        predicted_word, word_conf = self.predictor.predict_word(word_feats)
        predicted_alphabet, alphabet_conf = self.predictor.predict_fingerspelling(alphabet_feats)

        # 디버깅용 출력
        print(f"[⚙️Recognizer|Stats] WordConf={word_conf:.2f}, AlphaConf={alphabet_conf:.2f}, Movement={movement:.2f}, Idle={self.idle_counter}")
        
        newly_recognized_token = None
        # 지문자 신뢰도가 단어 신뢰도보다 0.1(10%) 이상 높을 때만 지문자로 인정
        if predicted_alphabet and alphabet_conf > self.config.get('CONF_THRESHOLD_ALPHABET', 0.8) and alphabet_conf > word_conf + 0.1:
            self.idle_counter = 0
            if not self.sentence_words or self.sentence_words[-1] != predicted_alphabet:
                self.sentence_words.append(predicted_alphabet)
                newly_recognized_token = predicted_alphabet
                print(f"    ✅ [Recognizer] Alphabet Appended: '{predicted_alphabet}' (Conf: {alphabet_conf:.2f}) -> Current: '{' '.join(self.sentence_words)}'")
            self.predictor.reset_word_buffer()
        elif predicted_word and word_conf > self.config.get('CONF_THRESHOLD_WORD', 0.89):
            self.idle_counter = 0
            if not self.sentence_words or self.sentence_words[-1] != predicted_word:
                self.sentence_words.append(predicted_word)
                newly_recognized_token = predicted_word
                print(f"    ✅ [Recognizer] Word Appended: '{predicted_word}' (Conf: {word_conf:.2f}) -> Current: '{' '.join(self.sentence_words)}'")
            self.predictor.reset_word_buffer()

        return newly_recognized_token

    
    def get_full_sentence(self) -> str:
        """현재까지 누적된 전체 문장을 문자열로 반환합니다."""
        return " ".join(self.sentence_words)

    def recognize_from_video_file(self, video_path: str) -> str:
        """
        하나의 비디오 파일 전체를 처리하여 최종 인식된 문장(또는 단어)을 반환합니다.
        (학습 및 동영상 파일 번역용 메소드)
        """
        self.reset()  # 메소드 실행 전 상태를 깨끗하게 초기화
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file at {video_path}")
            return "비디오 파일을 열 수 없습니다."

        frame_idx = 0 # ---  디버깅용 프레임 카운터  ---
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1
            
            
            
            if frame_idx % 100 == 0:
                print(f"    [ℹ️] Processing video file... Frame {frame_idx}")
            
            # 비디오 프레임을 뒤집어 처리 (필요시)
            frame = cv2.flip(frame, 1)
            self.process_frame(frame) # 각 프레임을 순서대로 처리하여 내부 상태에 단어를 축적

        cap.release()
        final_sentence = self.get_full_sentence()
        print(f"Video processing finished. Result: '{final_sentence}'")
        self.reset()  
        
        
        if not final_sentence:
            return "학습되지 않은 동작입니다"
            
        return final_sentence

    def close(self):
        """Mediapipe 리소스를 정리합니다."""
        self.feature_extractor.close()