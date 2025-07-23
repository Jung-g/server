# LSTM_video_OOP.py

from collections import deque
import cv2
import mediapipe as mp
import numpy as np
import torch
import tensorflow as tf
import json
import os
from PIL import ImageFont, ImageDraw, Image

CONFIG = {
        #"VIDEO_FILE_PATH": "C:/Users/bit/Desktop/KakaoTalk_20250716_211352901.mp4",
        "MODEL_DIR": "./model",
        
        # Word Model Parameters
        "SEQ_LEN_WORD": 60,
        "OVERLAP_LEN_WORD": 40,
        "STABLE_THRESHOLD_WORD": 1,
        "CONF_THRESHOLD_WORD": 0.89,
        
        # Fingerspelling Model Parameters
        "SEQ_LEN_ALPHABET": 10,
        "CONF_THRESHOLD_ALPHABET": 0.80,

        # General Parameters
        "IDLE_TIME_SECS": 2.5,
        "MOVEMENT_THRESHOLD": 0.5,
    }

class FeatureExtractor:
    """
    Extracts pose and hand keypoints from video frames for sign language recognition.
    It calculates features for both word and fingerspelling models and detects movement.
    """
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.pose = self.mp_pose.Pose(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.8)
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.8)
        
        self.previous_features = None
        self.previous_velocity = None

    def _get_75d_hand_features(self, joint):
        """Helper function to generate 75-dimensional hand features for the word model."""
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
        """Helper function to generate 55-dimensional hand features for the fingerspelling model."""
        v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :2]
        v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :2]
        v = v2 - v1
        v = v / (np.linalg.norm(v, axis=1)[:, np.newaxis] + 1e-6)
        angle = np.degrees(np.arccos(np.einsum('nt,nt->n',
            v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
            v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])))
        return np.concatenate([v.flatten(), angle])

    def extract(self, frame):
        """Processes a single frame to extract all required features."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_pose = self.pose.process(frame_rgb)
        results_hands = self.hands.process(frame_rgb)
        
        # <--- 변경된 부분: 감지된 손의 개수를 저장할 변수 ---
        num_hands_detected = 0
        if results_hands.multi_hand_landmarks:
            num_hands_detected = len(results_hands.multi_hand_landmarks)
        
        # --- 1. Word Model Feature Extraction (Pose + Hands) ---
        pose_features = np.zeros(16)
        word_model_features = np.zeros(166) # 16 (pose) + 75 (left) + 75 (right)
        
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
                
                # Feature for word model
                hand_features_75d = self._get_75d_hand_features(joint)
                if handedness == "Left":
                    left_hand_features = hand_features_75d
                elif handedness == "Right":
                    right_hand_features = hand_features_75d
                    # Feature for alphabet model (right hand only)
                    alphabet_feature = self._get_55d_hand_features(joint)

        word_model_features = np.concatenate([pose_features, left_hand_features, right_hand_features])

        # --- 2. Dynamic Feature Calculation (Velocity, Acceleration) ---
        if self.previous_features is None: self.previous_features = np.zeros_like(word_model_features)
        velocity = word_model_features - self.previous_features
        if self.previous_velocity is None: self.previous_velocity = np.zeros_like(velocity)
        acceleration = velocity - self.previous_velocity
        
        movement = np.sum(np.abs(velocity))
        self.previous_features = word_model_features
        self.previous_velocity = velocity

        dynamic_word_features = np.concatenate([word_model_features, velocity, acceleration])
        
        #return dynamic_word_features, alphabet_feature, movement, results_pose, results_hands
        return dynamic_word_features, alphabet_feature, movement, results_pose, results_hands, num_hands_detected


    def close(self):
        """Releases Mediapipe resources."""
        self.pose.close()
        self.hands.close()


class Predictor:
    """
    Manages loading and running both the word (PyTorch) and 
    fingerspelling (TFLite) models.
    """
    def __init__(self, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # --- Word Model (PyTorch) ---
        self.word_model = torch.jit.load(os.path.join(config['MODEL_DIR'], "lstm_sign_language_model_scripted.pt")).to(self.device)
        self.word_model.eval()
        self.data_mean = np.load(os.path.join(config['MODEL_DIR'], "data_mean.npy"))
        self.data_std = np.load(os.path.join(config['MODEL_DIR'], "data_std.npy"))
        with open(os.path.join(config['MODEL_DIR'], "label_map.json"), 'r', encoding='utf-8') as f:
            self.word_labels_map = {v: k for k, v in json.load(f).items()}
        
        # --- Fingerspelling Model (TFLite) ---
        tflite_path = os.path.join(config['MODEL_DIR'], "multi_hand_gesture_classifier.tflite")
        self.alphabet_interpreter = tf.lite.Interpreter(model_path=tflite_path)
        self.alphabet_interpreter.allocate_tensors()
        self.alphabet_input_details = self.alphabet_interpreter.get_input_details()
        self.alphabet_output_details = self.alphabet_interpreter.get_output_details()
        self.alphabet_actions = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ','ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ','ㅐ', 'ㅒ', 'ㅔ', 'ㅖ', 'ㅢ', 'ㅚ', 'ㅟ']

        # --- Buffers and Parameters ---
        self.word_buffer = deque()
        self.word_history = []
        self.alphabet_buffer = deque(maxlen=config['SEQ_LEN_ALPHABET'])
        
        self.alphabet_confirm_buffer = deque(maxlen=5) # 최근 3개 예측 저장용 버퍼
        self.last_confirmed_alphabet = None # 마지막으로 인식된 지문자를 저장
        
        self.config = config
        

    def predict_word(self, features):
        """
        특징(features) 시퀀스로부터 단어를 예측합니다.
        '호핑 윈도우' 방식을 사용하여 효율적으로 버퍼를 관리합니다.
        """
        # 1. 버퍼에 새로운 특징 추가
        self.word_buffer.append(features)
        
         # 2. 버퍼가 최대 길이를 초과하면 가장 오래된 데이터 제거
        if len(self.word_buffer) > self.config['SEQ_LEN_WORD']:
            self.word_buffer.popleft()

        # 3. 버퍼가 예측에 필요한 최소 길이에 도달했는지 확인
        if len(self.word_buffer) < self.config['SEQ_LEN_WORD']:
            return None, 0.0

        # 4. 예측 수행
        sequence = np.array(list(self.word_buffer))
        
        normalized_sequence = (sequence - self.data_mean) / (self.data_std + 1e-8)
        input_tensor = torch.tensor(normalized_sequence, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            probabilities = torch.softmax(self.word_model(input_tensor), dim=1)
        
        confidence, idx = torch.max(probabilities, 1)
        confidence_item = confidence.item()

        # 5. 예측 안정화 및 결과 반환 (이전과 동일)
        if confidence_item < self.config['CONF_THRESHOLD_WORD']:
            return None, 0.0
        
        label = self.word_labels_map.get(idx.item(), "Unknown")
        self.word_history.append(label)

        if len(self.word_history) > self.config['STABLE_THRESHOLD_WORD']:
            self.word_history.pop(0)

        if (len(self.word_history) == self.config['STABLE_THRESHOLD_WORD'] and
                len(set(self.word_history)) == 1):
            return self.word_history[0], confidence_item
        
        return None, 0.0

    def predict_fingerspelling(self, features):
        """
        특징 시퀀스로부터 지문자를 예측합니다. (최종 수정 버전)
        Debouncing과 마지막 인식 글자 비교를 통해 안정성을 대폭 향상시킵니다.
        """
        if features is None:
            self.alphabet_confirm_buffer.append(None)
            self.last_confirmed_alphabet = None 
            return None, 0.0

        self.alphabet_buffer.append(features)
        if len(self.alphabet_buffer) < self.config['SEQ_LEN_ALPHABET']:
            return None, 0.0

        input_data = np.expand_dims(np.array(self.alphabet_buffer, dtype=np.float32), axis=0)
        self.alphabet_interpreter.set_tensor(self.alphabet_input_details[0]['index'], input_data)
        self.alphabet_interpreter.invoke()
        y_pred = self.alphabet_interpreter.get_tensor(self.alphabet_output_details[0]['index'])

        i_pred = int(np.argmax(y_pred[0]))
        confidence = y_pred[0][i_pred]
        
        predicted_char_for_debug = self.alphabet_actions[i_pred]
        print(f"Debug: Char='{predicted_char_for_debug}', Conf={confidence:.4f}, Buffer={list(self.alphabet_confirm_buffer)}")

        if confidence > self.config['CONF_THRESHOLD_ALPHABET']:
            self.alphabet_confirm_buffer.append(self.alphabet_actions[i_pred])
        else:
            self.alphabet_confirm_buffer.append(None)
            self.last_confirmed_alphabet = None

        if (len(self.alphabet_confirm_buffer) == self.alphabet_confirm_buffer.maxlen and
                len(set(self.alphabet_confirm_buffer)) == 1 and
                self.alphabet_confirm_buffer[0] is not None):

            current_stable_prediction = self.alphabet_confirm_buffer[0]

            if current_stable_prediction != self.last_confirmed_alphabet:
                self.last_confirmed_alphabet = current_stable_prediction
                return current_stable_prediction, confidence

        return None, 0.0
    
    def reset_word_buffer(self):
        """Resets the word prediction buffer and history."""
        self.word_buffer.clear()
        self.word_history.clear()

""" 시각화부분 (필요없음)
class Visualizer:
    def __init__(self, font_path="C:/Windows/Fonts/malgunbd.ttf"):
        try:
            self.font = ImageFont.truetype(font_path, 30)
            self.font_small = ImageFont.truetype(font_path, 20)
        except IOError:
            self.font = ImageFont.load_default()
            self.font_small = ImageFont.load_default()
            print(f"Warning: Font not found at {font_path}. Using default font.")
        
        # <--- 변경된 부분: MediaPipe 그리기 유틸리티 초기화 ---
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands

    # <--- 변경된 부분: 메소드 시그니처에 landmark 결과 추가 ---
    def draw(self, frame, prediction, confidence, sentence, buffer_status, results_pose, results_hands):

        
        # <--- 변경된 부분: 랜드마크를 프레임에 먼저 그림 ---
        if results_pose and results_pose.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results_pose.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )

        if results_hands and results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                    self.mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                )

        # 랜드마크가 그려진 프레임 위에 텍스트를 그림
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        # Display current prediction
        display_text = "인식 결과: ..."
        if prediction:
            display_text = f"인식 결과: {prediction} ({confidence:.2f}%)"
        draw.text((10, 30), display_text, font=self.font, fill=(255, 255, 0)) # Yellow
        
        # Display buffer status
        draw.text((10, 65), buffer_status, font=self.font_small, fill=(255, 165, 0)) # Orange
        
        # Display the accumulating sentence
        draw.text((10, 95), f"문장: {' '.join(sentence)}", font=self.font, fill=(255, 255, 255)) # White
        
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

"""


class SignLanguageRecognizer:
    """
    Main class to orchestrate the sign language recognition process from a video file.
    """
    def __init__(self, config):
        self.config = config
        self.video_cap = cv2.VideoCapture(config['VIDEO_FILE_PATH'])
        if not self.video_cap.isOpened():
            raise IOError(f"Error: Could not open video file: {config['VIDEO_FILE_PATH']}")
        
        fps = self.video_cap.get(cv2.CAP_PROP_FPS)
        fps = 30 if fps == 0 else fps
        self.IDLE_TIME_THRESHOLD_FRAMES = int(config['IDLE_TIME_SECS'] * fps)
        
        # Initialize components
        self.feature_extractor = FeatureExtractor()
        self.predictor = Predictor(config)
        #self.visualizer = Visualizer()
        
        # State variables
        self.sentence_words = []
        self.current_prediction = None
        self.current_confidence = 0.0
        self.idle_counter = 0
        
        # 최근 15 프레임 동안의 손 개수를 저장합니다. (이 값은 튜닝 가능)
        self.HAND_HISTORY_LENGTH = 15 
        self.hand_presence_history = deque(maxlen=self.HAND_HISTORY_LENGTH)


    def run(self):
            """Starts the main recognition loop."""
            print(f"\n--- Starting Sign Language Recognition ---")
            print(f"Device: {self.predictor.device}")

            frame_process_counter = 0
            
            # <--- 변경된 부분: 랜드마크 결과를 저장할 변수 추가 ---
            results_pose, results_hands = None, None

            ret, frame = self.video_cap.read()
            if not ret:
                print("오류: 비디오 파일을 읽을 수 없습니다. 경로를 확인해 주세요.")
                self.cleanup()
                return
            
            # 초기 화면은 랜드마크 없이 텍스트만 그림
            #buffer_status = f"Word Buf: 0/{self.config['SEQ_LEN_WORD']} | Alpha Buf: 0/{self.config['SEQ_LEN_ALPHABET']}"
            #display_frame = self.visualizer.draw(cv2.flip(frame, 1), None, 0.0, [], buffer_status, None, None)


            while self.video_cap.isOpened():
                ret, frame = self.video_cap.read()
                if not ret:
                    print("End of video file reached.")
                    break
                
                frame = cv2.flip(frame, 1)
                frame_process_counter += 1

                if frame_process_counter % 2 == 0:
                    # <--- 변경된 부분: 특징 추출시 랜드마크 결과도 함께 받음 ---
                    #word_feats, alphabet_feats, movement, results_pose, results_hands = self.feature_extractor.extract(frame)
                    word_feats, alphabet_feats, movement, results_pose, results_hands, num_hands = self.feature_extractor.extract(frame)
                    
                    # <--- 변경된 부분: 손 감지 기록 버퍼 업데이트 ---
                    self.hand_presence_history.append(num_hands)

                    if movement < self.config['MOVEMENT_THRESHOLD']:
                        self.idle_counter += 1
                    else:
                        self.idle_counter = 0
                    
                    if self.idle_counter >= self.IDLE_TIME_THRESHOLD_FRAMES:
                        if self.sentence_words:
                            print(f"Final Sentence (due to inactivity): {' '.join(self.sentence_words)}")
                        self.sentence_words.clear()
                        self.predictor.reset_word_buffer()
                        self.idle_counter = 0
                        
                    # <--- ✨ 핵심 로직 변경 ✨ ---
                    # "최근에 두 손이 감지된 적이 있었는가?"를 확인
                    # hand_presence_history에 2(두 손)가 하나라도 있으면 True
                    #is_likely_word_gesture = any(h == 2 for h in self.hand_presence_history)
                    is_likely_word_gesture = self.hand_presence_history.count(2) > (self.HAND_HISTORY_LENGTH // 2)

                    predicted_alphabet, alphabet_conf = None, 0.0
                    
                    # 만약 최근 두 손이 감지된 적이 없거나, 오른손 특징(alphabet_feats)이 존재할 때만 지문자 예측 시도
                    if not is_likely_word_gesture and alphabet_feats is not None:
                        predicted_alphabet, alphabet_conf = self.predictor.predict_fingerspelling(alphabet_feats)
                    
                    if predicted_alphabet:
                        self.idle_counter = 0
                        self.current_prediction = predicted_alphabet
                        self.current_confidence = alphabet_conf * 100
                        
                        if not self.sentence_words or self.sentence_words[-1] != predicted_alphabet:
                            self.sentence_words.append(predicted_alphabet)
                            print(f"Fingerspelling Appended: {predicted_alphabet} | Current Sentence: {' '.join(self.sentence_words)}")
                        
                        self.predictor.reset_word_buffer()

                    else:
                        predicted_word, word_conf = self.predictor.predict_word(word_feats)
                        if predicted_word:
                            self.idle_counter = 0
                            self.current_prediction = predicted_word
                            self.current_confidence = word_conf * 100

                            if not self.sentence_words or self.sentence_words[-1] != predicted_word:
                                self.sentence_words.append(predicted_word)
                                print(f"Word Appended: {predicted_word} | Current Sentence: {' '.join(self.sentence_words)}")
                                
                                self.predictor.reset_word_buffer()

                
                # <--- 변경된 부분: 매 프레임마다 랜드마크와 텍스트를 다시 그림 ---
                # 로직이 처리되지 않는 프레임에서는 이전 랜드마크 결과를 그대로 사용해 부드러운 화면을 보여줌
                #buffer_status = f"Word Buf: {len(self.predictor.word_buffer)}/{self.config['SEQ_LEN_WORD']} | Alpha Buf: {len(self.predictor.alphabet_buffer)}/{self.config['SEQ_LEN_ALPHABET']}"
                #display_frame = self.visualizer.draw(frame, self.current_prediction, self.current_confidence, self.sentence_words, buffer_status, results_pose, results_hands)
                
                
                #cv2.imshow('Sign Language Recognition', display_frame)
                #if cv2.waitKey(1) & 0xFF == ord('q'):
                #    break
            
            final_result = " ".join(self.sentence_words)
            self.cleanup()
            return final_result
            
    def cleanup(self):
        """Cleans up resources and prints the final result."""
        if self.sentence_words:
            print(f"\n--- Video Finished ---")
            print(f"Final Sentence: {' '.join(self.sentence_words)}")
        
        self.feature_extractor.close()
        self.video_cap.release()
        #cv2.destroyAllWindows()
        print("--- Resources Released ---")


if __name__ == '__main__':
    # --- User Configuration ---
    

    try:
        recognizer = SignLanguageRecognizer(CONFIG)
        recognizer.run()
    except (IOError, FileNotFoundError) as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")