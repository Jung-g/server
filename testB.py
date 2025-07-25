# local_test_B_with_base64.py

import cv2
import os
import base64 # 
import numpy as np # 

# 수정된 핵심 로직과 설정을 가져옵니다.
from model.LSTM.LSTM_video_OOP2B import SignLanguageRecognizer
from model.LSTM.LSTM_video_OOP2A import CONFIG 

# 테스트에 사용할 비디오 파일 경로를 설정합니다.
TEST_VIDEO_PATH = "c:\\Users\\bit\\Desktop\\KakaoTalk_20250724_164924678.mp4" # ◀◀◀ 본인의 테스트 영상 파일 경로
MODEL_DIR = "C:/Users/bit/Desktop" # ◀◀◀ 모델 파일(pt, npy, json 등)이 있는 폴더 경로

# 💡 2. translate_routerB.py 에 있던 디코딩 함수 추가
def decode_base64_to_numpy(base64_string: str) -> np.ndarray:
    """Base64 문자열을 OpenCV 이미지(Numpy 배열)로 디코딩합니다."""
    try:
        img_bytes = base64.b64decode(base64_string)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception:
        return None

def run_streaming_simulation_with_base64():
    """
    서버 없이 방식 B(프레임 스트리밍)의 로직을 시뮬레이션합니다.
    (Base64 인코딩/디코딩 과정 포함)
    """
    print("--- 로컬 환경에서 Base64 변환을 포함한 스트리밍 테스트를 시작합니다. ---")
    
    current_config = CONFIG.copy()
    current_config["MODEL_DIR"] = MODEL_DIR

    if not os.path.exists(TEST_VIDEO_PATH):
        print(f"[오류] 테스트 비디오 파일을 찾을 수 없습니다: {TEST_VIDEO_PATH}")
        return
    if not os.path.exists(current_config["MODEL_DIR"]):
        print(f"[오류] 모델 파일 폴더를 찾을 수 없습니다: {current_config['MODEL_DIR']}")
        return

    recognizer = SignLanguageRecognizer(current_config)
    cap = cv2.VideoCapture(TEST_VIDEO_PATH)
    frame_count = 0
    cv2.namedWindow("Sign Language Recognition Test (Base64 Sim)", cv2.WINDOW_NORMAL)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # 💡 3. 실제 전송처럼 Base64 인코딩/디코딩 시뮬레이션
        # --------------------------------------------------------------------
        # (1) 프레임(numpy)을 jpg 포맷의 byte로 변환 후 base64로 '인코딩'
        _, buffer = cv2.imencode('.jpg', frame)
        base64_frame_string = base64.b64encode(buffer).decode('utf-8')
        
        # (2) Base64 문자열을 다시 프레임(numpy)으로 '디코딩'
        decoded_frame = decode_base64_to_numpy(base64_frame_string)
        
        if decoded_frame is None:
            print(f"Frame {frame_count}: Base64 디코딩 실패")
            continue
        # --------------------------------------------------------------------

        # 핵심: '디코딩된 프레임'을 recognizer로 처리합니다.
        newly_recognized_word = recognizer.process_frame(decoded_frame)
        
        if newly_recognized_word:
            print(f"  > Frame {frame_count}: 새로운 단어/지문자 인식! -> '{newly_recognized_word}'")

        # GUI 화면 업데이트 (디코딩된 프레임 기준)
        current_sentence = recognizer.get_full_sentence()
        cv2.putText(
            decoded_frame, f"Sentence: {current_sentence}", (10, 50), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA
        )
        cv2.imshow("Sign Language Recognition Test (Base64 Sim)", decoded_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    final_sentence = recognizer.get_full_sentence()
    print("\n" + "="*50)
    print("🎉 테스트 완료! 🎉")
    print(f"최종 인식된 문장: {final_sentence}")
    print("="*50)
    recognizer.reset()

if __name__ == '__main__':
    run_streaming_simulation_with_base64()