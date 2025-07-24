# local_test_B.py

import cv2
import os

# 💡 1. 수정된 핵심 로직과 설정을 가져옵니다.
#    방식 A의 CONFIG를 그대로 사용합니다.
from model.LSTM.LSTM_video_OOP2B import SignLanguageRecognizer
from model.LSTM.LSTM_video_OOP2A import CONFIG 

# 💡 2. 테스트에 사용할 비디오 파일 경로를 설정합니다.
TEST_VIDEO_PATH = "c:\\Users\\bit\\Desktop\\KakaoTalk_20250716_212034817.mp4" # ◀◀◀ 본인의 테스트 영상 파일 경로

def run_streaming_simulation():
    """
    서버 없이 방식 B(프레임 스트리밍)의 로직을 시뮬레이션합니다.
    """
    print("--- 로컬 환경에서 방식 B(스트리밍) 모델 성능 테스트를 시작합니다. ---")
    
    # 3. 서버 환경과 동일하게 CONFIG를 설정합니다.
    #    모델 파일 경로를 로컬 환경에 맞게 확인해주세요.
    current_config = CONFIG.copy()
    current_config["MODEL_DIR"] = "C:/Users/bit/Desktop" # ◀◀◀ 모델 파일(pt, npy, json 등)이 있는 폴더 경로

    if not os.path.exists(TEST_VIDEO_PATH):
        print(f"[오류] 테스트 비디오 파일을 찾을 수 없습니다: {TEST_VIDEO_PATH}")
        return
    if not os.path.exists(current_config["MODEL_DIR"]):
        print(f"[오류] 모델 파일 폴더를 찾을 수 없습니다: {current_config['MODEL_DIR']}")
        return

    # 4. (시뮬레이션) 클라이언트가 접속하여 Recognizer 인스턴스가 생성됩니다.
    print("\n[시뮬레이션] 새로운 사용자가 접속하여 Recognizer를 생성합니다.")
    recognizer = SignLanguageRecognizer(current_config)

    # 5. (시뮬레이션) 클라이언트가 비디오 프레임을 계속해서 보냅니다.
    #    이 과정은 `/translate/analyze_frames` API를 계속 호출하는 것과 같습니다.
    print("[시뮬레이션] 비디오 프레임 처리를 시작합니다... (/analyze_frames 호출)")
    
    cap = cv2.VideoCapture(TEST_VIDEO_PATH)
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        # 서버와 동일하게 프레임을 뒤집어줍니다.
        # routerB에서 flip을 안했다면 이 라인을 주석처리 하세요.
        frame = cv2.flip(frame, 1) # routerB.py에서 flip을 하지 않으므로 주석 처리
        
        # 핵심: process_frame()을 호출하여 프레임을 하나씩 처리합니다.
        newly_recognized_word = recognizer.process_frame(frame)
        
        # 새로운 단어가 인식될 때마다 로그를 출력합니다.
        if newly_recognized_word:
            print(f"  > Frame {frame_count}: 새로운 단어/지문자 인식! -> '{newly_recognized_word}'")

    cap.release()
    print("[시뮬레이션] 비디오 프레임 처리가 완료되었습니다.")

    # 6. (시뮬레이션) 클라이언트가 최종 번역 결과를 요청합니다.
    #    이 과정은 `/translate/translate_latest` API를 호출하는 것과 같습니다.
    print("\n[시뮬레이션] 최종 번역 결과를 요청합니다... (/translate_latest 호출)")
    final_sentence = recognizer.get_full_sentence()
    
    print("\n" + "="*50)
    print("🎉 테스트 완료! 🎉")
    print(f"최종 인식된 문장: {final_sentence}")
    print("="*50)
    
    # 7. (시뮬레이션) 결과를 받은 후 Recognizer 상태가 초기화됩니다.
    recognizer.reset()
    print(f"\n[시뮬레이션] Recognizer 상태가 초기화되었습니다. 현재 문장: '{recognizer.get_full_sentence()}'")


if __name__ == '__main__':
    run_streaming_simulation()