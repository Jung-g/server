# local_test.py

import os
# 💡 1. 수정된 핵심 로직과 설정을 가져옵니다.
#    파일 이름이 다르다면 이 부분을 수정하세요.
from model.LSTM.LSTM_video_OOP2A import SignLanguageRecognizer, CONFIG

# 💡 2. 테스트에 사용할 비디오 파일 경로를 설정합니다.
#    이 경로를 본인의 테스트 영상 파일 경로로 바꿔주세요.
#    윈도우 경로는 'C:/Users/...' 처럼 슬래시(/)를 사용하거나 'C:\\Users\\...' 처럼 이중 역슬래시를 사용해야 합니다.
TEST_VIDEO_PATH = "D:\\Bit\\server\\debug_videos\\aaa_20250811_142541.mp4" # ◀◀◀ 본인의 테스트 영상 파일 경로를 여기에 입력하세요!


def run_local_test():
    """
    서버 없이 SignLanguageRecognizer의 성능을 직접 테스트하는 함수.
    """
    print("--- 로컬 환경에서 수어 인식 모델 성능 테스트를 시작합니다. ---")

    # 3. CONFIG에 테스트 비디오 경로를 설정합니다.
    current_config = CONFIG.copy()
    current_config["VIDEO_FILE_PATH"] = TEST_VIDEO_PATH
    
    # 모델 디렉토리 경로도 서버 환경이 아닌 로컬에 맞게 확인합니다.
    # 예: "model" 또는 "C:/path/to/your/model_files"
    current_config["MODEL_DIR"] = "./model" # ◀◀◀ 모델 파일(pt, npy, json 등)이 있는 폴더 경로

    if not os.path.exists(TEST_VIDEO_PATH):
        print(f"[오류] 테스트 비디오 파일을 찾을 수 없습니다: {TEST_VIDEO_PATH}")
        return
    if not os.path.exists(current_config["MODEL_DIR"]):
        print(f"[오류] 모델 파일 폴더를 찾을 수 없습니다: {current_config['MODEL_DIR']}")
        return

    try:
        # 4. 수정된 Recognizer 클래스의 인스턴스를 생성합니다.
        recognizer = SignLanguageRecognizer(current_config)

        # 5. run() 메소드를 실행하여 예측을 수행합니다.
        #    이 과정에서 터미널에 "Word Appended: ..." 같은 로그가 출력될 것입니다.
        predicted_sentence = recognizer.run()

        print("\n" + "="*50)
        print("🎉 테스트 완료! 🎉")
        print(f"최종 인식된 문장: {predicted_sentence}")
        print("="*50)

    except Exception as e:
        print(f"\n[치명적 오류] 테스트 실행 중 예외가 발생했습니다: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    run_local_test()