import os
import tempfile
from dotenv import load_dotenv
import deepl
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, Response, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy.orm import Session
from DB_Table import Word
from core_method import get_db, verify_or_refresh_token

# --- 💡 1. 불필요한 import 정리 및 새로운 클래스 추가 ---
# 기존의 run_model, LSTM_frame 등을 모두 지우고 OOP2의 클래스를 가져옵니다.
# 파일 위치가 model/LSTM/LSTM_video_OOP2.py 라면 아래 경로가 맞습니다.
from model.LSTM.LSTM_video_OOP2A import SignLanguageRecognizer, CONFIG

router = APIRouter()

load_dotenv(dotenv_path="deepl_api_key.env")
AUTH_KEY = os.getenv("DEEPL_API_KEY")
VIDEO_DIR = "video"

# --- 수어 → 텍스트 (A 방식: 통 동영상 처리) ---
@router.post("/translate/sign_to_text")
async def translate_sign_to_text(request: Request, response: Response, expected_word: str = Form(...), file: UploadFile = File(...), db: Session = Depends(get_db)):
    user_id = verify_or_refresh_token(request, response)

    # 업로드된 영상 파일을 서버에 임시로 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(await file.read())
        video_path = tmp.name

    predicted_word = "인식 실패" # 기본값 설정
    try:

        current_config = CONFIG.copy()
        current_config["VIDEO_FILE_PATH"] = video_path

        # Recognizer 인스턴스 생성
        recognizer = SignLanguageRecognizer(current_config)

        # Recognizer 실행 및 결과 반환
        predicted_word = recognizer.run()
        
        print(f"### 디버깅: recognizer.run()의 실제 반환값: '{predicted_word}' (타입: {type(predicted_word)}) ###")

        print("예측 결과:", predicted_word)
        print("사용자 정답:", expected_word)
        

    except Exception as e:
        print(f"An error occurred during recognition: {e}")
        raise HTTPException(status_code=500, detail="수어 인식 중 오류가 발생했습니다.")
    finally:
        # 임시 파일 삭제
        if os.path.exists(video_path):
            os.remove(video_path)

    is_match = (predicted_word == expected_word)

    if not predicted_word or predicted_word in ['학습되지 않은 동작입니다', '인식실패 다시 시도해주세요']:
        return {
            "korean": predicted_word or "인식된 단어가 없습니다.",
            "match": False,
        }

    return {
        "korean": predicted_word,
        "match": is_match,
    }


# --- 텍스트 → 수어 (기존 코드 유지) ---
@router.get("/translate/text_to_sign")
async def get_sign_animation(request: Request, response: Response, word_text: str = Query(..., description="입력된 한국어 단어"), db: Session = Depends(get_db)):
    user_id = verify_or_refresh_token(request, response)
    

    # mBERT 이용해서 문장 -> list
    # ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
    words = []
    # ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
    
    from anime.motion_merge import check_merge, api_motion_merge
    
    try:
        motion_data = check_merge(words, send_type='api')
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f'{e}' # 클라이언트에게 보여줄 메시지
        )
    
    frame_generator = api_motion_merge(*motion_data)
    frame_list = list(frame_generator)

    return JSONResponse(content={"frames": frame_list})
# --- 💡 3. B 방식(프레임 스트림 처리) 관련 엔드포인트는 모두 삭제 ---
# "/translate/analyze_frames" 와 "/translate/translate_latest" 는 A 방식만 사용하므로 삭제합니다.