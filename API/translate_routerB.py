import os
from dotenv import load_dotenv
import deepl
from fastapi import APIRouter, Depends, Body, HTTPException, Query, Request, Response
from sqlalchemy.orm import Session
from core_method import get_db, verify_or_refresh_token
from typing import List
import numpy as np
import cv2
import base64

from cachetools import TTLCache


from model.LSTM.LSTM_video_OOP2B import SignLanguageRecognizer # 파일 이름과 경로 확인!
from model.LSTM.LSTM_video_OOP2A import CONFIG # 파일 이름과 경로 확인!

router = APIRouter()

load_dotenv(dotenv_path="deepl_api_key.env")
AUTH_KEY = os.getenv("DEEPL_API_KEY")

# --- 💡 2. 사용자별 Recognizer 객체를 저장할 딕셔너리 ---
# { "user_id": SignLanguageRecognizer_instance } 형태로 저장됩니다.
#user_recognizers = {}
user_recognizers = TTLCache(maxsize=100, ttl=300) 


def decode_base64_to_numpy(base64_string: str) -> np.ndarray:
    """Base64 문자열을 OpenCV 이미지(Numpy 배열)로 디코딩합니다."""
    try:
        img_bytes = base64.b64decode(base64_string)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception:
        return None

# --- 💡 3. `/analyze_frames` 엔드포인트 재구성 ---
@router.post("/translate/analyze_frames")
async def analyze_frames(request: Request, response: Response, frames: List[str] = Body(..., embed=True), db: Session = Depends(get_db)):
    user_id = verify_or_refresh_token(request, response)

    # 해당 유저의 Recognizer 객체가 없으면 새로 생성
    if user_id not in user_recognizers:
        print(f"--- New recognizer created for user: {user_id} ---")
        user_recognizers[user_id] = SignLanguageRecognizer(CONFIG)
    
    recognizer = user_recognizers[user_id]
    
    newly_recognized_words = []
    for base64_frame in frames:
        frame_np = decode_base64_to_numpy(base64_frame)
        if frame_np is None:
            continue
        
        # 프레임 하나를 처리하고, 새로 인식된 단어가 있으면 리스트에 추가
        result = recognizer.process_frame(frame_np)
        if result:
            newly_recognized_words.append(result)
            print(f"User {user_id} recognized new word: {result}")

    # 새로 인식된 단어들을 클라이언트에 즉시 반환 (선택사항)
    return {"status": "processing"}

# --- 💡 4. `/translate/translate_latest` 엔드포인트 재구성 ---
@router.get("/translate/translate_latest")
def translate_latest(request: Request, response: Response, db: Session = Depends(get_db)):
    user_id = verify_or_refresh_token(request, response)

    if user_id not in user_recognizers:
        return {"korean": "분석된 내용이 없습니다.", "english": "", "japanese": "", "chinese": ""}

    recognizer = user_recognizers[user_id]
    
    # Recognizer 객체에서 최종 문장 가져오기
    final_sentence = recognizer.get_full_sentence()
    
    
    if not final_sentence:
        return {"korean": "인식된 단어가 없습니다.", "english": "", "japanese": "", "chinese": ""}

    # DeepL 번역
    translator = deepl.Translator(AUTH_KEY)
    result = {
        "korean": final_sentence,
        "english": translator.translate_text(final_sentence, target_lang="EN-US").text,
        "japanese": translator.translate_text(final_sentence, target_lang="JA").text,
        "chinese": translator.translate_text(final_sentence, target_lang="ZH").text,
    }
    
    # 다음 문장 인식을 위해 해당 유저의 Recognizer 상태 초기화
    recognizer.reset()
    print(f"--- Recognizer for user {user_id} has been reset. ---")

    return result