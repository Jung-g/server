import os
import tempfile
import cv2
from dotenv import load_dotenv
import mediapipe as mp
import deepl
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, Response, UploadFile, Body
from sqlalchemy.orm import Session
from model.LSTM.LSTM import build_sequence_from_frames, predict_sign_language, result
from model.LSTM.LSTM_sentence import run_sentence_model
from model.LSTM.LSTM_video import run_model
from models import Word
from core_method import get_db, verify_or_refresh_token

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

router = APIRouter()

load_dotenv(dotenv_path="deepl_api_key.env")
AUTH_KEY = os.getenv("DEEPL_API_KEY")
VIDEO_DIR = "video"

# 수어 → 텍스트 → 번역
# @router.post("/translate/sign_to_text")
# async def translate_sign_to_text(request: Request, response: Response,file: UploadFile = File(...), db: Session = Depends(get_db)):
#     user_id = verify_or_refresh_token(request, response)
    
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
#         tmp.write(await file.read())
#         tmp_path = tmp.name
#     print(tmp_path)

#     recognized_korean = run_model(tmp_path)

#     print("최종 예측 결과:", recognized_korean)

#     word_text = recognized_korean
#     if (word_text == '학습되지 않은 동작입니다' or word_text == "인식실패 다시 시도해주세요"):
#         return {"korean": word_text}
    
#     translator = deepl.Translator(AUTH_KEY)
#     english = translator.translate_text(word_text, target_lang="EN-US").text
#     chinese = translator.translate_text(word_text, target_lang="ZH-HANS").text
#     japanese = translator.translate_text(word_text, target_lang="JA").text

#     return {
#         "korean": word_text,
#         "english": english,
#         "chinese": chinese,
#         "japanese": japanese
#     }
@router.post("/translate/sign_to_text")
async def translate_sign_to_text(request: Request, response: Response, expected_word: str = Form(...), file: UploadFile = File(...), db: Session = Depends(get_db)):
    user_id = verify_or_refresh_token(request, response)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    predicted_word = run_model(tmp_path)
    print("예측 결과:", predicted_word)
    print("사용자 정답:", expected_word)

    is_match = (predicted_word == expected_word)

    if predicted_word in ['학습되지 않은 동작입니다', '인식실패 다시 시도해주세요']:
        return {
            "korean": predicted_word,
            "match": False,
        }

    return {
        "korean": predicted_word,
        "match": is_match,
    }


# 텍스트 → 수어 애니메이션
@router.get("/translate/text_to_sign")
async def get_sign_animation(request: Request, response: Response,word_text: str = Query(..., description="입력된 한국어 단어"), db: Session = Depends(get_db)):
    user_id = verify_or_refresh_token(request, response)
    
    word = db.query(Word).filter(Word.Word == word_text).first()
    # if not word or not word.animations:
    #     raise HTTPException(status_code=404, detail="애니메이션이 존재하지 않습니다.")

    clean_word = word.Word.strip().replace("'", "").replace('"', "")
    file_name = f"{clean_word}.mp4"
    file_path = os.path.join(VIDEO_DIR, file_name)

    if os.path.isfile(file_path):
        video_url = f"http://10.101.92.18/video/{file_name}"
    else:
        video_url = "" 

    return {
        "URL": video_url
    }


# 속도 개선 테스트
import numpy as np
from typing import List
from collections import Counter
from collections import defaultdict, deque
from model.LSTM.LSTM_frame import decode_base64_to_numpy, extract_features_from_frame, predict_with_model, SEQ_LEN

# 예측 상태 저장 구조
user_keypoints = defaultdict(deque)  # deque of (feature, velocity, acceleration)
user_prediction_history = defaultdict(list)
user_latest_prediction = defaultdict(tuple)  # (word, confidence)

# 프레임 전송받아서 실시간으로 수어 -> 한글 단어 번역
@router.post("/translate/analyze_frames")
async def analyze_frames(request: Request, response: Response, frames: List[str] = Body(..., embed=True), db: Session = Depends(get_db)):
    user_id = verify_or_refresh_token(request, response)

    for base64_frame in frames:
        frame_np = decode_base64_to_numpy(base64_frame)
        feature = extract_features_from_frame(frame_np)
        if feature is None:
            continue

        buffer = user_keypoints[user_id]
        if buffer:
            velocity = feature - buffer[-1][0]
            acceleration = velocity - buffer[-1][1]
        else:
            velocity = np.zeros_like(feature)
            acceleration = np.zeros_like(feature)

        buffer.append((feature, velocity, acceleration))
        if len(buffer) > SEQ_LEN:
            buffer.popleft()

        if len(buffer) == SEQ_LEN:
            sequence = np.array([
                np.concatenate([f, v, a]) for f, v, a in buffer
            ])
            prediction, confidence = predict_with_model(sequence)

            history = user_prediction_history[user_id]
            history.append((prediction, confidence))
            user_latest_prediction[user_id] = (prediction, confidence)

            print(f"[{user_id}] 예측 결과: {prediction}, 확신도: {confidence:.2f}%")
            print(f"[{user_id}] 저장됨")

    return {"status": "processing"}

# 수어 -> 한글 단어 번역 결과를 다국어로 번역
@router.get("/translate/translate_latest")
def translate_latest(request: Request, response: Response, db: Session = Depends(get_db)):
    user_id = verify_or_refresh_token(request, response)

    history = user_prediction_history.get(user_id, [])
    high_confidence = [pred for pred, conf in history if conf >= 70.0]

    if not high_confidence:
        return {
            "korean": "신뢰도 70% 이상 결과 없음",
            "english": "",
            "japanese": "",
            "chinese": ""
        }

    top_word, _ = Counter(high_confidence).most_common(1)[0]

    translator = deepl.Translator(AUTH_KEY)
    result = {
        "korean": top_word,
        "english": translator.translate_text(top_word, target_lang="EN-US").text,
        "japanese": translator.translate_text(top_word, target_lang="JA").text,
        "chinese": translator.translate_text(top_word, target_lang="ZH").text,
    }

    user_prediction_history[user_id].clear()

    return result
