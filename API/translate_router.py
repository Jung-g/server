import datetime
import os
import tempfile
from cachetools import TTLCache
import cv2
from dotenv import load_dotenv
import mediapipe as mp
import deepl
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, Response, UploadFile, Body
from sqlalchemy.orm import Session
# from model.LSTM.LSTM import build_sequence_from_frames, predict_sign_language, result
# from model.LSTM.LSTM_sentence import run_sentence_model
# from model.LSTM.LSTM_video import run_model
from API.translate_router_F import serialize_result
from DB_Table import Word
from core_method import get_db, verify_or_refresh_token
from js_gloss_2_korean.hong_translate_main import translate_pipeline

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

router = APIRouter()

load_dotenv(dotenv_path="deepl_api_key.env")
AUTH_KEY = os.getenv("DEEPL_API_KEY")
VIDEO_DIR = "video"

# # 수어 → 텍스트 → 번역
# # @router.post("/translate/sign_to_text")
# # async def translate_sign_to_text(request: Request, response: Response,file: UploadFile = File(...), db: Session = Depends(get_db)):
# #     user_id = verify_or_refresh_token(request, response)
    
# #     with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
# #         tmp.write(await file.read())
# #         tmp_path = tmp.name
# #     print(tmp_path)

# #     recognized_korean = run_model(tmp_path)

# #     print("최종 예측 결과:", recognized_korean)

# #     word_text = recognized_korean
# #     if (word_text == '학습되지 않은 동작입니다' or word_text == "인식실패 다시 시도해주세요"):
# #         return {"korean": word_text}
    
# #     translator = deepl.Translator(AUTH_KEY)
# #     english = translator.translate_text(word_text, target_lang="EN-US").text
# #     chinese = translator.translate_text(word_text, target_lang="ZH-HANS").text
# #     japanese = translator.translate_text(word_text, target_lang="JA").text

# #     return {
# #         "korean": word_text,
# #         "english": english,
# #         "chinese": chinese,
# #         "japanese": japanese
# #     }
# @router.post("/translate/sign_to_text")
# async def translate_sign_to_text(request: Request, response: Response, expected_word: str = Form(...), file: UploadFile = File(...), db: Session = Depends(get_db)):
#     user_id = verify_or_refresh_token(request, response)

#     with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
#         tmp.write(await file.read())
#         tmp_path = tmp.name

#     predicted_word = run_model(tmp_path)
#     print("예측 결과:", predicted_word)
#     print("사용자 정답:", expected_word)

#     is_match = (predicted_word == expected_word)

#     if predicted_word in ['학습되지 않은 동작입니다', '인식실패 다시 시도해주세요']:
#         return {
#             "korean": predicted_word,
#             "match": False,
#         }

#     return {
#         "korean": predicted_word,
#         "match": is_match,
#     }


# # 텍스트 → 수어 애니메이션
# @router.get("/translate/text_to_sign")
# async def get_sign_animation(request: Request, response: Response,word_text: str = Query(..., description="입력된 한국어 단어"), db: Session = Depends(get_db)):
#     user_id = verify_or_refresh_token(request, response)
    
#     word = db.query(Word).filter(Word.Word == word_text).first()
#     # if not word or not word.animations:
#     #     raise HTTPException(status_code=404, detail="애니메이션이 존재하지 않습니다.")

#     clean_word = word.Word.strip().replace("'", "").replace('"', "")
#     file_name = f"{clean_word}.mp4"
#     file_path = os.path.join(VIDEO_DIR, file_name)

#     if os.path.isfile(file_path):
#         video_url = f"http://10.101.92.18/video/{file_name}"
#     else:
#         video_url = "" 

#     return {
#         "URL": video_url
#     }


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
user_full_sentence = defaultdict(list)  # user_id → List[str]

# --- 비디오 저장 로직 추가 ---
# 각 사용자별 비디오 파일을 관리하기 위한 캐시
user_video_writers = TTLCache(maxsize=100, ttl=300)
DEBUG_VIDEO_DIR = "debug_videos"
os.makedirs(DEBUG_VIDEO_DIR, exist_ok=True) # 저장할 폴더 생성

def get_full_sentence(user_id: str) -> str:
    return " ".join(user_full_sentence.get(user_id, []))

# 프레임 전송받아서 실시간으로 수어 -> 한글 단어 번역
@router.post("/translate/analyze_frames")
async def analyze_frames(request: Request, response: Response, frames: List[str] = Body(..., embed=True), db: Session = Depends(get_db)):
    user_id = verify_or_refresh_token(request, response)
    print(f"\n[ROUTER|/analyze_frames] User '{user_id}' sent a batch of {len(frames)} frames.")
    for base64_frame in frames:
        frame_np = decode_base64_to_numpy(base64_frame)
        feature = extract_features_from_frame(frame_np)
        if feature is None:
            continue

        # --- 비디오 저장 로직 추가 ---
        if user_id not in user_video_writers:
            # 해당 유저의 첫 프레임이면 비디오 파일 생성
            now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(DEBUG_VIDEO_DIR, f"{user_id}_{now}.mp4")
            height, width, _ = frame_np.shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # 클라이언트가 약 15fps로 보내므로, 저장할 비디오의 fps를 15로 설정
            writer = cv2.VideoWriter(filename, fourcc, 5, (width, height))
            user_video_writers[user_id] = writer
            print(f"--- Start recording debug video for user '{user_id}' to '{filename}' ---")
        
        # 프레임을 비디오 파일에 쓰기
        user_video_writers[user_id].write(frame_np)
        # -----------------------------

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

            if confidence >= 70.0:
                sentence = user_full_sentence[user_id]
                if not sentence or sentence[-1] != prediction:
                    sentence.append(prediction)

            print(f"[{user_id}] 예측 결과: {prediction}, 확신도: {confidence:.2f}%")
            print(f"[{user_id}] 저장됨")

    return {"status": "processing"}

# 수어 -> 한글 단어 번역 결과를 다국어로 번역
@router.get("/translate/translate_latest")
def translate_latest(request: Request, response: Response, db: Session = Depends(get_db)):
    user_id = verify_or_refresh_token(request, response)

    # --- 비디오 저장 종료 ---
    if user_id in user_video_writers:
        writer = user_video_writers[user_id]
        writer.release()
        print(f"--- Finished recording debug video for user '{user_id}' ---")
        del user_video_writers[user_id]

    # --- 박준수 수정: 문장 생성 + 번역 ---
    semi_sentence = get_full_sentence(user_id)
    print(f"디버깅용 semi_sentence: {semi_sentence}")

    final_sentence = translate_pipeline(semi_sentence) if semi_sentence else None
    print(f"디버깅용 final_sentence: {final_sentence}")
    # -----------------------------------

    # 캐시 초기화
    user_prediction_history[user_id].clear()
    user_full_sentence[user_id].clear()
    user_latest_prediction.pop(user_id, None)

    if not final_sentence:
        return {
            "korean": "인식된 단어가 없습니다.",
            "english": {"text": "", "원본언어": "KO"},
            "japanese": {"text": "", "원본언어": "KO"},
            "chinese": {"text": "", "원본언어": "KO"},
        }

    try:
        translator = deepl.Translator(AUTH_KEY)
        return {
            "korean": final_sentence,
            "english": serialize_result(translator.translate_text(final_sentence, target_lang="EN-US")),
            "japanese": serialize_result(translator.translate_text(final_sentence, target_lang="JA")),
            "chinese": serialize_result(translator.translate_text(final_sentence, target_lang="ZH")),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"번역 API 호출 중 오류 발생: {str(e)}")
