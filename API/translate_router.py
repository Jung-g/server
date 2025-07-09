import os
import tempfile
import cv2
from dotenv import load_dotenv
import mediapipe as mp
import deepl
from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, Response, UploadFile
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
@router.post("/translate/sign_to_text")
async def translate_sign_to_text(request: Request, response: Response,file: UploadFile = File(...), db: Session = Depends(get_db)):
    user_id = verify_or_refresh_token(request, response)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    print(tmp_path)

    recognized_korean = run_model(tmp_path)
    # recognized_korean = run_sentence_model(tmp_path)

    print("최종 예측 결과:", recognized_korean)

    # word = db.query(Word).filter(Word.Word == recognized_korean).first()
    # # word = '나무'
    # if not word:
    #     raise HTTPException(status_code=404, detail="해당 단어를 찾을 수 없습니다.")
    word_text = recognized_korean
    if (word_text == '학습되지 않은 동작입니다' or word_text == "인식실패 다시 시도해주세요"):
        return {"korean": word_text}
    
    # word_text = word.Word
    translator = deepl.Translator(AUTH_KEY)
    english = translator.translate_text(word_text, target_lang="EN-US").text
    chinese = translator.translate_text(word_text, target_lang="ZH-HANS").text
    japanese = translator.translate_text(word_text, target_lang="JA").text

    return {
        "korean": word_text,
        "english": english,
        "chinese": chinese,
        "japanese": japanese
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
