import os
import tempfile
import traceback
from typing import List
import datetime
import cv2
import deepl
import numpy as np
from cachetools import TTLCache
from dotenv import load_dotenv
from fastapi import (APIRouter, Body, Depends, File, Form, HTTPException, Query,
                     Request, Response, UploadFile)
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
import base64

from anime.motion_merge import api_motion_merge, check_merge
from core_method import get_db, verify_or_refresh_token

from js_gloss_2_korean.hong_translate_main import translate_pipeline
from js_korean_2_gloss.main_translate import main_translate
from model.LSTM.LSTM_sign import CONFIG, SignLanguageRecognizer

# --- 기본 설정 ---
router = APIRouter()
load_dotenv(dotenv_path="deepl_api_key.env")
AUTH_KEY = os.getenv("DEEPL_API_KEY")
user_recognizers = TTLCache(maxsize=100, ttl=300)


# --- 비디오 저장 로직 추가 ---
# 각 사용자별 비디오 파일을 관리하기 위한 캐시
user_video_writers = TTLCache(maxsize=100, ttl=300)
DEBUG_VIDEO_DIR = "debug_videos"
os.makedirs(DEBUG_VIDEO_DIR, exist_ok=True) # 저장할 폴더 생성

# 번역 결과 직렬화
def serialize_result(r):
    if isinstance(r, list):
        combined_text = " ".join([item.text for item in r])
        source_lang = r[0].detected_source_lang if r else None
        return {
            "text": combined_text,
            "원본언어": source_lang
        }
    else:
        return {
            "text": r.text,
            "원본언어": r.detected_source_lang
        }


@router.post("/translate/sign_to_text")
async def sign_to_text_handler(
    request: Request,
    response: Response,
    file: UploadFile = File(...),
    # expected_word를 선택적 파라미터로 설정하여 A타입, B타입 요청을 구분합니다.
    expected_word: str = Form(None),
    db: Session = Depends(get_db)
):
    """
    클라이언트로부터 수어 영상(.mp4)을 받아 텍스트로 변환합니다.
    - Form 데이터에 'expected_word'가 포함된 경우 (A타입: 학습):
      단일 단어를 예측하고, 정답과 일치하는지 판별하여 결과를 반환합니다.
    - 'expected_word'가 없는 경우 (B타입: 번역):
      영상 전체를 분석하여 문장으로 만들고, DeepL을 통해 다국어로 번역하여 반환합니다.
    """
    user_id = verify_or_refresh_token(request, response)
    
    # --- 디버깅용 출력 ---
    req_type = "A-Type (Learning)" if expected_word else "B-Type (Translation)"
    print(f"\n[ROUTER|/sign_to_text] Received request. User: '{user_id}', Type: {req_type}")
    if expected_word:
        print(f"    Expected Word: '{expected_word}'")

    # 업로드된 영상 파일을 서버에 임시로 저장합니다.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(await file.read())
        video_path = tmp.name

    # 요청마다 새로운 Recognizer 인스턴스를 생성하여 요청 간 상태가 섞이지 않도록 합니다.
    recognizer = SignLanguageRecognizer(CONFIG)
    
    try:
        # 통합된 모델의 비디오 파일 처리 메소드를 호출합니다.
        predicted_text = recognizer.recognize_from_video_file(video_path)
    except Exception as e:
        print(f"Error during sign recognition: {e}")
        raise HTTPException(status_code=500, detail="수어 인식 중 오류가 발생했습니다.")
    finally:
        if os.path.exists(video_path):
            os.remove(video_path) # 임시 파일 삭제

    # --- 분기 로직: expected_word의 존재 여부로 기능 구분 ---
    if expected_word is not None:
        # A타입 (학습) 요청 처리
        is_match = (predicted_text == expected_word)
        print(f"Learning Mode: Predicted='{predicted_text}', Expected='{expected_word}', Match={is_match}")
        if not predicted_text or predicted_text in ['학습되지 않은 동작입니다', '인식실패 다시 시도해주세요']:
             return JSONResponse(content={"korean": predicted_text, "match": False})
        return JSONResponse(content={"korean": predicted_text, "match": is_match})
    else:
        # B타입 (번역) 요청 처리
        print(f"    [ROUTER|Result] Predicted Sentence: '{predicted_text}'")
        if not predicted_text or predicted_text in ['학습되지 않은 동작입니다', '인식실패 다시 시도해주세요']:
            return JSONResponse(content={
                "korean": predicted_text or "인식된 단어가 없습니다.",
                "english": "", "japanese": "", "chinese": ""
            })

        try:
            translator = deepl.Translator(AUTH_KEY)
            return JSONResponse(content={
                "korean": predicted_text,
                "english": translator.translate_text(predicted_text, target_lang="EN-US").text,
                "japanese": translator.translate_text(predicted_text, target_lang="JA").text,
                "chinese": translator.translate_text(predicted_text, target_lang="ZH").text,
            })
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"번역 API 호출 중 오류 발생: {str(e)}")


@router.get("/translate/text_to_sign")
async def get_sign_animation(request: Request, response: Response, word_text: str = Query(..., description="입력된 한국어 단어"), db: Session = Depends(get_db)):
    user_id = verify_or_refresh_token(request, response)
    
    # mBERT 이용해서 문장 -> list / 박준수 수정
    # ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
    try:
        translator = deepl.Translator(AUTH_KEY)
        result = translator.translate_text(word_text, target_lang="KO")
        korean_words = result.text
        print(f"[DEBUG] Deepl 번역: {word_text} -> {korean_words}")
        words = main_translate(korean_words)
    # ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
    
        motion_data = check_merge(words, send_type='api')
    except ValueError as e:
        print("[ERROR] check_merge 에러 발생:", e)
        traceback.print_exc()
        raise HTTPException(
            status_code=500, 
            detail=f'{e}' # 클라이언트에게 보여줄 메시지
        )
    except Exception as e:
        print("[ERROR] 에상치 못한 에러 발생:", e)
        traceback.print_exc()
        raise HTTPException(
            status_code=500, 
            detail=f'{e}' # 클라이언트에게 보여줄 메시지
        )
        
    frame_generator = api_motion_merge(*motion_data)
    frame_list = list(frame_generator)

    return JSONResponse(content={"frames": frame_list})


# --- 실시간 스트리밍 관련 엔드포인트 ---

def decode_base64_to_numpy(base64_string: str) -> np.ndarray | None:
    """Base64 문자열을 OpenCV 이미지(Numpy 배열)로 디코딩합니다."""
    try:
        img_bytes = base64.b64decode(base64_string)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"[ERROR] Base64 decoding failed: {e}")
        return None

@router.post("/translate/analyze_frames")
async def analyze_frames(
    request: Request,
    response: Response,
    frames: List[str] = Body(..., embed=True),
    db: Session = Depends(get_db)
):
    """클라이언트로부터 받은 Base64 인코딩된 프레임들을 실시간으로 분석합니다."""
    user_id = verify_or_refresh_token(request, response)
    
    print(f"\n[ROUTER|/analyze_frames] User '{user_id}' sent a batch of {len(frames)} frames.")

    if user_id not in user_recognizers:
        print(f"--- New real-time recognizer created for user: {user_id} ---")
        user_recognizers[user_id] = SignLanguageRecognizer(CONFIG)
    
    recognizer = user_recognizers[user_id]
    
    for base64_frame in frames:
        frame_np = decode_base64_to_numpy(base64_frame)
        if frame_np is None:
            continue
        
        
        # --- 비디오 저장 로직 추가 ---
        if user_id not in user_video_writers:
            # 해당 유저의 첫 프레임이면 비디오 파일 생성
            now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(DEBUG_VIDEO_DIR, f"{user_id}_{now}.mp4")
            height, width, _ = frame_np.shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # 클라이언트가 약 15fps로 보내므로, 저장할 비디오의 fps를 15로 설정
            writer = cv2.VideoWriter(filename, fourcc, 15.0, (width, height))
            user_video_writers[user_id] = writer
            print(f"--- Start recording debug video for user '{user_id}' to '{filename}' ---")
        
        # 프레임을 비디오 파일에 쓰기
        user_video_writers[user_id].write(frame_np)
        # -----------------------------
        
        
        frame_np = cv2.flip(frame_np, 1)
        result = recognizer.process_frame(frame_np)
        if result:
            print(f"User '{user_id}' recognized new token: '{result}' -> Current sentence: '{recognizer.get_full_sentence()}'")
            
    return {"status": "processing"}


@router.get("/translate/translate_latest")
async def translate_latest(request: Request, response: Response, db: Session = Depends(get_db)):
    """실시간으로 분석된 최종 문장을 가져오고 번역합니다."""
    user_id = verify_or_refresh_token(request, response)

    if user_id not in user_recognizers:
        return {
            "korean": "인식된 단어가 없습니다.",
            "english": {"text": "", "원본언어": "KO"},
            "japanese": {"text": "", "원본언어": "KO"},
            "chinese": {"text": "", "원본언어": "KO"},
        }
    
    # --- 비디오 저장 로직 추가 ---
    # 번역 요청이 오면 비디오 녹화 종료
    if user_id in user_video_writers:
        writer = user_video_writers[user_id]
        writer.release()
        print(f"--- Finished recording debug video for user '{user_id}' ---")
        del user_video_writers[user_id]
    # -----------------------------

    recognizer = user_recognizers[user_id]
    # 박준수 수정 - 번역
    semi_sentence = recognizer.get_full_sentence()
    print(type(semi_sentence))
    print(f"디버깅용 semi_sentence: {semi_sentence}")
    final_sentence = translate_pipeline(semi_sentence) if semi_sentence else None
    print(f"디버깅용 semi_sentence: {final_sentence}")
    # ---
    
    print(f"    [ROUTER|Result] Retrieved sentence: '{final_sentence}'. Resetting state for user.")

    
    # 중요한: 결과를 가져온 후에는 해당 유저의 Recognizer 상태를 초기화하여 다음 문장을 받을 준비를 합니다.
    recognizer.reset()
    del user_recognizers[user_id] # 캐시에서 삭제하여 메모리 관리
    print(f"--- Recognizer for user {user_id} has been reset and removed from cache. ---")

    if not final_sentence:
        return {
            "korean": "인식된 단어가 없습니다.",
            "english": {"text": "", "원본언어": "KO"},
            "japanese": {"text": "", "원본언어": "KO"},
            "chinese": {"text": "", "원본언어": "KO"},
        }

    try:
        translator = deepl.Translator(AUTH_KEY)
        result = {
            "korean": final_sentence,
            "english": serialize_result(translator.translate_text(final_sentence, target_lang="EN-US")),
            "japanese": serialize_result(translator.translate_text(final_sentence, target_lang="JA")),
            "chinese": serialize_result(translator.translate_text(final_sentence, target_lang="ZH")),
        }
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"번역 API 호출 중 오류 발생: {str(e)}")