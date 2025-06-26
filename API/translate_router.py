import cv2
import mediapipe as mp
import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from sqlalchemy.orm import Session
from models import Word
from main import get_db
# 외부 번역 라이브러리나 API 서비스 추가하기

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

translate_router = APIRouter()

# 수어 → 텍스트 → 번역
@translate_router.post("/translate/sign-to-text")
async def translate_sign_to_text(file: UploadFile = File(...), db: Session = Depends(get_db)):
    # 수어 동작 분석 모델 연결하기

    # 이미지 (프레임) 로드
    image_bytes = await file.read()
    np_array = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    # 수어 인식 모델 실행 (1장짜리 프레임 기반 추론)
    recognized_korean = run_sign_model(frame, db)

    word = db.query(Word).filter(Word.Word == recognized_korean).first()
    if not word:
        raise HTTPException(status_code=404, detail="해당 단어를 찾을 수 없습니다.")

    # 외부 번역 API 호출 (번역 함수는 실제 구현 필요)
    # english = translate_text(recognized_korean, target_lang="en")
    # chinese = translate_text(recognized_korean, target_lang="zh")
    # japanese = translate_text(recognized_korean, target_lang="ja")

    return {
        "korean": recognized_korean,
        # "english": english,
        # "chinese": chinese,
        # "japanese": japanese
    }

# 텍스트 → 수어 애니메이션
@translate_router.get("/translate/text-to-sign")
def get_sign_animation(word_text: str = Query(..., description="입력된 한국어 단어"), db: Session = Depends(get_db)):
    word = db.query(Word).filter(Word.Word == word_text).first()
    if not word or not word.animations:
        raise HTTPException(status_code=404, detail="애니메이션이 존재하지 않습니다.")

    return {
        "word": word.Word,
        "animation_path": word.animations[0].AnimePath  # 상대경로 전송하기(클라이언트에서 애니메이션 다운받아둘 수 있도록 구성하기)
    }

# 사용자 실시간 영상에서 npy 추출
def run_sign_model(frame: np.ndarray, db: Session) -> str:
    # MediaPipe 초기화
    with mp_pose.Pose(static_image_mode=True) as pose, \
         mp_hands.Hands(static_image_mode=True, max_num_hands=2) as hands:
        
        # BGR → RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 포즈 추론
        pose_result = pose.process(rgb)
        pose_landmarks = pose_result.pose_landmarks

        # 손 추론
        hand_result = hands.process(rgb)
        left_hand = right_hand = None

        if hand_result.multi_handedness:
            for idx, handedness in enumerate(hand_result.multi_handedness):
                label = handedness.classification[0].label
                if label == "Left":
                    left_hand = hand_result.multi_hand_landmarks[idx]
                elif label == "Right":
                    right_hand = hand_result.multi_hand_landmarks[idx]

        # landmark를 numpy 배열로 변환 (없으면 0으로 채움)
        def to_array(landmarks, count):
            if not landmarks:
                return np.zeros((count, 3))
            return np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])

        pose_arr = to_array(pose_landmarks, 33)
        left_hand_arr = to_array(left_hand, 21)
        right_hand_arr = to_array(right_hand, 21)

        # concat → shape: (75, 3) → flatten: (225,)
        landmark_vector = np.concatenate([pose_arr, left_hand_arr, right_hand_arr], axis=0).flatten()
        input_array = landmark_vector[np.newaxis, :]  # shape: (1, 225)

        # 모델 추론

        predicted_index = 0  # 더미 인덱스

        # Word 테이블에서 라벨 추출 (모델 학습 라벨 순서와 동일하게 데이터를 테이블에 추가해두기)
        word_list = db.query(Word).order_by(Word.WID).all()
        label_map = {i: w.Word for i, w in enumerate(word_list)}

        recognized_korean = label_map.get(predicted_index, "알 수 없음")
        return recognized_korean
