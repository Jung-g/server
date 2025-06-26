from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from models import Word
from main import get_db
# 외부 번역 라이브러리나 API 서비스 추가하기

translate_router = APIRouter()

# 수어 → 텍스트 → 번역
@translate_router.post("/translate/sign-to-text")
def translate_sign_to_text(recognized_korean: str = Query(..., description="수어 분석된 한국어 단어"), db: Session = Depends(get_db)):
    word = db.query(Word).filter(Word.Word == recognized_korean).first()
    if not word:
        raise HTTPException(status_code=404, detail="해당 단어를 찾을 수 없습니다.")

    # 수어 동작 분석 모델 연결하기

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
