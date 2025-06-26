from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from models import Study, StudyWord, Word
from core_method import get_db

router = APIRouter()
course_router = APIRouter()

@course_router.get("/study/course")
def get_course_detail(course_name: str = Query(..., description="학습 코스 이름"), db: Session = Depends(get_db)):

    study = db.query(Study).filter(Study.Study_Course == course_name).first()
    if not study:
        raise HTTPException(status_code=404, detail="해당 코스를 찾을 수 없습니다.")

    words = (
        db.query(StudyWord, Word)
        .join(Word, StudyWord.WID == Word.WID)
        .filter(StudyWord.SID == study.SID)
        .order_by(StudyWord.Index)
        .all()
    )

    return {
        "sid": study.SID,
        "title": study.Study_Course,
        "words": [
            {
                "wid": w.WID,
                "word": w.Word,
                "index": sw.Index
            }
            for sw, w in words
        ]
    }
