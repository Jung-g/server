from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from core_method import get_db
from models import Word

router = APIRouter()

@router.get("/dictionary/words")
def get_words(query: str = Query(None), db: Session = Depends(get_db)):
    if query:
        words = db.query(Word).filter(Word.Word.contains(query)).all()
    else:
        words = db.query(Word).all()

    result = []
    for w in words:
        animation_path = w.animations[0].AnimePath if w.animations else ""
        
        result.append({
            "wid": w.WID,
            "word": w.Word,
            "animation": animation_path
        })

    return result 