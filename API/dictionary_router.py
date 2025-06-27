from fastapi import APIRouter, Depends, Query, Request, Response
from sqlalchemy.orm import Session
from core_method import get_db, verify_or_refresh_token
from models import Word

router = APIRouter()

@router.get("/dictionary/words")
async def get_words(request: Request, response: Response, query: str = Query(None), db: Session = Depends(get_db),):
    user_id = verify_or_refresh_token(request, response)

    if query:
        words = db.query(Word).filter(Word.Word.contains(query)).all()
    else:
        words = db.query(Word).all()

    result = []
    for w in words:
        # animation_path = w.animations[0].AnimePath if w.animations else ""
        result.append({
            "wid": w.WID,
            "word": w.Word,
        })

    return result