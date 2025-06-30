from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from core_method import verify_or_refresh_token, get_db
from models import BookMark, Word

router = APIRouter()

@router.get("/bookmark/list")
async def get_bookmarked_words(user_id: str = Depends(verify_or_refresh_token), db: Session = Depends(get_db)):
    results = (
        db.query(Word.WID, Word.Word)
        .join(BookMark, BookMark.WID == Word.WID)
        .filter(BookMark.UserID == user_id)
        .all()
    )

    return [{"wid": wid, "word": word} for wid, word in results]