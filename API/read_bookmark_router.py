from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from core_method import get_current_user_id, get_db
from models import BookMark, Word

router = APIRouter()
read_bookmark_router = APIRouter()

@read_bookmark_router.get("/bookmark/list")
def get_bookmarked_words(user_id: str = Depends(get_current_user_id), db: Session = Depends(get_db)):
    bookmarks = db.query(BookMark).filter_by(UserID=user_id).all()
    
    result = []
    for b in bookmarks:
        word = db.query(Word).filter_by(WID=b.WID).first()
        if word:
            anime = word.animations[0].AnimePath if word.animations else ""
            result.append({
                "wid": word.WID,
                "word": word.Word,
                "animation": anime
            })
    
    return result