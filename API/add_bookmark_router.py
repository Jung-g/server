from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from core_method import get_current_user_id, get_db
from models import BookMark

router = APIRouter()

class AddBookmarkRequest(BaseModel):
    user_id: str
    word_id: int

@router.post("/bookmark/add")
async def add_bookmark(req: AddBookmarkRequest, db: Session = Depends(get_db)):
    user_id = req.user_id
    wid = req.word_id
    bookmark = db.query(BookMark).filter_by(UserID=user_id, WID=wid).first()

    if bookmark:
        db.delete(bookmark)
        db.commit()
        return {"bookmarked": False}
    else:
        new_bookmark = BookMark(UserID=user_id, WID=wid)
        db.add(new_bookmark)
        db.commit()
        return {"bookmarked": True}