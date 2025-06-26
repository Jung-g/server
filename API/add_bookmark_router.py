from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from core_method import get_current_user_id, get_db
from models import BookMark

router = APIRouter()

@router.post("/bookmark/toggle")
def toggle_bookmark(wid: int, db: Session = Depends(get_db), user_id: str = Depends(get_current_user_id)):
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