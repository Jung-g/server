from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from core_method import verify_or_refresh_token, get_db
from models import BookMark

router = APIRouter()

@router.delete("/bookmark/remove/{word_id}")
async def remove_bookmark(word_id: int, db: Session = Depends(get_db), user_id: str = Depends(verify_or_refresh_token)):
    bookmark = db.query(BookMark)\
        .filter(BookMark.UserID == user_id, BookMark.WID == word_id)\
        .first()

    if not bookmark:
        raise HTTPException(status_code=404, detail="북마크가 없습니다.")

    db.delete(bookmark)
    db.commit()
    return {"success": True}
