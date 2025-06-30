from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from core_method import verify_or_refresh_token, get_db
from models import BookMark

router = APIRouter()

class AddBookmarkRequest(BaseModel):
    wid: int

@router.post("/bookmark/add")
async def add_bookmark(req: AddBookmarkRequest, db: Session = Depends(get_db), user_id: str = Depends(verify_or_refresh_token)):
    wid = req.wid
    exists = db.query(BookMark)\
        .filter_by(UserID=user_id, WID=wid)\
        .first()
    if exists:
        raise HTTPException(status_code=409, detail="이미 북마크된 단어입니다.")

    new_bm = BookMark(UserID=user_id, WID=wid)
    db.add(new_bm)
    db.commit()
    return {"success": True}