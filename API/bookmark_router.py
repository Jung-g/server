from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from core_method import verify_or_refresh_token, get_db
from DB_Table import BookMark, Word

router = APIRouter(tags=["Bookmark"])

class AddBookmarkRequest(BaseModel):
    wid: int

# 북마크 추가
@router.post("/bookmark/add", summary="해당 단어의 북마크 요청을 처리합니다.", description="해당 단어의 북마크 요청을 처리합니다.")
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

# 북마크 제거
@router.delete("/bookmark/remove/{word_id}", summary="해당 단어의 북마크 해제 요청을 처리합니다.", description="해당 단어의 북마크 해제 요청을 처리합니다.")
async def remove_bookmark(word_id: int, db: Session = Depends(get_db), user_id: str = Depends(verify_or_refresh_token)):
    bookmark = db.query(BookMark)\
        .filter(BookMark.UserID == user_id, BookMark.WID == word_id)\
        .first()

    if not bookmark:
        raise HTTPException(status_code=404, detail="북마크가 없습니다.")

    db.delete(bookmark)
    db.commit()
    return {"success": True}

# 저장된 북마크 보기
@router.get("/bookmark/list", summary="북마크 설정된 단어들만 가져옵니다.", description="북마크 설정된 단어들만 가져옵니다.")
async def get_bookmarked_words(user_id: str = Depends(verify_or_refresh_token), db: Session = Depends(get_db)):
    results = (
        db.query(Word.WID, Word.Word)
        .join(BookMark, BookMark.WID == Word.WID)
        .filter(BookMark.UserID == user_id)
        .all()
    )

    return [{"wid": wid, "word": word} for wid, word in results]