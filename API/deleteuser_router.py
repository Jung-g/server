from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from models import User, Token, BookMark, StudyRecord
from core_method import get_db, verify_or_refresh_token, pwd_context

router = APIRouter()

class DeleteUserRequest(BaseModel):
    password: str

@router.post("/delete_user")
def delete_user(data: DeleteUserRequest, db: Session = Depends(get_db), user_id: str = Depends(verify_or_refresh_token)):
    try:
        user = db.query(User).filter(User.UserID == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="유저를 찾을 수 없습니다")

        if not pwd_context.verify(data.password, user.PassWord):
            raise HTTPException(status_code=401, detail="비밀번호가 일치하지 않습니다")

        # DB에서 해당 user 데이터 삭제
        db.query(Token).filter(Token.UserID == user_id).delete()
        db.query(BookMark).filter(BookMark.UserID == user_id).delete()
        db.query(StudyRecord).filter(StudyRecord.UserID == user_id).delete()
        db.delete(user)
        db.commit()

        return {"success": True}
    except Exception as e:
        db.rollback()
        print("오류: ", e)
        return {"success": False}
