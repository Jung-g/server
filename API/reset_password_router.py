from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from core_method import get_db, pwd_context
from models import User

router = APIRouter()

class PasswordResetRequest(BaseModel):
    user_id: str
    new_password: str

@router.put("/user/reset_password")
async def reset_password(data: PasswordResetRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.UserID == data.user_id).first()
    if not user:
        return {"success": False, "message": "존재하지 않는 ID입니다."}

    user.PassWord = pwd_context.hash(data.new_password)
    db.commit()
    return {"success": True}
