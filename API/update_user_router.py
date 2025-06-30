from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from core_method import verify_or_refresh_token, get_db, pwd_context   
from models import User

router = APIRouter()

class UserUpdate(BaseModel):
    name: str | None = None
    new_password: str | None = None

@router.put("/user/update")
async def update_user(update_data: UserUpdate, db: Session = Depends(get_db), user_id: str = Depends(verify_or_refresh_token)):
    user = db.query(User).filter(User.UserID == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다.")

    updated = False
    if update_data.name:
        user.UserName = update_data.name
        updated = True
    if update_data.new_password:
        user.PassWord = pwd_context.hash(update_data.new_password)
        updated = True

    if updated:
        db.commit()
        return {
            "success": True,
            "nickname": user.UserName
        }
    else:
        return {"success": False}