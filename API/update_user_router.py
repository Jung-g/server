from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from core_method import get_current_user_id, get_db, pwd_context   
from models import User

update_user_router = APIRouter()

class UserUpdate(BaseModel):
    name: str = None
    new_password: str = None

@update_user_router.put("/user/update")
def update_user(
    update_data: UserUpdate,
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user_id)
):
    user = db.query(User).filter(User.UserID == user_id).first()
    if not user:
        return {"success": False}

    updated = False
    if update_data.name:
        user.UserName = update_data.name
        updated = True
    if update_data.new_password:
        user.PassWord = pwd_context.hash(update_data.new_password)
        updated = True

    if updated:
        db.commit()
        return {"success": True}
    else:
        return {"success": False}