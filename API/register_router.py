from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from main import get_db, pwd_context
from models import User

register_router = APIRouter()

@register_router.post("/user/register")
async def register_user(id: str, pw: str, name: str, db: Session = Depends(get_db)):
    if db.query(User).filter(User.UserID == id).first():
        return {"success": False}

    hashed_pw = pwd_context.hash(pw)
    user = User(UserID=id, PassWord=hashed_pw, UserName=name)
    db.add(user)
    db.commit()
    return {"success": True}