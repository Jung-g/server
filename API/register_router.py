from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from core_method import get_db, pwd_context
from models import User

router = APIRouter()

class RegisterRequest(BaseModel):
    id: str
    pw: str
    name: str

@router.post("/user/register")
async def register_user(req: RegisterRequest, db: Session = Depends(get_db)):
    id = req.id
    pw = req.pw
    name = req.name
    
    if db.query(User).filter(User.UserID == id).first():
        return {"success": False}

    hashed_pw = pwd_context.hash(pw)
    user = User(UserID=id, PassWord=hashed_pw, UserName=name)
    db.add(user)
    db.commit()
    return {"success": True}