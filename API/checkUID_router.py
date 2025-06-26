from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from main import get_db
from models import User

check_Userid_router = APIRouter()

@check_Userid_router.get("/user/check_id")
async def check_user_id(id: str, db: Session = Depends(get_db)):
    exists = db.query(User).filter(User.UserID == id).first()
    return {"available": not bool(exists)}