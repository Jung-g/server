from fastapi import APIRouter, Body, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from core_method import get_db
from models import Token

router = APIRouter()

class LogoutRequest(BaseModel):
    refresh_token: str

@router.post("/user/logout")
async def logout_user(req: LogoutRequest, db: Session = Depends(get_db)):
    refresh_token = req.refresh_token

    token_entry = db.query(Token).filter(Token.Refresh_token == refresh_token).first()

    if not token_entry:
        return {"success": False}

    db.delete(token_entry)
    db.commit()
    return {"success": True}
