from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from core_method import get_db
from models import Token
from datetime import datetime, timezone

router = APIRouter()

class AutoLoginRequest(BaseModel):
    refresh_token: str

@router.post("/user/auto_login")
async def auto_login(req: AutoLoginRequest, db: Session = Depends(get_db)):
    refresh_token = req.refresh_token

    token_entry = db.query(Token).filter(Token.Refresh_token == refresh_token).first()

    expires = token_entry.Expires.replace(tzinfo=timezone.utc)

    if not token_entry:
        return {"success": False}

    if expires < datetime.now(timezone.utc):
        return {"success": False}

    return {"success": True}
