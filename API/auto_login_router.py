from fastapi import APIRouter, Depends, HTTPException
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

    if not refresh_token:
        raise HTTPException(status_code=401, detail="Refresh token 누락")
    
    token_entry = db.query(Token).filter(Token.Refresh_token == refresh_token).first()

    if token_entry is None:
        raise HTTPException(status_code=401, detail="유효하지 않은 토큰")

    expires = token_entry.Expires.replace(tzinfo=timezone.utc)

    if expires < datetime.now(timezone.utc):
        db.delete(token_entry)
        db.commit()
        raise HTTPException(status_code=401, detail="토큰 만료")

    return {"success": True}
