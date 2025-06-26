from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from core_method import create_access_token, get_db
from models import Token

router = APIRouter()

class RefreshRequest(BaseModel):
    refresh_token: str

@router.post("/token/re_access")
async def new_access_token(req: RefreshRequest, db: Session = Depends(get_db)):
    refresh_token = req.refresh_token
    # DB에 해당 Refresh Token 있는지 확인
    token_entry = db.query(Token).filter(Token.Refresh_token == refresh_token).first()
    
    if not token_entry:
        raise HTTPException(status_code=401, detail="유효하지 않은 Refresh 토큰입니다.")
    
    # 만료 여부 확인
    if token_entry.Expires < datetime.now(timezone.utc):
        raise HTTPException(status_code=401, detail="Refresh 토큰이 만료되었습니다.")
    
    # 새로운 Access 토큰 생성
    new_access_token = create_access_token(data={"sub": token_entry.UserID})
    
    return {
        "access_token": new_access_token,
        "token_type": "bearer"
    }