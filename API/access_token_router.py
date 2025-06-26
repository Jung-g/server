from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from main import create_access_token, get_db
from models import Token

re_access_token_router = APIRouter()

@re_access_token_router.post("/token/re_access")
def new_access_token(refresh_token: str, db: Session = Depends(get_db)):
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