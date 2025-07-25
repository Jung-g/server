from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from core_method import create_access_token, create_refresh_token, get_db, pwd_context
from DB_Table import Token, User

router = APIRouter()

# 로그인
class LoginRequest(BaseModel):
    id: str
    pw: str
@router.post("/user/login")
async def login_user(req: LoginRequest, db: Session = Depends(get_db)):
    id = req.id
    pw = req.pw

    user = db.query(User).filter(User.UserID == id).first()

    if not user:
        return {"success": False}

    if not pwd_context.verify(pw, user.PassWord):
        return {"success": False}

    # Access 토큰 생성
    access_token = create_access_token(data={"sub": id})

    # Refresh 토큰 생성
    refresh_token, expires = create_refresh_token(data={"sub": id})

    # 기존 토큰 있는지 확인
    existing_token = db.query(Token).filter(Token.UserID == id).first()
    if existing_token:
        existing_token.Refresh_token = refresh_token
        existing_token.Expires = expires
    else:
        new_token = Token(
            UserID=id,
            Refresh_token=refresh_token,
            Expires=expires
        )
        db.add(new_token)

    db.commit()

    return {
        "success": True,
        "userID": id,
        "nickname": user.UserName,
        "access_token": access_token,
        "token_type": "bearer",
        "refresh_token": refresh_token,
        "expires_at": expires.isoformat()
    }

# 자동 로그인
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