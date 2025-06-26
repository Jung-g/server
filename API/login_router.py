from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from core_method import create_access_token, create_refresh_token, get_db, pwd_context
from models import Token, User

router = APIRouter()

@router.post("/user/login")
def login_user(id: str, pw: str, db: Session = Depends(get_db)):
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
        "access_token": access_token,
        "token_type": "bearer",
        "refresh_token": refresh_token,
        "expires_at": expires.isoformat()
    }