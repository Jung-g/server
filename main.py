import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, APIRouter, Request, Response, Depends, Body, Query
from contextlib import asynccontextmanager
from DB import init_db, SessionLocal
from sqlalchemy.orm import Session
from models import User, Token, Word, BookMark, StudyRecord
from datetime import datetime, timedelta, timezone
from passlib.context import CryptContext
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
from pydantic import BaseModel

@asynccontextmanager
async def lifespan(app: FastAPI):
    #init_db()
    print("DB 연결 완료")

    yield 

    print("앱 종료됨")

app = FastAPI(lifespan=lifespan)

#region 내부 메서드
# DB 세션 생성
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

#JWT 초기 설정
load_dotenv(dotenv_path="secret_key.env")
SECRET_KEY = os.getenv("SECRET_KEY")  # 토큰 서명, 검증용 비밀 키 반드시 .env로 분리
ALGORITHM = "HS256" # 암호화 알고리즘
ACCESS_TOKEN_EXPIRE_MINUTES = 15 # 토큰 만료시간 (단이 : 분)

# Access 토큰 생성
def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# Refresh 토큰 생성
def create_refresh_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(days=30))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM), expire

# 비밀번호 해시화
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT 인증
def get_current_user_id(request: Request):
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="인증 정보가 없습니다.")

    token = auth_header.split(" ")[1]
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="토큰에 사용자 정보가 없습니다.")
        return user_id
    except JWTError:
        raise HTTPException(status_code=401, detail="토큰이 유효하지 않습니다.")
#endregion

#region 회원가입
@app.post("/user/register")
async def register_user(id: str, pw: str, name: str, db: Session = Depends(get_db)):
    if db.query(User).filter(User.UserID == id).first():
        return {"success": False}  # 이미 존재하는 아이디

    hashed_pw = pwd_context.hash(pw)
    user = User(UserID=id, PassWord=hashed_pw, UserName=name)
    db.add(user)
    db.commit()
    return {"success": True}
#endregion

#region 회원가입시 아이디 중복 체크
@app.get("/user/check-id")
async def check_user_id(id: str, db: Session = Depends(get_db)):
    exists = db.query(User).filter(User.UserID == id).first()
    return {"available": not bool(exists)}
#endregion

#region 로그인
@app.post("/user/login")
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
#endregion

#region Access 토큰 만료시 재발급 (Refresh 토큰 일치시에만)
@app.post("/token/refresh")
def refresh_access_token(refresh_token: str, db: Session = Depends(get_db)):
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
#endregion

#region 자동 로그아웃 (Refresh 토큰 만료시)
@app.post("/token/refresh")
def refresh_access_token(refresh_token: str, db: Session = Depends(get_db)):
    token_entry = db.query(Token).filter(Token.Refresh_token == refresh_token).first()

    if not token_entry:
        return {"logged_in": False}

    if token_entry.Expires < datetime.now(timezone.utc):
        db.delete(token_entry)
        db.commit()
        return {"logged_in": False}

    return {"logged_in": True}
#endregion

#region 로그아웃
@app.post("/user/logout")
def logout_user(refresh_token: str = Body(...), db: Session = Depends(get_db)):
    token_entry = db.query(Token).filter(Token.Refresh_token == refresh_token).first()

    if not token_entry:
        return {"success": False}

    db.delete(token_entry)
    db.commit()
    return {"success": True}
#endregion

#region 회원정보수정
class UserUpdate(BaseModel):
    name: str = None
    new_password: str = None

@app.put("/user/update")
def update_user(
    update_data: UserUpdate,
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user_id)
):
    user = db.query(User).filter(User.UserID == user_id).first()
    if not user:
        return {"success": False}

    updated = False
    if update_data.name:
        user.UserName = update_data.name
        updated = True
    if update_data.new_password:
        user.PassWord = pwd_context.hash(update_data.new_password)
        updated = True

    if updated:
        db.commit()
        return {"success": True}
    else:
        return {"success": False}
#endregion

#region 사전
# app.include_router(dict_router)
# router = APIRouter()
@app.get("/dictionary/words")
def get_words(query: str = Query(None), db: Session = Depends(get_db)):
    if query:
        words = db.query(Word).filter(Word.Word.contains(query)).all()
    else:
        words = db.query(Word).all()

    result = []
    for w in words:
        animation_path = w.animations[0].AnimePath if w.animations else ""
        
        result.append({
            "wid": w.WID,
            "word": w.Word,
            "animation": animation_path
        })

    return result 
#endregion

#region 번역 (아마 외부 API 활용할 듯)

#endregion

#region 학습 코스 선택

#endregion

#region 단어 북마크 추가/삭제
@app.post("/bookmark/toggle")
def toggle_bookmark(wid: int, db: Session = Depends(get_db), user_id: str = Depends(get_current_user_id)):
    bookmark = db.query(BookMark).filter_by(UserID=user_id, WID=wid).first()

    if bookmark:
        db.delete(bookmark)
        db.commit()
        return {"bookmarked": False}
    else:
        new_bookmark = BookMark(UserID=user_id, WID=wid)
        db.add(new_bookmark)
        db.commit()
        return {"bookmarked": True}
#endregion

#region 단어 북마크 조회
@app.get("/bookmark/list")
def get_bookmarked_words(user_id: str = Depends(get_current_user_id), db: Session = Depends(get_db)):
    bookmarks = db.query(BookMark).filter_by(UserID=user_id).all()
    
    result = []
    for b in bookmarks:
        word = db.query(Word).filter_by(WID=b.WID).first()
        if word:
            anime = word.animations[0].AnimePath if word.animations else ""
            result.append({
                "wid": word.WID,
                "word": word.Word,
                "animation": anime
            })
    
    return result
#endregion

#region 학습 달력
@app.get("/study/calendar")
def get_study_records(
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user_id)
):
    records = db.query(StudyRecord).filter(StudyRecord.UserID == user_id).all()
    
    result = [
        {
            "date": record.Study_Date.data().isoformat(), # 이부분 하얀글씨 해결하기
            "sid": record.SID
        }
        for record in records
    ]
    
    return {"records": result}
#endregion







#region

#endregion