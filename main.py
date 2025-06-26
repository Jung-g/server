import os
from datetime import datetime, timedelta, timezone
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from DB import init_db, SessionLocal

from dotenv import load_dotenv
from jose import jwt, JWTError
from passlib.context import CryptContext

from API import (dictionary_router, register_router, checkUID_router, 
                 login_router, access_token_router, auto_logout_router, 
                 logout_router, study_course_router, update_user_router, 
                 add_bookmark_router, read_bookmark_router, calendar_router)

@asynccontextmanager
async def lifespan(app: FastAPI):
    #init_db()
    print("DB 연결 완료")

    yield 

    print("앱 종료됨")

app = FastAPI(lifespan=lifespan)

# 회원가입
app.include_router(register_router)

# 회원가입시 아이디 중복 체크
app.include_router(checkUID_router)

# 로그인
app.include_router(login_router)

# Access 토큰 만료시 재발급 (Refresh 토큰 일치시에만)
app.include_router(access_token_router)

# 자동 로그아웃 (Refresh 토큰 만료시)
app.include_router(auto_logout_router)

# 로그아웃
app.include_router(logout_router)

# 회원정보수정
app.include_router(update_user_router)

# 사전
app.include_router(dictionary_router)

# 번역 (아마 외부 API 같이 활용할 듯)


# 학습 코스 선택
app.include_router(study_course_router)

# 단어 북마크 추가/삭제
app.include_router(add_bookmark_router)

# 단어 북마크 조회
app.include_router(read_bookmark_router)

# 학습 달력
app.include_router(calendar_router)


#region 내부 메서드
# DB 세션 생성
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# 비밀번호 해시화
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

#JWT 초기 설정
load_dotenv(dotenv_path="secret_key.env")
SECRET_KEY = os.getenv("SECRET_KEY")  # 토큰 서명, 검증용 비밀 키 반드시 .env 파일로 분리
ALGORITHM = "HS256" # 암호화 알고리즘
ACCESS_TOKEN_EXPIRE_MINUTES = 15 # 토큰 만료시간 (단위 : 분)

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
#endregion
