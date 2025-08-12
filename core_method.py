import os
from datetime import datetime, timedelta, timezone
from fastapi import HTTPException, Request, Response
from DB import SessionLocal
from dotenv import load_dotenv
from jose import ExpiredSignatureError, jwt, JWTError
from passlib.context import CryptContext

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
load_dotenv(dotenv_path="keys.env")
SECRET_KEY = os.getenv("SECRET_KEY")
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

# 액세스 토큰 유효 여부 (미유효시 재발급)
def verify_or_refresh_token(request: Request, response: Response) -> str:
    auth_header = request.headers.get("Authorization")
    refresh_token = request.headers.get("X-Refresh-Token")

    print("Authorization 헤더:", auth_header)
    print("refresh_token:", refresh_token)

    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="액세스 토큰이 없습니다.")

    token = auth_header.split(" ")[1]
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        print("access token 유효")
        return payload.get("sub")
    except ExpiredSignatureError as e:
        print("access token 만료")
        if not refresh_token:
            print("refresh token 없음:", e)
            raise HTTPException(status_code=401, detail="리프레시 토큰이 없습니다.")

        try:
            refresh_payload = jwt.decode(refresh_token, SECRET_KEY, algorithms=[ALGORITHM])
            user_id = refresh_payload.get("sub")
            new_access_token = create_access_token({"sub": user_id})
            response.headers["X-New-Access-Token"] = new_access_token
            print("새 access token 발급:", new_access_token)
            return user_id
        except JWTError as e2:
            print("refresh token 디코딩 실패:", e2)
            raise HTTPException(status_code=401, detail="리프레시 토큰이 유효하지 않습니다.")
    except JWTError as e3:
        print("access token 디코딩 실패:", e3)
        raise HTTPException(status_code=401, detail="액세스 토큰이 유효하지 않습니다.")

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
