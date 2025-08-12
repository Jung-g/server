import os
import datetime as dt
from jose import jwt, JWTError
from passlib.context import CryptContext

ALGO = "HS256"
SECRET = os.getenv("JWT_SECRET", "change-me")  # 운영에서 꼭 변경
ACCESS_MIN = int(os.getenv("JWT_ACCESS_MIN", "30"))

_pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(p: str) -> str:
    return _pwd.hash(p)

def verify_password(p: str, hashed: str) -> bool:
    try:
        return _pwd.verify(p, hashed)
    except Exception:
        return False

def create_access(sub: str) -> str:
    now = dt.datetime.utcnow()
    payload = {"sub": sub, "iat": now, "exp": now + dt.timedelta(minutes=ACCESS_MIN)}
    return jwt.encode(payload, SECRET, algorithm=ALGO)

def decode_access(token: str) -> dict:
    return jwt.decode(token, SECRET, algorithms=[ALGO])
