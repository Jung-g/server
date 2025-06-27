from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Base

# DB 연결
SERVER = r"LAPTOP-5P5I1F6C\SQLEXPRESS"
DATABASE = "WB41"
UID = "aaa"
PWD = "1234"

# pyodbc + MSSQL 연결
DB_URL = f"mssql+pyodbc://{UID}:{PWD}@{SERVER}/{DATABASE}?driver=ODBC+Driver+17+for+SQL+Server"

# SQLAlchemy 엔진 생성
engine = create_engine(DB_URL, echo=True, future=True)

# 세션 생성 (트랜잭션 포함)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 테이블 자동 생성 함수 (이미 있으면 건너뜀)
def init_db():
    from models import (
        User, Word, Animation, Study, StudyWord,
        StudyRecord, BookMark, Token
    )
    Base.metadata.create_all(bind=engine)
