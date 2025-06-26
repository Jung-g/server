from contextlib import asynccontextmanager
from fastapi import FastAPI
from DB import init_db, SessionLocal
from API.dictionary_router import router as dictionary_router
from API.register_router import router as register_router
from API.checkUID_router import router as checkUID_router
from API.login_router import router as login_router
from API.access_token_router import router as access_token_router
from API.auto_logout_router import router as auto_logout_router
from API.logout_router import router as logout_router
from API.study_course_router import router as study_course_router
from API.translate_router import router as translate_router
from API.update_user_router import router as update_user_router
from API.add_bookmark_router import router as add_bookmark_router
from API.read_bookmark_router import router as read_bookmark_router
from API.calendar_router import router as calendar_router

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

# 번역
app.include_router(translate_router)

# 학습 코스 선택
app.include_router(study_course_router)

# 단어 북마크 추가/삭제
app.include_router(add_bookmark_router)

# 단어 북마크 조회
app.include_router(read_bookmark_router)

# 학습 달력
app.include_router(calendar_router)
