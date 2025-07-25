from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from DB import init_db
from API.dictionary_router import router as dictionary_router
from API.checkUID_router import router as checkUID_router
from API.login_router import router as login_router
from API.logout_router import router as logout_router
from API.study_course_router import router as study_course_router
# from API.translate_router import router as translate_router
from API.translate_routerA import router as translate_routerA
from API.translate_routerB import router as translate_routerB
from API.user_router import router as user_router
from API.bookmark_router import router as bookmark_router
from API.calendar_router import router as calendar_router
from API.animation_router import router as animation_router
# uvicorn main:app --host 0.0.0.0 --port 80 --log-level debug --reload

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    print("DB 연결 완료")

    yield 

    print("앱 종료됨")

app = FastAPI(lifespan=lifespan)
app.mount("/video", StaticFiles(directory="video"), name="video")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# user정보 수정 (회원가입 / 탈퇴 / 비밀번호, 닉네임 변경)
app.include_router(user_router)

print("user_router 등록 완료")

# 회원가입시 아이디 중복 체크
app.include_router(checkUID_router)

# 로그인 / 자동 로그인
app.include_router(login_router)

# 로그아웃 / 자동 로그아웃
app.include_router(logout_router)

# 사전
app.include_router(dictionary_router)

# 번역
# app.include_router(translate_router)
app.include_router(translate_routerA)
app.include_router(translate_routerB)

# 학습 코스 선택
app.include_router(study_course_router)

# 단어 북마크 추가 / 삭제 / 조회
app.include_router(bookmark_router)

# 학습 달력
app.include_router(calendar_router)

# 애니메이션
app.include_router(animation_router)