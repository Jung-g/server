from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from pydantic import BaseModel
from sqlalchemy import and_
from sqlalchemy.orm import Session
from API.calendar_router import calculate_streak
from DB_Table import Study, StudyRecord, StudyStep, StudyStepMeta, Word
from core_method import get_db, verify_or_refresh_token

router = APIRouter(tags=["Study"])

# 학습 코스
@router.get("/study/list", summary="학습 코스명을 가져옵니다.", description="학습 코스명을 가져옵니다.")
async def get_study_list(request: Request, response: Response, db: Session = Depends(get_db), _: str = Depends(verify_or_refresh_token)):
    studies = db.query(Study).all()
    return [{"SID": s.SID, "Study_Course": s.Study_Course} for s in studies]

# 학습 코스의 세부사항
@router.get("/study/course", summary="선택된 학습 코스의 세부정보를 가져옵니다.", description="선택된 학습 코스의 세부정보를 가져옵니다.")
async def get_course_detail(request: Request, response: Response, course_name: str = Query(..., description="학습 코스 이름"), db: Session = Depends(get_db), _: str = Depends(verify_or_refresh_token)):
    study = db.query(Study).filter(Study.Study_Course == course_name).first()
    if not study:
        raise HTTPException(status_code=404, detail="해당 코스를 찾을 수 없습니다.")

    steps = (
        db.query(
            StudyStep.Step,
            StudyStep.WordOrder,
            StudyStep.WID,
            Word.Word,
            StudyStepMeta.StepName
        )
        .join(Word, StudyStep.WID == Word.WID)
        .join(StudyStepMeta, and_(
            StudyStep.SID == StudyStepMeta.SID,
            StudyStep.Step == StudyStepMeta.Step
        ))
        .filter(StudyStep.SID == study.SID)
        .order_by(StudyStep.Step, StudyStep.WordOrder)
        .all()
    )

    seen_steps = set()
    steps_list = []
    for (step, _, _, _, step_name) in steps:
        if step not in seen_steps:
            seen_steps.add(step)
            steps_list.append({
                "step": step,
                "step_name": step_name
            })

    return {
        "sid": study.SID,
        "title": study.Study_Course,
        "words": [
            {
                "step": step,
                "step_name": step_name,
                "word_order": word_order,
                "wid": wid,
                "word": word,
            }
            for (step, word_order, wid, word, step_name) in steps
        ],
        "steps": steps_list
    }

# 학습 정보 불러오기
@router.get("/study/stats", summary="학습 정보를 가져옵니다.", description="학습 정보를 가져옵니다.")
async def get_study_stats(db: Session = Depends(get_db), user_id: str = Depends(verify_or_refresh_token)):
    records = (
        db.query(StudyRecord)
        .filter(
            StudyRecord.UserID == user_id,
            StudyRecord.Complate == True
        )
        .all()
    )

    date_list = [record.Study_Date.date() for record in records]
    learned_dates = [d.isoformat() for d in date_list]
    streak_days = calculate_streak(date_list)

    sid_step_pairs = {(r.SID, r.Step) for r in records}

    word_ids = set()
    for sid, step in sid_step_pairs:
        step_words = (
            db.query(StudyStep.WID)
            .filter(StudyStep.SID == sid, StudyStep.Step == step)
            .all()
        )
        word_ids.update(wid for (wid,) in step_words)

    learned_words_count = len(word_ids)

    completed_steps_by_sid = {}
    for sid, step in sid_step_pairs:
        completed_steps_by_sid.setdefault(sid, []).append(step)

    for sid in completed_steps_by_sid:
        completed_steps_by_sid[sid] = sorted(completed_steps_by_sid[sid])

    return {
        "learned_dates": learned_dates,
        "streak_days": streak_days,
        "learned_words_count": learned_words_count,
        "completed_steps": completed_steps_by_sid
    }

# 학습 진행도(%)
@router.get("/study/completion_rate", summary="전체 학습 코스에 대해서 학습 진행도를 가져옵니다.", description="전체 학습 코스에 대해서 학습 진행도를 가져옵니다.")
async def get_completion_rate(db: Session = Depends(get_db), user_id: str = Depends(verify_or_refresh_token)):
    from sqlalchemy import func

    # 전체 코스 개수
    total_courses = db.query(Study).count()

    # 사용자가 완료한 레코드 중, 1~5단계별로 묶기
    subquery = (
        db.query(
            StudyRecord.SID,
            func.count(StudyRecord.Step).label("completed_steps")
        )
        .filter(
            StudyRecord.UserID == user_id,
            StudyRecord.Complate == True,
            StudyRecord.Step.in_([1, 2, 3, 4, 5])
        )
        .group_by(StudyRecord.SID)
        .having(func.count(StudyRecord.Step) == 5)  # 5단계 전부 완료한 코스만
        .subquery()
    )

    completed_courses = db.query(func.count()).select_from(subquery).scalar()

    return {
        "total_courses": total_courses,
        "completed_courses": completed_courses,
        "completion_percent": round((completed_courses / total_courses) * 100, 1)
        if total_courses > 0 else 0.0
    }

# 학습 완료
class StudyCompleteRequest(BaseModel):
    sid: int
    step: int
@router.post("/study/complete", summary="학습 완료 요청을 처리합니다.", description="학습 완료 요청을 처리합니다.")
async def complete_study(request: Request, response: Response, req: StudyCompleteRequest, db: Session = Depends(get_db)):
    user_id = verify_or_refresh_token(request, response)

    record = db.query(StudyRecord).filter_by(
        UserID=user_id,
        SID=req.sid,
        Step=req.step
    ).first()

    now = datetime.now()

    if record:
        record.Complate = True
        record.Study_Date = now
    else:
        record = StudyRecord(
            UserID=user_id,
            SID=req.sid,
            Step=req.step,
            Study_Date=now,
            Complate=True
        )
        db.add(record)

    db.commit()
    return {"success": True}

# 학습 완료한 step5만 가져오기
@router.get("/study/review_words", summary="완료한 학습의 퀴즈 단계를 가져옵니다.", description="완료한 학습의 퀴즈 단계를 가져옵니다.")
def get_review_step5_words(request: Request, response: Response, db: Session = Depends(get_db)):
    user_id = verify_or_refresh_token(request, response)

    # 사용자가 step 5 완료한 sid만 가져오기
    completed_sids = (
        db.query(StudyRecord.SID)
        .filter(StudyRecord.UserID == user_id, StudyRecord.Step == 5, StudyRecord.Complate == True)
        .distinct()
        .all()
    )
    sid_list = [sid for (sid,) in completed_sids]

    if not sid_list:
        return []

    # 해당 SID들의 Step 5에 등록된 단어(WID), 단어명, 코스명 반환
    results = (
        db.query(StudyStep, Word, Study)
        .join(Word, StudyStep.WID == Word.WID)
        .join(Study, Study.SID == StudyStep.SID)
        .filter(
            StudyStep.Step == 5,
            StudyStep.SID.in_(sid_list)
        )
        .order_by(StudyStep.SID, StudyStep.WordOrder)
        .all()
    )

    response_data = []
    for step, word, study in results:
        response_data.append({
            "wid": word.WID,
            "word": word.Word,
            "step": step.Step,
            "sid": step.SID,
            "course": study.Study_Course,
        })

    return response_data