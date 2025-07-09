from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from sqlalchemy import and_
from sqlalchemy.orm import Session
from models import Study, StudyRecord, StudyStep, StudyStepMeta, Word
from core_method import get_db, verify_or_refresh_token

router = APIRouter()

@router.get("/study/list")
async def get_study_list(request: Request, response: Response, db: Session = Depends(get_db), _: str = Depends(verify_or_refresh_token)):
    studies = db.query(Study).all()
    return [{"SID": s.SID, "Study_Course": s.Study_Course} for s in studies]

@router.get("/study/course")
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
        ]
    }

@router.get("/study/completion_rate")
async def get_completion_rate(
    db: Session = Depends(get_db),
    user_id: str = Depends(verify_or_refresh_token)
):
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
