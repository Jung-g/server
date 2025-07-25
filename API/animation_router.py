import traceback
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse
from requests import Session
from anime.motion_merge import api_motion_merge, check_merge
from core_method import get_db, verify_or_refresh_token

router = APIRouter()

@router.get("/animation")
async def get_sign_animation(request: Request, response: Response, word_text: str = Query(..., description="입력된 한국어 단어"), db: Session = Depends(get_db)):
    user_id = verify_or_refresh_token(request, response)
    

    words = word_text.strip().split()
    
    try:
        motion_data = check_merge(words, send_type='api')
    except Exception as e:
        print("[ERROR] check_merge 에러 발생:", e)
        traceback.print_exc()
        raise HTTPException(
            status_code=500, 
            detail=f'{e}' # 클라이언트에게 보여줄 메시지
        )
    
    frame_generator = api_motion_merge(*motion_data)
    frame_list = list(frame_generator)

    return JSONResponse(content={"frames": frame_list})