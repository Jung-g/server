from fastapi import FastAPI , HTTPException
from fastapi.responses import StreamingResponse

from motion_merge import check_merge, api_motion_merge

#region 레디스사용

# import json
# import uuid
# import redis

# try:
#     redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
#     redis_client.ping() # 연결 테스트
#     print("Redis에 성공적으로 연결되었습니다.")
# except redis.exceptions.ConnectionError as e:
#     print(f"Redis 연결 오류: {e}")
#     # Redis 연결 실패 시 프로그램 종료 또는 대체 로직 수행
#     exit(1)

# # Redis 데이터 저장시간 10분
# JOB_EXPIRATION_SECONDS = 600 

#endregion

app = FastAPI()
# uvicorn api:app --reload --port 1234

@app.get("/stream/realtime")
async def stream_keypoints_video(words_json: str):
    words = words_json.split(',')
    for t in words:
        print(t)    
    try:
        motion_data = check_merge(words, send_type='api')
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f'{e}' # 클라이언트에게 보여줄 메시지
        )
    
    return StreamingResponse(
        api_motion_merge(*motion_data),        
         media_type='application/x-ndjson'
    )