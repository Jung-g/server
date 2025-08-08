import cv2
import os

# 확인할 영상 파일 경로
video_path = "debug_videos/aaa_20250807_142634.mp4"  # 예시 이름, 실제 파일명으로 수정하세요

# 영상 파일 열기
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("비디오 파일을 열 수 없습니다.")
    exit()

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 프레임 번호 표시
    frame_count += 1
    cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # 프레임 보여주기
    cv2.imshow("Frame", frame)

    # 'q' 키 누르면 종료
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
