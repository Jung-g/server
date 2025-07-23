import os
import cv2
import base64
import numpy as np
import pandas as pd
import mediapipe as mp
from typing import Literal


def linear_joint(video_path):
    """
    영상에서 양손과 몸 랜드마크를 추출하고, 누락된 프레임을 찾는 함수
    """
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,  
        min_detection_confidence=0.9,
        min_tracking_confidence=0.5        
    )
    
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,       
        model_complexity=2,             
        smooth_landmarks=True,          
        enable_segmentation=False,      
        min_detection_confidence=0.5,   
        min_tracking_confidence=0.5     
    )
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"오류: {video_path} 파일을 열 수 없습니다.")
        return None, None, (0, 0)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    hands_landmarks_data = []
    pose_landmarks_data = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = hands.process(image_rgb)
        pose_results = pose.process(image_rgb)
        
        hands_data = create_hands(hand_results)
        hands_landmarks_data.append(hands_data)
        
        pose_data = create_pose(pose_results)
        pose_landmarks_data.append(pose_data)
        
    cap.release()
    hands.close()
    pose.close()
    
    hands_original_df = pd.DataFrame(hands_landmarks_data)
    pose_original_df = pd.DataFrame(pose_landmarks_data)
    
    return hands_original_df, pose_original_df, (frame_width, frame_height)

def create_hands(results):
    
    frame_landmarks = {f'{side}_{h_type}_{i}': np.nan 
                           for side in ['left', 'right'] 
                           for h_type in ['landmark_x', 'landmark_y'] 
                           for i in range(21)}

    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            handedness = results.multi_handedness[idx].classification[0].label
            hand_side = 'left' if handedness == 'Left' else 'right'
            
            for i, landmark in enumerate(hand_landmarks.landmark):
                frame_landmarks[f'{hand_side}_landmark_x_{i}'] = landmark.x
                frame_landmarks[f'{hand_side}_landmark_y_{i}'] = landmark.y

    return frame_landmarks

def create_pose(results):
    
    frame_landmarks = {f'pose_{h_type}_{i}': np.nan 
                           for h_type in ['landmark_x', 'landmark_y'] 
                           for i in range(33)}

    if results.pose_landmarks:
                    
        for i, landmark in enumerate(results.pose_landmarks.landmark):
            frame_landmarks[f'pose_landmark_x_{i}'] = landmark.x
            frame_landmarks[f'pose_landmark_y_{i}'] = landmark.y

    return frame_landmarks

def draw_hand(frame, pose_df, hand_df, hand_side, frame_dims, frame_idx, is_debug = False):
    """한 손의 랜드마크와 연결선을 그리는 보조 함수."""
    if is_debug:
        original_df, interpolated_df = hand_df
        pose_origin, _ = pose_df
        
        original_row = original_df.loc[frame_idx]        
        is_original = pd.notna(original_row[f'{hand_side}_landmark_x_0'])        
    else:
        interpolated_df = hand_df
        pose_origin = pose_df
        
    landmark_row = interpolated_df.loc[frame_idx]
    pose_row = pose_origin.loc[frame_idx]
    
    # pose의 손목과 hands의 손목을 붙이는 계산
    if hand_side == 'left':
        first_pose = (pose_row['pose_landmark_x_16'] , pose_row['pose_landmark_y_16'])
    else:
        first_pose = (pose_row['pose_landmark_x_15'] , pose_row['pose_landmark_y_15'])
       
    cal = (first_pose[0] - landmark_row[f'{hand_side}_landmark_x_0'] , first_pose[1] - landmark_row[f'{hand_side}_landmark_y_0']) 
    
        
    mp_hands = mp.solutions.hands
    frame_width, frame_height = frame_dims

    up_pos = 0
    if is_debug:
        dot_color = (255, 0, 0) if is_original else (0, 0, 255)
        line_color = (0, 255, 0) if is_original else (0, 255, 255)  
    else:
        dot_color = (255, 0, 0)
        line_color = (0, 255, 0)
        up_pos = 0.1

    points = []
    for i in range(21):
        x = landmark_row[f'{hand_side}_landmark_x_{i}'] + cal[0]
        y = landmark_row[f'{hand_side}_landmark_y_{i}'] - up_pos + cal[1]
        if pd.notna(x) and pd.notna(y):
            points.append((int(x * frame_width), int(y * frame_height)))
        else:
            points.append(None)
    
    if points:
        for connection in mp_hands.HAND_CONNECTIONS:
            start_idx, end_idx = connection[0], connection[1]
            if points[start_idx] and points[end_idx]:
                cv2.line(frame, points[start_idx], points[end_idx], line_color, 2)
        for point in points:
            if point:
                cv2.circle(frame, point, 5, dot_color, -1)

def draw_pose(frame, pose_df, frame_dims, frame_idx, is_debug = False):
    """한 손의 랜드마크와 연결선을 그리는 보조 함수."""
    
    hide_point = [17 , 18 , 19 , 20 , 21 , 22]
    
    mp_pose = mp.solutions.pose
    frame_width, frame_height = frame_dims
    
    if is_debug:
        original_df , interpolated_df = pose_df
        original_row = original_df.loc[frame_idx]
        is_original = pd.notna(original_row[f'pose_landmark_x_0'])        
    else:
        interpolated_df = pose_df
    
    landmark_row = interpolated_df.loc[frame_idx]
    
    up_pos = 0
    if is_debug:
        dot_color = (255, 0, 0) if is_original else (0, 0, 255)
        line_color = (0, 255, 0) if is_original else (0, 255, 255)  
    else:
        dot_color = (255, 0, 0)
        line_color = (0, 255, 0)
        up_pos = 0.1

    points = []
    for i in range(33):
        x = landmark_row[f'pose_landmark_x_{i}']
        y = landmark_row[f'pose_landmark_y_{i}'] - up_pos
        if pd.notna(x) and pd.notna(y):
            points.append((int(x * frame_width), int(y * frame_height)))
        else:
            points.append(None)
    
    if points:
        for connection in mp_pose.POSE_CONNECTIONS:
            start_idx, end_idx = connection
            if start_idx in hide_point or end_idx in hide_point:
                continue
            if points[start_idx] and points[end_idx]:
                cv2.line(frame, points[start_idx], points[end_idx], line_color, 2)
        
        for idx in range(len(points)):
            if idx in hide_point:
                continue
            if points[idx]:
                cv2.circle(frame, points[idx], 5, dot_color, -1)
                
def realtime_visualize(video_path, hand_df, pose_df , frame_dimensions, loss_info):
    """
    보간된 양손의 결과와 프레임 손실 정보를 영상 위에 시각화하는 함수.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        
        draw_hand(frame, pose_df, hand_df, 'left', frame_dimensions , frame_idx, is_debug=True)
        draw_hand(frame, pose_df, hand_df, 'right', frame_dimensions , frame_idx, is_debug=True)
        
        draw_pose(frame, pose_df, frame_dimensions, frame_idx, is_debug= True)

        # --- 핵심 추가 부분: 영상에 손실 프레임 정보 표시 ---
        total_frames, left_lost, right_lost = loss_info
        text_left = f"Left Lost: {left_lost}/{total_frames} ({left_lost/total_frames:.1%})"
        text_right = f"Right Lost: {right_lost}/{total_frames} ({right_lost/total_frames:.1%})"
        
        cv2.putText(frame, text_left, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, text_right, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        # ---------------------------------------------------

        cv2.imshow('Both Hands Interpolation Visualization', frame)
        if cv2.waitKey(60) & 0xFF == ord('q'): break
        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

def create_keypoint_video(output_filename, hand_df, pose_df, frame_dimensions, fps=30 , frame_len = 0, out_type : Literal['Word', 'Sentence'] ='Word'):
    """
    키포인트 애니메이션을 시각화하고 MP4 파일로 저장하는 함수.
    """
    frame_width, frame_height = frame_dimensions

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # MP4 코덱
    base_dir = f'output//{out_type}'
    os.makedirs(base_dir, exist_ok=True)        
    video_path = os.path.join(base_dir , f'{output_filename}.mp4')
    
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (frame_width, frame_height))        
    
    for frame_idx in range(frame_len):
        black_canvas = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

        draw_pose(black_canvas, pose_df, frame_dimensions, frame_idx )
        
        draw_hand(black_canvas, pose_df, hand_df, 'left',  frame_dimensions, frame_idx)
        draw_hand(black_canvas, pose_df, hand_df, 'right', frame_dimensions, frame_idx) 

        video_writer.write(black_canvas)
        
    video_writer.release()
    cv2.destroyAllWindows()
    
def create_keypoint_debug(output_filename, hand_df, pose_df, frame_dimensions, fps=30 , frame_len = 0, idx = None):
    """
    키포인트 애니메이션을 시각화하고 MP4,PNG 두 파일로 저장하는 함수 -> 디버깅용
    ** 보간된 부분은 빨간색으로 강조됨 **
    """
    frame_width, frame_height = frame_dimensions

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # MP4 코덱
    
    base_dir = 'debug'                    
    os.makedirs(base_dir, exist_ok=True)        
    video_path = os.path.join(base_dir , f'{output_filename}{idx}.mp4')
    
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (frame_width, frame_height))
        
    for frame_idx in range(frame_len):
        black_canvas = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

        draw_pose(black_canvas, pose_df, frame_dimensions, frame_idx , is_debug=True)
        
        draw_hand(black_canvas, pose_df, hand_df, 'left',  frame_dimensions, frame_idx, is_debug=True)
        draw_hand(black_canvas, pose_df, hand_df, 'right', frame_dimensions, frame_idx, is_debug=True) 

        video_writer.write(black_canvas)
        
        #png 파일 생성
        base_dir = f'debug//{output_filename}{idx}'
        os.makedirs(base_dir, exist_ok=True)
        
        png_filename = os.path.join(base_dir, f'frame_{frame_idx:04d}.png')
        cv2.imwrite(png_filename, black_canvas)

    video_writer.release()
    cv2.destroyAllWindows()
    
def api_draw(hand_df, pose_df, frame_dimensions, frame_len):
    """
    키포인트 프레임을 한 장씩 생성하여 yield하는 제너레이터 함수.
    """
    frame_width, frame_height = frame_dimensions

    for frame_idx in range(frame_len):

        black_canvas = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        
        draw_pose(black_canvas, pose_df, frame_dimensions, frame_idx)
        draw_hand(black_canvas, pose_df, hand_df, 'left', frame_dimensions, frame_idx)
        draw_hand(black_canvas, pose_df, hand_df, 'right', frame_dimensions, frame_idx)

        (flag, encoded_image) = cv2.imencode(".jpg", black_canvas)
        if not flag:
            continue

        base64_bytes = base64.b64encode(encoded_image)
        base64_string = base64_bytes.decode('utf-8')

        yield base64_string
        # yield (b'--frame\r\n' 
        #        b'Content-Type: image/jpeg\r\n\r\n' + encoded_image.tobytes() + b'\r\n')
