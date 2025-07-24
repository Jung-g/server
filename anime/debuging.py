import os
import sys
import argparse
from tqdm import tqdm 

from .Utils.Spline import *
from .Utils.Interpolation import Detect_joint
from .Utils.Draw import linear_joint, create_keypoint_debug


def debug_check_video(video_path):
    
    _, file_extension = os.path.splitext(video_path)
    if not file_extension == '':
        raise FileNotFoundError('파일이 아닌 디렉토리로 설정을 해야합니다.')
    
    video_names = []
    for video_name in sorted(os.listdir(video_path)):
        _, file_extension = os.path.splitext(video_name)
        if file_extension == '':
            continue
        
        path = os.path.join(video_path , video_name)
        video_names.append(path)
    
    if len(video_names) == 0:
        raise FileNotFoundError('동영상 파일이 존재하지 않습니다.')
    return video_names

def debug_test(video_path,is_out, idx):
    video_name = video_path.split('\\')[-1].split('.')[0]
    hands_original_data, pose_original_data, dims = linear_joint(video_path)

    if hands_original_data is not None:
        
        total_frames = len(hands_original_data)
        left_hand_lost = hands_original_data['left_landmark_x_0'].isna().sum()
        right_hand_lost = hands_original_data['right_landmark_x_0'].isna().sum()

        debug_data = (idx , left_hand_lost , right_hand_lost)
         
        check_data = Detect_joint(hands_original_data , is_debug= True , debug_data = debug_data)
        hand_interpolated_data = spline(check_data.frames)
        pose_interpolated_data = spline_cal(pose_original_data)
        
        hand_df = ( hands_original_data, hand_interpolated_data)
        pose_df = ( pose_original_data,  pose_interpolated_data)
        
        if is_out:
            create_keypoint_debug(video_name, hand_df, pose_df, dims, frame_len=total_frames, is_debug = True, idx = debug_data[0])
    raise ValueError('모션이 감지되지 않았습니다.')

if __name__ == "__main__":
        
    video_path = None
    is_out = True
    
    #f5 로 디버깅 할 때
    if len(sys.argv) == 1:
        # --- 디버그용 ---
        video_path = r'C:\Users\User\Desktop\새 폴더' 
        # -----------------------------
    #명령어로 시작할 때
    else:
        
        parser = argparse.ArgumentParser(description="디버깅용 키포인트 추출 스크립트 . \n Debug파일에 전체적으로 저장이 됨.")
        parser.add_argument('--video', '-v',
                            type=str, 
                            required=True, 
                            help='처리할 비디오 폴더(디렉토리)경로를 입력 ** mp4 파일경로 x **')
        parser.add_argument('--out', '-o',
                            type=bool, 
                            required=False, 
                            default=False,
                            help='True: 비디오출력 , 기본값 : False \n비디오 출력을 할지 선택.디버깅용 영상은 보간된 구간이 빨간색으로 변함. ')

        args = parser.parse_args()
    
        video_path  = args.video 
        is_out = args.out
        
    video_names = debug_check_video(video_path)
    
    for idx in tqdm(range(len(video_names)) , desc='진행률'):     
        try:
            debug_test(video_names[idx], is_out, idx + 1)    
        except ValueError as e:
            print(e.with_traceback)

        
    
    