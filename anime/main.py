import os
import sys
import argparse
from tqdm import tqdm

from Utils.Draw import *
from Utils.Spline import *
from Utils.Csv import insert_csv, check_csv
from Utils.Interpolation import Detect_joint

def write_log(log_data, idx, is_csv):
    sus, loss_info_tuple = log_data
    if not loss_info_tuple:
        total_frames = 'Error'
        left_lost = 'Error'
        right_lost = 'Error'
    else:
        total_frames, left_lost, right_lost = loss_info_tuple
    if is_csv == 'Y' or is_csv =='y':
        base_dir = 'DataSet/Words'
    else:
        base_dir = 'output'
    path  = os.path.join(base_dir , 'out_log.txt')    
    os.makedirs(base_dir, exist_ok=True)
    
    if os.path.exists(path) and idx == 1:
        os.remove(path)
        
    with open(path, "a", encoding='utf-8') as file:
        
        file.write(f'{idx}번째 영상\n')
        file.write(f'프레임손실율(전체:왼손:오른손) : {total_frames} : {left_lost} : {right_lost}\n')
        file.write(f'영상 생성 : {sus}')
        file.write(f'\n\n')
                        
def check_video(video_path):
    """입력값이 디렉토리인지 MP4파일인지 체킹

    Raises:
        FileNotFoundError: 디렉토리를 입력했을 떄 mp4파일이 안보이면 에러발생

    Returns:
        List: 디렉토리에서 mp4이름을 출력
        
        **mp4 파일이라면 None값 리턴**
    """
    _, file_extension = os.path.splitext(video_path)
    if not file_extension == '':
        return None
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

def main(video_path :str, is_out: bool, is_csv: bool):
    video_name = video_path.split('\\')[-1].split('.')[0]
    hands_original_data, pose_original_data, dims = linear_joint(video_path)

    if hands_original_data is not None:

        total_frames = len(hands_original_data)
        left_hand_lost = hands_original_data['left_landmark_x_0'].isna().sum()
        right_hand_lost = hands_original_data['right_landmark_x_0'].isna().sum()
        loss_info_tuple = (total_frames, left_hand_lost, right_hand_lost)
        
        # 프레임 손실률이 25퍼 이상이면 제작 X
        if left_hand_lost > 25 or right_hand_lost > 25:
            return (False, loss_info_tuple)     
        
        check_data = Detect_joint(hands_original_data)
        hand_df = spline(check_data.frames)
        pose_df = spline_cal(pose_original_data)
        
        match is_out:
            case 0:
                realtime_visualize(video_path, hand_df, pose_df, dims, loss_info_tuple)
            case 1:
                create_keypoint_video(video_name, hand_df, pose_df, dims, frame_len=total_frames)
        
        if is_csv == 'Y' or is_csv == 'y':
            insert_csv(video_name, hand_df , pose_df)
            
        return (True , loss_info_tuple)
    
    return (False, None)

def run(video_path, is_out , is_log , is_csv):
    video_names = check_video(video_path)     
    
    if video_names:
        for idx in tqdm(range(len(video_names)) , desc= '진행률'): 
            if check_csv(video_names[idx]) and (is_csv == 'Y' or is_csv == 'y'):
                continue
            log_data = main(video_names[idx], is_out, is_csv)
            if is_log == 'Y' or is_log == 'y':
                write_log(log_data, idx + 1, is_csv)
    else:
        log_data = main(video_path, is_out, is_csv)
        if is_log == 'Y':
            write_log(log_data, 1, is_csv)

if __name__ == "__main__":
        
    video_path = None
    is_out = None
    is_log = None
    is_csv = None
    
    #f5 로 디버깅 할 때
    if len(sys.argv) == 1:
        # --- 디버그용 ---
        video_path = r'c:\Users\User\Desktop\원천데이터\REAL\WORD\01\NIA_SL_WORD1502_REAL01_F.mp4' 
        is_out = 1
        # -----------------------------
    #명령어 실행
    else:
        parser = argparse.ArgumentParser(description="키포인트 추출 -> 좌표 보간 -> 영상과 함께 출력 또는 KeyPoint만 따로 출력")

        parser.add_argument('--video', '-v',
                            metavar='VIDEO or DIRECTORY',
                            type=str, 
                            default= None,
                            required=True, 
                            help='처리할 비디오 파일의 경로 또는 폴더경로를 입력')
        parser.add_argument('--out','-o',
                            type=int,
                            required=False,
                            default=None,
                            help='0 : 실시간으로 화면 출력 , 1: Mp4 파일 생성 , 기본값 : 출력 X')
        parser.add_argument('--log','-l',
                            type=str,
                            metavar='Y/N',
                            required=False,
                            default=True,
                            help='로그 폴더를 작성하지 여부. 기본값: Y')
        parser.add_argument('--csv','-c',
                            type=str,
                            metavar='Y/N',
                            required=False,
                            default=False,
                            help='csv 데이터셋 설정할지 여부. 기본값: N')

        args = parser.parse_args()
        video_path = args.video 
        is_out  = args.out
        is_log = args.log
        is_csv = args.csv
        
        if not video_path:
            raise ValueError('비디오 파일이 없습니다.')
        
    run(video_path, is_out, is_log, is_csv)
        
