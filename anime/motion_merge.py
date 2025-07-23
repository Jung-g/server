import os
import sys
import argparse
import pandas as pd
from typing import Literal

from Utils.Spline import *
from Utils.Draw import create_keypoint_video, api_draw
from Utils.Csv import search_data, search_id 
   
def made_video(motion_data, output):
    hand_frame_df = []
    pose_frame_df = []
    
    for idx in range(len(motion_data)):
        if idx == 0:
            motion_type = 'start'            
        elif idx == len(motion_data)-1:
            motion_type = 'end'
        else:
            motion_type = 'middle'
        
        hand, pose = cutting_frame(motion_data[idx], 0.6, motion_type)
        
        hand_frame_df.append(hand)
        pose_frame_df.append(pose)
        
        if motion_type != 'end':
            blank_hand = pd.DataFrame(index=range(10), columns=hand.columns)
            blank_pose = pd.DataFrame(index=range(10), columns=pose.columns)
                        
            hand_frame_df.append(blank_hand)
            pose_frame_df.append(blank_pose)
        
    merged_hand = pd.concat(hand_frame_df, ignore_index=True)
    merged_pose = pd.concat(pose_frame_df, ignore_index=True)
        
    hand_df = spline(merged_hand)
    pose_df = spline_cal(merged_pose)
    dims = (1980,1020)
    
    create_keypoint_video(output, hand_df, pose_df, dims, frame_len=len(merged_hand), out_type = 'Sentence')

def cutting_frame(motion , standard = 0.8, motion_type : Literal['start', 'middle', 'end'] = None):
    """
    motion_type:
    
        start: 처음에 나오는 모션
        
        middle: 중간에 나오는 모션     
           
        end: 뒤에 나오는 모션    
    """        
    
    hand, pose = motion
    
    total_frame = len(hand)
    
    left_data = hand['left_landmark_y_0'].tolist()
    right_data = hand['right_landmark_y_0'].tolist()
    
    for idx in range(total_frame -1):
        if (left_data[idx] >= standard and left_data[idx+1] <standard) or (right_data[idx] >= standard and right_data[idx+1] <standard):
            start_idx = idx +1
        if (left_data[idx+1] >= standard and left_data[idx] <standard) or (right_data[idx+1] >= standard and right_data[idx] <standard):
            end_idx = idx
     
    if motion_type == 'start':
        piece_hand = hand.iloc[:end_idx - 1]
        piece_pose = pose.iloc[:end_idx - 1]

    elif motion_type == 'end':
        piece_hand = hand.iloc[start_idx:]
        piece_pose = pose.iloc[start_idx:]

    elif motion_type == 'middle':
        piece_hand = hand.iloc[start_idx:end_idx - 1]
        piece_pose = pose.iloc[start_idx:end_idx - 1]
    else:
        raise TypeError('모션타입 값이 불확실 합니다.')
        
    return piece_hand, piece_pose

def api_motion_merge(motion_data, word_type : Literal['Word','Sentence']):
    dims = (1980,1020)
    
    if word_type == 'Sentence':
        hand_frame_df = []
        pose_frame_df = []
        
        for idx in range(len(motion_data)):
            if idx == 0:
                motion_type = 'start'            
            elif idx == len(motion_data)-1:
                motion_type = 'end'
            else:
                motion_type = 'middle'
            
            hand_df, pose_df = cutting_frame(motion_data[idx], 0.6, motion_type)
            
            hand_frame_df.append(hand_df)
            pose_frame_df.append(pose_df)
            
            if motion_type != 'end':
                blank_hand = pd.DataFrame(index=range(10), columns=hand_df.columns)
                blank_pose = pd.DataFrame(index=range(10), columns=pose_df.columns)
                            
                hand_frame_df.append(blank_hand)
                pose_frame_df.append(blank_pose)
            
        merged_hand = pd.concat(hand_frame_df, ignore_index=True)
        merged_pose = pd.concat(pose_frame_df, ignore_index=True)
            
        hand_df = spline(merged_hand)
        pose_df = spline_cal(merged_pose)

        return api_draw(hand_df, pose_df, dims, frame_len=len(merged_hand))
    elif word_type == 'Word':
        hand_df, pose_df = motion_data
        return api_draw(hand_df , pose_df ,dims , frame_len=len(hand_df))

def check_merge(words, send_type : Literal['mp4','api'] = 'mp4'):
    out_name = ''
    motion_data = []
    fail_name = []
    for word in words:
        data = search_data(word)
        if data:
            motion_data.append(data)
            if out_name:
                out_name = out_name + '_' + word
            else:
                out_name = out_name + word
        else:
            fail_name.append(word)
    if len(motion_data)>=1:
        # 만들어진 단어 중에 해당 단어가 있으면 로드
        if len(motion_data) == 1:
            # API 요청일 때 (단어)
            if send_type == 'api':
                    return (motion_data[0], 'Word')
            elif send_type == 'mp4':
                name_path = f'output//Word//{search_id(out_name)}.mp4'
                if os.path.exists(name_path):                    
                    return name_path , fail_name                
            else:
                raise ValueError(f'데이터가 존재하지 않습니다.\n없는 단어 : {fail_name}')

        out_path = f'output//Sentence//{out_name}.mp4'
        # API 요청일 때 (문장)
        if send_type == 'api':
            return (motion_data, 'Sentence')
        elif send_type == 'mp4':
            if not os.path.exists(out_path):
                made_video(motion_data, out_name)
            
        return out_path , fail_name
    else:
        raise ValueError(f'데이터가 존재하지 않습니다.\n없는 단어 : {fail_name}')

if __name__ == "__main__":
    words = []
    
    #f5 로 디버깅 할 때
    if len(sys.argv) == 1:
        words = ['성토','없는단어','남매']
    #명령어 실행
    else:
        parser = argparse.ArgumentParser(description="키포인트 추출 -> 좌표 보간 -> 영상과 함께 출력 또는 KeyPoint만 따로 출력")

        parser.add_argument('--words', '-w',
                            type=str, 
                            default= None,
                            required=True, 
                            help='여러단어 출력 ex) (input)"A,B,C" (read)A B C  세 단어로 읽음.')

        args = parser.parse_args()
        words = args.words.split(',')
        if words == '' or words ==' ' or len(words) == 0 or not words:
            raise ValueError('단어가 없습니다.')

    try:
        video_path , fail_name = check_merge(words)
        print('영상이름: ',video_path)
        for fail in fail_name:
            print('없는단어: ',fail)
    except Exception as e:
        print(e.with_traceback)



