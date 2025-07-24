import os
import pandas as pd

# 현재 이 파일(Csv.py)의 경로 기준으로 DataSet 디렉토리 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # .../anime/Utils
DATASET_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'DataSet'))
WORDS_DIR = os.path.join(DATASET_DIR, 'Words')
WORDS_CSV_PATH = os.path.join(DATASET_DIR, 'words_data.csv')

def check_csv(video_path: str) -> bool:
    os.makedirs(WORDS_DIR, exist_ok=True)
    
    video_name = os.path.basename(video_path).split('.')[0]
    path = os.path.join(WORDS_DIR, f'{video_name}.csv')

    return os.path.exists(path)

def insert_csv(video_name, hand_df: pd.DataFrame, pose_df: pd.DataFrame):
    os.makedirs(WORDS_DIR, exist_ok=True)
    path = os.path.join(WORDS_DIR, f'{video_name}.csv')
    
    df = pd.concat([hand_df, pose_df], axis=1)    
    df.to_csv(path, index=False, encoding='utf-8-sig')

def search_data(wordname):
    """
    단어 -> 파일 이름 찾는 함수
    """
    check = pd.read_csv(WORDS_CSV_PATH)
    condition = (check['kor'] == wordname)
    results = check.loc[condition, 'id'].tolist()
        
    for name_id in results:
        csv_path = os.path.join(WORDS_DIR, f'{name_id}.csv')
        
        if not os.path.exists(csv_path):
            continue
        data = pd.read_csv(csv_path)
        pose_data = data.filter(like='pose', axis=1)
        
        left_hand_data = data.filter(like='left', axis=1)
        right_hand_data = data.filter(like='right', axis=1)
        hand_data = pd.concat([left_hand_data, right_hand_data], axis=1)
        
        return (hand_data, pose_data)
    return None

def search_id(wordname) -> str:
    check = pd.read_csv(WORDS_CSV_PATH)
    condition = (check['kor'] == wordname)
    results = check.loc[condition, 'id'].tolist()
        
    for name_id in results:
        return name_id
    return None
