import os
import pandas as pd

def check_csv(video_path:str)->bool:
    base = 'DataSet/Words'
    os.makedirs(base, exist_ok=True)
    
    video_name = os.path.split(video_path)[-1]    
    video_name = video_name.split('.')[0]
    path = os.path.join(base , f'{video_name}.csv')
    if os.path.exists(path):
        return True
    return False
    
def insert_csv(video_name, hand_df: pd.DataFrame,pose_df: pd.DataFrame):
    base = 'DataSet/Words'
    os.makedirs(base, exist_ok=True)
    path = os.path.join(base , f'{video_name}.csv')
    
    df = pd.concat([hand_df, pose_df], axis=1)    
    df.to_csv(path, index=False, encoding='utf-8-sig')

def search_data(wordname):
    """
    단어 -> 파일 이름 찾는 함수
    """
    
    check = pd.read_csv('DataSet//words_data.csv')
    condition = (check['kor'] == wordname)
    results = check.loc[condition, 'id'].tolist()
        
    for name_id in results:
        base = 'DataSet//Words'
        csv_path = os.path.join(base , f'{name_id}.csv')
        
        if not os.path.exists(csv_path):
            continue
        data = pd.read_csv(f'{csv_path}')
        pose_data = data.filter(like='pose', axis=1)
        
        left_hand_data = data.filter(like='left', axis=1)
        right_hand_data = data.filter(like='right', axis=1)
        hand_data = pd.concat([left_hand_data , right_hand_data], axis=1)
        
        return (hand_data , pose_data)
    return None

def search_id(wordname) -> str:
    check = pd.read_csv('DataSet//words_data.csv')
    condition = (check['kor'] == wordname)
    results = check.loc[condition, 'id'].tolist()
        
    for name_id in results:
        return name_id
    
    return None
