import os
import json
import numpy as np
import pandas as pd

class Detect_joint:         

    frames = None
    def __init__(self , df : pd.DataFrame, treshold : float = 8, is_debug = False, debug_data = None):       
        """MediaPipe에서 추출한 RawData를 알고리즘을 이용하여 일부 보간

        Args:
            df (pd.DataFrame): MediaPipe에서 출력한 데이터프레임
            treshold (float, optional): 임계값.해당 값이 크면 클 수록 이상치 값 감지가 둔해짐.권장 값 7 ~ 9
            is_debug (bool, optional): 디버깅용인지 여부. True라면 Log 작성
            debug_data (_type_, optional): 디버깅에 필요한 데이터
        """
        self.frames = df
        # 관절 이동거리에 비례해서 이상치 값 추출
        self.distance = Interpolation_Distance(df , treshold , is_debug , debug_data)
        
        # 뼈 길이에 비례해서 이상치 값 추출
        Interpolation_Bone(df , treshold ,is_debug , debug_data)
        
        self.delete_data()
        
    def delete_data(self):
        for idx in self.distance.left_last:
            delete_column = self.frames.loc[idx].filter(like='left',axis=0).index 
            self.frames.loc[idx, delete_column] = np.nan
                  
class Interpolation_Distance:
    """이동 거리에 비례해서 이상치 값을 추출합니다.
    """
    
    left_last = None
    right_last = None
    
    def __init__(self, frames :pd.DataFrame, treshold: float, is_debug = False , debug_data = None):
        """        
        Args:
            frames: 좌표 데이터.
            treshold: 이상치 판단을 위한 임계값 배수.
            bf: 이상치 주변을 체크할 프레임 범위.
        
        """
        
        가로 = 21
        세로 = len(frames)
        left_check = [[False for _ in range(가로)] for _ in range(세로)]
        right_check = [[False for _ in range(가로)] for _ in range(세로)]
        
        for bone_num in range(21):
            left_pos  = self.list_extend(frames, bone_num , 'left')
            right_pos = self.list_extend(frames, bone_num , 'right')

            left_check_idx , right_check_idx = self.main_cal(left_pos , right_pos , treshold)
            
            if len(left_check_idx):
                for num in left_check_idx:
                    left_check[num][bone_num] = True
            if len(right_check_idx):
                for num in right_check_idx:
                    right_check[num][bone_num] = True
                    
        left_result = [(num , sum(left_check[num])) for num in range(len(left_check)) if sum(left_check[num]) >= treshold + 5]
        right_result = [(num , sum(right_check[num])) for num in range(len(right_check)) if sum(right_check[num]) >= treshold + 5]

        self.left_last = self.chose_idx(left_result)
        self.right_last = self.chose_idx(right_result)
        
        if is_debug and debug_data :
            self.write_log(debug_data[0] , self.left_last , self.right_last , debug_data[1] , debug_data[2])
                
    def list_extend(self, frames : pd.DataFrame , idx : int , hand_type : str) -> list:        
        """
        MediaPipe에서 뽑은 좌표 정규화
        
        Args:
            idx: 뼈위치 인덱스값
            hand_type: 왼손 , 오른손 ( left , right)
            
        Returns:
            list : 특정관절의 프레임별 좌표 값 
            list[N] -> N프레임일 때의 특정 관절의 좌표 값
            [np.array[x , y, z]]            
        """
        
        position_x = []
        position_y = []
        
        position_x.extend(frames[f'{hand_type}_landmark_x_{idx}'].tolist())
        position_y.extend(frames[f'{hand_type}_landmark_y_{idx}'].tolist())
        
        positions = []                
        for num in range(len(position_x)):
            positions.append(np.array((position_x[num] , position_y[num])))

        return positions     
     
    def main_cal(self, left_pos: list , right_pos: list, treshold: float ) -> list:
        
        """
        주 거리 계산 알고리즘 , 평균거리 보다 임계값보다 큰 값의 인덱스 번호 저장
        
        Args:
            joints: 좌표 데이터 리스트.
            treshold: 이상치 판단을 위한 임계값 배수.
            bf: 이상치 주변을 체크할 프레임 범위.
        Returns:
            이상치로 판단된 프레임의 인덱스 리스트.
        """
        
        # 이동거리 값
        left_dis  = self.cal_distance(left_pos)
        right_dis = self.cal_distance(right_pos)
        
        # 평균 이동거리
        left_avg  = np.nanmean(left_dis)
        right_avg = np.nanmean(right_dis)
        
        # 평균 이동거리 * 임계값 보다 큰 이동거리 인덱스 번호 추출
        left_check_idx  = np.where(left_dis > left_avg * treshold)[0].tolist()
        right_check_idx = np.where(right_dis > right_avg * treshold)[0].tolist()
        
        return left_check_idx , right_check_idx
                    
    def cal_distance(self , joints : list) -> list:
        """ 
        이동거리 계산

        Args:
            joints (list): 좌표 정보 [( x , y, z )]

        Returns:
            list: 이동거리
        """
        distances = []
        
        for idx in range(len(joints) - 1):
            start = joints[idx]
            end   = joints[idx+1]
            
            if np.isnan(start).any() or np.isnan(end).any():
                distances.append(np.nan)
                continue
            
            이동거리 = np.linalg.norm(end - start)
            distances.append(이동거리)
            
        return np.array(distances)
   
    def chose_idx(self , last):
        
        def check():
            if len(group) > 1:
                    group.clear()
            elif len(group) != 0:
                result.append(group[0] + 1)
                    
        group = []
        result = []
            
        for i in range(0 , len(last) -1):
            if last[i+1][0] - last[i][0] == 1:
                group.append(last[i][0])
            else :
                check()
        if len(last) == 2:
            check()
        
        return result
        
    def write_log(self , idx , left_result ,right_result , left_lost , right_lost ):
        base_dir = 'debug//distance//'
        path  = os.path.join(base_dir , 'log.txt')
        
        os.makedirs(base_dir, exist_ok=True)
        
        if os.path.exists(path) and idx == 1:
            os.remove(path)
        
        if len(left_result) > 0 or len(right_result) > 0:
            with open(path, "a", encoding='utf-8') as file:
                
                file.write(f'{idx} : 프레임손실율(왼손/오른손) : {left_lost}/{right_lost}\n')
                if len(left_result) > 0:
                    left_data  = json.dumps(left_result)                    
                    file.write(f'왼손: {left_data}\n')
                if len(right_result) > 0:
                    right_data = json.dumps(right_result)
                    file.write(f'오른손: {right_data}\n')
                file.write(f'\n\n')

class Interpolation_Bone:
    """
    개발중
    """
    def __init__(self , frames :pd.DataFrame, treshold: float , is_debug = False , debug_data = None):
        
        """
        평균 뼈 길이를 계산 후 평균 뼈 길이보다 작을 시 이상치 값 판단.
        """
        
        self.bone_mapping = [ 
            (0,1) , (1,2) , (2,3) , (3,4),
            (0,5) , (5,6) , (6,7) , (7,8),
            (0,9) , (9,10) , (10,11) , (11,12),
            (0,13) , (13,14) , (14,15) , (15,16),
            (0,17) , (17,18) , (18,19) , (19,20)
        ]
            
        
        left_frames  = self.list_extend(frames, 'left')
        right_frames = self.list_extend(frames, 'right')
            
        left_bones  = self.cal_bones(left_frames)
        right_bones = self.cal_bones(right_frames)
        
        left_avges  = self.avg_bones(left_bones)
        right_avges = self.avg_bones(right_bones)

        가로 = 20
        세로 = len(frames)
        
        left_check = [[False for _ in range(가로)] for _ in range(세로)]
        right_check = [[False for _ in range(가로)] for _ in range(세로)]
        
        for bone_num in range(20):
            left_check_idx  = np.where(left_bones[:, bone_num] < left_avges[bone_num] - 0.015)[0].tolist()
            right_check_idx = np.where(right_bones[:, bone_num] < right_avges[bone_num] - 0.015)[0].tolist()
            
            if len(left_check_idx):
                for num in left_check_idx:
                    left_check[num][bone_num] = True
            if len(right_check_idx):
                for num in right_check_idx:
                    right_check[num][bone_num] = True
        
        left_result = [(num , sum(left_check[num])) for num in range(len(left_check)) if sum(left_check[num]) >= treshold]
        right_result = [(num , sum(right_check[num])) for num in range(len(right_check)) if sum(right_check[num]) >= treshold]
              
        if is_debug and debug_data :
            self.write_log(debug_data[0] , left_result , right_result , debug_data[1] , debug_data[2])
            
    def list_extend(self, frames : pd.DataFrame ,hand_type : str) -> list:        
        """
        MediaPipe에서 뽑은 좌표 정규화
        
        Args:
            idx: 뼈위치 인덱스값
            hand_type: 왼손 , 오른손 ( left , right)
            
        Returns:
            list : 특정관절의 프레임별 좌표 값 
            list[N] -> N프레임일 때의 모든 관절의 좌표 값
            [ [np.array[x , y, z]] . . . ]      
        """
        result = []
        joint_data = frames.filter(like=f'{hand_type}', axis=1)

        for idx in frames.index:
                
            joint = joint_data.loc[idx]
            
            pos = []
            for idx in range(21):
                pos_x = joint[f'{hand_type}_landmark_x_{idx}']
                pos_y = joint[f'{hand_type}_landmark_y_{idx}']
                # pos_z = joint[f'{hand_type}_landmark_z_{idx}']
                
                # joints.append(np.array((pos_x , pos_y , pos_z)))
                pos.append(np.array((pos_x , pos_y )))
            result.append(pos)                   
        
        return result     

    def cal_bones(self , frames : list) -> np.array:
        """ 
        뼈길이 계산

        Args:
            frames (list): 좌표 정보 [ [( x , y, z )] , , , ]

        Returns:
            np.array: N프레임의 모든 뼈 길이
        """
        
        bones = []
        
        for frame in frames:
            bone = []
            for start_idx , end_idx in self.bone_mapping:
                start = frame[start_idx]
                end = frame[end_idx]
                
                뼈길이 = np.linalg.norm(end - start)
                bone.append(뼈길이)
            
            bones.append(bone)
        return np.array(bones)
    
    def avg_bones(self , bones : np.array) -> list:
        """
        특정 뼈의 평균길이 값.

        Returns:
            list : N번째 인덱스의 평균 뼈 길이
        """
        average = []
        for idx in range(20):
            sel_bones = bones[:, idx]
            average.append(np.nanmean(sel_bones))
            
        return average
    
    def write_log(self , idx , left_result ,right_result , left_lost , right_lost ):
 
        base_dir = 'debug//bone//'
        path  = os.path.join(base_dir , 'log.txt')
        
        os.makedirs(base_dir, exist_ok=True)
        
        if os.path.exists(path) and idx == 1:
            os.remove(path)
        
        if len(left_result) > 0 or len(right_result) > 0:
            with open(path, "a", encoding='utf-8') as file:
                
                file.write(f'{idx}프레임손실율(왼손/오른손) : {left_lost}/{right_lost}\n')
                if len(left_result) > 0:
                    left_data  = json.dumps(left_result)                    
                    file.write(f'왼손: {left_data}\n')
                if len(right_result) > 0:
                    right_data = json.dumps(right_result)
                    file.write(f'오른손: {right_data}\n')
                file.write(f'\n\n')
    
