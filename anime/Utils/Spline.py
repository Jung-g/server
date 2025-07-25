import pandas as pd

def spline(df):
    """
    DataFrame의 각 열에 대해 CubicSpline 보간을 적용하는 함수.
    """
    data = df.copy()
    
    left_df  = data.filter(like='left', axis=1)
    right_df = data.filter(like='right', axis=1)
    
    left_df = spline_cal(left_df)
    right_df = spline_cal(right_df)
    
    return pd.concat([left_df, right_df], axis=1)
    
def spline_cal(interpolated_df):
    original_df = pd.DataFrame(interpolated_df)
    interpolated_df = original_df.interpolate(method='akima') 
        
    return interpolated_df
    
