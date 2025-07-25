import argparse
from main import run

run(r'C:\Users\User\Desktop\새 폴더', is_out= None , is_log='Y',is_csv='Y')

if __name__ == "__main__":
    video_path = ''
    
    parser = argparse.ArgumentParser(description="데이터셋 준비")

    parser.add_argument('--video', '-v',
                        metavar='VIDEO or DIRECTORY',
                        type=str, 
                        default= None,
                        required=True, 
                        help='처리할 비디오 파일의 경로 또는 폴더경로를 입력')
   
    args = parser.parse_args()
    video_path = args.video 
    
    if not video_path:
        raise ValueError('비디오 파일이 없습니다.')
    
    run(video_path, is_out= None , is_log='Y',is_csv='Y')
    