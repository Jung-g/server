import re

# 초성 19개
L_TABLE = list('ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ')
# 중성 21개
V_TABLE = list('ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ')
# 받침 목록 + 공백(받침이 없는 경우)
T_TABLE = [''] + ['ㄱ','ㄲ','ㄳ','ㄴ','ㄵ','ㄶ','ㄷ','ㄹ','ㄺ','ㄻ','ㄼ','ㄽ','ㄾ','ㄿ','ㅀ','ㅁ','ㅂ','ㅄ','ㅅ','ㅆ','ㅇ','ㅈ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ']
# 겹받침 목록
COMPOUND_TAILS = {
    ('ㄱ','ㅅ'):'ㄳ', ('ㄴ','ㅈ'):'ㄵ', ('ㄴ','ㅎ'):'ㄶ', ('ㄹ','ㄱ'):'ㄺ', ('ㄹ','ㅁ'):'ㄻ', ('ㄹ','ㅂ'):'ㄼ',
    ('ㄹ','ㅅ'):'ㄽ', ('ㄹ','ㅌ'):'ㄾ', ('ㄹ','ㅍ'):'ㄿ', ('ㄹ','ㅎ'):'ㅀ', ('ㅂ','ㅅ'):'ㅄ',
}

def contains_jamo(text: str) -> bool:
    """문자열 안에 한글 자음 또는 모음(자모)이 포함되어 있는지 확인합니다."""
    pattern = r'[ㄱ-ㅎㅏ-ㅣ]'
    return bool(re.search(pattern, text))

def is_jamo_or_numeric_only(text: str) -> bool:
    """ [ㄱ-ㅎㅏ-ㅣ\d\s]+: 자모(ㄱ-ㅎ, ㅏ-ㅣ), 숫자(\d), 공백(\s)이 하나 이상(+) 있는지 검사합니다 """
    pattern = r'^[ㄱ-ㅎㅏ-ㅣ\d\s]+$'
    return bool(re.fullmatch(pattern, text))

def preprocess_input(text: str):
    """ 자모 사이의 공백을 제거합니다"""
    previous_text = text
    while True:
        current_text = re.sub(r'([ㄱ-ㅎㅏ-ㅣ])\s+([ㄱ-ㅎㅏ-ㅣ])', r'\1\2', previous_text)
        
        if current_text == previous_text:
            break

        previous_text = current_text
        
    return current_text

def assemble_jamos(text):
    """ 주어진 자모 문자열을 조합하여 완성형 한글 문자열로 변환합니다."""
    result = []
    i = 0
    n = len(text)
    
    while i < n:
        #1. 초성 찾기
        # 현재 글자가 초성 리스트(L_TABLE)에 없으면 추가 (예: 공백, 영어, 이미 완성된 한글)
        if text[i] not in L_TABLE:
            result.append(text[i]) 
            i += 1                 
            continue
        cho = text[i]
        i += 1

        # 2. 종성 찾기
        if i >= n or text[i] not in V_TABLE:
            result.append(cho)
            continue
       
        jung = text[i]
        i += 1

        # 3. 종성(받침) 찾기
        jong = ''
        
        # (3-1) 겹받침 확인
        if i+1 < n and (text[i], text[i+1]) in COMPOUND_TAILS:
            if (i+2 < n and text[i+1] in L_TABLE and text[i+2] in V_TABLE):
                jong = text[i]
                i += 1
            else: 
                jong = COMPOUND_TAILS[(text[i], text[i+1])]
                i += 2
        # (3-2) 홑받침 확인
        elif i < n and text[i] in T_TABLE[1:]:
            if not (i+1 < n and text[i+1] in V_TABLE):
                jong = text[i]
                i += 1

        # 4. 글자 조립 (유니코드 공식 활용)
        syllable = chr(0xAC00 + L_TABLE.index(cho)*21*28 + V_TABLE.index(jung)*28 + T_TABLE.index(jong))
        result.append(syllable)
    
    return ''.join(result)