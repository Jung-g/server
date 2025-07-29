# -*- coding: utf-8 -*-
import re

# 초성(ㄱ,ㄴ,ㄷ...)으로 사용될 수 있는 자음 리스트
L_TABLE = list('ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ')
# 중성(ㅏ,ㅑ,ㅓ...)으로 사용될 수 있는 모음 리스트
V_TABLE = list('ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ')
# 종성(받침)으로 사용될 수 있는 자음 리스트. 받침이 없는 경우를 위해 맨 앞에 빈 문자('')를 추가
T_TABLE = [''] + ['ㄱ','ㄲ','ㄳ','ㄴ','ㄵ','ㄶ','ㄷ','ㄹ','ㄺ','ㄻ','ㄼ','ㄽ','ㄾ','ㄿ','ㅀ','ㅁ','ㅂ','ㅄ','ㅅ','ㅆ','ㅇ','ㅈ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ']
# 'ㄹㄱ', 'ㄴㅈ' 등 2개의 자음이 합쳐져 만들어지는 겹받침 목록
COMPOUND_TAILS = {
    ('ㄱ','ㅅ'):'ㄳ', ('ㄴ','ㅈ'):'ㄵ', ('ㄴ','ㅎ'):'ㄶ', ('ㄹ','ㄱ'):'ㄺ', ('ㄹ','ㅁ'):'ㄻ', ('ㄹ','ㅂ'):'ㄼ',
    ('ㄹ','ㅅ'):'ㄽ', ('ㄹ','ㅌ'):'ㄾ', ('ㄹ','ㅍ'):'ㄿ', ('ㄹ','ㅎ'):'ㅀ', ('ㅂ','ㅅ'):'ㅄ',
}

def contains_jamo(text: str) -> bool:
    """문자열 안에 한글 자음 또는 모음(자모)이 포함되어 있는지 확인합니다."""
    # [ㄱ-ㅎㅏ-ㅣ] : 한글 자음 또는 모음
    # re.search()는 문자열의 어느 위치에서든 패턴이 일치하는지 찾습니다.
    pattern = r'[ㄱ-ㅎㅏ-ㅣ]'
    return bool(re.search(pattern, text))

def is_jamo_or_numeric_only(text: str) -> bool:
    # ^: 문자열 시작, $: 문자열 끝
    # [ㄱ-ㅎㅏ-ㅣ\d\s]+: 자모(ㄱ-ㅎ, ㅏ-ㅣ), 숫자(\d), 공백(\s)이 하나 이상(+) 있는지 검사
    pattern = r'^[ㄱ-ㅎㅏ-ㅣ\d\s]+$'
    return bool(re.fullmatch(pattern, text))

# 자모 사이의 공백을 제거하는 전처리 함수
def preprocess_input(text):
    # 나중에 비교하기 위해, 함수에 들어온 초기 텍스트를 'previous_text'에 저장
    previous_text = text
    # 무한 루프 시작. 나중에 조건이 만족되면 'break'로 탈출
    while True:
        # 정규 표현식을 사용해 '(자모)(공백)(자모)' 패턴을 찾아 중간의 공백만 제거
        current_text = re.sub(r'([ㄱ-ㅎㅏ-ㅣ])\s+([ㄱ-ㅎㅏ-ㅣ])', r'\1\2', previous_text)
        
        # 공백 제거 후의 텍스트(current_text)와 이전 텍스트(previous_text)가 완전히 같다면,
        # 더 이상 제거할 자모 사이의 공백이 없다는 뜻이므로 'break'로 루프를 멈춤
        if current_text == previous_text:
            break
            
        # 만약 공백이 제거되어 텍스트에 변화가 있었다면,
        # 변경된 텍스트를 'previous_text'에 저장하고 루프를 다시 실행
        previous_text = current_text
        
    # 루프가 끝나면, 모든 자모 사이 공백이 제거된 최종 텍스트를 반환
    return current_text

# 자모 조립 함수
def assemble_jamos(text):
    # 완성된 글자들을 담을 빈 리스트 생성
    result = []
    # 문자열을 처음부터 끝까지 순회하기 위한 인덱스 변수
    i = 0
    n = len(text)
    
    # i가 문자열 길이보다 작은 동안 계속 반복
    while i < n:
        # 1. 초성 찾기
        # 현재 글자가 초성 리스트(L_TABLE)에 없으면 (예: 공백, 영어, 이미 완성된 한글)
        if text[i] not in L_TABLE:
            result.append(text[i]) # 그냥 결과에 추가하고
            i += 1                 # 다음 글자로 이동
            continue
        # 초성으로 쓸 자음을 'cho' 변수에 저장
        cho = text[i]
        i += 1

        # 2. 중성 찾기
        # 문자열 끝에 도달했거나 다음 글자가 중성 리스트(V_TABLE)에 없으면
        if i >= n or text[i] not in V_TABLE:
            result.append(cho) # 모음이 없으면 글자를 만들 수 없으므로, 그냥 초성만 결과에 추가
            continue
        # 중성으로 쓸 모음을 'jung' 변수에 저장
        jung = text[i]
        i += 1

        # 3. 종성(받침) 찾기
        # 우선 종성은 없다고 가정하고 빈 문자열로 시작
        jong = ''
        
        # (3-1) 겹받침 확인: 남은 문자열이 2자 이상이고, 다음 두 글자가 겹받침 목록에 있다면
        if i+1 < n and (text[i], text[i+1]) in COMPOUND_TAILS:
            # 연음 법칙 확인: 겹받침 바로 뒤에 또 모음이 오는지 확인 (예: '닭이' -> '달기')
            if (i+2 < n and text[i+1] in L_TABLE and text[i+2] in V_TABLE):
                # 연음 법칙이 적용되면 겹받침을 만들지 않고, 첫 번째 자음만 받침으로 사용
                jong = text[i]
                i += 1
            else: # 연음 법칙이 적용되지 않으면 겹받침을 생성
                jong = COMPOUND_TAILS[(text[i], text[i+1])]
                i += 2
        # (3-2) 홑받침 확인: 다음 글자가 받침 리스트(T_TABLE)에 있다면
        elif i < n and text[i] in T_TABLE[1:]:
            # 단, 그 바로 뒤에 모음이 오면 안 됨 (연음 법칙)
            if not (i+1 < n and text[i+1] in V_TABLE):
                jong = text[i]
                i += 1

        # 4. 글자 조립 (유니코드 공식 활용)
        # '가'의 유니코드(AC00)를 기준으로, 초/중/종성의 순서에 따라 값을 더해 완성형 글자의 코드를 계산
        syllable = chr(0xAC00 + L_TABLE.index(cho)*21*28 + V_TABLE.index(jung)*28 + T_TABLE.index(jong))
        # 완성된 글자를 결과 리스트에 추가
        result.append(syllable)
    
    # 리스트에 담긴 모든 글자들을 하나의 문자열로 합쳐서 반환
    return ''.join(result)