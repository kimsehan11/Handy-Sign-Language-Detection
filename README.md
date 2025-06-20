# Handy Sign Language Detection

이 프로젝트는 웹캠을 이용해 손 동작(수어/숫자 제스처)을 인식하고, 실시간으로 자막 및 음성 출력, 단어/문장 추천, 자동완성 기능을 제공하는 Python 기반 수어 인식 시스템입니다.

## 환경 설정 방법

1. **Python 환경**
   - Anaconda 가상환경에서 Python 3.11 버전으로 실행했습니다.

2. **가상환경(권장)**
   - 아래 중 하나의 방법으로 가상환경을 생성 및 활성화할 수 있습니다. 저는 Conda에서 했습니다.

   **(1) conda 환경 사용 시**
   ```powershell
   conda create -n handy python=3.11
   conda activate handy
   ```
   **(2) venv(PowerShell) 사용 시**
   ```powershell
   python -m venv handy
   .\handy\Scripts\Activate.ps1
   ```

3. **필수 라이브러리 설치**
   - 프로젝트 폴더에서 아래 명령어로 필요한 패키지를 설치하세요.
   ```powershell
   pip install -r requirements.txt

   **** 혹시 requirements.txt에 빠진 라이브러리가 있을 수 있습니다. 
   새로운 환경에서 다시 실행해봤을 때, 문제없이 되긴 했지만 혹시 라이브러리가 존재하지 않는다고 pip 으로 설치 해주시면 감사하겠습니다.
   ```


4. **모델 및 데이터 파일 준비**
   - 아래 파일들이 프로젝트 폴더에 있어야 합니다.
     - `model_digit_cnn.h5`: 숫자(0~9) 손동작 인식용 딥러닝 모델 파일
     - `model_sentences_cnn.h5`: 알파벳(a~z) 손동작 인식용 딥러닝 모델 파일
     - `unigram_freq.csv`: 단어 추천을 위한 단어 빈도 데이터 파일
     - `sentences_db.txt`: 문장 자동완성/추천을 위한 문장 데이터베이스 파일(최대 1000개, 음성 출력 시 자동 추가)
     - `data_digit.pickle`: 숫자 손동작 학습/평가용 랜드마크 데이터
     - `data_sentences.pickle`: 알파벳 손동작 학습/평가용 랜드마크 데이터

5. **실행**
   - 가상환경이 활성화된 상태에서 아래 명령어로 실행합니다.
   ```powershell
   python classifier_w_audio_output.py
   ```

## ---------------------------------------------------------------------------

## 따로 실행할 필요 없는 파일들/설명용

1. **데이터 수집 (생략/ 데이터 이미 모아놓음)**
   - `img collect.py`를 실행하여 손 동작 이미지 데이터를 수집합니다. (이건 안 하셔도 됩니다. 이미 data 모아놨어요)
   - 각 숫자/알파벳 폴더에 이미지를 저장합니다.

2. **음성 모델 성능&자연스러움 비교** 
   -`experiment_tts_compare.py`를 실행하여 `edge_tts_output.mp3`,`gtts_output.mp3`,`pyttsx3_output.mp3`를 만들고 각 tts model의 성능을 테스트함

3. **랜드마크 추출**
   - `landmarks.py`를 실행하여 수집한 이미지에서 손 랜드마크 좌표를 추출하고, pickle 파일(`data_digit.pickle`, `data_sentences.pickle`)로 저장합니다. (이미 추출된 파일이 있으므로 별도 실행 불필요)

4. **모델 학습**
   - `train.py`를 실행하여 손 동작(숫자/알파벳) 인식용 CNN 모델(`model_digit_cnn.h5`, `model_sentences_cnn.h5`)을 학습합니다. (이미 학습된 모델 파일이 있으므로 별도 실행 불필요)

5. **추천/자동완성 기능 구현 코드**
   - `recommend_utils.py`, `recommend_llm_utils.py`는 추천 단어/문장 및 LLM 기반 자동완성 기능을 위한 코드로, 직접 실행할 필요 없습니다. (메인 프로그램에서 import되어 사용됨)


## --------------------------------------------------------------------

## 전체 실행 절차

1. **데이터 및 환경 준비**
   - Python 3.11 환경 및 가상환경(Conda) 생성
   - `pip install -r requirements.txt`로 필수 라이브러리 설치
   - `data_digit.pickle`, `data_sentences.pickle`, `unigram_freq.csv`, `sentences_db.txt` 등 데이터 파일 준비(미리 다 준비 해놓았습니다.폴더에 존재하는지만 확인해 주세요)

5. **메인 프로그램 실행**
   - 가상환경 활성화 후 `python classifier_w_audio_output.py` 실행
   - 웹캠을 통해 실시간 손 동작 인식, 자막/음성 출력, 추천/자동완성 기능 사용

## 주요 기능
- Mediapipe와 딥러닝 모델(CNN) 기반 손 동작(알파벳/숫자) 인식
- 실시간 자막 표시 및 음성 출력 (pyttsx3)
- 자주 쓰는 단어 추천(빈도 기반, unigram_freq.csv)
- 문장 자동완성 추천 (sentences_db.txt, SBERT, LLM)
- 추천 단어/문장 터치 및 제스처(1~6)로 선택 가능
- 문장 음성 출력 시 DB에 자동 저장(최대 1000개, 중복 없음)

## 사용법
- 손 제스처(오른손: 숫자, 왼손: 알파벳)로 입력

- 추천 단어/문장: 화면 상단/하단에 표시, 손가락 터치 또는 숫자 제스처 오른손(1,2,3,4,5,6)로 선택

    - 1/2/3: 추천 단어 선택 (**수화 3은 엄지와 검지 중지를 피고 약지 새끼를 접은 것입니다.)
    - 4: 문장 자동완성 tail PUSH
    - 5: 음성 출력(문장 DB 저장)
    - 6: 프로그램 종료 (**수화 6은 흔히 우리가 3을 손으로 표현할 때와 동일하게 검지,중지,약지를 피고 엄지 새끼를 접은 것입니다.)

- 자막은 띄어쓰기를 `_`로 표시, 실제 DB/음성에는 공백

