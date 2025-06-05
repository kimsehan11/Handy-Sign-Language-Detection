# DEPENDENCIES ------>
import pickle
import cv2
import mediapipe as mp
import numpy as np
import time
import pyttsx3
from tensorflow.keras.models import load_model
import string
from sentence_transformers import SentenceTransformer, util
from recommend_utils import load_common_sentences, get_sentence_recommendation
from recommend_utils import load_unigram_words, get_unigram_word_recommendations
from recommend_llm_utils import load_llm_pipeline, get_llm_sentence_recommendation


class TextToAudio:
    engine: pyttsx3.Engine

    def __init__(self, voice, rate: int, volume: float):
        self.engine = pyttsx3.init()
        if voice:
            self.engine.setProperty("voice", voice)
        self.engine.setProperty("rate", rate)
        self.engine.setProperty("volume", volume)

    def text_to_audio(self, text: str):
        self.engine.say(text)
        print(f"Signer: {predicted_sign}")
        self.engine.runAndWait()


# digit/sentence 모델 각각 불러오기
model_digit = load_model("model_digit_cnn.h5")
model_sentence = load_model("model_sentences_cnn.h5")

# digit: 0~9, sentence: 10~35 (sentence 모델은 0~25로 예측, 실제 라벨은 10~35)
labels_digit = {i: str(i) for i in range(10)}  # 0~9
labels_sentence = {i: chr(ord('a') + i) for i in range(26)}  # 0~25 → 'a'~'z'
capture = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.9, min_tracking_confidence=0.9)
#문장 단위로 실행하기 위해서
sentences = []
words = []  # 현재 입력 중인 단어(알파벳 시퀀스) 리스트 추가
prev_time = time.time()
last_sign = None
sign_hold_start = None
prev_landmarks = None  # 이전 landmark 좌표 저장용
landmark_move_threshold = 0.005  # 손 움직임 허용 임계값(조절 가능)

# 단어/문장 DB 파일에서 불러오기
common_sentences = load_common_sentences('sentences_db.txt')
unigram_words = load_unigram_words('unigram_freq.csv')

# 문장 임베딩 모델 로드 (최초 실행 시 다운로드)
sbert_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
llm_pipe = load_llm_pipeline()

prev_prefix = ""
cached_llm_tail = ""

# 문장 추천 tail 결정 함수 (SBERT 우선, 없으면 LLM)
def get_best_sentence_tail(prefix, common_sentences, sbert_model, llm_pipe):
    sbert_tail = get_sentence_recommendation(prefix, common_sentences, sbert_model)
    if sbert_tail:
        return sbert_tail
    return get_llm_sentence_recommendation(prefix, llm_pipe)

while True:
    data_aux = []
    x_ = []
    y_ = []
    ret, frame = capture.read()
    # --- 웹캠 좌우 반전 ---
    frame = cv2.flip(frame, 1)
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # 손 방향 정보 가져오기
            hand_label = None
            if results.multi_handedness:
                hand_label = results.multi_handedness[idx].classification[0].label  # 'Left' 또는 'Right'
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x)
                data_aux.append(y)
                x_.append(x)
                y_.append(y)
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            input_data = np.array(data_aux).reshape(1, 21, 2)
            # 오른손이면 digit(0~9), 왼손이면 sentence(0~9)
            if hand_label == 'Right':
                pred_digit = model_digit.predict(input_data, verbose=0)
                pred_idx = int(np.argmax(pred_digit))
                predicted_sign = labels_digit.get(pred_idx, 'Unknown')
            elif hand_label == 'Left':
                pred_sentence = model_sentence.predict(input_data, verbose=0)
                pred_idx = int(np.argmax(pred_sentence))
                # sentence 모델은 0~25로 예측, 실제 라벨은 10~35이므로 변환
                label_idx = pred_idx + 10
                predicted_sign = labels_sentence.get(pred_idx, 'Unknown')
            else:
                predicted_sign = 'Unknown'

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
            cv2.putText(frame, predicted_sign, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
            data_aux = []
            x_ = []
            y_ = []

        # landmark 변화량 계산
        landmark_diff = None
        if prev_landmarks is not None and len(prev_landmarks) == 42 and len(input_data.flatten()) == 42:
            landmark_diff = np.linalg.norm(np.array(prev_landmarks) - input_data.flatten()) / 42
        prev_landmarks = input_data.flatten().tolist()

        # 오른손 digit이 '0', '9', '4', '5', '6', '1', '2', '3' 처리 (0.5초 이상 유지 시에만 동작)
        if hand_label == 'Right' and predicted_sign in ['0', '9', '4', '5', '6', '1', '2', '3']:
            if last_sign != predicted_sign:
                last_sign = predicted_sign
                sign_hold_start = time.time()
            elif sign_hold_start is not None and time.time() - sign_hold_start >= 0.5:
                if predicted_sign == '0':
                    if sentences:
                        sentences.pop()
                    if words:
                        words.pop()  # 지우개 동작 시 words 마지막 글자만 삭제
                    sign_hold_start = None
                elif predicted_sign == '9':
                    if not sentences or sentences[-1] != ' ':
                        sentences.append(' ')
                    words = []  # 띄어쓰기 동작 시 words도 초기화
                    sign_hold_start = None
                elif predicted_sign == '4':
                    # PUSH(문장 자동완성 tail 추가, SBERT 우선, 없으면 LLM 추천 사용)
                    prefix = "".join(sentences).strip()
                    if prefix != prev_prefix:
                        rec_sentence_tail = get_best_sentence_tail(prefix, common_sentences, sbert_model, llm_pipe)
                        cached_llm_tail = rec_sentence_tail
                        prev_prefix = prefix
                    else:
                        rec_sentence_tail = cached_llm_tail
                    if rec_sentence_tail:
                        for ch in rec_sentence_tail:
                            if ch == ' ':
                                sentences.append('_')
                            else:
                                sentences.append(ch)
                    sign_hold_start = None
                elif predicted_sign == '5':
                    # 엔터(문장 음성 출력)
                    if sentences:
                        sentence_str = "".join(sentences)
                        tta = TextToAudio("com.apple.speech.synthesis.voice.Alex", 200, 1.0)
                        tta.text_to_audio(sentence_str)
                        # --- 문장 DB에 저장 (중복 없이, 최대 1000개 유지) ---
                        try:
                            db_path = 'sentences_db.txt'
                            # 파일 읽기
                            with open(db_path, 'r', encoding='utf-8') as f:
                                lines = [line.strip() for line in f if line.strip()]
                            # 중복 제거(기존 문장 우선, 새 문장은 맨 뒤에 추가)
                            if sentence_str not in lines:
                                lines.append(sentence_str)
                            # 1000개 초과 시 오래된 것부터 삭제
                            if len(lines) > 1000:
                                lines = lines[-1000:]
                            # 파일 저장
                            with open(db_path, 'w', encoding='utf-8') as f:
                                f.write("\n".join(lines) + "\n")
                        except Exception as e:
                            print(f"[문장 DB 저장 오류] {e}")
                        sentences.clear()
                        words = []
                        last_sign = None
                        sign_hold_start = None
                    else:
                        sign_hold_start = None
                elif predicted_sign == '6':
                    # 종료(웹캠 나가고 프로그램 종료)
                    capture.release()
                    cv2.destroyAllWindows()
                    exit(0)
                elif predicted_sign in ['1', '2', '3']:
                    # 단어 추천 선택 기능 (1,2,3)
                    prefix = "".join(words)
                    rec_words = get_unigram_word_recommendations(prefix, unigram_words)
                    idx = int(predicted_sign) - 1
                    if 0 <= idx < len(rec_words):
                        # 기존 words 제거
                        for _ in range(len(words)):
                            if sentences:
                                sentences.pop()
                        for ch in rec_words[idx]:
                            sentences.append(ch)
                        words.clear()
                        # 단어 뒤에 공백 추가 (자막에서는 '_'로 보임)
                        sentences.append(' ')
                        sign_hold_start = None
       
        else:
            # 같은 sign이 2초 이상 연속 인식되고, landmark 변화량이 임계값 이하일 때만 추가 (중복 허용, Unknown 제외)
            if predicted_sign == last_sign:
                if sign_hold_start is None:
                    sign_hold_start = time.time()
                elif time.time() - sign_hold_start >= 2.0:
                    if predicted_sign != 'Unknown' and (landmark_diff is not None and landmark_diff < landmark_move_threshold):
                        sentences.append(predicted_sign)
                        # 알파벳이면 words에 추가, 아니면 words 초기화
                        if predicted_sign.isalpha():
                            words.append(predicted_sign)
                        else:
                            words = []
                        sign_hold_start = time.time()  # 연속 추가를 위해 시간 초기화
            else:
                last_sign = predicted_sign
                sign_hold_start = time.time()
    else:
        last_sign = None
        sign_hold_start = None
        prev_landmarks = None
    # 화면 밖으로 손이 나가도 자동 출력하지 않음 (엔터로만 출력)

    # 자막(문장 리스트) 실시간 표시 (여러 줄로 나누기)
    if sentences:
        # 화면에는 띄어쓰기를 '_'로 표시, sentences에는 실제로는 ' '
        subtitle = "".join(sentences).replace(' ', '_')
        max_chars_per_line = 21
        lines = [subtitle[i:i+max_chars_per_line] for i in range(0, len(subtitle), max_chars_per_line)]
        for idx, line in enumerate(lines):
            y_pos = H - 30 - (len(lines)-1-idx)*40  # 여러 줄이 아래에서 위로 쌓이게
            cv2.putText(frame, line, (30, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)
        
        # --- 문장 추천 기능 ---
        # sentences를 공백 기준으로 합쳐서 prefix로 사용
        prefix = "".join(sentences).strip()
        if "".join(sentences).count(' ') >= 3:
            if prefix != prev_prefix:
                rec_sentence_tail = get_sentence_recommendation(prefix, common_sentences, sbert_model)
                cached_llm_tail = rec_sentence_tail
                prev_prefix = prefix
            else:
                rec_sentence_tail = cached_llm_tail
        else:
            rec_sentence_tail = ""
        if rec_sentence_tail:
            # 추천 문장 뒷부분을 회색으로 표시
            rec_text = rec_sentence_tail.replace(' ', '_')
            # 마지막 자막 줄의 끝 위치 계산
            last_line = lines[-1]
            last_line_size = cv2.getTextSize(last_line, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
            rec_x = 30 + last_line_size[0] + 10
            rec_y = H - 30  # 마지막 줄 y_pos
            cv2.putText(frame, rec_text, (rec_x, rec_y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (180, 180, 180), 3, cv2.LINE_AA)
            # PUSH 버튼 표시
            push_text = " PUSH "
            push_size = cv2.getTextSize(push_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
            push_x = rec_x + cv2.getTextSize(rec_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0][0] + 20
            push_y = rec_y
            # 버튼 사각형
            btn_left = push_x - 10
            btn_top = push_y - push_size[1] - 10
            btn_right = push_x + push_size[0] + 10
            btn_bottom = push_y + 15
            cv2.rectangle(frame, (btn_left, btn_top), (btn_right, btn_bottom), (180,180,180), -1)
            cv2.putText(frame, push_text, (push_x, push_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80,80,80), 2, cv2.LINE_AA)
            # 손가락 tip(검지:8, 중지:12)로 PUSH 버튼 터치 감지
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for tip_idx in [8, 12]:
                        tip = hand_landmarks.landmark[tip_idx]
                        tip_x = int(tip.x * W)
                        tip_y = int(tip.y * H)
                        if btn_left <= tip_x <= btn_right and btn_top <= tip_y <= btn_bottom:
                            # 추천 문장 뒷부분을 sentences에 추가
                            for ch in rec_sentence_tail:
                                if ch == ' ':
                                    sentences.append('_')
                                else:
                                    sentences.append(ch)
                            break

    # --- 자주 쓰이는 단어 추천 기능 (빠르고 렉 없음, words 기준) ---
    if words:
        prefix = "".join(words)
        rec_words = get_unigram_word_recommendations(prefix, unigram_words)
        # 화면 크기 기준 위치 계산
        positions = []
        if len(rec_words) == 1:
            # 중앙
            text = f"1. {rec_words[0]}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
            x = (W - text_size[0]) // 2
            positions = [(x, 40)]
        elif len(rec_words) == 2:
            # 왼쪽, 오른쪽
            text0 = f"1. {rec_words[0]}"
            text1 = f"2. {rec_words[1]}"
            text0_size = cv2.getTextSize(text0, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
            text1_size = cv2.getTextSize(text1, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
            positions = [
                (30, 40),
                (W - text1_size[0] - 30, 40)
            ]
        elif len(rec_words) == 3:
            # 왼쪽, 중앙, 오른쪽
            text0 = f"1. {rec_words[0]}"
            text1 = f"2. {rec_words[1]}"
            text2 = f"3. {rec_words[2]}"
            text0_size = cv2.getTextSize(text0, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
            text1_size = cv2.getTextSize(text1, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
            text2_size = cv2.getTextSize(text2, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
            positions = [
                (30, 40),
                ((W - text1_size[0]) // 2, 40),
                (W - text2_size[0] - 30, 40)
            ]
        # --- 추천 단어 터치 선택 기능 ---
        # 추천 단어 rect 정보 저장
        rec_word_rects = []
        font_scale = 1.3  # 글자 크기 약간 줄임
        font_thickness = 3
        padding = 35  # 히트 박스(터치 판정 범위) 유지
        y_offset = 40  # 추천 리스트 위쪽 패딩 유지
        for idx, word in enumerate(rec_words):
            text = f"{idx+1}. {word}"
            # y 위치에 y_offset 추가
            base_pos = positions[idx] if idx < len(positions) else (30, 40 + idx*50)
            x, y = base_pos[0], base_pos[1] + y_offset
            # --- 동적 폰트 크기 조정 ---
            max_width = int(W * 0.32)  # 추천 박스 최대 너비(프레임의 1/3)
            font_scale = 1.3
            font_thickness = 3
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
            while text_size[0] > max_width and font_scale > 0.5:
                font_scale -= 0.1
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
            rect = (x - padding, y - text_size[1] - padding, x + text_size[0] + padding, y + 15 + padding)
            rec_word_rects.append(rect)
            cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), font_thickness, cv2.LINE_AA)

        # 손가락 tip(검지:8, 중지:12)로 추천 단어 터치 감지
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for tip_idx in [8, 12]:  # 8: 검지, 12: 중지
                    tip = hand_landmarks.landmark[tip_idx]
                    tip_x = int(tip.x * W)
                    tip_y = int(tip.y * H)
                    for idx, rect in enumerate(rec_word_rects):
                        left, top, right, bottom = rect
                        if left <= tip_x <= right and top <= tip_y <= bottom:
                            # 해당 추천 단어 선택
                            # 중복 선택 방지: words가 비어있지 않을 때만 동작
                            if words:
                                # 기존 words 제거
                                for _ in range(len(words)):
                                    if sentences:
                                        sentences.pop()
                                for ch in rec_words[idx]:
                                    sentences.append(ch)
                                words.clear()
                                # 단어 뒤에 공백 추가 (자막에는 '_'로 보임)
                                sentences.append(' ')
                            break
    cv2.imshow("Handy", frame)
    key = cv2.waitKey(1)

    # 엔터(Enter) 키(13) 누르면 현재 자막을 음성 출력하고, 문장 DB에 저장
    if key == 13 and sentences:
        sentence_str = "".join(sentences)  # 자막 리스트를 문자열로 변환
        tta = TextToAudio("com.apple.speech.synthesis.voice.Alex", 200, 1.0)
        tta.text_to_audio(sentence_str)

        # --- 문장 DB에 저장 (중복 없이, 최대 1000개 유지) ---
        try:
            db_path = 'sentences_db.txt'
            # 파일에서 기존 문장 읽기
            with open(db_path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
            # 중복 방지: 새 문장이 없으면 추가
            if sentence_str not in lines:
                lines.append(sentence_str)
            # 1000개 초과 시 오래된 것부터 삭제
            if len(lines) > 1000:
                lines = lines[-1000:]
            # 파일에 저장
            with open(db_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(lines) + "\n")
        except Exception as e:
            print(f"[문장 DB 저장 오류] {e}")

        # 자막, 단어, 상태 초기화
        sentences.clear()
        words = []
        last_sign = None
        sign_hold_start = None

capture.release()
cv2.destroyAllWindows()
