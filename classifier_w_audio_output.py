# DEPENDENCIES ------>
import pickle
import cv2
import mediapipe as mp
import numpy as np
import time
import pyttsx3


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


model_dict = pickle.load(open("./model.p", "rb"))
model = model_dict["model"]
capture = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.9, min_tracking_confidence=0.9)
labels_dict = {0: "hello", 1: "i love you", 2: "yes", 3: "good", 4: "bad", 5: "okay", 6: "you", 7: "i", 8: "why", 9: "no", 10 : "0", 11: "1", 12: "2", 13: "3", 14: "4", 15: "5", 16: "6", 17: "7", 18: "8", 19: "9"}
#문장 단위로 실행하기 위해서
sentences = []
prev_time = time.time()
last_sign = None
sign_hold_start = None

while True:
    data_aux = []
    x_ = []
    y_ = []
    ret, frame = capture.read()
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

            prediction = model.predict([np.asarray(data_aux)])
            print("prediction:", prediction)
            pred_idx = int(prediction[0])
            print("pred_idx[0]:", prediction[0])

            # 오른손이면 10~19, 왼손이면 0~9만 분류 (기존과 반대로)
            if hand_label == 'Left' and 0 <= pred_idx <= 9:
                predicted_sign = labels_dict[pred_idx]
            elif hand_label == 'Right' and 10 <= pred_idx <= 19:
                predicted_sign = labels_dict[pred_idx]
            else:
                predicted_sign = 'Unknown'

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
            cv2.putText(frame, predicted_sign, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
            data_aux = []
            x_ = []
            y_ = []

        #predicted_sign이 0.5초 이상 유지되면 문장에 추가
        if predicted_sign == last_sign:
            if sign_hold_start is None:
                sign_hold_start = time.time()
            elif time.time() - sign_hold_start >= 1.5:
                if not sentences:
                    sentences.append(predicted_sign)
                    sign_hold_start = None  # 첫 append는 0.5초만 유지되면 됨
                elif sentences[-1] != predicted_sign:
                    sentences.append(predicted_sign)
                    sign_hold_start = None  # 두 번째부터는 0.5초+이전 단어와 달라야 함
        else:
            last_sign = predicted_sign
            sign_hold_start = time.time()
    else:
        last_sign = None
        sign_hold_start = None
    # 화면 밖으로 손이 나가도 자동 출력하지 않음 (엔터로만 출력)

    # 자막(문장 리스트) 실시간 표시 (여러 줄로 나누기)
    if sentences:
        subtitle = " ".join(sentences)
        max_chars_per_line = 40
        lines = [subtitle[i:i+max_chars_per_line] for i in range(0, len(subtitle), max_chars_per_line)]
        for idx, line in enumerate(lines):
            y_pos = H - 30 - (len(lines)-1-idx)*40  # 여러 줄이 아래에서 위로 쌓이게
            cv2.putText(frame, line, (30, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)

    cv2.imshow("Handy", frame)
    key = cv2.waitKey(1)
    # 엔터(Enter) 키(13) 누르면 sentence 음성 출력
    if key == 13 and sentences:
        sentence_str = " ".join(sentences)
        tta = TextToAudio("com.apple.speech.synthesis.voice.Alex", 200, 1.0)
        tta.text_to_audio(sentence_str)
        sentences.clear()
        last_sign = None
        sign_hold_start = None

capture.release()
cv2.destroyAllWindows()
