# DEPENDENCIES ------>
import os
import pickle
import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)
data_dir = "./data"
data = []
labels = []

for dir_ in os.listdir(data_dir):
    for img_path in os.listdir(os.path.join(data_dir, dir_)):
        img = cv2.imread(os.path.join(data_dir, dir_, img_path))
        # 증강: 원본, 좌우반전, 밝기 증가
        aug_imgs = [img]
        # 좌우반전
        flipped = cv2.flip(img, 1)
        aug_imgs.append(flipped)
        # 밝기 증가
        brighter = cv2.convertScaleAbs(img, alpha=1, beta=40)
        aug_imgs.append(brighter)
        # 증강 이미지들에 대해 랜드마크 추출
        for aug_img in aug_imgs:
            data_aux = []
            img_rgb = cv2.cvtColor(aug_img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x)
                        data_aux.append(y)
                if len(data_aux) == 42:
                    data.append(data_aux)
                    labels.append(dir_)
file = open("data.pickle", "wb")
pickle.dump({"data": data, "labels": labels}, file)
file.close()
