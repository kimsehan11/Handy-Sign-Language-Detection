# DEPENDENCIES ------>
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten, Dropout, BatchNormalization, MaxPooling1D
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


data_digit_dict = pickle.load(open("./data_digit.pickle", "rb"))
data_digit = np.asarray(data_digit_dict["data"])
labels_digit = np.asarray(data_digit_dict["labels"]).astype(int)

# digit: 0~9만 사용
mask = (labels_digit >= 0) & (labels_digit <= 9)
data_digit = data_digit[mask]
labels_digit = labels_digit[mask]

num_classes = 10
labels_cat = to_categorical(labels_digit, num_classes)

# train/val/test 분할
x_temp, x_test, y_temp, y_test = train_test_split(data_digit, labels_cat, test_size=0.2, shuffle=True, stratify=labels_digit)
x_train, x_val, y_train, y_val = train_test_split(x_temp, y_temp, test_size=0.1, shuffle=True, stratify=np.argmax(y_temp, axis=1))

x_train = x_train.reshape((-1, 21, 2))
x_val = x_val.reshape((-1, 21, 2))
x_test = x_test.reshape((-1, 21, 2))

model = Sequential([
    Conv1D(64, 3, activation='relu', input_shape=(21, 2)),
    BatchNormalization(),
    MaxPooling1D(2),
    Conv1D(128, 3, activation='relu'),
    BatchNormalization(),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=20, batch_size=16, validation_data=(x_val, y_val))

# 검증셋 평가
val_loss, val_acc = model.evaluate(x_val, y_val)
print(f"Validation accuracy: {val_acc*100:.2f}%")

# 테스트셋 평가
loss, acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {acc*100:.2f}%")

model.save("model_digit_cnn.h5")

# ----- sentences(10~35) 모델 학습용 -----
data_sentences_dict = pickle.load(open("./data_sentences.pickle", "rb"))
data_sentences = np.asarray(data_sentences_dict["data"])
labels_sentences = np.asarray(data_sentences_dict["labels"]).astype(int)

# sentences: 10~35만 사용
mask_s = (labels_sentences >= 10) & (labels_sentences <= 35)
data_sentences = data_sentences[mask_s]
labels_sentences = labels_sentences[mask_s]

# 라벨을 0~25로 변환 (10~35 → 0~25)
labels_sentences = labels_sentences - 10
num_classes_sent = 26
labels_cat_sent = to_categorical(labels_sentences, num_classes_sent)

x_temp_s, x_test_s, y_temp_s, y_test_s = train_test_split(data_sentences, labels_cat_sent, test_size=0.2, shuffle=True, stratify=labels_sentences)
x_train_s, x_val_s, y_train_s, y_val_s = train_test_split(x_temp_s, y_temp_s, test_size=0.1, shuffle=True, stratify=np.argmax(y_temp_s, axis=1))

x_train_s = x_train_s.reshape((-1, 21, 2))
x_val_s = x_val_s.reshape((-1, 21, 2))
x_test_s = x_test_s.reshape((-1, 21, 2))

model_sent = Sequential([
    Conv1D(64, 3, activation='relu', input_shape=(21, 2)),
    BatchNormalization(),
    MaxPooling1D(2),
    Conv1D(128, 3, activation='relu'),
    BatchNormalization(),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(num_classes_sent, activation='softmax')
])

model_sent.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_sent.fit(x_train_s, y_train_s, epochs=20, batch_size=16, validation_data=(x_val_s, y_val_s))

val_loss_s, val_acc_s = model_sent.evaluate(x_val_s, y_val_s)
print(f"[SENTENCES] Validation accuracy: {val_acc_s*100:.2f}%")

loss_s, acc_s = model_sent.evaluate(x_test_s, y_test_s)
print(f"[SENTENCES] Test accuracy: {acc_s*100:.2f}%")

model_sent.save("model_sentences_cnn.h5")
