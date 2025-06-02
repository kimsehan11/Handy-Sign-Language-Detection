# DEPENDENCIES ------>
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten, Dropout, BatchNormalization, MaxPooling1D
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


data_dict = pickle.load(open("./data.pickle", "rb"))
data = np.asarray(data_dict["data"])
labels = np.asarray(data_dict["labels"]).astype(int)

num_classes = len(np.unique(labels))
labels_cat = to_categorical(labels, num_classes)

# train/val/test 분할
x_temp, x_test, y_temp, y_test = train_test_split(data, labels_cat, test_size=0.2, shuffle=True, stratify=labels)
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

model.save("model_cnn.h5")
