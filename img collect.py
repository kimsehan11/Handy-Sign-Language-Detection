# DEPENDENCIES ------>
import os
import cv2


DATA_DIR = "./data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

num_classes = 10
dataset_size = 200

cap = cv2.VideoCapture(0)

while True:
    class_num = input("저장할 클래스 번호를 입력하세요(종료하려면 q): ")
    if class_num.lower() == 'q':
        break
    if not class_num.isdigit():
        print("숫자를 입력하세요.")
        continue
    class_num = int(class_num)
    class_dir = os.path.join(DATA_DIR, str(class_num))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)
    print(f"Collecting data for class {class_num}")

    while True:
        ret, frame = cap.read()
        cv2.putText(frame, "Press '1' to start collecting", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(25) == ord("1"):
            break

    # count를 기존 파일 개수에서 시작하도록 변경
    existing_files = os.listdir(class_dir)
    existing_indices = [int(f.split('.')[0]) for f in existing_files if f.endswith('.jpg') and f.split('.')[0].isdigit()]
    if existing_indices:
        count = max(existing_indices) + 1
    else:
        count = 0
    while count < (len(existing_indices) + dataset_size):
        ret, frame = cap.read()
        cv2.imshow("Frame", frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(class_dir, f"{count}.jpg"), frame)
        count += 1

cap.release()
cv2.destroyAllWindows()
