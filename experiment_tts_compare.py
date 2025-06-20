import time
import numpy as np
import matplotlib.pyplot as plt
from gtts import gTTS
import pyttsx3
import asyncio
import edge_tts

text = "안녕하세요, 테스트입니다."
repeat = 5  # 반복 횟수(더 늘려도 됨)

gtts_times = []
pyttsx3_times = []
edge_times = []

for _ in range(repeat):
    # gTTS
    start = time.time()
    tts = gTTS(text=text, lang='ko')
    tts.save("gtts_output.mp3")
    gtts_times.append(time.time() - start)

    # pyttsx3
    engine = pyttsx3.init()
    start = time.time()
    engine.save_to_file(text, "pyttsx3_output.mp3")
    engine.runAndWait()
    pyttsx3_times.append(time.time() - start)

    # edge-tts (비동기)
    start = time.time()
    communicate = edge_tts.Communicate(text, "ko-KR-SunHiNeural")
    asyncio.get_event_loop().run_until_complete(
        communicate.save("edge_tts_output.mp3")
    )
    edge_times.append(time.time() - start)

# 평균/표준편차 계산
labels = ['gTTS', 'pyttsx3', 'edge-tts']
times_mean = [np.mean(gtts_times), np.mean(pyttsx3_times), np.mean(edge_times)]
times_std = [np.std(gtts_times), np.std(pyttsx3_times), np.std(edge_times)]

plt.bar(labels, times_mean, yerr=times_std, color=['skyblue', 'orange', 'limegreen'], capsize=10)
plt.ylabel('응답 시간(초)')
plt.title('TTS 모델별 응답 시간 비교')
plt.tight_layout()
plt.show()

print(f"gTTS 평균: {times_mean[0]:.2f}s, 표준편차: {times_std[0]:.2f}s")
print(f"pyttsx3 평균: {times_mean[1]:.2f}s, 표준편차: {times_std[1]:.2f}s")
print(f"edge-tts 평균: {times_mean[2]:.2f}s, 표준편차: {times_std[2]:.2f}s")
