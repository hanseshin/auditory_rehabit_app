import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

mp3_file = "/home/ubuntu/learn_server/learn/root/mnt/upload/wave_form.mp3"
y, sr = librosa.load(mp3_file)

# 스펙트로그램
D = librosa.amplitude_to_db(librosa.stft(y), ref=np.max)
plt.figure(figsize=(10, 4))
librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram')
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('/home/ubuntu/learn_server/learn/root/mnt/spectrogram/spectrogram.png')  # 수정된 경로

# 진폭 그래프
plt.figure(figsize=(10, 4))
librosa.display.waveshow(y, sr=sr)
plt.title('Amplitude Envelope')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.savefig('/home/ubuntu/learn_server/learn/root/mnt/amplitude_graph/amplitude_graph.png')  # 수정된 경로

