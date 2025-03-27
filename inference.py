import os
import torch
import torch.nn as nn
from torch import Tensor, FloatTensor
from dataloader.vocabulary import KsponSpeechVocabulary
from metric.metric import CharacterErrorRate
import numpy as np
import torchaudio
import glob
import librosa
import mysql.connector
import requests
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def parse_audio(audio_path: str):
    signal, _ = librosa.load(audio_path, sr=16000, mono=True)
    feature = torchaudio.compliance.kaldi.fbank(
        waveform=Tensor(signal).unsqueeze(0),
        num_mel_bins=80,
        frame_length=20,
        frame_shift=10,
        window_type='hamming'
    ).transpose(0, 1).numpy()

    feature -= feature.mean()
    feature /= np.std(feature)

    feature = FloatTensor(feature).transpose(0, 1)
    return feature
    
def parse_video(video_path: str):
    video = np.load(video_path)
    video = torch.from_numpy(video).float()

    video -= torch.mean(video)
    video /= torch.std(video)
    video_feature = video.permute(3, 0, 1, 2)  # T H W C --> C T H W
    return video_feature

def connect_to_database():
    conn = mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD")
    )
    cur = conn.cursor()
    return conn, cur

def close_database_connection(conn, cur):
    cur.close()
    conn.close()

def send_text_to_flutter(text):
    url = os.getenv("FLUTTER_SERVER_URL")
    payload = {'text': text}
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, json=payload, headers=headers)
    
    if response.status_code == 200:
        print("Text sent to Flutter successfully!")
        return response.json()
    else:
        print("Failed to send text to Flutter.")
        return None

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    
    multi_gpu = False
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    vocab = KsponSpeechVocabulary('dataset/labels.csv')
    
    test_model = "./temp_model/Docker_model.pt"
    model = torch.load(test_model, map_location=lambda storage, loc: storage).to(device)

    if multi_gpu:
        model = nn.DataParallel(model)

    model.eval()
   
    val_metric = CharacterErrorRate(vocab)
    print(model)
    print(count_parameters(model))

    mp4_lists = sorted(glob.glob('./temp_data/*.npy'))
    mp3_lists = sorted(glob.glob('./temp_data/*.mp3'))
    
    print('Inference start!!!')

    conn, cur = connect_to_database()

    if len(mp3_lists) != len(mp4_lists):
        print("Error! Mismatched number of audio and video files.")
        exit()

    for i in range(len(mp4_lists)):
        audio_inputs = parse_audio(mp3_lists[i])
        audio_input_lengths = torch.IntTensor([audio_inputs.shape[0]])
        video_inputs = parse_video(mp4_lists[i])
        video_input_lengths = torch.IntTensor([video_inputs.shape[1]])

        audio_inputs = audio_inputs.unsqueeze(0).to(device)
        video_inputs = video_inputs.unsqueeze(0).to(device)
        video_input_lengths = video_input_lengths.to(device)
        audio_input_lengths = audio_input_lengths.to(device)

        outputs = model.recognize(video_inputs, video_input_lengths, audio_inputs, audio_input_lengths)
        y_hats = outputs.max(-1)[1]
        sentence = vocab.label_to_string(y_hats.cpu().detach().numpy())
        
        print(sentence)
        send_text_to_flutter(str(sentence[0]))

        try:
            query = "INSERT INTO inference_results (result) VALUES (%s);"
            cur.execute(query, [str(sentence[0])])
            conn.commit()
        except mysql.connector.Error as err:
            print("MySQL 오류:", err)
            
    close_database_connection(conn, cur)
