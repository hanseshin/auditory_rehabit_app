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
import pdb
import subprocess
import getHttp
import requests

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
    video_feature  = video
    video_feature = video_feature.permute(3, 0, 1, 2)  # T H W C --> C T H W
    return video_feature

def connect_to_database(host, database, user, password):
    conn = mysql.connector.connect(host=host, database=database, user=user, password=password)
    cur = conn.cursor()
    return conn, cur

def close_database_connection(conn, cur):
    cur.close()
    conn.close()
# inference.py의 sentence 반환 함수 추가

def send_text_to_flutter(text):
    # Flutter 앱으로 텍스트를 동기적으로 전송
    url = "http://127.0.0.1:8400/receive-data"
    payload = {'text': text}
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, json=payload, headers=headers)
    
    # 응답 확인
    if response.status_code == 200:
        print("Text sent to Flutter successfully!")
        return response.json()  # 텍스트가 성공적으로 전송되었을 때의 응답을 반환
    else:
        print("Failed to send text to Flutter.")
        return None



if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    
    multi_gpu = False
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu' )
   
    vocab = KsponSpeechVocabulary('dataset/labels.csv')
    
    test_model = "./temp_model/Docker_model.pt"
    
    model = torch.load(test_model, map_location=lambda storage, loc: storage).to(device)
    
    if multi_gpu:
        model = nn.DataParallel(model)
    
    model.eval()
   
    val_metric = CharacterErrorRate(vocab)
    print(model)
    print(count_parameters(model))
    model.eval()
    
    
    #mp4_lists = glob.glob('./upload/*.npy')
    mp4_lists = glob.glob('./temp_data/*.npy')
    mp4_lists = sorted(mp4_lists)

    #mp3_lists = glob.glob('/home/ubuntu/upload/*.mp3')
    mp3_lists = glob.glob('./temp_data/*.mp3')
    mp3_lists = sorted(mp3_lists)
    print('Inference start!!!')

    # MySQL 연결 정보
    host = ''
    database = ''
    user = ''
    password = ''

    conn, cur = connect_to_database(host, database, user, password)

    if len(mp3_lists) != len(mp4_lists):
        print("Error!!!!!!!!!!")
        pdb.set_trace()
        
    for i in range(len(mp4_lists)):
        audio_inputs = parse_audio(mp3_lists[i])
        audio_input_lengths = torch.IntTensor([audio_inputs.shape[0]])
        video_inputs = parse_video(mp4_lists[i])
        video_input_lengths = torch.IntTensor([video_inputs.shape[1]])

        audio_inputs = audio_inputs.unsqueeze(0)
        video_inputs = video_inputs.unsqueeze(0)

        video_inputs = video_inputs.to(device)
        audio_inputs = audio_inputs.to(device)
        video_input_lengths = video_input_lengths.to(device)
        audio_input_lengths = audio_input_lengths.to(device)

        outputs = model.recognize(video_inputs, 
                                video_input_lengths, 
                                audio_inputs,
                                audio_input_lengths,
                                )
        y_hats = outputs.max(-1)[1]
        sentence = vocab.label_to_string(y_hats.cpu().detach().numpy())
        print(sentence)
        send_text_to_flutter(str(sentence[0]))
        # 결과값을 테이블에 삽입
        try:
            query = "INSERT INTO inference_results (result) VALUES (%s);"
            cur.execute(query, [str(sentence[0])])  # 첫 번째 항목을 추출하여 문자열로 변환하여 전달
            conn.commit()
        except mysql.connector.Error as err:
            print("MySQL 오류: {}".format(err))
            
      
         
        
    # 연결 종료
    close_database_connection(conn, cur)