import os
from flask import Flask, send_from_directory, request, jsonify
from werkzeug.utils import secure_filename
import time
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import threading
import mysql.connector
from mysql.connector import Error

app = Flask(__name__)



UPLOAD_FOLDER = '/home/ubuntu/learn_server/learn/root/mnt/upload'
SPD_FOLDER='/home/ubuntu/learn_server/learn/root/mnt/spectrogram'
AMP_FOLDER='/home/ubuntu/learn_server/learn/root/mnt/amplitude_graph'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi', 'mov', 'mkv'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def run_inference():
    os.system('python /home/ubuntu/learn_server/learn/root/mnt/inference.py')
def run_wave():
    os.system('python /home/ubuntu/learn_server/learn/root/mnt/wave_form.py')
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



class MyHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory:
            if event.src_path.endswith('.mp4'):
                print(f"Detected new video file: {event.src_path}, extracting audio...")
                convert_video_to_numpy_and_wav(event.src_path,'/home/ubuntu/learn_server/learn/root/mnt/upload')
                wave_thread = threading.Thread(target=run_wave)
                inference_thread = threading.Thread(target=run_inference)
                time.sleep(3)
                wave_thread.start()
                wave_thread.join()  
                inference_thread.start()
                
               
                
def convert_video_to_numpy_and_wav(src_file, dest_dir):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    filename = os.path.basename(src_file)
    base_name = "wave_form"  # 여기에서 원하는 파일명 형식을 설정
    dest_path_numpy = os.path.join(dest_dir, base_name + '.npy')
    dest_path_mp3 = os.path.join(dest_dir, base_name + '.mp3')

    command_numpy = [
        "video2numpy",
        src_file,
        "--dest='/home/ubuntu/learn_server/learn/root/mnt/upload'",
        "--take_every_nth=1",
        "--resize_size=224",
        "--workers=1",
        "--memory_size=4"
    ]
    result = subprocess.run(command_numpy, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print(f"Error executing video2numpy: {result.stderr.decode()}")

    command_ffmpeg = [
        "ffmpeg",
        "-i", src_file,
        "-vn",
        "-acodec", "libmp3lame",
        "-ar", "44100",
        "-ac", "2",
        dest_path_mp3
    ]
    subprocess.run(command_ffmpeg)

    print(f"Converted {src_file} to numpy and extracted audio")



def connect_to_database(host, database, user, password):
    """데이터베이스 연결을 설정합니다."""
    try:
        connection = mysql.connector.connect(
            host=host,
            database=database,
            user=user,
            password=password
        )
        if connection.is_connected():
            db_info = connection.get_server_info()
            print("MySQL server version:", db_info)
            cursor = connection.cursor()
            cursor.execute("select database();")
            record = cursor.fetchone()
            print("Connected to database:", record)
            return connection, cursor
    except Error as e:
        print("Error while connecting to MySQL", e)
        return None, None



def close_database_connection(connection, cursor):
    """데이터베이스 연결을 종료합니다."""
    if connection.is_connected():
        cursor.close()
        connection.close()
        print("MySQL connection is closed")
 
                
def watch_directory(directory):
    observer = Observer()
    observer.schedule(MyHandler(), directory)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

@app.route('/', methods=['GET'])
def index():
    return 'Welcome to the upload server!'

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return jsonify({'message': 'File uploaded successfully'}), 200
    else:
        return jsonify({'error': 'Allowed video types are - mp4, avi, mov, mkv'}), 400
  
@app.route('/result', methods=['GET'])
def send_result():
    conn, cur = connect_to_database('learn.c5oaaoww6rbz.ap-northeast-2.rds.amazonaws.com', 'testDB', 'sub0913', 's2154759329')
    cur.execute("SELECT result FROM inference_results ORDER BY id DESC LIMIT 1;")
    result = cur.fetchone()
    close_database_connection(conn, cur)
    if result:
        return jsonify({'result': result[0]}), 200
    else:
        return jsonify({'error': 'No result found'}), 404
    

@app.route('/download/spectrogram', methods=['GET'])
def download_spectrogram():
    directory = '/home/ubuntu/learn_server/learn/root/mnt/spectrogram'  # 파일이 저장된 서버의 폴더 경로
    filename = 'spectrogram.png'
    if os.path.exists(os.path.join(directory, filename)):
        return send_from_directory(directory, filename, as_attachment=True)
    else:
        return jsonify({'error': 'File not found'}), 404
    
@app.route('/download/amplitude', methods=['GET'])
def download_amplitude():
    directory = '/home/ubuntu/learn_server/learn/root/mnt/amplitude_graph'  # 파일이 저장된 서버의 폴더 경로
    filename = 'amplitude_graph.png'
    if os.path.exists(os.path.join(directory, filename)):
        return send_from_directory(directory, filename, as_attachment=True)
    else:
        return jsonify({'error': 'File not found'}), 404
    

if __name__ == '__main__':
    watch_thread = threading.Thread(target=watch_directory, args=(UPLOAD_FOLDER,))
    watch_thread.start()
    app.run(host='0.0.0.0', port=5000, debug=False)

