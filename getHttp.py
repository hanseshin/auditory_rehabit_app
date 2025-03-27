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
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

app = Flask(__name__)

UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "/home/ubuntu/learn_server/learn/root/mnt/upload")
SPD_FOLDER = os.getenv("SPD_FOLDER", "/home/ubuntu/learn_server/learn/root/mnt/spectrogram")
AMP_FOLDER = os.getenv("AMP_FOLDER", "/home/ubuntu/learn_server/learn/root/mnt/amplitude_graph")
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
        if not event.is_directory and event.src_path.endswith('.mp4'):
            print(f"Detected new video file: {event.src_path}, extracting audio...")
            convert_video_to_numpy_and_wav(event.src_path, UPLOAD_FOLDER)
            time.sleep(3)
            threading.Thread(target=run_wave).start()
            threading.Thread(target=run_inference).start()

def convert_video_to_numpy_and_wav(src_file, dest_dir):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    dest_path_mp3 = os.path.join(dest_dir, 'wave_form.mp3')

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
    print(f"Extracted audio from {src_file}")

def connect_to_database():
    try:
        conn = mysql.connector.connect(
            host=os.getenv("DB_HOST"),
            database=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD")
        )
        return conn, conn.cursor()
    except Error as e:
        print("Database connection error:", e)
        return None, None

def close_database_connection(conn, cur):
    if conn and conn.is_connected():
        cur.close()
        conn.close()
        print("MySQL connection closed")

@app.route('/result', methods=['GET'])
def send_result():
    conn, cur = connect_to_database()
    cur.execute("SELECT result FROM inference_results ORDER BY id DESC LIMIT 1;")
    result = cur.fetchone()
    close_database_connection(conn, cur)
    
    return jsonify({'result': result[0]}) if result else jsonify({'error': 'No result found'}), 200

if __name__ == '__main__':
    threading.Thread(target=lambda: Observer().schedule(MyHandler(), UPLOAD_FOLDER).start()).start()
    app.run(host='0.0.0.0', port=5000, debug=False)
