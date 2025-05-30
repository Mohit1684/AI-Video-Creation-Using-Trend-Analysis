import torch

# Add this function
def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU memory cleared")

import os
print(f"Checkpoint exists: {os.path.exists('E:\\SGP-4\\Wav2Lip\\checkpoints\\wav2lip_gan.pth')}")
print(f"Checkpoint size: {os.path.getsize('E:\\SGP-4\\Wav2Lip\\checkpoints\\wav2lip_gan.pth')} bytes")

import sys
print("Python path:", sys.executable)
print("Python version:", sys.version)
try:
    import torch
    print("PyTorch version:", torch.__version__)
    print("PyTorch location:", torch.__file__)
except ImportError:
    print("PyTorch not found in this environment")

from flask import Flask, render_template, request, jsonify, send_file
import os
import subprocess
import urllib.request
import asyncio
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from duckduckgo_search import DDGS
import edge_tts
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
import time

app = Flask(__name__)

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY")

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create necessary directories
os.makedirs('uploads', exist_ok=True)
os.makedirs('static/output', exist_ok=True)
os.makedirs('temp', exist_ok=True)
os.makedirs(os.path.join('Wav2Lip', 'checkpoints'), exist_ok=True)

# File paths
WAV2LIP_MODEL = os.path.join('Wav2Lip', 'checkpoints', 'wav2lip_gan.pth')
AUDIO_FILE = "static/output/generated_audio.mp3"
VIDEO_OUTPUT = "static/output/output_video.mp4"
FINAL_VIDEO = "static/output/final_video.mp4"

print("System PATH:", os.environ.get('PATH'))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Fetch trending topics from GNews API
def fetch_trending_topics(country='in', category='general'):
    try:
        url = "https://gnews.io/api/v4/top-headlines"
        params = {
            'token': GNEWS_API_KEY,
            'country': country,
            'lang': 'en',
            'max': 20,
            'category': category
        }
        response = requests.get(url, params=params)
        data = response.json()
        print("GNews response:", data)
        if response.status_code == 200 and "articles" in data:
            return [article["title"] for article in data["articles"][:20]]
        else:
            return ["No trending topics available"]
    except Exception as e:
        return [f"Error fetching trends: {e}"]

# Add these functions to app.py after the fetch_trending_topics function

# Fetch related news from GNews API
def fetch_trend_news(trend, country='in'):
    try:
        url = "https://gnews.io/api/v4/search"
        params = {
            'token': GNEWS_API_KEY,
            'q': trend,
            'country': country,
            'lang': 'en',
            'max': 15
        }
        response = requests.get(url, params=params)
        data = response.json()
        print("GNews response:", data)
        if response.status_code == 200 and "articles" in data:
            return [article["title"] + " - " + (article.get("description") or "") for article in data["articles"][:15]]
        else:
            return ["No related news found"]
    except Exception as e:
        return [f"Error fetching news: {e}"]
# Perform sentiment analysis on news
def analyze_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    score = analyzer.polarity_scores(text)
    if score["compound"] >= 0.05:
        return "positive"
    elif score["compound"] <= -0.05:
        return "negative"
    else:
        return "neutral"

# Generate video script using AI
def generate_script(trend, news, sentiment):
    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)

    context = f"Trend: {trend}\nSentiment: {sentiment}\nRelated News:\n{' '.join(news)}"
    prompt = f"{context}\nCreate a concise, conversational 4-minute news script that directly presents the information. The script should be written as a continuous monologue without any scene directions, character changes, or technical instructions. Focus only on the actual spoken content."

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system", 
                "content": "You are a news script writer. Write natural, flowing scripts that can be read in 4 minutes. Do not include any directions, scene changes, or speaker labels. Write only the actual spoken content in a conversational tone."
            },
            {"role": "user", "content": prompt}
        ],
        model="llama-3.3-70b-versatile",
    )
    return chat_completion.choices[0].message.content

# Update the generate_content route
@app.route('/generate_content', methods=['POST'])
def generate_content():
    try:
        selected_trend = request.form.get('trend')
        if not selected_trend:
            return jsonify({'error': 'No trend selected'}), 400

        # Fetch news and generate script
        news = fetch_trend_news(selected_trend)
        if not news:
            return jsonify({'error': 'No news found for the selected trend'}), 400

        sentiment = analyze_sentiment(' '.join(news))
        script = generate_script(selected_trend, news, sentiment)
        
        return jsonify({
            'success': True,
            'script': script
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Convert text to speech using Edge-TTS
async def text_to_speech(script, output_file):
    tts = edge_tts.Communicate(script, "en-US-JennyNeural")
    await tts.save(output_file)

def convert_text_to_speech(script, output_file):
    asyncio.run(text_to_speech(script, output_file))

# Generate lip-synced video using Wav2Lip
def generate_video(image_path, audio_path, output_video):
    try:
        wav2lip_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Wav2Lip')
        inference_path = os.path.join(wav2lip_dir, 'inference.py')
        
        # Create temp directory inside Wav2Lip
        os.makedirs(os.path.join(wav2lip_dir, 'temp'), exist_ok=True)
        
        # Copy FFmpeg executables to Wav2Lip directory
        for exe in ['ffmpeg.exe', 'ffprobe.exe', 'ffplay.exe']:
            src = os.path.join(os.getcwd(), exe)
            dst = os.path.join(wav2lip_dir, exe)
            if os.path.exists(src) and not os.path.exists(dst):
                import shutil
                shutil.copy2(src, dst)

        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Get the current Python executable path
        python_executable = sys.executable
        
        # Update the checkpoint path to use Wav2Lip directory
        checkpoint_path = os.path.join(wav2lip_dir, 'checkpoints', 'wav2lip_gan.pth')
        
        # Convert all paths to absolute paths
        image_path = os.path.abspath(image_path)
        audio_path = os.path.abspath(audio_path)
        output_video = os.path.abspath(output_video)

        # Print debug information
        print("Debug Info:")
        print(f"Working Directory: {os.getcwd()}")
        print(f"Wav2Lip Directory: {wav2lip_dir}")
        print(f"Inference Path: {inference_path}")
        print(f"Checkpoint Path: {checkpoint_path}")
        print(f"Image Path: {image_path}")
        print(f"Audio Path: {audio_path}")
        print(f"Output Video Path: {output_video}")

        # Verify files exist
        print("\nFile Existence Checks:")
        print(f"Wav2Lip Directory exists: {os.path.exists(wav2lip_dir)}")
        print(f"Inference script exists: {os.path.exists(inference_path)}")
        print(f"Checkpoint file exists: {os.path.exists(checkpoint_path)}")
        print(f"Input image exists: {os.path.exists(image_path)}")
        print(f"Input audio exists: {os.path.exists(audio_path)}")

        # Add current directory to PATH
        current_dir = os.getcwd()
        os.environ['PATH'] = current_dir + os.pathsep + os.environ.get('PATH', '')

        # Change working directory to Wav2Lip directory
        os.chdir(wav2lip_dir)

        # Add Wav2Lip directory to PATH as well
        os.environ['PATH'] = wav2lip_dir + os.pathsep + os.environ['PATH']

        # Set CUDA environment variables
        env = os.environ.copy()
        env['CUDA_LAUNCH_BLOCKING'] = '1'

        # Construct command with additional parameters for stability
        command = [
            python_executable,
            inference_path,
            "--checkpoint_path", checkpoint_path,
            "--face", image_path,
            "--audio", audio_path,
            "--outfile", output_video,
            "--resize_factor", "2",  # Add resize factor to prevent CUDA memory issues
            "--wav2lip_batch_size", "128",  # Adjust batch size
            "--nosmooth"  # Disable smoothing for better stability
        ]
        
        print("\nExecuting command:")
        print(" ".join(command))

        # Run the command with output capture and environment variables
        process = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env
        )

        # Print any output
        if process.stdout:
            print("\nCommand output:")
            print(process.stdout)
        
        if process.stderr:
            print("\nCommand errors:")
            print(process.stderr)

        # Change back to original directory
        os.chdir(current_dir)
        print("✅ Video generated successfully at:", output_video)
        
    except subprocess.CalledProcessError as e:
        print("❌ Error generating video:", e)
        print("Command output:", e.output if hasattr(e, 'output') else 'No output')
        print("Command stderr:", e.stderr if hasattr(e, 'stderr') else 'No stderr')
        raise Exception(str(e))
    except Exception as e:
        print("❌ Error:", str(e))
        print("Exception type:", type(e))
        print("Exception args:", e.args)
        raise
    finally:
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# Merge video with correct audio using FFmpeg
def merge_audio(video_file, audio_file, final_output):
    try:
        # Get FFmpeg path from system
        ffmpeg_path = "ffmpeg"  # Default command
        if os.name == 'nt':  # If on Windows
            # Try common FFmpeg installation locations
            possible_paths = [
                os.path.join(os.getcwd(), 'ffmpeg.exe'),  # Current directory
                r"C:\ffmpeg\bin\ffmpeg.exe",
                r"D:\ffmpeg\bin\ffmpeg.exe",
                os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ffmpeg.exe')
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    ffmpeg_path = path
                    break

        print(f"Using FFmpeg from: {ffmpeg_path}")
        
        subprocess.run([
            ffmpeg_path,
            "-i", video_file,
            "-i", audio_file,
            "-c:v", "copy",
            "-c:a", "aac",
            final_output,
            "-y"
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        print("✅ Final video saved at:", final_output)
    except subprocess.CalledProcessError as e:
        print("❌ Error merging audio:", e)
        print("Command stderr:", e.stderr.decode() if hasattr(e, 'stderr') else 'No stderr')
        raise Exception(f"FFmpeg error: {str(e)}")
    except Exception as e:
        print("❌ Error:", str(e))
        raise

def preprocess_audio(input_audio, output_audio):
    """Preprocess audio without duration limit"""
    try:
        command = [
            'ffmpeg',
            '-i', input_audio,
            '-ar', '16000',  # Set sample rate to 16kHz
            '-ac', '1',      # Convert to mono
            output_audio,
            '-y'
        ]
        subprocess.run(command, check=True, capture_output=True)
        return True
    except Exception as e:
        print(f"Error preprocessing audio: {e}")
        return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/fetch_trends', methods=['GET'])
def get_trends():
    country = request.args.get('country', 'in')
    category = request.args.get('category', 'general')
    trends = fetch_trending_topics(country, category)
    return jsonify(trends)

@app.route('/video_generation')
def video_generation():
    return render_template('video.html')

@app.route('/generate_video', methods=['POST'])
def create_video():
    if 'image' not in request.files or 'audio' not in request.files:
        return jsonify({'error': 'Missing files'}), 400
    
    image_file = request.files['image']
    audio_file = request.files['audio']
    
    if image_file and allowed_file(image_file.filename):
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'input_image.jpg')
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], 'input_audio.mp3')
        processed_audio = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_audio.wav')
        
        image_file.save(image_path)
        audio_file.save(audio_path)
        
        try:
            # Get audio duration before processing
            duration_cmd = [
                'ffprobe', 
                '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                audio_path
            ]
            duration = float(subprocess.check_output(duration_cmd).decode().strip())
            print(f"Audio duration: {duration} seconds")
            
            # Preprocess audio without duration limit
            if not preprocess_audio(audio_path, processed_audio):
                return jsonify({'error': 'Audio preprocessing failed'}), 500
                
            # Generate video using processed audio
            generate_video(image_path, processed_audio, VIDEO_OUTPUT)
            merge_audio(VIDEO_OUTPUT, audio_path, FINAL_VIDEO)
            
            return jsonify({
                'success': True,
                'video_path': FINAL_VIDEO,
                'duration': duration
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/generate_audio', methods=['POST'])
def generate_audio():
    try:
        script = request.form.get('script')
        if not script:
            return jsonify({'error': 'No script provided'}), 400

        # Generate audio
        convert_text_to_speech(script, AUDIO_FILE)
        
        return jsonify({
            'success': True,
            'audio_path': '/static/output/generated_audio.mp3'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
