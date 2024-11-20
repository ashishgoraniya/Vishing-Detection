from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import logging
from preprocessor import AudioProcessor
import time
from werkzeug.utils import secure_filename
from concurrent.futures import ThreadPoolExecutor
import asyncio

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize AudioProcessor and ThreadPoolExecutor
processor = AudioProcessor(
    whisper_api_key="Whisper API goes here....",
    huggingface_api_key="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"
)
executor = ThreadPoolExecutor(max_workers=8)

def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed"""
    ALLOWED_EXTENSIONS = {'mp3', 'wav', 'm4a', 'ogg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_safe_filepath(filename: str) -> str:
    """Create a safe filepath with timestamp"""
    timestamp = int(time.time())
    safe_filename = secure_filename(f"{timestamp}_{filename}")
    return os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)

@app.route('/')
def home():
    """Serve the main page"""
    return send_from_directory('.', 'index.html')

def run_async_task(coroutine):
    """Run an async task in a new event loop"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coroutine)
    finally:
        loop.close()

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle audio file upload and processing"""
    try:
        if 'audio' not in request.files:
            logger.warning("No file part in request")
            return jsonify({"error": "No audio file provided"}), 400
            
        file = request.files['audio']
        
        if file.filename == '':
            logger.warning("No file selected")
            return jsonify({"error": "No selected file"}), 400
            
        if not allowed_file(file.filename):
            logger.warning(f"Invalid file type: {file.filename}")
            return jsonify({"error": "File type not allowed"}), 400

        filepath = get_safe_filepath(file.filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        try:
            logger.info(f"Saving file: {filepath}")
            file.save(filepath)

            # Process audio using thread pool
            future = executor.submit(
                processor.process_audio,  # Using synchronous wrapper
                filepath
            )
            
            # Get results with timeout
            transcription, prediction, confidence = future.result(timeout=300)
            
            logger.info(f"Processing complete: {prediction} ({confidence:.2f})")
            
            return jsonify({
                "transcription": transcription,
                "prediction": prediction,
                "confidence": round(confidence * 100, 2)
            })

        except asyncio.TimeoutError:
            logger.error("Processing timeout")
            return jsonify({
                "error": "Processing timeout",
                "details": "Audio processing took too long"
            }), 504

        except Exception as e:
            logger.error(f"Processing error: {str(e)}")
            return jsonify({
                "error": "Error processing audio",
                "details": str(e)
            }), 500

        finally:
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                    logger.debug(f"Cleaned up file: {filepath}")
                except Exception as e:
                    logger.error(f"Error cleaning up file {filepath}: {str(e)}")

    except Exception as e:
        logger.error(f"Request handling error: {str(e)}")
        return jsonify({
            "error": "Server error",
            "details": str(e)
        }), 500

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error"""
    return jsonify({
        "error": "File too large",
        "details": "Maximum file size is 32MB"
    }), 413

@app.errorhandler(500)
def internal_server_error(error):
    """Handle internal server errors"""
    return jsonify({
        "error": "Internal server error",
        "details": "An unexpected error occurred"
    }), 500

@app.errorhandler(Exception)
def handle_exception(e):
    """Handle all other exceptions"""
    logger.error(f"Unhandled exception: {str(e)}")
    return jsonify({
        "error": "Server error",
        "details": str(e)
    }), 500

if __name__ == '__main__':
    # Ensure uploads directory exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Run with standard Flask development server
    app.run(debug=True, port=5000)