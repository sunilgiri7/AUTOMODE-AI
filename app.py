from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from werkzeug.utils import secure_filename
import logging
import os
import fitz 
import docx
from pathlib import Path
from chatbot_backend import get_session_history, handle_query, update_session_history
from faiss_manager import DocumentManager
from chatbot_backend import build_production_chain, vector_store

# Initialize Flask app
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

qa_chain = build_production_chain(vector_store)

# Configure upload settings
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'doc', 'docx', 'md'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize document manager
doc_manager = DocumentManager()

def allowed_file(filename):
    """Check if the file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_pdf_text(filepath):
    """Extract text from PDF file."""
    try:
        text = ""
        pdf_document = fitz.open(filepath)
        for page in pdf_document:
            text += page.get_text()
        pdf_document.close()
        return text
    except Exception as e:
        logger.error(f"Error extracting PDF text: {str(e)}")
        return None

def extract_docx_text(filepath):
    """Extract text from DOCX file."""
    try:
        doc = docx.Document(filepath)
        text = []
        for paragraph in doc.paragraphs:
            text.append(paragraph.text)
        return '\n'.join(text)
    except Exception as e:
        logger.error(f"Error extracting DOCX text: {str(e)}")
        return None
    
def process_uploaded_file(file):
    """Process uploaded file based on its type."""
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        file_ext = filename.rsplit('.', 1)[1].lower()
        text = None
        
        if file_ext == 'pdf':
            text = extract_pdf_text(filepath)
        elif file_ext in ['doc', 'docx']:
            text = extract_docx_text(filepath)
        elif file_ext in ['txt', 'md']:
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
        
        if text is None:
            return False, f"Failed to extract text from {file_ext.upper()} file"
            
        # Save extracted text to a temp file
        text_filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{filename}.txt")
        with open(text_filepath, 'w', encoding='utf-8') as f:
            f.write(text)
            
        # Clean up original file
        try:
            os.remove(filepath)
        except Exception as e:
            logger.error(f"Error removing original file: {str(e)}")
            
        return True, text_filepath
        
    except Exception as e:
        logger.error(f"Error processing file {file.filename}: {str(e)}")
        return False, str(e)

@app.route('/')
def index():
    """Render the main chat interface."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle document uploads and add them to the vector store."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        files = request.files.getlist('file')
        uploaded_files = []
        failed_files = []
        
        for file in files:
            if file and allowed_file(file.filename):
                success, result = process_uploaded_file(file)
                if success:
                    uploaded_files.append(result)
                else:
                    failed_files.append(f"{file.filename} ({result})")
            else:
                failed_files.append(f"{file.filename} (unsupported file type)")
        
        if uploaded_files:
            try:
                doc_manager.add_documents(uploaded_files)
                
                # Clean up processed text files
                for filepath in uploaded_files:
                    try:
                        os.remove(filepath)
                    except Exception as e:
                        logger.error(f"Error removing file {filepath}: {str(e)}")
                
                response = {
                    'message': f'Successfully processed {len(uploaded_files)} files',
                    'processed_files': [Path(f).name.replace('.txt', '') for f in uploaded_files]
                }
                if failed_files:
                    response['failed_files'] = failed_files
                return jsonify(response), 200
            except Exception as e:
                logger.error(f"Error processing documents: {str(e)}")
                return jsonify({'error': 'Error processing documents', 'details': str(e)}), 500
        
        if failed_files:
            return jsonify({
                'error': 'No valid files were uploaded',
                'failed_files': failed_files
            }), 400
        
        return jsonify({'error': 'No files were uploaded'}), 400
    
    except Exception as e:
        logger.error(f"Unexpected error in upload_file: {str(e)}")
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500
    
# Message counter for feedback
message_counters = {}

# Add helper function to check if file is binary
def is_binary_file(filepath):
    """Check if a file is binary."""
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            file.read()
            return False
    except UnicodeDecodeError:
        return True
    
# Add favicon route
@app.route('/favicon.ico')
def favicon():
    return '', 204  # Return no content for favicon requests

# In app.py, update the socket handler:
@socketio.on('user_message')
def handle_message(data):
    try:
        query = data.get("message", "").strip()
        if not query:
            emit('bot_response', {'message': 'Please provide a valid question.'})
            return
            
        session_id = data.get("session_id", "default_session")
        
        # Update message counter for this session
        if session_id not in message_counters:
            message_counters[session_id] = 0
        message_counters[session_id] += 1
        
        # Determine if we should show feedback
        show_feedback = message_counters[session_id] % 10 == 0
        
        # Emit typing indicator
        emit('bot_typing', {'status': True})
        
        chat_history = get_session_history(session_id)
        response = handle_query(
            query=query,
            history=chat_history,
            qa_chain=qa_chain
        )
        
        # Stop typing indicator and send response
        emit('bot_typing', {'status': False})
        emit('bot_response', {
            'message': response,
            'show_feedback': show_feedback
        })
        update_session_history(session_id, query, response)
        
    except Exception as e:
        logger.error(f"Socket error: {str(e)}", exc_info=True)
        emit('bot_typing', {'status': False})
        emit('bot_response', {'message': 'I apologize, but I encountered an error processing your request.'})

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error."""
    return jsonify({
        'error': 'File too large',
        'message': f'File size exceeds maximum limit of {MAX_CONTENT_LENGTH / (1024 * 1024)}MB'
    }), 413

@app.errorhandler(Exception)
def handle_exception(e):
    """Handle any unhandled exceptions."""
    logger.error(f"Unhandled exception: {str(e)}")
    return jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred'
    }), 500

def start_server():
    """Start the Flask server with Socket.IO integration."""
    try:
        logger.info("Starting server...")
        socketio.run(app, host='0.0.0.0', port=5000, debug=True)
    except Exception as e:
        logger.error(f"Error starting server: {str(e)}")
        raise

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)