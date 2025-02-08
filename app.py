from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from werkzeug.utils import secure_filename
import logging
import os
import fitz 
import docx
from pathlib import Path
from typing import Tuple, List, Dict, Union, Optional
from chatbot_backend import chatbot, qa_chain
from faiss_manager import DocumentManager

class FlaskChatApplication:
    def __init__(self):
        # Initialize Flask app
        self.app = Flask(__name__)
        CORS(self.app)
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Configure upload settings
        self.UPLOAD_FOLDER = 'uploads'
        self.ALLOWED_EXTENSIONS = {'txt', 'pdf', 'doc', 'docx', 'md'}
        self.MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
        
        # Configure Flask app
        self._configure_app()
        
        # Initialize components
        self.doc_manager = DocumentManager()
        self.message_counters = {}
        
        # Register routes and handlers
        self._register_routes()
        self._register_socket_handlers()
        self._register_error_handlers()

    def _configure_app(self):
        """Configure Flask application settings"""
        self.app.config['UPLOAD_FOLDER'] = self.UPLOAD_FOLDER
        self.app.config['MAX_CONTENT_LENGTH'] = self.MAX_CONTENT_LENGTH
        os.makedirs(self.UPLOAD_FOLDER, exist_ok=True)

    def _register_routes(self):
        """Register HTTP routes"""
        self.app.route('/')(self.index)
        self.app.route('/upload', methods=['POST'])(self.upload_file)
        self.app.route('/favicon.ico')(self.favicon)

    def _register_socket_handlers(self):
        """Register Socket.IO event handlers"""
        self.socketio.on('user_message')(self.handle_message)

    def _register_error_handlers(self):
        """Register error handlers"""
        self.app.errorhandler(413)(self.too_large)
        self.app.errorhandler(Exception)(self.handle_exception)

    def allowed_file(self, filename: str) -> bool:
        """Check if the file extension is allowed."""
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in self.ALLOWED_EXTENSIONS

    def extract_pdf_text(self, filepath: str) -> Optional[str]:
        """Extract text from PDF file."""
        try:
            text = ""
            pdf_document = fitz.open(filepath)
            for page in pdf_document:
                text += page.get_text()
            pdf_document.close()
            return text
        except Exception as e:
            self.logger.error(f"Error extracting PDF text: {str(e)}")
            return None

    def extract_docx_text(self, filepath: str) -> Optional[str]:
        """Extract text from DOCX file."""
        try:
            doc = docx.Document(filepath)
            text = [paragraph.text for paragraph in doc.paragraphs]
            return '\n'.join(text)
        except Exception as e:
            self.logger.error(f"Error extracting DOCX text: {str(e)}")
            return None

    def is_binary_file(self, filepath: str) -> bool:
        """Check if a file is binary."""
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                file.read()
                return False
        except UnicodeDecodeError:
            return True

    def process_uploaded_file(self, file) -> Tuple[bool, str]:
        """Process uploaded file based on its type."""
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(self.app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            file_ext = filename.rsplit('.', 1)[1].lower()
            text = None
            
            # Extract text based on file type
            if file_ext == 'pdf':
                text = self.extract_pdf_text(filepath)
            elif file_ext in ['doc', 'docx']:
                text = self.extract_docx_text(filepath)
            elif file_ext in ['txt', 'md']:
                with open(filepath, 'r', encoding='utf-8') as f:
                    text = f.read()
            
            if text is None:
                return False, f"Failed to extract text from {file_ext.upper()} file"
                
            # Save extracted text
            text_filepath = os.path.join(self.app.config['UPLOAD_FOLDER'], f"{filename}.txt")
            with open(text_filepath, 'w', encoding='utf-8') as f:
                f.write(text)
                
            # Clean up original file
            try:
                os.remove(filepath)
            except Exception as e:
                self.logger.error(f"Error removing original file: {str(e)}")
                
            return True, text_filepath
            
        except Exception as e:
            self.logger.error(f"Error processing file {file.filename}: {str(e)}")
            return False, str(e)

    # Route handlers
    def index(self):
        """Render the main chat interface."""
        return render_template('index.html')

    def upload_file(self):
        """Handle document uploads and add them to the vector store."""
        try:
            if 'file' not in request.files:
                return jsonify({'error': 'No file part'}), 400
            
            files = request.files.getlist('file')
            uploaded_files = []
            failed_files = []
            
            # Process each uploaded file
            for file in files:
                if file and self.allowed_file(file.filename):
                    success, result = self.process_uploaded_file(file)
                    if success:
                        uploaded_files.append(result)
                    else:
                        failed_files.append(f"{file.filename} ({result})")
                else:
                    failed_files.append(f"{file.filename} (unsupported file type)")
            
            # Handle successful uploads
            if uploaded_files:
                try:
                    self.doc_manager.add_documents(uploaded_files)
                    
                    # Clean up processed text files
                    for filepath in uploaded_files:
                        try:
                            os.remove(filepath)
                        except Exception as e:
                            self.logger.error(f"Error removing file {filepath}: {str(e)}")
                    
                    response = {
                        'message': f'Successfully processed {len(uploaded_files)} files',
                        'processed_files': [Path(f).name.replace('.txt', '') for f in uploaded_files]
                    }
                    if failed_files:
                        response['failed_files'] = failed_files
                    return jsonify(response), 200
                except Exception as e:
                    self.logger.error(f"Error processing documents: {str(e)}")
                    return jsonify({'error': 'Error processing documents', 'details': str(e)}), 500
            
            # Handle failed uploads
            if failed_files:
                return jsonify({
                    'error': 'No valid files were uploaded',
                    'failed_files': failed_files
                }), 400
            
            return jsonify({'error': 'No files were uploaded'}), 400
        
        except Exception as e:
            self.logger.error(f"Unexpected error in upload_file: {str(e)}")
            return jsonify({'error': 'Internal server error', 'details': str(e)}), 500

    def favicon(self):
        """Handle favicon requests."""
        return '', 204

    def handle_message(self, data: Dict[str, str]):
        """Handle incoming socket messages."""
        try:
            query = data.get("message", "").strip()
            if not query:
                emit('bot_response', {'message': 'Please provide a valid question.'})
                return
                
            session_id = data.get("session_id", "default_session")
            
            # Update message counter
            if session_id not in self.message_counters:
                self.message_counters[session_id] = 0
            self.message_counters[session_id] += 1
            
            # Check if feedback should be shown
            show_feedback = self.message_counters[session_id] % 10 == 0
            
            # Process message
            emit('bot_typing', {'status': True})
            
            chat_history = chatbot.get_session_history(session_id)
            response = chatbot.handle_query(
                query=query,
                history=chat_history,
                qa_chain=qa_chain
            )
            
            # Send response
            emit('bot_typing', {'status': False})
            emit('bot_response', {
                'message': response,
                'show_feedback': show_feedback
            })
            chatbot.update_session_history(session_id, query, response)
            
        except Exception as e:
            self.logger.error(f"Socket error: {str(e)}", exc_info=True)
            emit('bot_typing', {'status': False})
            emit('bot_response', {
                'message': 'I apologize, but I encountered an error processing your request.'
            })

    # Error handlers
    def too_large(self, e):
        """Handle file too large error."""
        return jsonify({
            'error': 'File too large',
            'message': f'File size exceeds maximum limit of {self.MAX_CONTENT_LENGTH / (1024 * 1024)}MB'
        }), 413

    def handle_exception(self, e):
        """Handle any unhandled exceptions."""
        self.logger.error(f"Unhandled exception: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'message': 'An unexpected error occurred'
        }), 500

    def start(self):
        """Start the Flask server with Socket.IO integration."""
        try:
            self.logger.info("Starting server...")
            self.socketio.run(self.app, host='0.0.0.0', port=5000, debug=True)
        except Exception as e:
            self.logger.error(f"Error starting server: {str(e)}")
            raise

# Create and run application
if __name__ == '__main__':
    chat_app = FlaskChatApplication()
    chat_app.start()