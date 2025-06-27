import os
from flask import Flask, request, send_file, jsonify
from werkzeug.utils import secure_filename
import step1 # Import the processing logic from step1.py

app = Flask(__name__) # Initialize the Flask application

# Configuration for uploaded and generated files
UPLOAD_FOLDER = 'uploads' # Directory for temporarily storing uploaded files
GENERATED_FILES_DIR = 'generated_files' # Directory for storing generated JSON and PDF files
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg','docx'} # Allowed file extensions for uploads

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['GENERATED_FILES_DIR'] = GENERATED_FILES_DIR

# Ensure the upload and generated directories exist when the app starts
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GENERATED_FILES_DIR, exist_ok=True)

def allowed_file(filename):
    """
    Checks if the uploaded file's extension is among the allowed types.
    """
    # Check if there's a '.' in the filename and if the part after the last '.' is allowed
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    """
    Route for the application's home page.
    Serves the home.html file to the user's browser.
    """
    return send_file('home.html')

@app.route('/demographics')
def demographics_page():
    """
    Route for the medical demographics processing page.
    Serves the index.html file (the document upload interface).
    """
    return send_file('index.html')

@app.route('/upload', methods=['POST'])
def upload_document():
    """
    Handles file uploads from the frontend.
    It expects a 'document' file in the POST request.
    """
    # Check if the 'document' file is present in the request
    if 'document' not in request.files:
        return jsonify({"error": "No document part in the request"}), 400

    file = request.files['document'] # Get the uploaded file

    # Check if a file was actually selected
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Check if the file type is allowed
    if file and allowed_file(file.filename):
        # Securely save the uploaded file temporarily to prevent path traversal issues
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path) # Save the file to the uploads directory

        # Call the document processing logic defined in step1.py
        # Pass the temporary file path and original filename
        processing_result = step1.process_document(file_path, filename)

        # Check if the processing resulted in an error
        if processing_result.get("error"):
            # If an error occurred, attempt to clean up the uploaded file
            if os.path.exists(file_path):
                os.remove(file_path)
            return jsonify(processing_result), 500 # Return the error message with a 500 status
        else:
            return jsonify(processing_result), 200 # Return success message and filenames with a 200 status
    else:
        # Return error if the file type is not allowed
        return jsonify({"error": "File type not allowed"}), 400

@app.route('/download_pdf', methods=['GET'])
def download_pdf():
    """
    Allows users to download the generated PDF reports.
    It expects a 'filename' query parameter in the GET request.
    """
    filename = request.args.get('filename') # Get the filename from the URL query parameter
    if not filename:
        return jsonify({"error": "Filename not provided"}), 400

    # Construct the full path to the PDF file in the generated_files directory
    pdf_path = os.path.join(app.config['GENERATED_FILES_DIR'], filename)

    # Check if the file exists before attempting to send it
    if os.path.exists(pdf_path):
        # Send the file as an attachment, which prompts the user to download it
        return send_file(pdf_path, as_attachment=True)
    else:
        # Return error if the PDF file is not found
        return jsonify({"error": "PDF not found"}), 404

@app.route('/download_fhir_json', methods=['GET']) # NEW ROUTE FOR FHIR JSON
def download_fhir_json():
    """
    Allows users to download the generated FHIR R4 JSON bundle.
    It expects a 'filename' query parameter in the GET request.
    """
    filename = request.args.get('filename')
    if not filename:
        return jsonify({"error": "FHIR JSON filename not provided"}), 400

    fhir_json_path = os.path.join(app.config['GENERATED_FILES_DIR'], filename)

    if os.path.exists(fhir_json_path):
        # Send the file as an attachment with appropriate MIME type
        return send_file(fhir_json_path, as_attachment=True, mimetype='application/json')
    else:
        return jsonify({"error": "FHIR JSON bundle not found"}), 404


if __name__ == '__main__':
    # This block executes only when app.py is run directly (e.g., python app.py)
    # It starts the Flask development server.
    try:
        app.run(debug=True) # Run in debug mode (useful for development, automatically reloads on code changes)
                            # Set debug=False for production deployment.
    except Exception as e:
        # Catch and print any exceptions that occur during the app's startup phase
        print(f"An error occurred during Flask application startup: {e}")
