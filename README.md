Medical Document Processor

This project develops a functional proof-of-concept web service that automates the extraction of data from medical requisition documents (PDFs, images, and DOCX files) and transforms it into structured demographic and clinical information. It leverages the Gemini API for robust data extraction and generates a user-friendly PDF report.

Features

Document Upload: Web interface to upload medical requisition documents in PDF, PNG, JPG, JPEG, and DOCX formats.

Gemini API Integration: Utilizes Google's Gemini 2.0 Flash model for advanced Optical Character Recognition (OCR) and intelligent data extraction.

Structured Data Extraction: Extracts key demographic information (full name, date of birth, gender, address, contact details, medical record number) and clinical context (service requests, specimen info, coverage, encounter details, observations, procedures, medication history).

JSON Output: Saves the extracted structured data as a JSON file, named after the patient's full name.

PDF Report Generation: Creates a well-designed, readable PDF document summarizing all extracted demographic and clinical information.

PDF Download: Allows users to download the generated PDF report directly from the web interface.

Two-Page Interface: A welcome page to navigate to the main document processing utility.

Technologies Used

Backend: Flask (Python Web Framework)

OCR & AI: Google Gemini 2.0 Flash API

Frontend: HTML, Tailwind CSS (for styling), JavaScript

Document Processing (Python):

PyPDF2: For reading PDF files.

python-docx: For reading DOCX files.

Pillow (PIL): For basic image handling (though Gemini handles image OCR).

reportlab: For generating PDF reports.

Server Gateway Interface: gunicorn (for production deployment hint in Procfile)

Project Structure

medical-doc-processor/
├── app.py                  # Main Flask application file
├── home.html               # Welcome page with navigation button
├── index.html              # Document upload and processing page
├── step1.py                # Core logic for document processing (Gemini call, PDF gen)
├── config.json             # Stores the Gemini API key securely (for local run)
├── uploads/                # Directory for temporary uploaded files (created automatically)
└── generated_files/        # Directory for saving JSON and PDF outputs (created automatically)


Setup Instructions

Follow these steps to get the project up and running on your local machine.

1. Clone the Repository (or create files manually)

If you have a Git repository:

git clone <your-repository-url>
cd medical-doc-processor


If you're creating files manually, ensure you have the medical-doc-processor folder and all *.py and *.html files inside it, as provided in the previous responses.

2. Create and Activate a Virtual Environment

It's highly recommended to use a virtual environment to manage project dependencies.

python3 -m venv venv
# OR, if 'python3' is not found, try:
# python -m venv venv


Activate the virtual environment:

On macOS / Linux:

source venv/bin/activate


On Windows (Command Prompt):

.\venv\Scripts\activate.bat


On Windows (PowerShell):

.\venv\Scripts\Activate.ps1


You'll see (venv) appear at the beginning of your terminal prompt, indicating the environment is active.

3. Install Required Python Libraries

With your virtual environment activated, install all necessary packages:

pip install Flask requests Pillow reportlab PyPDF2 python-docx gunicorn


(gunicorn is included here for potential future deployment, but not strictly needed for local app.run)

4. Configure Gemini API Key

You need a Gemini API key to use the model.

Get your API Key: Go to Google AI Studio and generate a new API key.

Create config.json: In your medical-doc-processor root directory, create a file named config.json and add your API key to it:

{
    "GEMINI_API_KEY": "YOUR_ACTUAL_GEMINI_API_KEY_HERE"
}


Replace "YOUR_ACTUAL_GEMINI_API_KEY_HERE" with the key you obtained from Google AI Studio.

Security Note: For production deployments or public repositories, avoid committing config.json directly. Instead, use environment variables on your hosting platform. For local development, this config.json approach is convenient.

How to Run the Application

With your virtual environment activated and config.json set up, run the Flask application:

python app.py


You should see output indicating that the Flask development server is running, typically on http://127.0.0.1:5000/.

Usage

Access the Application: Open your web browser and go to http://127.0.0.1:5000/.

Welcome Page: You will land on the home.html welcome page.

Navigate to Demographics: Click the "Go to Medical Demographics" button. This will take you to the index.html page.

Upload Document: On the "Medical Document Processor" page, click the "Click to upload" area or drag and drop a medical requisition document (PDF, PNG, JPG, or DOCX).

Process Document: Click the "Process Document" button. The application will send the document to the Gemini API for analysis.

View Status & Download:

A status message will indicate if the processing was successful or if an error occurred.

If successful, a "Download Demographics PDF" button will appear. Click it to download the generated PDF report containing the extracted information.

A JSON file with the extracted data will also be saved in the generated_files/ directory on your server.

Important Notes

Ephemeral File System on Free Hosting: If you plan to host this application on a free tier platform (like Render's free web service), be aware that the uploads/ and generated_files/ directories will likely not persist data across restarts or "sleep" cycles. For persistent storage in a deployed environment, consider integrating with cloud storage solutions (e.g., Google Cloud Storage, AWS S3).

Gemini API Quotas: Be mindful of the free tier quotas for the Gemini API. Excessive usage might incur costs or hit rate limits.

Accuracy: The accuracy of data extraction depends heavily on the quality, format, and content of the uploaded documents, as well as the capabilities of the Gemini model.

PDF to Image Conversion: The current step1.py extracts text from PDFs and DOCX files. For rich image-based OCR from PDF pages (e.g., if the PDF is just an image scan), you would typically need to integrate with a library like pdf2image (which requires poppler-utils system dependency) or an OCR service that handles PDFs directly. The current implementation relies on Gemini's multimodal capabilities for images.