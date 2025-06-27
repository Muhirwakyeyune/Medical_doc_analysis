import os
import io
import json
import base64
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Frame, PageTemplate
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import PyPDF2
import requests
import uuid
from docx import Document # Import for handling .docx files

# Define the directory for saving processed files
GENERATED_FILES_DIR = 'generated_files'
# Create the directory if it doesn't already exist
os.makedirs(GENERATED_FILES_DIR, exist_ok=True)

def extract_text_from_pdf(pdf_path):
    """
    Extracts text content from a PDF document using PyPDF2.
    """
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            # Iterate through each page and extract text
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text() or "" # Append extracted text, handle None for empty pages
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return None

def extract_text_from_docx(docx_path):
    """
    Extracts text content from a .docx document using python-docx.
    """
    text = ""
    try:
        document = Document(docx_path)
        # Iterate through each paragraph in the document and append its text
        for paragraph in document.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        print(f"Error extracting text from DOCX: {e}")
        return None

def convert_pdf_to_images(pdf_path):
    """
    Placeholder for converting PDF pages to images.
    NOTE: This function is not fully implemented here as it typically requires 'pdf2image'
    and 'poppler-utils' system dependencies, which are outside the scope of basic Python installs.
    For this project, if a PDF is uploaded, its text content is primarily used for Gemini analysis.
    """
    images_base64 = []
    # No direct image conversion from PDF pages for simplicity without external binaries.
    return images_base64

def get_gemini_api_key():
    """
    Retrieves the Gemini API key from the 'config.json' file.
    This is a common practice for managing API keys in development environments.
    """
    # Get the directory where step1.py is located to find config.json
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, 'config.json')
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            api_key = config.get("GEMINI_API_KEY", "") # Get the key, default to empty string if not found
            if not api_key:
                print("Warning: 'GEMINI_API_KEY' not found in config.json. Please ensure it's present.")
            return api_key
    except FileNotFoundError:
        print(f"Error: config.json not found at {config_path}. Please create it with your API key.")
        return ""
    except json.JSONDecodeError:
        print(f"Error: Could not decode config.json at {config_path}. Check JSON format.")
        return ""
    except Exception as e:
        print(f"An unexpected error occurred while reading config.json: {e}")
        return ""

def call_gemini_api(prompt_text, file_content_base64=None, mime_type=None):
    """
    Makes a call to the Google Gemini API to extract structured information.
    It can handle text-only prompts or multimodal prompts with image data.
    """
    api_key = get_gemini_api_key()
    if not api_key:
        # Return an error if the API key is not available
        return {"error": "Gemini API Key is missing. Please ensure it's in config.json."}

    # Gemini API endpoint for gemini-2.0-flash model
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"

    chat_history = []
    parts = [{"text": prompt_text}] # Initial part of the prompt is text

    # If image data is provided, add it to the parts for multimodal input
    if file_content_base64 and mime_type:
        parts.append({
            "inlineData": {
                "mimeType": mime_type,
                "data": file_content_base64
            }
        })

    chat_history.append({"role": "user", "parts": parts}) # Add user message to chat history

    # Define the JSON schema for the expected structured output from Gemini
    # This guides the model to return data in a predictable format
    response_schema = {
        "type": "OBJECT",
        "properties": {
            "fullName": {"type": "STRING", "description": "Patient's full name."},
            "dateOfBirth": {"type": "STRING", "description": "Patient's date of birth (e.g.,YYYY-MM-DD)."},
            "gender": {"type": "STRING", "description": "Patient's gender."},
            "address": {"type": "STRING", "description": "Patient's full address."},
            "phoneNumber": {"type": "STRING", "description": "Patient's phone number."},
            "email": {"type": "STRING", "description": "Patient's email address."},
            "insuranceProvider": {"type": "STRING", "description": "Patient's insurance provider."},
            "policyNumber": {"type": "STRING", "description": "Patient's insurance policy number."},
            "medicalRecordNumber": {"type": "STRING", "description": "Patient's medical record number or patient ID."},
            "serviceRequestDetails": {"type": "STRING", "description": "Details about ordered tests or procedures."},
            "specimenInfo": {"type": "STRING", "description": "Information about collected specimen(s), including collection/received times and types."},
            "coverageInfo": {"type": "STRING", "description": "Patient's insurance coverage information."},
            "encounterInfo": {"type": "STRING", "description": "Clinical context, including diagnoses."},
            "observationInfo": {"type": "STRING", "description": "Specific clinical findings like LMP (Last Menstrual Period) or pregnancy status."},
            "procedureHistory": {"type": "STRING", "description": "Patient's surgical or medical procedure history."},
            "medicationRequestHistory": {"type": "STRING", "description": "Patient's hormone therapy or other medication request history."},
            "otherRelevantInfo": {"type": "STRING", "description": "Any other key demographic or clinical information found, summarized."}
        },
        "propertyOrdering": [ # Defines the preferred order of fields in the output JSON
            "fullName", "dateOfBirth", "gender", "address", "phoneNumber", "email",
            "insuranceProvider", "policyNumber", "medicalRecordNumber",
            "serviceRequestDetails", "specimenInfo", "coverageInfo", "encounterInfo",
            "observationInfo", "procedureHistory", "medicationRequestHistory",
            "otherRelevantInfo"
        ]
    }

    # Construct the payload for the API request
    payload = {
        "contents": chat_history,
        "generationConfig": {
            "responseMimeType": "application/json", # Requesting JSON output
            "responseSchema": response_schema # Providing the schema for structured response
        }
    }

    headers = {'Content-Type': 'application/json'} # Set content type for JSON payload

    try:
        # Send POST request to Gemini API
        response = requests.post(api_url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
        result = response.json() # Parse the JSON response from the API

        # Extract the text content (which is a JSON string) from the API response
        if result.get("candidates") and result["candidates"][0].get("content") and \
           result["candidates"][0]["content"].get("parts") and result["candidates"][0]["content"]["parts"][0].get("text"):
            json_string = result["candidates"][0]["content"]["parts"][0]["text"]
            parsed_json = json.loads(json_string) # Parse the JSON string into a Python dictionary
            return parsed_json
        else:
            print("Gemini API did not return structured content or expected format.")
            return {"error": "Failed to extract structured data from Gemini."}
    except requests.exceptions.HTTPError as err:
        print(f"HTTP error occurred during Gemini API call: {err} - Response: {err.response.text}")
        return {"error": f"Gemini API HTTP Error: {err.response.text}"}
    except requests.exceptions.ConnectionError as err:
        print(f"Connection error occurred during Gemini API call: {err}")
        return {"error": "Gemini API Connection Error."}
    except json.JSONDecodeError as err:
        print(f"JSON decode error from Gemini API response: {err} - Response: {response.text}")
        return {"error": "Failed to decode JSON from Gemini API response."}
    except Exception as e:
        print(f"An unexpected error occurred during Gemini API call: {e}")
        return {"error": f"Unexpected error with Gemini API: {e}"}

def process_document(file_path, original_filename):
    """
    Orchestrates the document processing workflow:
    1. Determines file type.
    2. Extracts text/prepares image data.
    3. Calls Gemini API for data extraction.
    4. Saves extracted data to a JSON file.
    5. Generates a PDF report.
    6. Returns status and filenames.
    """
    file_extension = original_filename.split('.')[-1].lower()
    demographics_data = {}
    extracted_text = None
    file_base64 = None
    mime_type = None

    # Base prompt for Gemini API, requesting specific structured information
    prompt_base = (
        "Extract the following demographic and clinical information from this medical requisition document. "
        "Provide the output as a JSON object with these fields: "
        "'fullName', 'dateOfBirth', 'gender', 'address', 'phoneNumber', 'email', "
        "'insuranceProvider', 'policyNumber', 'medicalRecordNumber', "
        "'serviceRequestDetails', 'specimenInfo', 'coverageInfo', 'encounterInfo', "
        "'observationInfo', 'procedureHistory', 'medicationRequestHistory', 'otherRelevantInfo'. "
        "If a field is not found or not applicable, set its value to 'NA'. "
        "Focus on patient demographics and then medical information. "
    )

    try:
        if file_extension in ['png', 'jpg', 'jpeg']:
            # Handle image files: read, base64 encode, set MIME type
            with open(file_path, 'rb') as f:
                file_base64 = base64.b64encode(f.read()).decode('utf-8')
            mime_type = f'image/{file_extension}'
            prompt = prompt_base # Image input uses the base prompt
            demographics_data = call_gemini_api(prompt, file_base64, mime_type)

        elif file_extension == 'pdf':
            # Handle PDF files: extract text, then pass text to Gemini
            extracted_text = extract_text_from_pdf(file_path)
            if extracted_text:
                prompt = f"{prompt_base} Document Content:\n{extracted_text}" # Text appended to base prompt
                demographics_data = call_gemini_api(prompt) # No image data for text-only analysis
            else:
                return {"error": "Failed to extract text from PDF or PDF is empty."}
        
        elif file_extension == 'docx': # Handle .docx files: extract text, then pass text to Gemini
            extracted_text = extract_text_from_docx(file_path)
            if extracted_text:
                prompt = f"{prompt_base} Document Content:\n{extracted_text}"
                demographics_data = call_gemini_api(prompt)
            else:
                return {"error": "Failed to extract text from DOCX or DOCX is empty."}

        else:
            # Return error for unsupported file types
            return {"error": "Unsupported file type. Please upload a PDF, image (PNG, JPG, JPEG), or DOCX."}

        # Check if demographics data was successfully extracted without errors
        if demographics_data and not demographics_data.get("error"):
            # Determine filenames for output JSON and PDF based on patient's full name
            full_name = demographics_data.get('fullName', 'NA').replace(' ', '_').replace('/', '_')
            if full_name == 'NA' or not full_name.strip():
                # If name not found, use a UUID for unique filenames
                json_filename = f"demographics_{uuid.uuid4().hex}.json"
                pdf_filename = f"demographics_{uuid.uuid4().hex}.pdf"
            else:
                json_filename = f"{full_name}.json"
                pdf_filename = f"{full_name}.pdf"

            # Construct full paths for saving files
            json_path = os.path.join(GENERATED_FILES_DIR, json_filename)
            pdf_path_output = os.path.join(GENERATED_FILES_DIR, pdf_filename)

            # Save extracted demographics to a JSON file
            with open(json_path, 'w') as f:
                json.dump(demographics_data, f, indent=4)
            print(f"Demographics JSON saved to: {json_path}")

            # Generate the PDF report from the extracted data
            generate_demographics_pdf(demographics_data, pdf_path_output)
            print(f"Demographics PDF generated at: {pdf_path_output}")

            # Return success message and generated filenames to the frontend
            return {
                "message": "Document processed successfully!",
                "json_filename": json_filename,
                "pdf_filename": pdf_filename
            }
        else:
            # If Gemini API returned an error or no data, propagate that error
            return demographics_data # Contains error message from API call

    except Exception as e:
        print(f"Error during overall document processing: {e}")
        return {"error": f"An error occurred during processing: {e}"}
    finally:
        # Clean up the uploaded temporary file after processing (whether successful or not)
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Cleaned up temporary uploaded file: {file_path}")


def generate_demographics_pdf(data, output_pdf_path):
    """
    Generates a PDF document displaying the extracted demographic and clinical data.
    """
    doc = SimpleDocTemplate(output_pdf_path, pagesize=letter) # Create a new PDF document
    styles = getSampleStyleSheet() # Get standard ReportLab paragraph styles

    # Define custom paragraph styles for consistent formatting in the PDF
    h1_style = ParagraphStyle(
        'h1_custom',
        parent=styles['h1'],
        fontSize=24,
        leading=28,
        alignment=TA_CENTER,
        spaceAfter=20,
        textColor='#1e40af' # Blue color for main title
    )
    h2_style = ParagraphStyle(
        'h2_custom',
        parent=styles['h2'],
        fontSize=18,
        leading=22,
        alignment=TA_LEFT,
        spaceAfter=12,
        textColor='#1f2937' # Dark gray color for section headings
    )
    p_style = ParagraphStyle(
        'p_custom',
        parent=styles['Normal'],
        fontSize=12,
        leading=16,
        spaceAfter=6,
        textColor='#374151' # Medium gray for normal text
    )
    key_style = ParagraphStyle( # Style for the key/label part of data fields
        'key_custom',
        parent=p_style,
        fontName='Helvetica-Bold', # Bold font for labels
        textColor='#1f2937' # Dark gray for labels
    )

    story = [] # List to hold all flowables (content elements) for the PDF

    # Add Title to PDF
    story.append(Paragraph("Patient Demographics Report", h1_style))
    story.append(Spacer(1, 0.2 * inch)) # Add some vertical space

    # --- Basic Information Section ---
    story.append(Paragraph("Basic Information", h2_style))
    # Loop through predefined keys and display names for this section
    for key, display_name in [
        ("fullName", "Full Name"),
        ("dateOfBirth", "Date of Birth"),
        ("gender", "Gender"),
        ("medicalRecordNumber", "Medical Record Number")
    ]:
        value = data.get(key, 'NA') # Get value from data, default to 'NA' if not found
        story.append(Paragraph(f"<font name='Helvetica-Bold'>{display_name}:</font> {value}", p_style))

    story.append(Spacer(1, 0.2 * inch))

    # --- Contact Information Section ---
    story.append(Paragraph("Contact Information", h2_style))
    for key, display_name in [
        ("address", "Address"),
        ("phoneNumber", "Phone Number"),
        ("email", "Email")
    ]:
        value = data.get(key, 'NA')
        story.append(Paragraph(f"<font name='Helvetica-Bold'>{display_name}:</font> {value}", p_style))

    story.append(Spacer(1, 0.2 * inch))

    # --- Insurance Information Section ---
    story.append(Paragraph("Insurance Information", h2_style))
    for key, display_name in [
        ("insuranceProvider", "Provider"),
        ("policyNumber", "Policy Number"),
        ("coverageInfo", "General Coverage Details")
    ]:
        value = data.get(key, 'NA')
        story.append(Paragraph(f"<font name='Helvetica-Bold'>{display_name}:</font> {value}", p_style))

    story.append(Spacer(1, 0.2 * inch))

    # --- Medical Service Details Section ---
    story.append(Paragraph("Medical Service Details", h2_style))
    for key, display_name in [
        ("serviceRequestDetails", "Service Request Details"),
        ("specimenInfo", "Specimen Information")
    ]:
        value = data.get(key, 'NA')
        story.append(Paragraph(f"<font name='Helvetica-Bold'>{display_name}:</font> {value}", p_style))

    story.append(Spacer(1, 0.2 * inch))

    # --- Clinical Context & History Section ---
    story.append(Paragraph("Clinical Context & History", h2_style))
    for key, display_name in [
        ("encounterInfo", "Encounter Information (Diagnoses etc.)"),
        ("observationInfo", "Observation Information (LMP, Pregnancy Status etc.)"),
        ("procedureHistory", "Procedure History"),
        ("medicationRequestHistory", "Medication Request History (Hormone Therapy etc.)")
    ]:
        value = data.get(key, 'NA')
        story.append(Paragraph(f"<font name='Helvetica-Bold'>{display_name}:</font> {value}", p_style))

    story.append(Spacer(1, 0.2 * inch))

    # --- Other Relevant Information Section (only if content exists) ---
    other_info = data.get("otherRelevantInfo", "NA")
    if other_info and other_info != 'NA':
        story.append(Paragraph("Other Relevant Information", h2_style))
        story.append(Paragraph(other_info, p_style))

    try:
        doc.build(story) # Build the PDF document from the story elements
    except Exception as e:
        print(f"Error building PDF: {e}")
        raise # Re-raise any error to ensure it's noticed

if __name__ == '__main__':
    # This block is for testing step1.py directly, not used when run via Flask.
    # When app.py imports step1, this __name__ == '__main__' block is skipped.
    print("This script is intended to be imported and used by app.py.")
    print("You can add test logic here if you want to run it standalone for debugging.")
