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
from pdf2image import convert_from_path # NEW: For converting PDF to images
import datetime # For FHIR timestamps

# Define the directory for saving processed files
GENERATED_FILES_DIR = 'generated_files'
# Create the directory if it doesn't already exist
os.makedirs(GENERATED_FILES_DIR, exist_ok=True)

# Define a threshold for "low text density" to guess if a PDF is scanned
# This is a heuristic; adjust as needed. A very small number means mostly image.
MIN_TEXT_LENGTH_FOR_OCR_ASSUMPTION = 50

def extract_text_from_pdf(pdf_path):
    """
    Extracts text content from a PDF document using PyPDF2.
    Returns the extracted text and a boolean indicating if significant text was found.
    """
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text() or ""
        # Return text and whether a substantial amount of text was extracted
        return text, len(text.strip()) > MIN_TEXT_LENGTH_FOR_OCR_ASSUMPTION
    except Exception as e:
        print(f"Error extracting text from PDF with PyPDF2: {e}")
        return None, False

def extract_text_from_docx(docx_path):
    """
    Extracts text content from a .docx document using python-docx.
    """
    text = ""
    try:
        document = Document(docx_path)
        for paragraph in document.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        print(f"Error extracting text from DOCX: {e}")
        return None

def get_gemini_api_key():
    """
    Retrieves the Gemini API key from the 'config.json' file.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, 'config.json')
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            api_key = config.get("GEMINI_API_KEY", "")
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

def call_gemini_api(prompt_text, file_content_base64=None, mime_type=None, image_parts=None):
    """
    Makes a call to the Google Gemini API to extract structured information.
    It can handle text-only prompts, multimodal prompts with single image data,
    or multimodal prompts with multiple image parts (for scanned PDFs).
    """
    api_key = get_gemini_api_key()
    if not api_key:
        return {"error": "Gemini API Key is missing. Please ensure it's in config.json."}

    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"

    chat_history = []
    parts = [{"text": prompt_text}]

    if file_content_base64 and mime_type: # For single image uploads (JPG/PNG)
        parts.append({
            "inlineData": {
                "mimeType": mime_type,
                "data": file_content_base64
            }
        })
    elif image_parts: # For multiple image pages from PDF conversion
        parts.extend(image_parts) # Add all image parts to the prompt

    chat_history.append({"role": "user", "parts": parts})

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
        "propertyOrdering": [
            "fullName", "dateOfBirth", "gender", "address", "phoneNumber", "email",
            "insuranceProvider", "policyNumber", "medicalRecordNumber",
            "serviceRequestDetails", "specimenInfo", "coverageInfo", "encounterInfo",
            "observationInfo", "procedureHistory", "medicationRequestHistory",
            "otherRelevantInfo"
        ]
    }

    payload = {
        "contents": chat_history,
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": response_schema
        }
    }

    headers = {'Content-Type': 'application/json'}

    try:
        response = requests.post(api_url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        result = response.json()

        if result.get("candidates") and result["candidates"][0].get("content") and \
           result["candidates"][0]["content"].get("parts") and result["candidates"][0]["content"]["parts"][0].get("text"):
            json_string = result["candidates"][0]["content"]["parts"][0]["text"]
            parsed_json = json.loads(json_string)
            return parsed_json
        else:
            print("Gemini API did not return structured content or expected format.")
            return {"error": "Failed to extract structured data from Gemini."}
    except requests.exceptions.HTTPError as err:
        print(f"HTTP error occurred during Gemini API call: {err} - {err.response.text}")
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

def construct_fhir_bundle(data):
    """
    Constructs a simplified FHIR R4 Bundle JSON from the extracted demographic and clinical data.
    This is a basic construction and does not cover all FHIR complexities or validations.
    """
    get_val = lambda key: data.get(key, "NA")

    patient_resource = {
        "resourceType": "Patient",
        "id": str(uuid.uuid4()),
        "meta": {
            "profile": ["http://hl7.org/fhir/StructureDefinition/Patient"]
        },
        "identifier": [],
        "name": [],
        "gender": get_val("gender").lower() if get_val("gender") != "NA" else "unknown",
        "birthDate": get_val("dateOfBirth") if get_val("dateOfBirth") != "NA" else None,
        "address": [],
        "telecom": []
    }

    if get_val("fullName") != "NA":
        parts = get_val("fullName").split(" ")
        family = parts[-1] if len(parts) > 1 else get_val("fullName")
        given = parts[:-1] if len(parts) > 1 else []
        patient_resource["name"].append({
            "use": "official",
            "family": family,
            "given": given
        })

    if get_val("medicalRecordNumber") != "NA":
        patient_resource["identifier"].append({
            "use": "usual",
            "type": {
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/v2-0203",
                    "code": "MR"
                }]
            },
            "value": get_val("medicalRecordNumber")
        })

    if get_val("address") != "NA":
        patient_resource["address"].append({
            "use": "home",
            "text": get_val("address")
        })

    if get_val("phoneNumber") != "NA":
        patient_resource["telecom"].append({
            "system": "phone",
            "value": get_val("phoneNumber"),
            "use": "home"
        })

    if get_val("email") != "NA":
        patient_resource["telecom"].append({
            "system": "email",
            "value": get_val("email"),
            "use": "home"
        })

    coverage_resource = None
    if get_val("insuranceProvider") != "NA" or get_val("policyNumber") != "NA" or get_val("coverageInfo") != "NA":
        coverage_resource = {
            "resourceType": "Coverage",
            "id": str(uuid.uuid4()),
            "meta": {
                "profile": ["http://hl7.org/fhir/StructureDefinition/Coverage"]
            },
            "status": "active",
            "type": {
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/v3-ActCode",
                    "code": "RDP"
                }],
                "text": "Medical Plan"
            },
            "beneficiary": {
                "reference": f"Patient/{patient_resource['id']}"
            }
        }
        if get_val("policyNumber") != "NA":
            coverage_resource["identifier"] = [{
                "system": "http://example.org/fhir/sid/policy-number",
                "value": get_val("policyNumber")
            }]
        if get_val("insuranceProvider") != "NA":
            coverage_resource["payor"] = [{
                "display": get_val("insuranceProvider")
            }]
        if get_val("coverageInfo") != "NA":
             if "extension" not in coverage_resource:
                 coverage_resource["extension"] = []
             coverage_resource["extension"].append({
                 "url": "http://example.org/fhir/StructureDefinition/coverage-info",
                 "valueString": get_val("coverageInfo")
             })

    service_request_resource = None
    if get_val("serviceRequestDetails") != "NA":
        service_request_resource = {
            "resourceType": "ServiceRequest",
            "id": str(uuid.uuid4()),
            "meta": {
                "profile": ["http://hl7.org/fhir/StructureDefinition/ServiceRequest"]
            },
            "status": "active",
            "intent": "order",
            "subject": {
                "reference": f"Patient/{patient_resource['id']}"
            },
            "requester": {
                "display": "Extracted from Document"
            },
            "code": {
                "coding": [{
                    "system": "http://snomed.info/sct",
                    "code": "394581000",
                    "display": "Medical service"
                }],
                "text": get_val("serviceRequestDetails")
            },
             "authoredOn": datetime.datetime.now(datetime.timezone.utc).isoformat()
        }

    specimen_resource = None
    if get_val("specimenInfo") != "NA":
        specimen_resource = {
            "resourceType": "Specimen",
            "id": str(uuid.uuid4()),
            "meta": {
                "profile": ["http://hl7.org/fhir/StructureDefinition/Specimen"]
            },
            "status": "available",
            "subject": {
                "reference": f"Patient/{patient_resource['id']}"
            },
            "type": {
                "text": get_val("specimenInfo")
            },
            "collection": {
                "collectedDateTime": datetime.datetime.now(datetime.timezone.utc).isoformat()
            }
        }

    encounter_resource = None
    if get_val("encounterInfo") != "NA":
        encounter_resource = {
            "resourceType": "Encounter",
            "id": str(uuid.uuid4()),
            "meta": {
                "profile": ["http://hl7.org/fhir/StructureDefinition/Encounter"]
            },
            "status": "finished",
            "class": {
                "system": "http://terminology.hl7.org/CodeSystem/v3-ActCode",
                "code": "AMB",
                "display": "ambulatory"
            },
            "subject": {
                "reference": f"Patient/{patient_resource['id']}"
            },
            "period": {
                "start": datetime.datetime.now(datetime.timezone.utc).isoformat()
            },
            "reasonCode": [{
                "text": get_val("encounterInfo")
            }]
        }

    observation_resource = None
    if get_val("observationInfo") != "NA":
        observation_resource = {
            "resourceType": "Observation",
            "id": str(uuid.uuid4()),
            "meta": {
                "profile": ["http://hl7.org/fhir/StructureDefinition/Observation"]
            },
            "status": "final",
            "code": {
                "text": get_val("observationInfo")
            },
            "subject": {
                "reference": f"Patient/{patient_resource['id']}"
            },
            "effectiveDateTime": datetime.datetime.now(datetime.timezone.utc).isoformat()
        }

    procedure_resource = None
    if get_val("procedureHistory") != "NA":
        procedure_resource = {
            "resourceType": "Procedure",
            "id": str(uuid.uuid4()),
            "meta": {
                "profile": ["http://hl7.org/fhir/StructureDefinition/Procedure"]
            },
            "status": "completed",
            "code": {
                "text": get_val("procedureHistory")
            },
            "subject": {
                "reference": f"Patient/{patient_resource['id']}"
            },
            "performedDateTime": datetime.datetime.now(datetime.timezone.utc).isoformat()
        }
    
    medication_request_resource = None
    if get_val("medicationRequestHistory") != "NA":
        medication_request_resource = {
            "resourceType": "MedicationRequest",
            "id": str(uuid.uuid4()),
            "meta": {
                "profile": ["http://hl7.org/fhir/StructureDefinition/MedicationRequest"]
            },
            "status": "active",
            "intent": "order",
            "medicationCodeableConcept": {
                "text": get_val("medicationRequestHistory")
            },
            "subject": {
                "reference": f"Patient/{patient_resource['id']}"
            },
            "authoredOn": datetime.datetime.now(datetime.timezone.utc).isoformat()
        }

    bundle_entry = [
        {
            "fullUrl": f"urn:uuid:{patient_resource['id']}",
            "resource": patient_resource
        }
    ]
    if coverage_resource:
        bundle_entry.append({
            "fullUrl": f"urn:uuid:{coverage_resource['id']}",
            "resource": coverage_resource
        })
    if service_request_resource:
        bundle_entry.append({
            "fullUrl": f"urn:uuid:{service_request_resource['id']}",
            "resource": service_request_resource
        })
    if specimen_resource:
        bundle_entry.append({
            "fullUrl": f"urn:uuid:{specimen_resource['id']}",
            "resource": specimen_resource
        })
    if encounter_resource:
        bundle_entry.append({
            "fullUrl": f"urn:uuid:{encounter_resource['id']}",
            "resource": encounter_resource
        })
    if observation_resource:
        bundle_entry.append({
            "fullUrl": f"urn:uuid:{observation_resource['id']}",
            "resource": observation_resource
        })
    if procedure_resource:
        bundle_entry.append({
            "fullUrl": f"urn:uuid:{procedure_resource['id']}",
            "resource": procedure_resource
        })
    if medication_request_resource:
        bundle_entry.append({
            "fullUrl": f"urn:uuid:{medication_request_resource['id']}",
            "resource": medication_request_resource
        })

    fhir_bundle = {
        "resourceType": "Bundle",
        "id": str(uuid.uuid4()),
        "meta": {
            "lastUpdated": datetime.datetime.now(datetime.timezone.utc).isoformat()
        },
        "type": "collection",
        "entry": bundle_entry
    }

    return fhir_bundle

def process_document(file_path, original_filename):
    """
    Orchestrates the document processing workflow:
    1. Determines file type.
    2. Extracts text/prepares image data or converts PDF pages to images.
    3. Calls Gemini API for data extraction.
    4. Saves extracted data to a JSON file.
    5. Generates a FHIR R4 JSON bundle.
    6. Generates a PDF report.
    7. Returns status and filenames.
    """
    file_extension = original_filename.split('.')[-1].lower()
    demographics_data = {}
    extracted_text = None
    file_base64 = None
    mime_type = None
    image_parts_for_gemini = [] # To hold multiple image parts for Gemini (e.g., from PDF pages)

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
            # Handle single image files: read, base64 encode, set MIME type
            with open(file_path, 'rb') as f:
                file_base64 = base64.b64encode(f.read()).decode('utf-8')
            mime_type = f'image/{file_extension}'
            prompt = prompt_base
            demographics_data = call_gemini_api(prompt, file_content_base64=file_base64, mime_type=mime_type)

        elif file_extension == 'pdf':
            # Try extracting text first
            pdf_text, has_significant_text = extract_text_from_pdf(file_path)

            if has_significant_text:
                # If text is substantial, send text to Gemini (for text-searchable PDFs)
                prompt = f"{prompt_base} Document Content:\n{pdf_text}"
                demographics_data = call_gemini_api(prompt)
            else:
                # If text is minimal or absent (likely a scanned PDF), convert pages to images
                print("No significant text found in PDF. Attempting image-based OCR via pdf2image...")
                try:
                    # convert_from_path requires Poppler to be installed on the system
                    # This will convert each page of the PDF into a PIL Image object
                    images = convert_from_path(file_path)
                    
                    # Prepare each image as an inlineData part for Gemini
                    for i, img in enumerate(images):
                        buffered = io.BytesIO()
                        img.save(buffered, format="PNG") # Save image to a byte buffer
                        image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                        image_parts_for_gemini.append({
                            "inlineData": {
                                "mimeType": "image/png",
                                "data": image_base64
                            }
                        })
                        # Limit pages sent to Gemini to avoid context window issues for very large PDFs
                        if i >= 4: # Process first 5 pages (0-indexed) to avoid very long requests
                            print("Limiting scanned PDF processing to the first 5 pages.")
                            break
                    
                    if not image_parts_for_gemini:
                        return {"error": "Failed to convert PDF pages to images. Ensure Poppler is installed and accessible."}

                    # Send the prompt with multiple image parts to Gemini
                    demographics_data = call_gemini_api(prompt_base, image_parts=image_parts_for_gemini)

                except Exception as e:
                    print(f"Error converting PDF to images or sending to Gemini: {e}")
                    return {"error": f"Failed to process scanned PDF: {e}. Ensure 'Poppler' is installed and its path is configured if needed."}
        
        elif file_extension == 'docx':
            extracted_text = extract_text_from_docx(file_path)
            if extracted_text:
                prompt = f"{prompt_base} Document Content:\n{extracted_text}"
                demographics_data = call_gemini_api(prompt)
            else:
                return {"error": "Failed to extract text from DOCX or DOCX is empty."}

        else:
            return {"error": "Unsupported file type. Please upload a PDF, image (PNG, JPG, JPEG), or DOCX."}

        if demographics_data and not demographics_data.get("error"):
            full_name = demographics_data.get('fullName', 'NA').strip()
            
            # --- Updated filename logic for 'no_name.pdf/json/fhir.json' ---
            if full_name.lower() == 'na' or not full_name:
                base_filename = "no_name"
            else:
                # Sanitize the name to create a valid filename: keep alphanumeric and underscores
                sanitized_name = ''.join(c if c.isalnum() else '_' for c in full_name)
                # Replace multiple underscores with a single one, and strip leading/trailing
                sanitized_name = '_'.join(filter(None, sanitized_name.split('_')))
                if not sanitized_name:
                    sanitized_name = "demographics_unknown" # Fallback if sanitization results in empty string
                base_filename = sanitized_name
            # --- End filename logic ---

            json_filename = f"{base_filename}.json"
            pdf_filename = f"{base_filename}.pdf"
            fhir_json_filename = f"{base_filename}_fhir.json"

            json_path = os.path.join(GENERATED_FILES_DIR, json_filename)
            pdf_path_output = os.path.join(GENERATED_FILES_DIR, pdf_filename)
            fhir_json_path = os.path.join(GENERATED_FILES_DIR, fhir_json_filename)

            with open(json_path, 'w') as f:
                json.dump(demographics_data, f, indent=4)
            print(f"Demographics JSON saved to: {json_path}")

            fhir_bundle = construct_fhir_bundle(demographics_data)
            with open(fhir_json_path, 'w') as f:
                json.dump(fhir_bundle, f, indent=4)
            print(f"FHIR R4 JSON bundle generated and saved to: {fhir_json_path}")

            generate_demographics_pdf(demographics_data, pdf_path_output)
            print(f"Demographics PDF generated at: {pdf_path_output}")

            return {
                "message": "Document processed successfully!",
                "json_filename": json_filename,
                "pdf_filename": pdf_filename,
                "fhir_json_filename": fhir_json_filename
            }
        else:
            return demographics_data

    except Exception as e:
        print(f"Error during overall document processing: {e}")
        return {"error": f"An error occurred during processing: {e}"}
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Cleaned up temporary uploaded file: {file_path}")


def generate_demographics_pdf(data, output_pdf_path):
    """
    Generates a PDF document displaying the extracted demographic and clinical data.
    """
    doc = SimpleDocTemplate(output_pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()

    h1_style = ParagraphStyle(
        'h1_custom',
        parent=styles['h1'],
        fontSize=24,
        leading=28,
        alignment=TA_CENTER,
        spaceAfter=20,
        textColor='#1e40af'
    )
    h2_style = ParagraphStyle(
        'h2_custom',
        parent=styles['h2'],
        fontSize=18,
        leading=22,
        alignment=TA_LEFT,
        spaceAfter=12,
        textColor='#1f2937'
    )
    p_style = ParagraphStyle(
        'p_custom',
        parent=styles['Normal'],
        fontSize=12,
        leading=16,
        spaceAfter=6,
        textColor='#374151'
    )
    key_style = ParagraphStyle(
        'key_custom',
        parent=p_style,
        fontName='Helvetica-Bold',
        textColor='#1f2937'
    )

    story = []

    story.append(Paragraph("Patient Demographics and Clinical Report", h1_style))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("Basic Information", h2_style))
    for key, display_name in [
        ("fullName", "Full Name"),
        ("dateOfBirth", "Date of Birth"),
        ("gender", "Gender"),
        ("medicalRecordNumber", "Medical Record Number")
    ]:
        value = data.get(key, 'NA')
        story.append(Paragraph(f"<font name='Helvetica-Bold'>{display_name}:</font> {value}", p_style))

    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("Contact Information", h2_style))
    for key, display_name in [
        ("address", "Address"),
        ("phoneNumber", "Phone Number"),
        ("email", "Email")
    ]:
        value = data.get(key, 'NA')
        story.append(Paragraph(f"<font name='Helvetica-Bold'>{display_name}:</font> {value}", p_style))

    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("Insurance Information", h2_style))
    for key, display_name in [
        ("insuranceProvider", "Provider"),
        ("policyNumber", "Policy Number"),
        ("coverageInfo", "General Coverage Details")
    ]:
        value = data.get(key, 'NA')
        story.append(Paragraph(f"<font name='Helvetica-Bold'>{display_name}:</font> {value}", p_style))

    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("Medical Service Details", h2_style))
    for key, display_name in [
        ("serviceRequestDetails", "Service Request Details"),
        ("specimenInfo", "Specimen Information")
    ]:
        value = data.get(key, 'NA')
        story.append(Paragraph(f"<font name='Helvetica-Bold'>{display_name}:</font> {value}", p_style))

    story.append(Spacer(1, 0.2 * inch))

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

    other_info = data.get("otherRelevantInfo", "NA")
    if other_info and other_info != 'NA':
        story.append(Paragraph("Other Relevant Information", h2_style))
        story.append(Paragraph(other_info, p_style))

    try:
        doc.build(story)
    except Exception as e:
        print(f"Error building PDF: {e}")
        raise

if __name__ == '__main__':
    print("This script is intended to be imported and used by app.py.")
