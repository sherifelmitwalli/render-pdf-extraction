# app.py
import os
import base64
import tempfile
import warnings
import shutil
import logging
from io import BytesIO
from pathlib import Path
from typing import List, Any

from flask import Flask, request, jsonify, send_file
from PIL import Image
from pdf2image import convert_from_path
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential
from openai import OpenAI

# --- Configuration & Logging ---
import logging
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS  # Import CORS

logging.basicConfig(level=logging.INFO)
app = Flask(__name__)
# Enable CORS with explicit settings
CORS(app, resources={
    r"/*": {
        "origins": "*",  # Allow all origins in development
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Accept", "Origin"],
        "expose_headers": ["Content-Type", "X-Request-ID"],
        "supports_credentials": False,
        "max_age": 600  # Cache preflight requests for 10 minutes
    }
})

# Add CORS headers to all responses
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Accept,Origin')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return response

# Log all requests for debugging
@app.before_request
def log_request_info():
    logging.info('Headers: %s', dict(request.headers))
    logging.info('Body: %s', request.get_data())


# Environment configuration (set these in Render’s environment variables)
DEEPINFRA_API_KEY = os.environ.get("DEEPINFRA_API_KEY")
MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.2-11B-Vision-Instruct")
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.deepinfra.com/v1/openai")

# Constants
MAX_IMAGE_SIZE = 2000
DPI = 200
MAX_WIDTH = 1700
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Suppress PIL warnings and allow large images
warnings.filterwarnings('ignore', category=Image.DecompressionBombWarning)
Image.MAX_IMAGE_PIXELS = None

# Initialize OpenAI client
try:
    client = OpenAI(
        api_key=DEEPINFRA_API_KEY,
        base_url=API_BASE_URL,
        http_client=httpx.Client(
            timeout=60.0,
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
            follow_redirects=True
        ),
        max_retries=3
    )
except Exception as e:
    logging.error(f"Failed to initialize API client: {str(e)}")
    raise

# --- Helper Functions ---

def resize_image(image: Image.Image, max_size: int = MAX_IMAGE_SIZE) -> Image.Image:
    """Resize image while maintaining aspect ratio."""
    if not isinstance(image, Image.Image):
        raise ValueError("Input must be a PIL Image object")
    try:
        ratio = min(max_size/float(image.size[0]), max_size/float(image.size[1]))
        if ratio < 1:
            new_size = tuple(int(dim * ratio) for dim in image.size)
            return image.resize(new_size, Image.Resampling.LANCZOS)
        return image
    except Exception as e:
        raise Exception(f"Image resize failed: {str(e)}")

def convert_pdf_to_images(pdf_path: str) -> List[Image.Image]:
    """Convert PDF pages to images."""
    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    try:
        pages = convert_from_path(
            pdf_path,
            dpi=DPI,
            size=(MAX_WIDTH, None),
            grayscale=True,
            fmt='jpeg'
        )
        return [resize_image(page) for page in pages]
    except Exception as e:
        raise Exception(f"PDF conversion failed: {str(e)}")

def encode_image(image: Any) -> str:
    """Convert image to a base64 string."""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def describe_image_with_vision(client: OpenAI, image: Any, page_num: int) -> str:
    """Send image to vision model for text extraction."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a text extraction tool. Extract ALL text EXACTLY as shown in the image. "
                        "Do NOT explain, interpret, or provide instructions. Only output the exact text found in the image."
                    )
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "You are a text extraction tool. Your ONLY task is to extract ALL text from this document EXACTLY as it appears, with special attention to headers and tables. "
                                "Follow these STRICT rules:\n\n"
                                "1. **Headers and Page Information**:\n"
                                "   - Always extract headers at the top of pages\n"
                                "   - Include page numbers, dates, or any other metadata\n"
                                "   - Preserve header formatting and position\n"
                                "   - Extract running headers and footers\n\n"
                                "2. **Table Handling**:\n"
                                "   - Extract ALL table content cell by cell\n"
                                "   - Maintain table structure using tabs or spaces\n"
                                "   - Preserve column headers and row labels\n"
                                "   - Keep numerical data exactly as shown\n"
                                "   - Include table borders and separators using ASCII characters\n"
                                "   - Format multi-line cells accurately\n\n"
                                "3. **Exact Text Only**: Extract every character, word, number, symbol, and punctuation mark exactly as it appears. Do NOT:\n"
                                "   - Add any text not present in the document\n"
                                "   - Remove any text present in the document\n"
                                "   - Change any text present in the document\n"
                                "   - Include any commentary, analysis, or interpretation\n\n"
                                "4. **Preserve Formatting**: Maintain the exact:\n"
                                "   - Line breaks and spacing\n"
                                "   - Indentation and alignment\n"
                                "   - Text styles (bold, italics, underline)\n"
                                "   - Font sizes and styles\n"
                                "   - Page layout and structure\n\n"
                                "5. **Order and Structure**:\n"
                                "   - Begin with page headers/metadata\n"
                                "   - Follow the document's natural flow\n"
                                "   - Extract text in reading order (top to bottom, left to right)\n"
                                "   - Preserve paragraph breaks and section spacing\n"
                                "   - Maintain hierarchical structure of headings\n\n"
                                "6. **Table-Specific Output Format**:\n"
                                "   - Use consistent spacing for columns\n"
                                "   - Align numerical data properly\n"
                                "   - Preserve column widths where possible\n"
                                "   - Use ASCII characters for table borders (│, ─, ┌, ┐, └, ┘)\n"
                                "   - Include table captions and notes\n\n"
                                "7. **Special Elements**:\n"
                                "   - Mark footnotes and endnotes appropriately\n"
                                "   - Preserve bullet points and numbered lists\n"
                                "   - Include figure captions and references\n"
                                "   - Extract sidebar content in position\n\n"
                                "8. **Clarity Rules**:\n"
                                "   - Mark unclear text as [UNREADABLE]\n"
                                "   - Indicate merged cells in tables with [MERGED]\n"
                                "   - Note rotated or vertical text with [ROTATED]\n"
                                "   - Flag complex formatting that can't be fully preserved\n\n"
                                "9. **Strict Prohibitions**: Do NOT:\n"
                                "   - Summarize or paraphrase\n"
                                "   - Analyze or interpret content\n"
                                "   - Rearrange table data\n"
                                "   - Skip any text, even if it seems irrelevant\n"
                                "   - Add explanations or descriptions\n"
                                "   - Make assumptions about unclear content\n\n"
                                "10. **Verification**: If the page is blank, return: \"[NO TEXT FOUND]\"\n\n"
                                "Extract the text exactly as it appears in the document:"
                            )
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{encode_image(image)}"}
                        }
                    ]
                }
            ],
            max_tokens=8192,
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        raise Exception(f"API Error: {str(e)}")

def check_dependencies() -> bool:
    """Check if required system dependencies are installed (e.g. poppler)."""
    if not shutil.which('pdftoppm'):
        logging.error("pdftoppm not found. Please install poppler.")
        return False
    return True

# --- API Endpoint ---

@app.route("/extract", methods=["POST", "OPTIONS"])
def extract_text():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200
        
    # For POST requests
    # Check dependencies first
    if not check_dependencies():
        return jsonify({"error": "Required system dependency not installed (poppler)."}), 500

    # Log request details
    logging.info("Processing PDF extraction request")
    logging.info(f"Content-Type: {request.content_type}")
    logging.info(f"Headers: {dict(request.headers)}")

    if "file" not in request.files:
        logging.error("No file in request")
        return jsonify({"error": "No file part in the request."}), 400

    file = request.files["file"]

    if file.filename == "" or not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "A valid PDF file is required."}), 400

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    if file_size > MAX_FILE_SIZE:
        return jsonify({"error": "File size exceeds 10MB limit."}), 400

    # Save the uploaded PDF temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf:
        file.save(temp_pdf)
        temp_pdf_path = temp_pdf.name

    try:
        pages = convert_pdf_to_images(temp_pdf_path)
        if len(pages) == 0:
            return jsonify({"error": "No pages found in PDF."}), 400

        extracted_texts = []
        total_pages = len(pages)
        for i, page in enumerate(pages):
            logging.info(f"Processing page {i + 1}/{total_pages}")
            text = describe_image_with_vision(client, page, i)
            extracted_texts.append(f"=== Page {i + 1} ===\n{text}")

        final_text = "\n\n".join(extracted_texts)

        return jsonify({
            "extracted_text": final_text,
            "page_count": total_pages
        })

    except Exception as e:
        logging.error(f"Error processing PDF: {str(e)}")
        return jsonify({"error": f"Error processing PDF: {str(e)}"}), 500

    finally:
        try:
            os.unlink(temp_pdf_path)
        except Exception:
            pass

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    # Use host=0.0.0.0 for Render
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
