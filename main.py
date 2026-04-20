from flask import Flask, request, jsonify, send_file, render_template
from werkzeug.utils import secure_filename
import os
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import sqlite3
from docx import Document
from zipfile import ZipFile
from io import BytesIO
import cv2
import numpy as np
import re
import subprocess

# Initialize the Flask app
app = Flask(__name__)

# Configure upload folder and maximum file size
app.config['UPLOAD_FOLDER'] = 'pdf_search'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # Max 100MB files

# NOTE: Do NOT set pytesseract.tesseract_cmd on Linux/Render
# Tesseract is installed via apt and found automatically in PATH

# Valid file extensions
VALID_EXTENSIONS = ['pdf', 'jpg', 'jpeg', 'png', 'bmp', 'gif', 'doc', 'docx']

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# SQLite Database setup
DATABASE = 'extracted_text_final_new.db'

# Function to initialize the SQLite database and create table
def init_db():
    with sqlite3.connect(DATABASE) as conn:
        conn.execute(''' 
            CREATE TABLE IF NOT EXISTS pdf_text (
                pdf_filename TEXT NOT NULL,
                page_num INTEGER NOT NULL,
                extracted_text TEXT NOT NULL,
                PRIMARY KEY (pdf_filename, page_num)
            )
        ''')

# Function to insert extracted text into the SQLite database
def insert_text(pdf_filename, page_num, extracted_text):
    with sqlite3.connect(DATABASE) as conn:
        conn.execute('''
            INSERT OR REPLACE INTO pdf_text (pdf_filename, page_num, extracted_text)
            VALUES (?, ?, ?)
        ''', (pdf_filename, page_num, extracted_text))

# Function to retrieve extracted text from the SQLite database
def get_extracted_text(pdf_filename, page_num):
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.execute('''
            SELECT extracted_text FROM pdf_text
            WHERE pdf_filename = ? AND page_num = ?
        ''', (pdf_filename, page_num))
        result = cursor.fetchone()
        return result[0] if result else None

# Function to retrieve all extracted text for a specific PDF from the SQLite database
def get_all_extracted_text(pdf_filename):
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.execute('''
            SELECT page_num, extracted_text FROM pdf_text
            WHERE pdf_filename = ?
            ORDER BY page_num
        ''', (pdf_filename,))
        return cursor.fetchall()

# Initialize database
init_db()

@app.route('/')
def index():
    return render_template('index.html')


# Extract text from DOCX
def extract_text_from_docx(file_path):
    doc = Document(file_path)
    text = []
    images = []

    for para in doc.paragraphs:
        text.append(para.text)

    for rel in doc.part.rels.values():
        if "image" in rel.target_ref:
            img_data = rel.target_part._blob
            image = Image.open(BytesIO(img_data))
            images.append(image)

    extracted_text_from_images = []
    for image in images:
        text_from_image = pytesseract.image_to_string(image)
        extracted_text_from_images.append(text_from_image)

    return '\n'.join(text), extracted_text_from_images


# Extract text from DOC using LibreOffice (cross-platform, works on Linux/Render)
def extract_text_from_doc(file_path):
    try:
        out_dir = os.path.dirname(file_path)
        # Convert .doc to .docx using LibreOffice headless
        result = subprocess.run(
            ['libreoffice', '--headless', '--convert-to', 'docx', file_path, '--outdir', out_dir],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode != 0:
            raise Exception(f"LibreOffice conversion failed: {result.stderr}")

        docx_path = os.path.splitext(file_path)[0] + '.docx'
        if not os.path.exists(docx_path):
            raise Exception("Converted .docx file not found after LibreOffice conversion.")

        text, images_text = extract_text_from_docx(docx_path)
        return text, images_text

    except FileNotFoundError:
        raise Exception("LibreOffice is not installed. Cannot process .doc files.")


# Process and extract text from images
def process_image(file_path):
    img = cv2.imread(file_path)
    if img is None:
        raise Exception(f"Could not read image: {file_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)[1]
    cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not cnts:
        # Fallback: OCR the image directly without contour masking
        pil_img = Image.open(file_path)
        return pytesseract.image_to_string(pil_img)

    H, W = img.shape[:2]
    cnt = cnts[0]
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if cv2.contourArea(c) > 100 and (0.7 < w / h < 1.3) and \
           (W / 4 < x + w // 2 < W * 3 / 4) and (H / 4 < y + h // 2 < H * 3 / 4):
            cnt = c
            break

    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, -1)
    dst = cv2.bitwise_and(img, img, mask=mask)

    gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    scanned_file_name = os.path.join(
        app.config['UPLOAD_FOLDER'],
        os.path.splitext(os.path.basename(file_path))[0] + "-Scanned.png"
    )
    cv2.imwrite(scanned_file_name, dst)

    file_text = pytesseract.image_to_string(Image.open(scanned_file_name))
    return file_text


# Extract text from PDF (with OCR)
def extract_text_from_pdf_main(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)

    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=300)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        page_text = pytesseract.image_to_string(img)
        text += page_text

    return text


@app.route('/api/documentsOCR', methods=['POST'])
def api_documentsOCR():
    if 'files' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    files = request.files.getlist('files')
    if not files or all(f.filename == '' for f in files):
        return jsonify({'error': 'No selected files'}), 400

    text_results = []
    file_types = []

    for file in files:
        file_extension = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''

        if file_extension not in VALID_EXTENSIONS:
            return jsonify({'error': 'Invalid file format. Please upload valid files.'}), 400

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        text = ""
        try:
            if file_extension == 'pdf':
                file_types.append("PDF")
                text += extract_text_from_pdf_main(file_path)
            elif file_extension in ['jpg', 'jpeg', 'png', 'bmp', 'gif']:
                file_types.append("Image")
                text = process_image(file_path)
            elif file_extension == 'docx':
                file_types.append("DOCX")
                text, extra = extract_text_from_docx(file_path)
                if extra:
                    text += '\n' + '\n'.join(extra)
            elif file_extension == 'doc':
                file_types.append("DOC")
                text, extra = extract_text_from_doc(file_path)
                if extra:
                    text += '\n' + '\n'.join(extra)

        except Exception as e:
            return jsonify({'error': str(e)}), 500

        if not text.strip():
            return jsonify({'error': 'No text found in the document.'}), 400

        text_results.append(text)

    combined_text = "\n\n".join(text_results)
    return jsonify({'text': combined_text, 'file_types': file_types})


@app.route('/api/view_db', methods=['GET'])
def view_db():
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.execute('SELECT * FROM pdf_text')
        rows = cursor.fetchall()

    data = []
    for row in rows:
        data.append({
            'pdf_filename': row[0],
            'page_num': row[1],
            'extracted_text': row[2]
        })

    return jsonify({'data': data}), 200


if __name__ == '__main__':
    app.run(debug=True)
