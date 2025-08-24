
from PyPDF2 import PdfReader
import docx

def extract_text_from_file(file_storage):
    filename = file_storage.filename.lower()
    content = ""
    if filename.endswith(".pdf"):
        reader = PdfReader(file_storage.stream)
        for page in reader.pages:
            content += page.extract_text() or ""
    elif filename.endswith(".docx") or filename.endswith(".doc"):
        document = docx.Document(file_storage)
        for p in document.paragraphs:
            content += p.text + "\n"
    else:
        content = file_storage.stream.read().decode("utf-8", errors="ignore")
    return content
