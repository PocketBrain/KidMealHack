from docx import Document
from pdf2docx import Converter

def convert_pdf_to_docx(pdf_path, docx_path):
    """
    Конвертирует PDF-документ в формат DOCX.

    :param pdf_path: Путь к исходному PDF-файлу.
    :param docx_path: Путь к создаваемому DOCX-файлу.
    """
    cv = Converter(pdf_path)
    cv.convert(docx_path, start=0, end=None)
    cv.close()