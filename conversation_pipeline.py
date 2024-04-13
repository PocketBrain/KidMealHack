from core.document_conversion import extract_images
from llm.utiils import pipeline
from core.utilities import  preprocess_image
from core.document_generator import generate_output_text
import numpy as np
from PIL import Image
import io

def get_data(files: np.array = None):
    # Удалить путь, и поставить на вход np.array
    if files == None:
        file_path = "test_files/cola.jpg"
        image_np = np.array(Image.open(file_path))
    else:
        image_np = files
    image_np = preprocess_image(image_np)
    image_np = (image_np * 255).astype(np.uint8)
    with io.BytesIO() as output:
        Image.fromarray(image_np).save(output, format='PNG')
        files = output.getvalue()
    return files

def result_pipeline(files: np.array) -> str:
    files = get_data()
    images = extract_images(files=[files])
    text_from_ocr = generate_output_text(images)
    rules, json = pipeline(text_from_ocr)
    return json, rules

if __name__ == '__main__':
    print(result_pipeline(None))
