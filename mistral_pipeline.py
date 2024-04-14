from core.document_conversion import extract_images
from llm.utiils import interact
from core.utilities import  preprocess_image
from core.document_generator import generate_output_text
from constant import MODEL_PATH, FROM_TEXT_2_JSON_PROMPT, FROM_JSON_2_RULE_PROMPT
import numpy as np
from PIL import Image
import io

def get_data() -> bytes:
    """
    Получает данные изображения и возвращает их в формате PNG.

    Возвращает:
    bytes: Байтовое представление изображения в формате PNG.
    """
    file_path = "test_files/Йогурт Агуша с персиком с 8 месяцев 2.7% 200 г.jpg"
    image_np = np.array(Image.open(file_path))
    image_np = preprocess_image(image_np)
    image_np = (image_np * 255).astype(np.uint8)
    with io.BytesIO() as output:
        Image.fromarray(image_np).save(output, format='PNG')
        files = output.getvalue()
    return files

def result_pipeline() -> str:
    """
    Выполняет пайплайн обработки изображения и генерации ответа.

    Возвращает:
    str: Сгенерированный ответ.
    """
    files = get_data()
    images = extract_images(files=[files])
    text_from_ocr = generate_output_text(images)
    result_answer_llm_json = interact(model_path=MODEL_PATH, user_prompt=FROM_TEXT_2_JSON_PROMPT + text_from_ocr)
    result_answer = interact(model_path=MODEL_PATH, user_prompt=FROM_JSON_2_RULE_PROMPT + result_answer_llm_json)
    return result_answer

if __name__ == "__main__":
    print(result_pipeline())
