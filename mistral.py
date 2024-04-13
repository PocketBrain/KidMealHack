from core.document_conversion import extract_images
from llm.utiils import interact
from core.document_generator import generate_output_text
from constant import MODEL_PATH, FROM_TEXT_2_JSON_PROMPT, FROM_JSON_2_RULE_PROMPT
from skimage import exposure
import numpy as np
from PIL import Image
import io

def preprocess_image(image_np):
    try:
        image_eq = exposure.equalize_adapthist(image_np)
        v_min, v_max = np.percentile(image_eq, (0.2, 99.8))
        image_contrast = exposure.rescale_intensity(image_eq, in_range=(v_min, v_max))
        return image_contrast

    except Exception as e:
        print("Ошибка обработки изображения:", e)


def get_result_answer():
    file_path = "test_files/Йогурт Агуша с персиком с 8 месяцев 2.7% 200 г.jpg"
    #Удалить путь, и поставить на вход np.array
    image_np = np.array(Image.open(file_path))
    image_np = preprocess_image(image_np)
    image_np = (image_np * 255).astype(np.uint8)
    with io.BytesIO() as output:
        Image.fromarray(image_np).save(output, format='PNG')
        files = output.getvalue()
    images = extract_images(files=[files])
    result_text = generate_output_text(images)
    return result_text


result_text = get_result_answer()
result_answer_llm_json = interact(model_path=MODEL_PATH, user_prompt=FROM_TEXT_2_JSON_PROMPT + result_text)
print(result_answer_llm_json)
result_answer = interact(model_path=MODEL_PATH, user_prompt=FROM_JSON_2_RULE_PROMPT + result_answer_llm_json)
print(result_answer)