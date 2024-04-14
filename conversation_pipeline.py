from core.document_conversion import extract_images
from llm.utiils import pipeline
from core.utilities import  preprocess_image
from core.document_generator import generate_output_text
import numpy as np
from PIL import Image
import io
import time
from typing import List, Tuple

def get_data(image_np: np.array) -> bytes:
    """
    Предварительная обработка изображения и преобразование его в байтовый формат PNG.

    Параметры:
    - image_np (np.array): Массив NumPy, представляющий изображение.

    Возвращает:
    bytes: Байтовое представление обработанного изображения в формате PNG.

    Пример использования:
    ```python
    image_array = np.array(Image.open("image.jpg"))
    image_bytes = get_data(image_array)
    ```

    Подробности:
    - Функция принимает массив NumPy, представляющий изображение.
    - Изображение подвергается предварительной обработке.
    - Затем изображение преобразуется в формат PNG и сохраняется в байтовом формате.
    - Байтовое представление изображения в формате PNG возвращается в качестве результата.
    """
    # Удалить путь и применить предварительную обработку к изображению
    image_np = preprocess_image(image_np)
    
    # Преобразовать массив в формат uint8
    image_np = (image_np * 255).astype(np.uint8)
    
    # Сохранить изображение в формате PNG в байтовом представлении
    with io.BytesIO() as output:
        Image.fromarray(image_np).save(output, format='PNG')
        image_bytes = output.getvalue()
    
    return image_bytes

def result_pipeline(files: List[np.array]) -> Tuple[str, str]:
    """
    Обработка изображений и генерация JSON на основе распознанного текста.

    Параметры:
    - files (List[np.array]): Список массивов NumPy, представляющих изображения для обработки.

    Возвращает:
    Tuple[str, str]: Кортеж, содержащий сгенерированный JSON и правила обработки текста.

    Если текст не удалось распознать, возвращается кортеж с сообщением об ошибке и пустой строкой.

    Пример использования:
    ```python
    files = [np.array(Image.open("image1.jpg")), np.array(Image.open("image2.jpg"))]
    json_data, rules = result_pipeline(files)
    ```

    Подробности:
    - Функция принимает список массивов NumPy, представляющих изображения для обработки.
    - Если список файлов пуст, будет использован тестовый файл по умолчанию.
    - Если в списке файлов только один элемент, он будет преобразован в список.
    - Для каждого файла выполняется предварительная обработка для подготовки к извлечению данных.
    - Изображения извлекаются из предварительно обработанных файлов.
    - Производится генерация текста из изображений с использованием определенного порога распознавания.
    - Если распознанный текст короче или равен 40 символам, возвращается сообщение об ошибке.
    - В противном случае применяются правила обработки и создается JSON на основе распознанного текста.
    """
    # Проверка наличия файлов
    if not files:
        print("Запускаем тестовый файл")
        file_path = "test_files/Йогурт Агуша с персиком с 8 месяцев 2.7% 200 г.jpg"
        files = [np.array(Image.open(file_path))]
    elif len(files) == 1:
        files = [files]
    
    # Предварительная обработка файлов
    preprocess_files = [get_data(file) for file in files]
    
    # Извлечение изображений
    images = extract_images(files=preprocess_files)
    
    # Генерация текста из изображений
    text_from_ocr = generate_output_text(images, detection_threshold=0)
    print(text_from_ocr)
    
    # Проверка длины распознанного текста
    if len(text_from_ocr.strip()) <= 40:
        return "Не удалось распознать текст", ""
    
    # Применение правил и создание JSON
    rules, json_data, answer = pipeline(text_from_ocr)
    
    return json_data, rules, answer

if __name__ == '__main__':
    start = time.time()
    result = result_pipeline([])
    print(time.time() - start)
    if len(result) == 3:
        print("json:")
        print(result[0])
        print("rules:")
        print(result[1])
        print("answer:")
        print(result[2])
    else:
        print(result)
