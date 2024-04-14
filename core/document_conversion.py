from PIL import Image
import io
from typing import List

def png_to_images(data: bytes) -> List[Image.Image]:
    """
    Преобразует PNG-данные в список изображений в формате PIL.Image.Image.

    :param data: Данные PNG-файла в виде байт.
    :return: Список изображений.
    """
    try:
        file = io.BytesIO(data)
        image = Image.open(file)
        return [image]
    except:
        return []


def extract_images(files: List[bytes]) -> List[Image.Image]:
    """
    Извлекает изображения из словаря файлов.

    :param files: Словарь с именем файла в качестве ключа и данными файла в качестве значения.
    :return: Список изображений.
    """
    images = []
    for data in files:
        images.extend(png_to_images(data))
    return images
