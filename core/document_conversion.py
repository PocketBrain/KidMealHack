from PIL import Image
import io
import zipfile
from pdf2image import convert_from_bytes


def pdf_to_images(data: bytes) -> list[Image.Image]:
    """
    Преобразует PDF-данные в список изображений в формате PIL.Image.Image.

    :param data: Данные PDF-файла в виде байт.
    :return: Список изображений.
    """
    try:
        return convert_from_bytes(data)
    except:
        return []

def zip_to_images(data: bytes) -> list[Image.Image]:
    """
    Извлекает изображения из данных ZIP-архива.

    :param data: Данные ZIP-архива в виде байт.
    :return: Список изображений.
    """
    try:
        file = io.BytesIO(data)

        files = {}
        with zipfile.ZipFile(file, mode='r') as zf:
            for name in zf.namelist():
                data = zf.read(name)
                files[name] = data
        return extract_images(files)
    except:
        return []

def png_to_images(data: bytes) -> list[Image.Image]:
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

def extract_images(files: [bytes]) -> list[Image.Image]:
    """
    Извлекает изображения из словаря файлов.

    :param files: Словарь с именем файла в качестве ключа и данными файла в качестве значения.
    :return: Список изображений.
    """
    images = []
    for data in files:
        images.extend(png_to_images(data))
    return images