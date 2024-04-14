from skimage import exposure
import numpy as np


def preprocess_image(image_np):
    """
    Предварительно обрабатывает изображение для улучшения контраста.

    Параметры:
    - image_np: Массив NumPy, представляющий изображение.

    Возвращает:
    np.array: Предварительно обработанное изображение с улучшенным контрастом.
    """
    try:
        # Улучшение контраста изображения
        image_eq = exposure.equalize_adapthist(image_np)
        
        # Вычисление минимального и максимального значения для контраста
        v_min, v_max = np.percentile(image_eq, (0.2, 99.8))
        
        # Масштабирование интенсивности изображения в заданный диапазон
        image_contrast = exposure.rescale_intensity(image_eq, in_range=(v_min, v_max))
        
        return image_contrast

    except Exception as e:
        print("Ошибка обработки изображения:", e)


def get_dpi(image):
    """
    Получает разрешение (dpi) изображения.

    :param image: Изображение.
    :return: Разрешение в точках на дюйм.
    """
    info = image.info
    dpi = info.get('dpi', (300, 300))
    return dpi[0]
