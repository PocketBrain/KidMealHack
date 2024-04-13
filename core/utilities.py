from skimage import exposure
import numpy as np


def preprocess_image(image_np):
    try:
        image_eq = exposure.equalize_adapthist(image_np)
        v_min, v_max = np.percentile(image_eq, (0.2, 99.8))
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
