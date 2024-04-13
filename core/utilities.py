from docx.shared import Inches

a4_width = Inches(210/25.4)
a4_height = Inches(297/25.4)


def pixels_to_inches(pixels, dpi):
    """
    Преобразует пиксели в дюймы.

    :param pixels: Количество пикселей.
    :param dpi: Разрешение изображения (точек на дюйм).
    :return: Размер в дюймах.
    """
    inches = pixels / dpi
    return Inches(inches)


def get_dpi(image):
    """
    Получает разрешение (dpi) изображения.

    :param image: Изображение.
    :return: Разрешение в точках на дюйм.
    """
    info = image.info
    dpi = info.get('dpi', (300, 300))
    return dpi[0]


def set_section_parameters(section, width, height, dpi):
    """
    Устанавливает параметры секции документа.

    :param section: Секция документа.
    :param width: Ширина в пикселях.
    :param height: Высота в пикселях.
    :param dpi: Разрешение изображения (точек на дюйм).
    :return: Обновленная секция документа.
    """
    section.page_width = pixels_to_inches(width, dpi)
    section.page_height = pixels_to_inches(height, dpi)
    section.left_margin = Inches(0)
    section.right_margin = Inches(0)
    section.top_margin = Inches(0)
    section.bottom_margin = Inches(0)
    return section


def add_paragraph(doc, text, left_indent, space_before, font_size):
    """
    Добавляет параграф в документ.

    :param doc: Объект документа.
    :param text: Текст параграфа.
    :param left_indent: Отступ слева.
    :param space_before: Пространство перед параграфом.
    :param font_size: Размер шрифта.
    """
    if text:
        paragraph = doc.add_paragraph(text)
        
        if paragraph.runs:
            paragraph.runs[0].font.size = font_size
            paragraph.paragraph_format.left_indent = left_indent
            paragraph.paragraph_format.space_before = space_before
            paragraph.paragraph_format.space_after = Inches(0)


def scale_to_A4(width, height, dpi):
    """
    Масштабирует размеры изображения до формата A4.

    :param width: Ширина в пикселях.
    :param height: Высота в пикселях.
    :param dpi: Разрешение изображения (точек на дюйм).
    :return: Коэффициенты масштабирования для ширины и высоты.
    """
    if is_horizontal(width, height):
        scale_width = a4_width / pixels_to_inches(height, dpi)
        scale_height = a4_height / pixels_to_inches(width, dpi)
    else:
        scale_width = a4_width / pixels_to_inches(width, dpi)
        scale_height = a4_height / pixels_to_inches(height, dpi)
    return scale_width, scale_height


def is_horizontal(width, height):
    """
    Проверяет, является ли изображение горизонтальным.

    :param width: Ширина в пикселях.
    :param height: Высота в пикселях.
    :return: True, если горизонтальное, иначе False.
    """
    return width > height
