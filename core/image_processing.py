import cv2
import numpy as np
import pytesseract
import pandas as pd
from core.utilities import get_dpi


def recognize_text(image: np.ndarray, lang: str = "rus"):
    """
    Распознает текст на изображении с использованием pytesseract.

    :param image: Изображение в формате NumPy array.
    :param lang: Язык распознавания текста (по умолчанию "rus").
    :return: DataFrame с результатами распознавания.
    """
    df = pytesseract.image_to_data(image, lang=lang, output_type=pytesseract.Output.DATAFRAME, config="--oem 1 --psm 3 -c tessedit_char_blacklist=_")
    return df


def filter_dataframe(df: pd.DataFrame, threshold=25):
    """
    Преобразует DataFrame с результатами распознавания текста в удобный для создания docx файла формат.

    :param df: DataFrame с результатами распознавания текста.
    :param threshold: Пороговое значение уверенности распознавания (по умолчанию 25).
    :return: Отфильтрованный DataFrame.
    """
    df = df[(0 >= df['conf']) | (df['conf'] >= threshold)]
    df.reset_index(drop=True, inplace=True)

    df_concatenated = df.groupby(['block_num', 'par_num', 'line_num']).apply(lambda group: group.sort_values(by='left')).reset_index(drop=True)
    df_concatenated = df_concatenated.groupby(['block_num', 'par_num', 'line_num']).agg({
        'text': lambda x: [str(e).strip() for e in x if not pd.isna(e) and str(e).strip()],
        'left': 'first',
        'top': 'first',
        'width': 'first',
        'height': 'first',
        'field_id': lambda x: [e for e in x if not pd.isna(e)] if [e for e in x if not pd.isna(e)] else None
    }).reset_index()
    df_filtered = df_concatenated[df_concatenated['text'].apply(lambda x: any(x))]
    df_filtered = df_filtered.sort_values(by='top')
    df_filtered.reset_index(drop=True, inplace=True)
    df_filtered = df_filtered.drop(['block_num', 'par_num', 'line_num'], axis=1)

    return df_filtered


def process_image(image, horizontal_shift_threshold=50, lang = "rus"):
    """
    Обрабатывает изображение, распознает текст и привязывает графику к текстовым блокам.

    :param image: Изображение в формате NumPy array.
    :param horizontal_shift_threshold: Пороговое значение горизонтального смещения (по умолчанию 50).
    :param lang: Язык распознавания текста (по умолчанию "rus").
    :return: DataFrame с результатами распознавания, информацией о полях и разрешение изображения.
    """
    dpi = get_dpi(image)

    df = recognize_text(image, lang)
    fields = pd.DataFrame(crop_fields(image))
    for _, text_line in df[df["level"] == 4].iterrows():
        for index, graph in fields.iterrows():
            if assign_graph_to_line(graph, text_line, horizontal_shift_threshold):
                fields.at[index, 'block_num'] = text_line['block_num']
                fields.at[index, 'par_num'] = text_line['par_num']
                fields.at[index, 'line_num'] = text_line['line_num']
    df["field_id"] = None
    df = pd.concat([df, fields], ignore_index=True)
    df = df.drop(['level', 'page_num'], axis=1).reset_index(drop=True)

    return df, fields, dpi


def crop_fields(image):
    """
    Выделяет графы на изображении.

    :param image: Изображение в формате NumPy array.
    :return: Список словарей с информацией о выделенных областях.
    """
    gray = cv2.cvtColor(np.array(image),cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20,1))
    
    detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    contours, _ = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    fields_info = []
    for i, contour in enumerate(contours, start=1):
        x, y, w, h = cv2.boundingRect(contour)
        fields_info.append({
            'text': " ",
            'left': x,
            'top': y,
            'width': w,
            'height': h,
            "field_id": i,
            'block_num': -i,
            'par_num': -i,
            'line_num': -i,
            "conf": 90,
        })
    return fields_info


def assign_graph_to_line(graph, text_line, horizontal_shift_threshold=50):
    """
    Определяет относится ли объект (граф) к данной текстовой линии.

    :param graph: Информация об объекте (графе).
    :param text_line: Информация о текстовой линии.
    :param horizontal_shift_threshold: Пороговое значение горизонтального смещения (по умолчанию 50).
    :return: True, если объект находится на линии, False в противном случае.
    """
    graph_left = graph['left']
    graph_top = graph['top']
    graph_right = graph['left'] + graph['width']
    graph_bottom = graph['top'] + graph['height']


    for corner in [(graph_left, graph_top), (graph_right, graph_top), (graph_left, graph_bottom), (graph_right, graph_bottom)]:
        if (text_line['left'] - horizontal_shift_threshold) <= corner[0] <= (text_line['left'] + text_line['width'] + horizontal_shift_threshold) and \
           text_line['top'] <= corner[1] <= text_line['top'] + text_line['height']:
            return True
    return False