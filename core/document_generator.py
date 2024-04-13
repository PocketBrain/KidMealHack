from core.image_processing import process_image, filter_dataframe


def generate_output_text(images, lang="rus", detection_threshold=25) -> str:
    """
    Генерирует текстовую строку на основе изображений с текстом.

    :param images: Список изображений в формате PIL.Image.Image.
    :param lang: Язык распознавания текста (по умолчанию "rus").
    :param detection_threshold: Пороговое значение уверенности распознавания текста (по умолчанию 25).
    :return: Текстовая строка.
    """
    output_text = ""

    for image_index, image in enumerate(images):
        df, _, dpi = process_image(image, horizontal_shift_threshold=50, lang=lang)
        df_filtered = filter_dataframe(df, threshold=detection_threshold)
        found_start = False
        for _, row in df_filtered.iterrows():
            if found_start:
                text = " ".join(row['text'])
                output_text += text + "\n"
            else:
                text = " ".join(row['text'])
                output_text += text + "\n"
                found_start = True

    return output_text
