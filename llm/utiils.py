import json
from llama_cpp import Llama
from constant import (
    SYSTEM_PROMPT,
    BOT_TOKEN,
    LINEBREAK_TOKEN,
    ROLE_TOKENS,
    FROM_TEXT_2_JSON_PROMPT,
    FROM_JSON_2_RULE_PROMPT,
    MODEL_PATH,
)
from typing import List, Tuple, Any, Dict


def parse_json_string(json_str: str) -> Dict[str, str]:
    """
    Преобразует строку JSON в словарь, приводя ключи и значения к строковому типу.

    Параметры:
    - json_str (str): Строка JSON для разбора.

    Возвращает:
    dict: Словарь, содержащий ключи и значения из строки JSON в строковом формате.
    Если строка JSON не может быть разобрана, возвращается пустой словарь.

    Пример использования:
    ```python
    json_str = '{"key": "value"}'
    parsed_json = parse_json_string(json_str)
    ```

    Подробности:
    - Функция разбирает строку JSON и приводит ключи и значения к строковому типу.
    - В случае ошибки разбора JSON возвращается пустой словарь.
    """
    try:
        parsed_json = json.loads(json_str)
        parsed_json_str = {str(key): str(value) for key, value in parsed_json.items()}
        return parsed_json_str
    except json.JSONDecodeError as e:
        print("Error JSON:", e)
        return {}


def get_message_tokens(model: Any, role: str, content: str) -> List[int]:
    """
    Создает токены для сообщения с учетом роли и содержания.

    Параметры:
    - model (Any): Модель токенизатора.
    - role (str): Роль сообщения.
    - content (str): Содержание сообщения.

    Возвращает:
    List[int]: Список токенов сообщения.

    Пример использования:
    ```python
    model = SomeTokenizer()
    role = "user"
    content = "Hello, world!"
    message_tokens = get_message_tokens(model, role, content)
    ```

    Подробности:
    - Функция токенизирует содержание сообщения с учетом роли и вставляет соответствующие токены.
    - В конце сообщения добавляется токен окончания строки.
    """
    message_tokens = model.tokenize(content.encode("utf-8"))
    message_tokens.insert(1, ROLE_TOKENS[role])
    message_tokens.insert(2, LINEBREAK_TOKEN)
    message_tokens.append(model.token_eos())
    return message_tokens


def get_system_tokens(model: Any) -> List[int]:
    """
    Создает токены для системного сообщения.

    Параметры:
    - model (Any): Модель токенизатора.

    Возвращает:
    List[int]: Список токенов системного сообщения.

    Пример использования:
    ```python
    model = SomeTokenizer()
    system_tokens = get_system_tokens(model)
    ```

    Подробности:
    - Функция создает токены для системного сообщения, добавляя соответствующие маркеры.
    """
    system_message = {"role": "system", "content": SYSTEM_PROMPT}
    return get_message_tokens(model, **system_message)


def interact(
    model_path: str,
    user_prompt: str,
    n_ctx: int = 4096,
    top_k: int = 30,
    top_p: float = 0.9,
    temperature: float = 0.2,
    repeat_penalty: float = 1.1,
) -> str:
    """
    Взаимодействие с моделью на основе LLAMA для генерации ответов на пользовательские запросы.

    Параметры:
    - model_path (str): Путь к предварительно обученной модели LLAMA.
    - user_prompt (str): Пользовательский запрос для генерации ответа.
    - n_ctx (int): Максимальная длина контекста.
    - top_k (int): Количество наиболее вероятных токенов для рассмотрения в генерации.
    - top_p (float): Порог отсечения для выбора токенов в генерации на основе вероятностей.
    - temperature (float): Параметр температуры для разнообразия в генерации.
    - repeat_penalty (float): Штраф за повторение токенов в генерации.

    Возвращает:
    str: Сгенерированный ответ на основе пользовательского запроса.

    Пример использования:
    ```python
    model_path = "path/to/model"
    user_prompt = "Привет, как дела?"
    response = interact(model_path, user_prompt)
    ```

    Подробности:
    - Функция использует модель LLAMA для генерации ответов на пользовательские запросы.
    - Задает параметры генерации, такие как ограничения токенов, температура и штраф за повторения.
    - Генерирует ответ на основе пользовательского запроса и возвращает его в виде строки.
    """
    # Инициализация модели
    model = Llama(
        model_path=model_path,
        n_gpu_layers=-1,
        n_batch=512,
        n_ctx=n_ctx,
        n_parts=1,
    )

    # Получение токенов системного сообщения
    system_tokens = get_system_tokens(model)
    tokens = system_tokens
    model.eval(tokens)

    # Получение токенов пользовательского сообщения
    message_tokens = get_message_tokens(model=model, role="user", content=user_prompt)
    token_str = ""
    role_tokens = [model.token_bos(), BOT_TOKEN, LINEBREAK_TOKEN]
    tokens += message_tokens + role_tokens

    # Генерация ответа на основе токенов
    generator = model.generate(
        tokens,
        top_k=top_k,
        top_p=top_p,
        temp=temperature,
        repeat_penalty=repeat_penalty,
    )

    # Преобразование токенов в строку
    for token in generator:
        token_str += model.detokenize([token]).decode("utf-8", errors="ignore")
        tokens.append(token)
        if token == model.token_eos():
            break

    return token_str


def pipeline(
    ocr_text: str,
    model_path: str = MODEL_PATH,
    n_ctx: int = 4096,
    top_k: int = 30,
    top_p: float = 0.9,
    temperature: float = 0.2,
    repeat_penalty: float = 1.1,
) -> Tuple[str, str, str]:
    """
    Обработка текста с помощью модели LLAMA для генерации правил и JSON на основе распознанного текста OCR.

    Параметры:
    - ocr_text (str): Распознанный текст из OCR.
    - model_path (str): Путь к предварительно обученной модели LLAMA.
    - n_ctx (int): Максимальная длина контекста.
    - top_k (int): Количество наиболее вероятных токенов для рассмотрения в генерации.
    - top_p (float): Порог отсечения для выбора токенов в генерации на основе вероятностей.
    - temperature (float): Параметр температуры для разнообразия в генерации.
    - repeat_penalty (float): Штраф за повторение токенов в генерации.

    Возвращает:
    Tuple[str, str, str]: Кортеж, содержащий сгенерированные правила, JSON и ответ.

    Пример использования:
    ```python
    ocr_text = "Some OCR text"
    rules, json_data, answer = pipeline(ocr_text)
    ```

    Подробности:
    - Функция использует модель LLAMA для генерации правил, JSON и ответа на основе распознанного текста OCR.
    - Задает параметры генерации, такие как ограничения токенов, температура и штраф за повторения.
    - Генерирует правила, JSON и ответ на основе текста OCR и возвращает их в виде кортежа.
    """
    # Инициализация модели
    model = Llama(
        model_path=model_path,
        n_gpu_layers=-1,
        n_batch=512,
        n_ctx=n_ctx,
        n_parts=1,
    )

    # Получение токенов системного сообщения
    system_tokens = get_system_tokens(model)
    tokens = system_tokens
    model.eval(tokens)

    # Генерация JSON на основе распознанного текста OCR
    message_tokens = get_message_tokens(
        model=model, role="user", content=FROM_TEXT_2_JSON_PROMPT + ocr_text
    )
    json_str = ""
    role_tokens = [model.token_bos(), BOT_TOKEN, LINEBREAK_TOKEN]
    tokens += message_tokens + role_tokens
    generator = model.generate(
        tokens,
        top_k=top_k,
        top_p=top_p,
        temp=temperature,
        repeat_penalty=repeat_penalty,
    )
    for token in generator:
        json_str += model.detokenize([token]).decode("utf-8", errors="ignore")
        tokens.append(token)
        if token == model.token_eos():
            break

    # Генерация правил на основе JSON
    tokens.extend(
        get_message_tokens(model=model, role="user", content=FROM_JSON_2_RULE_PROMPT)
    )
    tokens += role_tokens
    rules = ""
    generator = model.generate(
        tokens,
        top_k=top_k,
        top_p=top_p,
        temp=temperature,
        repeat_penalty=repeat_penalty,
    )
    for token in generator:
        rules += model.detokenize([token]).decode("utf-8", errors="ignore")
        tokens.append(token)
        if token == model.token_eos():
            break

    # Генерация ответа на основе правил
    tokens.extend(
        get_message_tokens(
            model=model,
            role="user",
            content="WRITE ONLY ONE BOOLEAN VALUE: true if the product matches the declaration and false otherwise.",
        )
    )
    tokens += role_tokens
    answer = ""
    generator = model.generate(
        tokens,
        top_k=top_k,
        top_p=top_p,
        temp=temperature,
        repeat_penalty=repeat_penalty,
    )
    for token in generator:
        answer += model.detokenize([token]).decode("utf-8", errors="ignore")
        tokens.append(token)
        if token == model.token_eos():
            break

    return rules, json_str, answer
