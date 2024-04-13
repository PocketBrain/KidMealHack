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


def parse_json_string(json_str: str) -> dict:
    try:
        parsed_json = json.loads(json_str)
        parsed_json_str = {str(key): str(value) for key, value in parsed_json.items()}

        return parsed_json_str
    except json.JSONDecodeError as e:
        print("Error JSON:", e)
        return {}


def get_message_tokens(model, role, content):
    message_tokens = model.tokenize(content.encode("utf-8"))
    message_tokens.insert(1, ROLE_TOKENS[role])
    message_tokens.insert(2, LINEBREAK_TOKEN)
    message_tokens.append(model.token_eos())
    return message_tokens


def get_system_tokens(model):
    system_message = {"role": "system", "content": SYSTEM_PROMPT}
    return get_message_tokens(model, **system_message)


def interact(
    model_path,
    user_prompt,
    n_ctx=4096,
    top_k=30,
    top_p=0.9,
    temperature=0.2,
    repeat_penalty=1.1,
):
    model = Llama(
        model_path=model_path,
        n_gpu_layers=-1,
        n_batch=512,
        n_ctx=n_ctx,
        n_parts=1,
    )

    system_tokens = get_system_tokens(model)
    tokens = system_tokens
    model.eval(tokens)

    message_tokens = get_message_tokens(model=model, role="user", content=user_prompt)
    token_str = ""
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
        token_str += model.detokenize([token]).decode("utf-8", errors="ignore")
        tokens.append(token)
        if token == model.token_eos():
            break
    return token_str


def pipeline(
    ocr_text,
    model_path=MODEL_PATH,
    n_ctx=4096,
    top_k=30,
    top_p=0.9,
    temperature=0.2,
    repeat_penalty=1.1,
):
    model = Llama(
        model_path=model_path,
        n_gpu_layers=-1,
        n_batch=512,
        n_ctx=n_ctx,
        n_parts=1,
    )
    system_tokens = get_system_tokens(model)
    tokens = system_tokens
    model.eval(tokens)
    message_tokens = get_message_tokens(model=model, role="user", content=FROM_TEXT_2_JSON_PROMPT+ocr_text)
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
    return rules, json_str
