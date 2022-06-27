from loguru import logger
from transformers import AutoTokenizer

from settings import MODEL_CLASS


def build_tokenizer(token_name: str):
    """
    :param token_name: tokenizer name which corresponds in MODEL_CLASS, settings.py
    :return: tokenizer
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            'bert-base-uncased', do_lower_case=True
        )
        return tokenizer
    except Exception as e:
        error_message = f"token_name {token_name} is not fount"
        raise ValueError(error_message)




