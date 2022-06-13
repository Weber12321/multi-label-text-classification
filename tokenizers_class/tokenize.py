from loguru import logger

from settings import MODEL_CLASS


def build_tokenizer(token_name: str, log: logger):
    """
    :param token_name: tokenizer name which corresponds in MODEL_CLASS, settings.py
    :param log: logger class
    :return: tokenizer
    """
    log.debug(f"loading {token_name} ...")

    ckpt = MODEL_CLASS.get(token_name)

    if ckpt == 'bert-base-uncased':
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained(
            'bert-base-uncased', do_lower_case=True
        )
    elif ckpt == 'xlnet-base-cased':
        from transformers import XLNetTokenizer
        tokenizer = XLNetTokenizer.from_pretrained(ckpt, do_lower_case=False)
    elif ckpt == 'roberta-base':
        from transformers import RobertaTokenizer
        tokenizer = RobertaTokenizer.from_pretrained(ckpt, do_lower_case=False)
    elif ckpt == 'albert-base-v2':
        from transformers import AlbertTokenizer
        tokenizer = AlbertTokenizer.from_pretrained(ckpt)
    elif ckpt == 'xlm-roberta-base':
        from transformers import XLMRobertaTokenizer
        tokenizer = XLMRobertaTokenizer.from_pretrained(ckpt)
    else:
        error_message = f"token_name {token_name} is not fount"
        log.error(error_message)
        raise ValueError(error_message)

    return tokenizer


