import os
# from dotenv import load_dotenv

# load_dotenv('.env')

DEBUG: bool = True if os.getenv('LEVEL') == 'debug' else False

REDIS = os.getenv('REDIS')


class LogDir:
    training = 'training'
    inference = 'inference'


class LogVar:
    level = 'DEBUG' if DEBUG else 'INFO'
    format = '{time:HH:mm:ss.SS} | {level} | {message}'
    color = True
    serialize = False  # True if you want to save it as json format to NoSQL db
    enqueue = True
    catch = True if DEBUG else False
    backtrace = True if DEBUG else False
    diagnose = True if DEBUG else False
    rotation = '00:00'


class TrainingFileName:
    dataset = 'au_2234_p.json'
    labels = 'labels.json'


MODEL_CKPT = {
    'bert-chinese': 'bert-base-chinese',
    'chinese-bert-wwm': 'hfl/chinese-bert-wwm-ext',
    'chinese-roberta-wwm': 'hfl/chinese-roberta-wwm-ext'
}

# corresponds to the config.pbtxt inside the model-repo
INFERENCE_TYPE = {
    'pytorch': {
        'bert': {
            'input_name': [
                'input__0', 'input__1'
            ],
            'output_name': 'output__0',
        },
    },
}


