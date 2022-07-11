import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report

from utils.train_helper import sigmoid_logits_to_one_hot, one_hot_to_label


def get_classification_report(model, test_data_loader, path, device, label_col):
    y_review_texts, y_pred, y_test = get_prediction(model, test_data_loader, path, device)
    report = classification_report(y_test, y_pred, target_names=label_col, output_dict=True)
    report = pd.DataFrame(report).transpose()
    report = report.reset_index().rename(columns={'index': 'label'})
    false_prediction = [['text', 'label', 'predict']]
    for i in range(len(y_pred)):
        if not (y_pred[i] == y_test[i]).all():
            false_prediction.append(
                [
                    y_review_texts[i],
                    one_hot_to_label(y_test[i], label_col),
                    one_hot_to_label(y_pred[i], label_col)
                ]
            )
    pd.set_option('display.float_format', lambda x: '%.3f' % x)
    false_prediction_df = pd.DataFrame(false_prediction[1:], columns=false_prediction[0])

    return report, false_prediction_df


def get_prediction(model, data_loader, path, device):
    model.load_state_dict(torch.load(path))
    model = model.eval()
    review_texts = []
    predictions = []
    # prediction_probability = []
    real_values = []
    with torch.no_grad():
        for d in data_loader:
            texts = d['review_text']
            input_ids = d['input_ids'].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d['targets'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # logits = outputs.logits
            logits = outputs[-1]
            review_texts.extend(texts)
            predictions.extend(logits.sigmoid())
            real_values.extend(targets)

        y_pred = np.array([b.cpu().detach().numpy() for b in predictions])
        y_true = np.array([b.cpu().detach().numpy() for b in real_values])

        y_pred = sigmoid_logits_to_one_hot(y_pred)
        # predictions = torch.stack(predictions).cpu().detach()
        # real_values = torch.stack(real_values).cpu().detach()
    return review_texts, y_pred, y_true


# https://discuss.pytorch.org/t/finding-model-size/130275/2
def get_model_size(model):
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()

    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()

    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    size_dict = {
        'param_size': param_size,
        'param_sum': param_sum,
        'buffer_size': buffer_size,
        'buffer_sum': buffer_sum,
        'total_size(MB)': size_all_mb
    }
    pd.set_option('display.float_format', lambda x: '%.3f' % x)
    size_df = pd.DataFrame(list(size_dict.items()), columns=['name', 'value'])
    size_df = size_df.round(2)
    return size_df


# https://pytorch.org/tutorials/beginner/saving_loading_models.html#export-load-model-in-torchscript-format
# https://huggingface.co/docs/transformers/serialization#saving-a-model
def save_pt(model, dummy_input, state_dict_path, pt_output_path):
    model.load_state_dict(torch.load(state_dict_path))
    traced_model = torch.jit.trace(model, dummy_input)
    torch.jit.save(traced_model, pt_output_path)
