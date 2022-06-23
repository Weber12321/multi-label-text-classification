import numpy as np
import torch


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
            logits = outputs.logits
            review_texts.extend(texts)
            predictions.extend(logits.sigmoid())
            real_values.extend(targets)

        y_pred = np.array([b.cpu().detach().numpy() for b in predictions])
        y_true = np.array([b.cpu().detach().numpy() for b in real_values])
        # predictions = torch.stack(predictions).cpu().detach()
        # real_values = torch.stack(real_values).cpu().detach()
    return review_texts, y_pred, y_true
