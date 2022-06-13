from time import time

import numpy as np

from metrics.nultilabel_metrics import accuracy_thresh


def train_epoch(
        model, loader, optimizer, device, lr_scheduler, num_labels, loss_func=None
):
    start = time()
    model = model.train()
    pred_list = []
    label = []
    for d in loader:
        optimizer.zero_grad()
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=targets.float()
        )

        logits = outputs.logits
        if loss_func:
            loss = loss_func(
                logits.view(-1, num_labels),
                targets.float().view(-1, num_labels)
            )
        else:
            loss = outputs.loss

        pred_list.append(logits.sigmoid())
        label.append(targets)

        loss.backward()
        optimizer.step()
        lr_scheduler.step()

    y_pred = np.array([l for b in pred_list for l in b.cpu().detach().numpy()])
    y_true = np.array([l for b in label for l in b.cpu().detach().numpy()])

    acc = accuracy_thresh(
        y_pred=y_pred,
        y_true=y_true,
        thresh=0.5
    )
    return loss, acc, (time() - start) / 60

