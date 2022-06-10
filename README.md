# Multi-Label Text Classification
###### created by Weber Huang 2022-05-23

**Table of Content**

1. [Overview](#overview)
   + [Requirements](#requirements)
   + [Usage](#usage)
2. [Workflow](#workflow)
3. [Datasets](#datasets)
4. [Models](#models)
5. [Appendix](#appendix)

## Overview

This is a project for text multi-label text classification implementation.

### Requirements

+ [Celery](https://docs.celeryq.dev/en/stable/index.html)
+ [FastAPI](https://fastapi.tiangolo.com/)
+ [Huggingface Transformers](https://huggingface.co/docs/transformers/index)
+ [Poetry 1.1.13](https://python-poetry.org/docs/)

+ [Python 3.8.10](https://www.python.org/downloads/release/python-3810/)
+ [Pytorch](https://pytorch.org/)
+ [Sqlmodel](https://sqlmodel.tiangolo.com/)

### Usage

Download and setup the environment:

```shell
$ git clone https://gitting.eland.com.tw/rd2/models/multi-label-classification.git
$ cd <project directory>
$ poetry install
```

Create a `.env` file

+ By default, the debug environment (`DEBUG=TRUE`) is using the sqlite database to store the training information, you can switch to other mysql or postgresql database. Try to modify the `settings.py` to change the configuration.

```bash
#Basic configuration
DEBUG=TRUE

#Training device configuration
DEVICE=cpu

#DATABASE: using mysql
USER=<user>
PASSWORD=<password>
HOST=<host>
PORT=<port>
SCHEMA=<schema>
```

Run the service

+ You can modify the celery command in `Makefile`, since we use cpu device configuration as default, the pooling command of celery worker is set to `solo`.

```bash
$ make run_api
$ make run_celery
```

Access the experimental docs of swagger user interface and start the experiment by http://127.0.0.1:8000/docs

### API

+ `/` 

  + GET

    + response all training records

    + response data

      ```bash
      [
        {
          "status": "failed",
          "id": 2,
          "dataset_name": "go_emotion",
          "model_name": "distil_bert",
          "create_time": "2022-06-01 16:55:49",
          "total_time": null,
          "training_args": "TrainingArguments(\n_n_gpu=0,\nadafactor=False,\nadam_beta1=0.9,\nadam_beta2=0.999,\nadam_epsilon=1e-08,\nauto_find_batch_size=False,\nbf16=False,\nbf16_full_eval=False,\ndata_seed=None,\ndataloader_drop_last=False,\ndataloader_num_workers=0,\ndataloader_pin_memory=True,\nddp_bucket_cap_mb=None,\nddp_find_unused_parameters=None,\ndebug=[],\ndeepspeed=None,\ndisable_tqdm=False,\ndo_eval=True,\ndo_predict=False,\ndo_train=False,\neval_accumulation_steps=None,\neval_delay=0,\neval_steps=None,\nevaluation_strategy=IntervalStrategy.EPOCH,\nfp16=False,\nfp16_backend=auto,\nfp16_full_eval=False,\nfp16_opt_level=O1,\nfsdp=[],\nfsdp_min_num_params=0,\nfull_determinism=False,\ngradient_accumulation_steps=1,\ngradient_checkpointing=False,\ngreater_is_better=None,\ngroup_by_length=False,\nhalf_precision_backend=auto,\nhub_model_id=None,\nhub_private_repo=False,\nhub_strategy=HubStrategy.EVERY_SAVE,\nhub_token=<HUB_TOKEN>,\nignore_data_skip=False,\ninclude_inputs_for_metrics=False,\nlabel_names=None,\nlabel_smoothing_factor=0.0,\nlearning_rate=2e-05,\nlength_column_name=length,\nload_best_model_at_end=False,\nlocal_rank=-1,\nlog_level=-1,\nlog_level_replica=-1,\nlog_on_each_node=True,\nlogging_dir=models\\runs\\Jun01_16-55-49_nuc373,\nlogging_first_step=False,\nlogging_nan_inf_filter=True,\nlogging_steps=12,\nlogging_strategy=IntervalStrategy.STEPS,\nlr_scheduler_type=SchedulerType.LINEAR,\nmax_grad_norm=1.0,\nmax_steps=-1,\nmetric_for_best_model=None,\nmp_parameters=,\nno_cuda=False,\nnum_train_epochs=3,\noptim=OptimizerNames.ADAMW_HF,\noutput_dir=models,\noverwrite_output_dir=False,\npast_index=-1,\nper_device_eval_batch_size=32,\nper_device_train_batch_size=32,\nprediction_loss_only=False,\npush_to_hub=False,\npush_to_hub_model_id=None,\npush_to_hub_organization=None,\npush_to_hub_token=<PUSH_TO_HUB_TOKEN>,\nremove_unused_columns=True,\nreport_to=[],\nresume_from_checkpoint=None,\nrun_name=models,\nsave_on_each_node=False,\nsave_steps=500,\nsave_strategy=IntervalStrategy.EPOCH,\nsave_total_limit=None,\nseed=42,\nsharded_ddp=[],\nskip_memory_metrics=True,\ntf32=None,\ntpu_metrics_debug=False,\ntpu_num_cores=None,\nuse_legacy_prediction_loop=False,\nwarmup_ratio=0.0,\nwarmup_steps=0,\nweight_decay=0.01,\nxpu_backend=None,\n)",
          "training_result": null,
          "evaluate_result": null,
          "error_message": "(sqlite3.InterfaceError) Error binding parameter 2 - probably unsupported type.\n[SQL: UPDATE trainingtask SET status=?, total_time=?, training_result=?, evaluate_result=? WHERE trainingtask.id = ?]\n[parameters: ('finished', 1.6925429, TrainOutput(global_step=39, training_loss=0.41692265486105895, metrics={'train_runtime': 99.5837, 'train_samples_per_second': 12.291, 'train_steps_per_second': 0.392, 'train_loss': 0.41692265486105895, 'epoch': 3.0}), {'eval_loss': 0.3040769100189209, 'eval_accuracy_thresh': 0.9545807242393494, 'eval_runtime': 1.922, 'eval_samples_per_second': 47.867, 'eval_steps_per_second': 1.561, 'epoch': 3.0}, 2)]\n(Background on this error at: https://sqlalche.me/e/14/rvf5)"
        },
        {
          "status": "finished",
          "id": 3,
          "dataset_name": "go_emotion",
          "model_name": "distil_bert",
          "create_time": "2022-06-01 17:11:29",
          "total_time": 1.7146706333333335,
          "training_args": "TrainingArguments(\n_n_gpu=0,\nadafactor=False,\nadam_beta1=0.9,\nadam_beta2=0.999,\nadam_epsilon=1e-08,\nauto_find_batch_size=False,\nbf16=False,\nbf16_full_eval=False,\ndata_seed=None,\ndataloader_drop_last=False,\ndataloader_num_workers=0,\ndataloader_pin_memory=True,\nddp_bucket_cap_mb=None,\nddp_find_unused_parameters=None,\ndebug=[],\ndeepspeed=None,\ndisable_tqdm=False,\ndo_eval=True,\ndo_predict=False,\ndo_train=False,\neval_accumulation_steps=None,\neval_delay=0,\neval_steps=None,\nevaluation_strategy=IntervalStrategy.EPOCH,\nfp16=False,\nfp16_backend=auto,\nfp16_full_eval=False,\nfp16_opt_level=O1,\nfsdp=[],\nfsdp_min_num_params=0,\nfull_determinism=False,\ngradient_accumulation_steps=1,\ngradient_checkpointing=False,\ngreater_is_better=None,\ngroup_by_length=False,\nhalf_precision_backend=auto,\nhub_model_id=None,\nhub_private_repo=False,\nhub_strategy=HubStrategy.EVERY_SAVE,\nhub_token=<HUB_TOKEN>,\nignore_data_skip=False,\ninclude_inputs_for_metrics=False,\nlabel_names=None,\nlabel_smoothing_factor=0.0,\nlearning_rate=2e-05,\nlength_column_name=length,\nload_best_model_at_end=False,\nlocal_rank=-1,\nlog_level=-1,\nlog_level_replica=-1,\nlog_on_each_node=True,\nlogging_dir=models\\runs\\Jun01_17-11-29_nuc373,\nlogging_first_step=False,\nlogging_nan_inf_filter=True,\nlogging_steps=12,\nlogging_strategy=IntervalStrategy.STEPS,\nlr_scheduler_type=SchedulerType.LINEAR,\nmax_grad_norm=1.0,\nmax_steps=-1,\nmetric_for_best_model=None,\nmp_parameters=,\nno_cuda=False,\nnum_train_epochs=3,\noptim=OptimizerNames.ADAMW_HF,\noutput_dir=models,\noverwrite_output_dir=False,\npast_index=-1,\nper_device_eval_batch_size=32,\nper_device_train_batch_size=32,\nprediction_loss_only=False,\npush_to_hub=False,\npush_to_hub_model_id=None,\npush_to_hub_organization=None,\npush_to_hub_token=<PUSH_TO_HUB_TOKEN>,\nremove_unused_columns=True,\nreport_to=[],\nresume_from_checkpoint=None,\nrun_name=models,\nsave_on_each_node=False,\nsave_steps=500,\nsave_strategy=IntervalStrategy.EPOCH,\nsave_total_limit=None,\nseed=42,\nsharded_ddp=[],\nskip_memory_metrics=True,\ntf32=None,\ntpu_metrics_debug=False,\ntpu_num_cores=None,\nuse_legacy_prediction_loop=False,\nwarmup_ratio=0.0,\nwarmup_steps=0,\nweight_decay=0.01,\nxpu_backend=None,\n)",
          "training_result": "TrainOutput(global_step=39, training_loss=0.4436204127776317, metrics={'train_runtime': 100.6878, 'train_samples_per_second': 11.769, 'train_steps_per_second': 0.387, 'train_loss': 0.4436204127776317, 'epoch': 3.0})",
          "evaluate_result": "{'eval_loss': 0.31640884280204773, 'eval_accuracy_thresh': 0.9588435292243958, 'eval_runtime': 2.142, 'eval_samples_per_second': 49.02, 'eval_steps_per_second': 1.867, 'epoch': 3.0}",
          "error_message": null
        }
      ```

  + POST

    + request data

      ```bash
      {
        "DATASET_NAME": "go_emotion",
        "MODEL_NAME": "distil_bert",
        "N_SAMPLE": 500,
        "IS_TRAINER": 1,
        "EPOCH": 3,
        "BATCH_SIZE": 32,
        "WEIGHT_DECAY": 0.01
      }
      ```

    - response data

      ```bash
      OK
      ```

+ `/{id}`

  + GET

    + return the specific task information by task `id`

    + response

      ```bash
      {
        "status": "finished",
        "id": 3,
        "dataset_name": "go_emotion",
        "model_name": "distil_bert",
        "create_time": "2022-06-01 17:11:29",
        "total_time": 1.7146706333333335,
        "training_args": "TrainingArguments(\n_n_gpu=0,\nadafactor=False,\nadam_beta1=0.9,\nadam_beta2=0.999,\nadam_epsilon=1e-08,\nauto_find_batch_size=False,\nbf16=False,\nbf16_full_eval=False,\ndata_seed=None,\ndataloader_drop_last=False,\ndataloader_num_workers=0,\ndataloader_pin_memory=True,\nddp_bucket_cap_mb=None,\nddp_find_unused_parameters=None,\ndebug=[],\ndeepspeed=None,\ndisable_tqdm=False,\ndo_eval=True,\ndo_predict=False,\ndo_train=False,\neval_accumulation_steps=None,\neval_delay=0,\neval_steps=None,\nevaluation_strategy=IntervalStrategy.EPOCH,\nfp16=False,\nfp16_backend=auto,\nfp16_full_eval=False,\nfp16_opt_level=O1,\nfsdp=[],\nfsdp_min_num_params=0,\nfull_determinism=False,\ngradient_accumulation_steps=1,\ngradient_checkpointing=False,\ngreater_is_better=None,\ngroup_by_length=False,\nhalf_precision_backend=auto,\nhub_model_id=None,\nhub_private_repo=False,\nhub_strategy=HubStrategy.EVERY_SAVE,\nhub_token=<HUB_TOKEN>,\nignore_data_skip=False,\ninclude_inputs_for_metrics=False,\nlabel_names=None,\nlabel_smoothing_factor=0.0,\nlearning_rate=2e-05,\nlength_column_name=length,\nload_best_model_at_end=False,\nlocal_rank=-1,\nlog_level=-1,\nlog_level_replica=-1,\nlog_on_each_node=True,\nlogging_dir=models\\runs\\Jun01_17-11-29_nuc373,\nlogging_first_step=False,\nlogging_nan_inf_filter=True,\nlogging_steps=12,\nlogging_strategy=IntervalStrategy.STEPS,\nlr_scheduler_type=SchedulerType.LINEAR,\nmax_grad_norm=1.0,\nmax_steps=-1,\nmetric_for_best_model=None,\nmp_parameters=,\nno_cuda=False,\nnum_train_epochs=3,\noptim=OptimizerNames.ADAMW_HF,\noutput_dir=models,\noverwrite_output_dir=False,\npast_index=-1,\nper_device_eval_batch_size=32,\nper_device_train_batch_size=32,\nprediction_loss_only=False,\npush_to_hub=False,\npush_to_hub_model_id=None,\npush_to_hub_organization=None,\npush_to_hub_token=<PUSH_TO_HUB_TOKEN>,\nremove_unused_columns=True,\nreport_to=[],\nresume_from_checkpoint=None,\nrun_name=models,\nsave_on_each_node=False,\nsave_steps=500,\nsave_strategy=IntervalStrategy.EPOCH,\nsave_total_limit=None,\nseed=42,\nsharded_ddp=[],\nskip_memory_metrics=True,\ntf32=None,\ntpu_metrics_debug=False,\ntpu_num_cores=None,\nuse_legacy_prediction_loop=False,\nwarmup_ratio=0.0,\nwarmup_steps=0,\nweight_decay=0.01,\nxpu_backend=None,\n)",
        "training_result": "TrainOutput(global_step=39, training_loss=0.4436204127776317, metrics={'train_runtime': 100.6878, 'train_samples_per_second': 11.769, 'train_steps_per_second': 0.387, 'train_loss': 0.4436204127776317, 'epoch': 3.0})",
        "evaluate_result": "{'eval_loss': 0.31640884280204773, 'eval_accuracy_thresh': 0.9588435292243958, 'eval_runtime': 2.142, 'eval_samples_per_second': 49.02, 'eval_steps_per_second': 1.867, 'epoch': 3.0}",
        "error_message": null
      }
      ```

      

## Workflow

![](graphs/workflow.png)

+ The model worker will select the data preprocessing and model class by the input arguments.
+ You can add new model class and data preprocess flow in this project.

## Datasets

| Name                                          | Description                       | Size (row)               | Source      | Link                                                         |
| --------------------------------------------- | --------------------------------- | ------------------------ | ----------- | ------------------------------------------------------------ |
| Questions from Cross Validated Stack Exchange | Q&A dataset                       | 85.1 k                   | Kaggle      | [Questions from Cross Validated Stack Exchange ](https://www.kaggle.com/datasets/stackoverflow/statsquestions?resource=download&select=Questions.csv) |
| go_emotions                                   | emotions data from reddit comment | 43.41k / 5.426k / 5,427k | Huggingface | [go emotions](https://huggingface.co/datasets/go_emotions)   |
|                                               |                                   |                          |             |                                                              |

+ **Questions from Cross Validated Stack Exchange** dataset contained three datasets with question, answer and tags tables and the site also includes a `.sqlite` file. Join the question table with tags table to handle the text classification.

+ Download the **go_emotion** via Python `datasets` package: 

  ```Python
  from datasets import load_dataset
  
  dataset = load_dataset("go_emotions")
  ```



## Models

| Name                    | dataset     | args                                                         | accuracy                                                     |
| ----------------------- | ----------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| distilbert-base-uncased | go_emotions | "DATASET_NAME": "go_emotion",<br/> "MODEL_NAME": "distil_bert",<br/> "N_SAMPLE": 500,<br/> "IS_TRAINER": 1,<br/> "EPOCH": 3,<br/> "BATCH_SIZE": 32,<br/> "WEIGHT_DECAY": 0.01 | 'eval_loss': 0.3100494146347046, <br />'eval_accuracy_thresh': 0.9567669034004211, <br />'eval_runtime': 2.1132, <br />'eval_samples_per_second': 44.956, <br />'eval_steps_per_second': 1.42, <br />'epoch': 3.0 |
|                         |             |                                                              |                                                              |
|                         |             |                                                              |                                                              |



## Appendix

### Experiment

###### Dataset

+ go_emotions

| Name          | Description                                                  | length |
| ------------- | ------------------------------------------------------------ | ------ |
| go_emotions   | raw dataset without any preprocess, only remove the duplicated row based on text | 57732  |
| go_emotions_s | small dataset with size 1000 rows sampling                   | 1000   |
| *             | only extract top 8 occurrence of label classes (original size of label classes is 28) | -      |



###### Baseline

+ metrics: accuracy and f1_score

| Name      | dataset     | method                   | accuracy                                                     |
| --------- | ----------- | ------------------------ | ------------------------------------------------------------ |
| RF        | go_emotions |                          | acc: 95.7194 <br />micro avg 17.667207 <br />macro avg 10.599551 <br />weighted avg 14.287957 |
| kNN       | go_emotions |                          | acc: 95.7184 <br />micro avg 16.128085 <br />macro avg 9.279948 <br />weighted avg 13.678768 |
| RF        | go_emotions | sklearn multi output     | acc: 95.7370 <br />micro avg 17.362550 <br />macro avg 10.471720 <br />weighted avg 13.940626 |
| LinearSVC | go_emotions | sklearn multi output     | acc: 95.9625 <br />micro avg 14.273205 <br />macro avg 8.714187 <br />weighted avg 10.206662 |
| LR        | go_emotions | sklearn multi output     | acc: 95.9509 <br />micro avg 14.033272 <br />macro avg 9.029469 <br />weighted avg 10.311118 |
| RF        | go_emotions | sklearn classifier chain | acc: 94.6698<br/>micro avg 32.595297<br/>macro avg 13.468366<br/>weighted avg 23.166349 |
| LinearSVC | go_emotions | sklearn classifier chain | acc: 94.7122<br/>micro avg 32.461811<br/>macro avg 10.715990<br/>weighted avg 20.726128 |
| LR        | go_emotions | sklearn classifier chain | acc: 94.73414<br/>micro avg 32.733890<br/>macro avg 11.458806<br/>weighted avg 21.258039 |



###### BERT

+ metrics: precision, recall and f1_score

| Name      | dataset        | params                                               | best_accuracy                                                |
| --------- | -------------- | ---------------------------------------------------- | ------------------------------------------------------------ |
| bert-base | go_emotions_s* | epoch: 10<br/>batch_size: 32<br/>learning_rate: 5e-5 | training acc {'precision': 86.39340162318044, 'recall': 62.52927400468384, 'f1': 68.35319614875736}<br/><br/>validation acc {'precision': 63.61616021620872, 'recall': 40.09216589861751, 'f1': 47.05389188894794} |
| bert-base | go_emotions*   | epoch: 10<br/>batch_size: 32<br/>learning_rate: 5e-5 |                                                              |
| XLNet     | go_emotions_s* |                                                      |                                                              |
| XLNet     | go_emotions*   |                                                      |                                                              |
| roBERTa   | go_emotions_s* |                                                      |                                                              |
| roBERTa   | go_emotions*   |                                                      |                                                              |
| albert    | go_emotions_s* |                                                      |                                                              |
| albert    |                |                                                      |                                                              |

