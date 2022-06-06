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

#DATABASE
USER=rd2
PASSWORD=eland4321
HOST=172.18.20.190
PORT=3306
SCHEMA=audience_dl
```

Run the service

+ You can modify the celery command in `Makefile`, since we use cpu device configuration as default, the pooling command of celery worker is set to `solo`.

```bash
$ make run_api
$ make run_celery
```

Access the experimental docs of swagger user interface and start the experiment by http://127.0.0.1:8000/docs

## Workflow





## Datasets

| Name                                          | Description                       | Size (row)               | Source      | Link                                                         |
| --------------------------------------------- | --------------------------------- | ------------------------ | ----------- | ------------------------------------------------------------ |
| Questions from Cross Validated Stack Exchange | Q&A dataset                       | 85.1 k                   | Kaggle      | [Questions from Cross Validated Stack Exchange ](https://www.kaggle.com/datasets/stackoverflow/statsquestions?resource=download&select=Questions.csv) |
| go_emotions                                   | emotions data from reddit comment | 43.41k / 5.426k / 5,427k | Huggingface | [go emotions](https://huggingface.co/datasets/go_emotions)   |
|                                               |                                   |                          |             |                                                              |

### Usage

+ **Questions from Cross Validated Stack Exchange** dataset contained three datasets with question, answer and tags tables and the site also includes a `.sqlite` file. Join the question table with tags table to handle the text classification.

+ Download the **go_emotion** via Python `datasets` package: 

  ```Python
  from datasets import load_dataset
  
  dataset = load_dataset("go_emotions")
  ```



## Models



## Appendix

