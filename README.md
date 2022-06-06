# Multi-Label Text Classification
###### created by Weber Huang 2022-05-23

**Table of Content**

[Overview](#overview)

[Datasets](#datasets)

[Models](#models)

[Appendix](#appendix)

## Overview

This is a project for text multi-label text classification implementation.

### Requirements

+ [Celery](https://docs.celeryq.dev/en/stable/index.html)
+ [FastAPI](https://fastapi.tiangolo.com/)

+ [Python 3.8.10](https://www.python.org/downloads/release/python-3810/)
+ [Poetry 1.1.13](https://python-poetry.org/docs/)
+ 
+ 

### Usage

Download and setup the environment:

```shell
$ git clone https://gitting.eland.com.tw/rd2/models/multi-label-classification.git

$ cd 

$ poetry install
```



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

