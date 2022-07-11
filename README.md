# Bert Celery Triton

###### created by Weber Huang 

## Description

This repo is refer to [triton-inference-server](https://github.com/triton-inference-server/server) and [Triton Inference Server 介紹與範例](https://roychou121.github.io/2020/07/20/nvidia-triton-inference-server/)  for building a text classification training task using Pytorch, Transformer BERT, celery worker and deploying with Triton Inference Sever.

Noted that this repo doesn't contain client code, you can refer to [triton-inference-server client docs](https://github.com/triton-inference-server/client) to setup your client machine or using celery cluster to develop inference client, in this way you have to bind the same massage broker.

## Usage

+ Clone the repo

```bash
$ git clone <this repo>

$ cd <this repo>
```

+ create a `.env` file with

```bash
# set LEVEL to info if you dont wanna log with verbose mode
# change the redis localhost to your ip
LEVEL=debug
REDIS=redis://localhost:6379/0
```

+ run service, this will setup
  + celery_server: for model training and validation
  + redis: message broker and backend for celery_worker
  + triton server: for inference usage, model deployment 
  
    + run triton server respectively
  
      ```bash
      docker run --rm --name test_triton_server --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -p 8000:8000 -p 8001:8001 -p 8002:8002 -v "$(pwd)"/model/torch_script:/models nvcr.io/nvidia/tritonserver:22.06-py3 tritonserver --model-store=/models --model-control-mode=poll
      ```
  
      

```bash
$ docker-compose -f docker-compose.yml --env-file .env up
```

+ Test if the model is loaded
  + 8000: HTTP protocol
  + 8001: GRPC protocol

```bash
$ curl -v localhost:8000/v2/health/ready

$ curl localhost:8000/v2/models/audience_bert/versions/1/stats

$ GET v2/models[/${MODEL_NAME}[/versions/${MODEL_VERSION}]]/stats
```
