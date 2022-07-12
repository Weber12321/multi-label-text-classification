# Bert Celery Triton

###### created and updated by Weber Huang

## Description

This repo is refer to [triton-inference-server](https://github.com/triton-inference-server/server) for building a text classification training task using Pytorch, Transformers, celery and deploying the inference with Triton Inference Sever.

Noted that this repo doesn't contain client code, you can refer to [triton-inference-server client docs](https://github.com/triton-inference-server/client) to setup your client machine or using celery cluster to develop inference client, in this way you have to bind the same massage broker.

## Work flow

![](graph/Audience DL-model_service.png)

## Usage

+ Clone the repo

    ```bash
    $ git clone <this repo>
    
    $ cd <this repo>
    ```

+ create a `.env` file with

    ```bash
    # set LEVEL to info if you dont wanna log with verbose mode
    LEVEL=debug
    ```

+ run `docker-compose`, this will setup following services

  ```bash
  $ docker-compose -f docker-compose.yml --env-file .env up
  ```

  + celery_server: for model training and validation

  + redis: message broker for celery_worker

  + triton server: for inference usage, model deployment 

    + if you wanted to run triton server respectively: 

      ```bash
      docker run --rm --name test_triton_server --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -p 8000:8000 -p 8001:8001 -p 8002:8002 -v "$(pwd)"/model/torch_script:/models nvcr.io/nvidia/tritonserver:22.06-py3 tritonserver --model-store=/models --model-control-mode=poll
      ```

      + 8000: HTTP protocol
      + 8001: GRPC protocol

+ model-repository

    ```
    model\
    	...
    	torch_script\
    		...
    		<model_name>\
    			<version>\
                	model.pt
    			config.pbtxt
    ```

    + model_name: model training task name
    + version: training task version



+ Test if the model was loaded

```bash
$ curl -v localhost:8000/v2/health/ready

$ curl localhost:8000/v2/models/audience_bert/versions/1/stats

$ GET v2/models[/${MODEL_NAME}[/versions/${MODEL_VERSION}]]/stats
```
