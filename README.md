# Bert Celery Triton

###### created and updated by Weber Huang 

## Description

This repository is refer to [triton-inference-server](https://github.com/triton-inference-server/server) for building a text classification training task using Pytorch, Transformers, Celery and deploying the inference with Triton Inference Sever.

You can refer to [triton-inference-server client docs](https://github.com/triton-inference-server/client) for more details to set up your client machine or using [Celery](https://docs.celeryq.dev/en/stable/getting-started/introduction.html) to develop inference client, in this way you have to bind the same massage broker. 

In this project, we have created a simple inference client with training task by celery:    
1. training task: 
   + users can place the training dataset with training arguments to start the training flow, the result will be saved into the  `model_report` and `eval_report` table in `audience_result`.
   + input dataset from audience-API deep model worker with model parameters, return training information like classification report, f1_score and false prediction data.
2. predicting task (triton client):
   + receive the inference request with dataset and arguments (these are fixed value in client side) and return the predicted probabilities.
   + input dataset from audience-API predict worker with inference parameters, return inference probabilities.

Please refer to `bert_celery_app.py` for more details.

## Work flow

![](graph/Audience DL-model_service.png)

## Build with

+ Celery [redis] 5.2.7
+ Docker 20.10.7
+ Python 3.8
+ Torch 1.11.0
+ Triton Client 2.23.0 
+ Triton Inference Sever <u>nvcr.io/nvidia/tritonserver:22.06-py3</u> 

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
    model
    	...
    	torch_script
    		...
    		<model_name>
    			<version>
                	model.pt
    			config.pbtxt
    ```

    + model_name: model training task name
    + version: training task version



+ Test if the model was loaded

```bash
$ curl -v localhost:8000/v2/health/ready

$ curl localhost:8000/v2/models/${MODEL_NAME}/versions/1/stats

$ GET v2/models[/${MODEL_NAME}[/versions/${MODEL_VERSION}]]/stats
```

## Appendix

#### How to add model?

1. Adding BERT family model
   + Please refer to abstract classes of  `interface.model_interface.model_interface.DeepModelInterface` and `interface.model_interface.BertModelInterface` beforehand
   + Create new module for new model under `worker/train/` with naming `{name}_{task_type}` , for example`chinese_bert_classification`
2. Adding other deep model
   + Please refer to abstract classes of  `interface.model_interface.model_interface.DeepModelInterface`
   + Create new class for new model under `worker/train/` with a proper naming.

#### How to add inference?

+ Please refer to abstract classes of  `interface.inference_interface.bert_inference_interface.BertInferenceInterface` and `interface.inference_interface.inference_interface.InferenceInterface` beforehand
+ Create new module under `worker/inference/` with a proper naming. 

#### Model-repository structure

+ `torch_script` folder will be a sharing volume in both `celery_server` and `triton server`  containers in order to let the user saving training models in different name and version, at the same time triton will dynamically load/unload the inference models to serve the prediction ETL.
+ the `tokenizer` repository structure is designed same as `torch_script`, to let the inference task to load the correct version of tokenizer.

```
/model
	/bin
		<model training file>
    /tokenizer
    	/<model_name>
    		/<version>
    			...
    /torch_script
    	/<model_name>
    		/<version>
    			model.pt
            config.pbtxt
		
```

#### Managing triton server configuration file

Since there may be multiple models serve on triton container, for every model in model repository, it requires a `config.pbtxt` to describe the model inference information. You can manage the configuration file saved in `/configuration/` (Noted the number value of directory name represents the number of classification)

For current version this project serve the <u>2-classes, 4-classes and 8-classes</u> classification.

Please refer to [Minimal Model Configuration](https://github.com/triton-inference-server/server/blob/main/docs/model_configuration.md#minimal-model-configuration) for more details.

```
/configuration
	/2
		config.pbtxt
    /4
    	config.pbtxt
    /8
    	config.pbtxt
```

