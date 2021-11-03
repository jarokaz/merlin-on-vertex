# [WiP] NVIDIA Merlin on Vertex AI

This repository compiles best practices and code samples for operationalization of [NVIDIA Merlin framework](https://developer.nvidia.com/nvidia-merlin) on [Google Cloud Vertex AI](https://cloud.google.com/vertex-ai).

[NVIDIA Merlin](https://developer.nvidia.com/nvidia-merlin) is and open-source framework for building large-scale deep learning recommender system. [Vertex AI](https://cloud.google.com/vertex-ai) is Google Cloud's unified Machine Learning platform to help data scientists and machine learning engineers increase experimentation, deploy faster, and manage models with confidence.

The content is this repository centers around five core areas:

- Setting up Merlin experimentation and development environment in [Vertex AI Workbench](https://cloud.google.com/vertex-ai-workbench). 
- Operationalizing large scale data preprocessing pipelines with [NVIDIA Merlin NVTabular](https://developer.nvidia.com/nvidia-merlin/nvtabular) and [Vertex AI Pipelines](https://cloud.google.com/vertex-ai/docs/pipelines/introduction).
- Training large-scale deep learning ranking models with [NVIDIA Merlin HugeCTR](https://github.com/NVIDIA-Merlin/HugeCTR) and [Vertex AI Training](https://cloud.google.com/vertex-ai/docs/training/custom-training).
- Deploying models and serving predictions with [NVIDIA Triton Inference Server](https://developer.nvidia.com/nvidia-triton-inference-server) and [Vertex AI Prediction](https://cloud.google.com/vertex-ai/docs/predictions/getting-predictions)
- Implementing end to end data preprocessing, training, and deployment pipelines with [Vertex AI Pipelines](https://cloud.google.com/vertex-ai/docs/pipelines/introduction).

This repository provides examples on how to run [NVIDIA Merlin](https://developer.nvidia.com/nvidia-merlin) framework for building large-scale deep learning recommender systems using [Verex AI](https://cloud.google.com/vertex-ai) managed services on Google Cloud. The example covers:

* Data preprocessing and feature engineering using [NVIDIA NVTabular](https://developer.nvidia.com/nvidia-merlin/nvtabular).
* Training and evaluating deep learning recommender models using TensorFlow and [NVIDIA HugeCTR](https://developer.nvidia.com/nvidia-merlin/hugectr).
* Serving the models using [NVIDIA Triton](https://developer.nvidia.com/nvidia-triton-inference-server) inference server.
* Scaling and automating the system using [Vertex AI](https://cloud.google.com/vertex-ai) training, prediction, and pipeline services, as well as [Cloud GPUs](https://cloud.google.com/gpu). 


![NVIDIA Merlin](images/overview.png)



## Overview

With the rapid growth in scale of industry datasets, Deep Learning (DL) recommender models have started to gain advantages over traditional methods. At the same time, the complexity of building such systems has grown to the point where special thought must be taken in both the preparation of the data and in the training methods used in order to avoid performance issues that can slow down the total training iteration time by orders of magnitude.

The combination of more sophisticated models and rapid data growth has raised the bar for computational resources required for data preprocessing and training while also placing new burdens on production systems. The current challenges for training large-scale recommenders include:

* **Huge datasets**: Commercial recommenders are trained on huge datasets, often several terabytes in scale. At this scale, data ETL and preprocessing steps often take much more time than training the DL model.
* **Complex data preprocessing and feature engineering pipelines**: Datasets need to be preprocessed and transformed into a form relevant to be used with DL models and frameworks. In addition, feature engineering creates an extensive set of new features from existing ones, requiring multiple iterations to arrive at an optimal solution.
* **Input bottleneck**: Data loading, if not well optimized, can be the slowest part of the training process, leading to under-utilization of high-throughput computing devices such as GPUs.
* **Extensive repeated experimentation**: The whole data engineering, training, and evaluation process is generally repeated many times, requiring significant time and computational resources.

To meet the computational demands for large-scale DL recommender systems training and inference, NVIDIA introduced Merlin, an application framework and ecosystem created to facilitate all phases of building a DL recommender system, accelerated on NVIDIA GPUs. NVIDIA Merlin includes the following components:

1. [NVTabular](https://developer.nvidia.com/nvidia-merlin/nvtabular) - a feature engineering and preprocessing library designed to effectively manipulate terabytes of recommender system datasets and significantly reduce data preparation time.
2. [HugeCTR](https://developer.nvidia.com/nvidia-merlin/hugectr) -  a deep neural network training framework designed for recommender systems. It provides distributed training with model-parallel embedding tables and data-parallel neural networks across multiple GPUs and nodes for maximum performance.
3. [Triton](https://developer.nvidia.com/nvidia-triton-inference-server) - an inference server to serve model efficiently on GPUs by maximizing throughput with the right combination of latency and GPU utilization.

In addition, you can use NVIDIA Merlin components when you are building your TensorFlow or PyTorch DL recommender models, by using NVTabular to preprocess your data, using NVTabular data loaders to feed the data to your model efficiently during training, and using Triton server to serve your trained model.

![NVIDIA Merlin](images/nvidia-merlin.png)


[Verex AI](https://cloud.google.com/vertex-ai) is Google Cloud's unified data science and ML platform to build, deploy, and operationalize custom AI systems at scale.
Vertex AI provides a suite of managed service for MLOps processes, including experimentation, model training, model serving, metadata tracking, and model monitoring,
as well as pipeline workflow orchestration. 

This code repository shows how to use Vertex AI services to run the various components of NVIDIA Merlin framework: NVTabular, HugeCTR, and Triton, to build and deploy large scale DL recommender models on Google Cloud.

![NVIDIA Merlin](images/vertexai_componentes.png)


## Repository structure

The source code for the data preprocessing, model training, and model inference is provided in [src](src) directory. However, we provide the following notebook 
to drive the execution of the of different steps of the system:

1. [00-dataset-management](00-dataset-management.ipynb) describes and explore the dataset used in our examples, and load it to BigQuery.
2. [01-dataset-preprocessing](01-dataset-preprocessing.ipynb) shows how to use NVTabular to preprocess the CSV data on GCS, as well as BigQuery data, to Parquet files with Vertex AI. 
3. [02-model-training-hugectr](02-model-training-hugectr.ipynb) shows how to train a HugeCTR model using Vertex AI. 
4. [03-model-inference-hugectr](03-model-inference-triton.ipynb) shows how to use Triton Inference Server to serve the model on Vertex AI.
5. [04-e2e-pipeline](04-e2e-pipeline.ipynb) shows how to deploy and run and end-to-end NVIDIA Merlin pipeline with Vertex AI.


## Getting started
### Enabling the required services
From Cloud Shell, run the following `gcloud` command to enable the required Cloud APIs:
```
PROJECT_ID=merlin-on-gcp
gcloud services enable \
    aiplatform.googleapis.com         \
    bigquery.googleapis.com           \
    bigquerystorage.googleapis.com    \
    cloudapis.googleapis.com          \
    cloudbuild.googleapis.com         \
    compute.googleapis.com            \
    containerregistry.googleapis.com  \
    notebooks.googleapis.com          \
    storage.googleapis.com            \
    --project=${PROJECT_ID}
```


### Creating Merlin development container image
From Cloud Shell

1. Get Dockerfile for the Merlin development image:
```
SRC_REPO=https://github.com/GoogleCloudPlatform/merlin-on-vertex
LOCAL_DIR=merlin-env-setup
kpt pkg get $SRC_REPO/env@main $LOCAL_DIR
cd $LOCAL_DIR
```
2. Build and push the development image
```
PROJECT_ID=merlin-on-vertex # change to your project id.
IMAGE_URI=gcr.io/${PROJECT_ID}/merlin-dev-vertex
gcloud builds submit --timeout "2h" --tag ${IMAGE_URI} . --machine-type=e2-highcpu-8
```

### Creating and configuring an instance of Vertex Workbench managed notebook

1. Follow the instructions in the [Create a managed notebooks instance how-to guide](https://cloud.google.com/vertex-ai/docs/workbench/managed/create-instance):
    * In the [Use custom Docker images settings](https://cloud.google.com/vertex-ai/docs/workbench/managed/create-instance#expandable-2) enter a name of the image you created in the previous step: `gcr.io/<your-project-id>/merlin-dev-vertex:latest`
    * In the [Configure hardware settings](https://cloud.google.com/vertex-ai/docs/workbench/managed/create-instance#expandable-3) select your GPU configuration. We recommend a machine with two `NVIDIA Tesla T4` or `NVIDIA Tesla A100` GPUs. 

### Run the example
After the Vertex Workbench managed notebook is created, peform the following steps:

1. Click on the OPEN JUPYTERLAB link on the notebook instance.
2. Click on the New Launcher button, then start a new terminal session.
3. Clone the repository to your notebook instance:
    ```
    git clone https://github.com/GoogleCloudPlatform/merlin-on-vertex.git
    cd merlin-on-vertex
    ```
    

