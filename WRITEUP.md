# Project Write-Up

In this section, I document important detials regarding the model selection research, performance evaluation and how to deploy the 'app' including use cases and industries where applicable.

## Model Selection

This project utilised ssd_mobilenet_v2_coco_2018_03_29 to deploy the "People_counter_app", the model selected should not have been converted Intermediate Representation (IR) format, so I seleted the model from [public_model_zoo](https://github.com/opencv/open_model_zoo/tree/master/models/public/) used for this project. I chose this model because it provided a good detection with regards to the number of person in the video frame per time, prediction accuracy and fast infereence time.

- ssd_mobilenet_v2_coco

SSD_MObilenet is a Single-Shot multibox Detection (SSD) network model used for object detection. The model input is a blob that consists of a single image of 1x3x300x300 in RGB format. This means that I have to process the model before delpoying the app. The original model input is in RGB format while converted model is in BGR format. The pre-process model also shows the number of detection boxes, channels, width and height (n, c, w, h) which is documented in the main.py.

## Downloading the model

The folowing steps where taken to download the model from the public_model_zoo

1. Download the model using the command: wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz

2. Extract the tar file using: tar -xvf ssd_mobilenet_v2_coco_2018_03_29.tar.gz

## Converting the model into Intermediate Representation model

1. cd into the extracted model folder

2. Convert the model to IR using the following command argument:

python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json


## Explaining Custom Layers

The process behind converting custom layers involves...

Some of the potential reasons for handling custom layers are...

## Comparing Model Performance

My method(s) to compare models before and after conversion to Intermediate Representations
were...

The difference between model accuracy pre- and post-conversion was...

The size of the model pre- and post-conversion was...

The inference time of the model pre- and post-conversion was...

## Assess Model Use Cases

Some of the potential use cases of the people counter app are...

Each of these use cases would be useful because...

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows...

## Model Research

[This heading is only required if a suitable model was not found after trying out at least three
different models. However, you may also use this heading to detail how you converted 
a successful model.]

In investigating potential people counter models, I tried each of the following three models:

- Model 1: [Name]
  - [Model Source]
  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...
  
- Model 2: [Name]
  - [Model Source]
  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...

- Model 3: [Name]
  - [Model Source]
  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...
