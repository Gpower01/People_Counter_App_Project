# Project Write-Up

In this section, I document important detials regarding the model selection research, performance evaluation and how to deploy the 'app' including use cases and industries where applicable.

## Explaining Custom Layers

Custom layers are a neccessary and important feature to have in the OpneVINO ToolKkit, however it is not often used due to the fast coverage of the supported layers but it is useful to know about its existence and how to use it if the need arises. The list of supported layers is shown [here](https://docs.openvinotoolkit.org/2019_R3/_docs_MO_DG_prepare_model_Supported_Frameworks_Layers.html). Any layer not listed is classified as custom layer by the Model Optimizer. In order to add custom layers, there are a few differences depending on the original model framework. In both TensorFlow and Caffe models, the first option is to register the custom layers as extensions to the Model Optimizer.

- For Caffe, the second option is to register the layers as Custom, then use Caffe to calculate the output shape of the layer. More information on how to do that [here](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_customize_model_optimizer_Customize_Model_Optimizer.html). You'll require Caffe installed on your system for this option.

- For TensorFlow, the second option is to replace the unsupported subgraph with a different subgraph. More information [here](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_customize_model_optimizer_Customize_Model_Optimizer.html).
This feature is helpful for many TensorFlow models and more information on how to do that [here](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_customize_model_optimizer_Subgraph_Replacement_Model_Optimizer.html). The final TensorFlow option is to actually offload the computation of the subgraph back to TensorFlow during inference. More information on offloading sugbraph inference to TensorFlow [here](https://docs.openvinotoolkit.org/2019_R3/_docs_MO_DG_prepare_model_customize_model_optimizer_Offloading_Sub_Graph_Inference.html). Checkout the developer documentation [here](https://docs.openvinotoolkit.org/2019_R3/_docs_MO_DG_prepare_model_customize_model_optimizer_Customize_Model_Optimizer.html) for more information on Custom Lyers in the Model Optimizer.

Potential reasons for handling custom layers may be that the IR does not support all of the layers from original framework. Sometimes because of the hardware, for example on CPU there are few IR model which are directly supported while others may not be supported.


## Converting a TensorFlow model to Intermediate Representation (IR)

Before converting a model, you must configure the Model Optimizer for the framework that was used to train the moddel. You can learn more about how to configure the Model Optimizer [here](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_Config_Model_Optimizer.html). To converting a Model to Intermediate Representation (IR), use the the mo.py script from the <INSTALL_DIR>/deployement_tools/model_optimizer directory to run the Model Optimizer and convert the model to the Intermediate Representation (IR). 

The mo.py is the universal entry point that can deduce the framework that has produced the input model by a standard extension of the model file. More information [here](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Converting_Model.html).

- .caffemodel - Caffe models
- .pb - TensorFlow models
- .params - MXNet models
- .onnx - ONNX models
- .nnet - Kaldi models

Converting TensorFlow Object Detection API Model, go to the <INSATLL_DIR>/deployment_tools/mode_optimizer directory and run the mo_tf.py script with the following required parameters:

- --input_model <path_to_frozen.pb>

- --transformations_config <path_to_subgraph_replacement_configuration_file.json> - A subgraph replacement configuration file with transformations description. For models downloaded from the TensorFlow Object Detection API zoo, the confiuguration file can be found in the <INSTALL_DIR>/deployment_tools/model_optimizer/extensions/front/tf directory. Use:

- ssd_v2_support.json - for frozen SSD topologies from the model zoo

More Details on Converting TensorFlow Object Detection API Models [here](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_Object_Detection_API_Models.html).

## Model Selection

This project utilised ssd_mobilenet_v2_coco_2018_03_29 to deploy the "People_counter_app", the model selected should not have been converted Intermediate Representation (IR) format, so I seleted the model from [public_model_zoo](https://github.com/opencv/open_model_zoo/tree/master/models/public/) used for this project. I chose this model because it provided a good detection with regards to the number of person in the video frame per time, prediction accuracy and fast infereence time.

- ssd_mobilenet_v2_coco

SSD_MObilenet is a Single-Shot multibox Detection (SSD) network model used for object detection. The model input is a blob that consists of a single image of 1x3x300x300 in RGB format. This means that I have to process the model before delpoying the app. The original model input is in RGB format while converted model is in BGR format. The pre-process model also shows the number of detection boxes, channels, width and height (n, c, w, h) which is documented in the main.py.


## Downloading the model used for this project

The folowing steps where taken to download the model from the public_model_zoo

1. Download the model using the command: 

"wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz"

2. Extract the tar file using: "tar -xvf ssd_mobilenet_v2_coco_2018_03_29.tar.gz"

## Converting the model into Intermediate Representation model

1. cd into the extracted model folder

2. Convert the model to IR using the following command argument:

"python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json"


- Details of supported Frozen Topologies from TensorFlow Object Detection zoo including the SSD_MobileNet_v2_Coco is shown [here](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_TensorFlow.html) and the convertion parameters and steps for converting the model to IR is shown [here](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_customize_model_optimizer_TensorFlow_SSD_ObjectDetection_API.html).


## Runing the People_Counter_APP

To run the people_counter_app, the following command argument was used: 

" python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m ssd_mobilenet_v2_model/frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm"

- Note: that you must check the path for the fozen_inference_graph.xml which in this project was located in the ssd_mobilenet_ve_model/

- Also you will need to open four terminals to run the people_counter_app. Please see the information on 'README' for further steps on running the application.


## Comparing Model Performance

In order to evaluate the model performance, I compared the model size, prediction accuracy and inference time of the model before and after conversion.

My method(s) to compare models before and after conversion to Intermediate Representations
were:


The size of the model pre- and post-conversion was:

- Model: ssd_mobilenet_v2_coco_2018_03_29 size before: 180MB and after 65MB

The inference time of the model pre- and post-conversion:

- Inference time of the model ranges from 68ms - 74ms after conversion


## Assess Model Use Cases

The people_counter_app have an important application in many business sector where customer servivce is of great importance. The applicaton can help business owners to monitor their customer habit and use the generated statistics to make informed decision on how to better serve their customers.

Some of the potential use cases of the people counter app inlucde but limited to 

1. Retail stores: where this can be deployed to monitor customers shopping habits and used to that optimised thair staffing needs to know when the busiest hours are and place more staff on shopping grounds. It can also be used to optimse merchandising. 

2. Security Industry: It can be used to monitor trespassers from entering secured areas or premises 

3. Hospital: To maintain the number of patient that can be allowed to enter a particular area, so that if the number is decreasing, they can be notified to call for more patient.

4. Train Stations: For example the people_counter_app can be used to monitor the number of people in train stations if they are observing social distancing particulary during COVID-19 


## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model.

Lighter, quicker models are helpful for the edge and certain optimizations like lower precision that help with these will also impact on accuracy. No amnount of post-processing and attempt to extract useful data from the output will make up for a poor model choice or one where too many sacrifices were made for speed and of course these choices are dependent on how much loss of accuracy is accepted. More information [here](https://docs.openvinotoolkit.org/2019_R3/_docs_IE_DG_Intro_to_Performance.html).

The consideration of speed, size and network impact are very important to deploy AI applications at the Edge. Faster models can help free up computation for other tasks, leading to less power usage which also can allow for cheaper hardware to be use.

SSD model used for this project was trianed for detecting object in images using a single deep neural network. It discretizes the output sapces of the bounding boxes into a set of default boxes over different aspect ratios and scales per feature map locationw which allows the model to generate scores for the presence of each object category in each default bounding boxes and produce adjustments to the box to better match the object shape.

The model also have features that allows it to combine predictions from multiple feature maps with different resolution to better handle objects of different sizes. These features makes the SSD model easy to train and integrate into a system that requires a detection component and provides better accuracy even with a smaller input image size. For 300 x 300 input (which is used in for this project), SSD can achieve 72.1%  accuracy and for 500 x 500 input, it can achieve up to 75.1% acuuracy. More information on SSD models can be found [here](https://arxiv.org/abs/1512.02325).

