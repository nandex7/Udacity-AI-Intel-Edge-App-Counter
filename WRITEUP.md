# People Counter App at the Edge
In this application, app counts the number of people in the current frame, the duration that a person is in the frame (time elapsed between entering and exiting a frame) and the total count of people for this app, we send the information to a MOSCA server like total counted people, current count and the average duration of person.
This application is a part of the Udacity scholarship Nanodegree . Where we apply AI with Intel OpenVino Toolkit.
If you want to see the video processed check at the following link:  [video](https://drive.google.com/file/d/1yxZObQsYWdDzng7duC3vJPqXF4A8iP3T/view?usp=sharing)

## Explaining Custom Layers  

The custom layers help us to create new models for specific operations. To create a Custom Layer is needed to Generate  an extension template file using the Model Extension Generator and after using the Model Optimizer to generate the IR files  , also you can edit the CPU extension template file or TPU and Execute the model with the custom layer.

## Comparing Model Performance 
The difference between model accuracy pre- and post-conversion was :

ssd_mobilenet_v1_coco (Tensorflow) -> Inference Time  89ms and Size 57MB and using IR Conversion we reduce to 44ms and 28 MBs almost half faster_rcnn_inception_v2_coco (Tensor flow) Inference Time  342 and Size 202MB and using IR Conversion we reduce to 155ms and 98 MBs more than halfssd_mobilenet_v2_coco (Tensor flow) Inference Time  104 and Size 135MB and using IR Conversion we reduce to 68ms and 67 MBs almost half

##### Comparing Edge and Cloud 
Cloud computing models are good for big data processes but the Edge computing can give us the opportunity to execute powerful Deep learning models with less resources.
## Assess Model Use Cases
The potential uses cases of this app People Counter are :
For Retail companies we can create metrics to check what 's the conversion of people coming to the store and Who are the conversion people buying products.For Security we can see the number of people entering an office or industrial company and checking if we have some different pattern that could be in risk for the security.For Airports we can make statistics of people coming to some places to improve the operations performance.

## Assess Effects on End User Needs
The camera Angle can affect the frame images and the lighting effect. If there is much dark it can also affect the frame images and the model accuracy. 

## Model Research
I used Tensorflow models for Object detection Model and I also Installed a virtual Machine to try the models with Ubuntu and OpenVino toolkit, [TF detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md), [OpenVINO public models](https://github.com/opencv/open_model_zoo/blob/master/models/public/index.md)

### Model1: ssd_mobilenet_v1_coco  To Download model:   
```  wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz  ```  Extracting the tar.gz file: `tar -xzvf ssd_mobilenet_v1_coco_2018_01_28.tar.gz`  Change path to `cd ssd_mobilenet_v1_coco_2018_01_28`  Convert the model to an Intermediate Representation with the following arguments:   ```  python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json  ```  The model was insufficient.It was not dectinting person properly. I tried to improve the model for the app changing parameters but I had to try with other models I explained later.  ### Model 2:faster_rcnn_inception_v2_coco To Download model:  ``` wget http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz ```Extracting the tar.gz file: `tar -xzvf ssd_inception_v2_coco_2018_01_28.tar.gz`  Change path to `cd ssd_inception_v2_coco_2018_01_28`  Convert the model to an Intermediate Representation with the following arguments:   ```  python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json  ```  The model was insufficient also. Because it was a slow fps.



### Model 3: ssd_mobilenet_v2_coco  Download model:  
``` wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz ```  Extracting the tar.gz file: `tar -xzvf ssd_mobilenet_v2_coco_2018_03_29.tar.gz`  Change path to `cd ssd_mobilenet_v2_coco_2018_03_29`  Convert the model to an Intermediate Representation with the following arguments:   ```  python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json  ```  Finally the SSD mobilenetv2_coco model is better than others and gives expected results with 0.2 probability threshold 
## Run the application
From the main directory:
### Step 1 - Start the Mosca server
```cd webservice/server/node-servernode ./server.js
```
You should see the following message, if successful:
```Mosca server started.
```
### Step 2 - Start the GUI
Open new terminal and run below commands.
```cd webservice/uinpm run dev
```
You should see the following message in the terminal.
```webpack: Compiled successfully
```
### Step 3 - FFmpeg Server
Open new terminal and run the below commands.
```sudo ffserver -f ./ffmpeg/server.conf
```
### Step 4 - Run the code
Open a new terminal to run the code.
#### Setup the environment
You must configure the environment to use the Intel® Distribution of OpenVINO™ toolkit one time per session by running the following command:
```source /opt/intel/openvino/bin/setupvars.sh -pyver 3.5```You should also be able to run the application with Python 3.6, although newer versions of Python will not work with the app.
#### Running on the CPU
When running Intel® Distribution of OpenVINO™ toolkit Python applications on the CPU, the CPU extension library is required. This can be found at:
```/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/```#### Run app```python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.2 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
```To see the output on a web based interface, open the link  [http://0.0.0.0:3004](http://0.0.0.0:3004/)  in a browser.