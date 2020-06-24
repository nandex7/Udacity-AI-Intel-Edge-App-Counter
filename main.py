"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
 
import os
import sys
import time
import socket
import json
import cv2
import numpy as np
import time
# from imutils.video import FPS

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.
    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client() 
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)  
    return client

def output_box(frame, result, args, width, height):   
    count = 0     
    c_id = 0
    for obj_input in result[0][0]:  
        confidence = obj_input[2]  
      
        c_id = int(obj_input[1]) 
        if c_id == 1: 
            if confidence >= args.prob_threshold: 
                x_min = int(obj_input[3] * width)
                y_min = int(obj_input[4] * height)
                x_max = int(obj_input[5] * width)
                y_max = int(obj_input[6] * height)  
                #yellow rgb is 255,255,0, but in opencv bgr  
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 255), 1) 
                Person_confidence = '%s: %.1f%%' % ("Person", round(confidence * 100, 1))
                cv2.putText(frame, Person_confidence, (10, 90), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1)
                count += 1    
                
    return frame, count

def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.
    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold 
    ### TODO: Load the model through `infer_network` ###
    
    model=args.model
    video_input=args.input    
    cpu_extension=args.cpu_extension
    device=args.device
    infer_network.load_model(model, device, cpu_extension)
    
    
    net_input_shape = infer_network.get_input_shape()  
    ### TODO: Handle the input stream ###
    image_flag = False
    
    # If input is CAM 
    if video_input == 'CAM':
        input_stream = 0 
    # If input is image
    elif video_input.endswith('.jpg') or args.input.endswith('.bmp') :
        image_flag = True
        input_stream = video_input
    # input is video file
    else:
        input_stream = video_input
        assert os.path.isfile(video_input), " Video file is not found"
    try:
        cap=cv2.VideoCapture(video_input)
    except FileNotFoundError:
        print("Video cannot be found: "+ video_input)
    except Exception as e:
        print("Possible wrong format with the video file: ", e)
    
    if input_stream and not image_flag: 
        cap = cv2.VideoCapture(video_input)
        cap.open(video_input) 
        
    width = int(cap.get(3))
    height = int(cap.get(4))
    # Process frames until the video ends, or process is exited
    counter = 0
    duration = 0
    total_count = 0
    current_count = 0 
    total_inference_time = 0
    last_count = 0 
    threshold_value = 2  
    ### TODO: Loop until stream is over ###
    color = (255,0,0)

    while cap.isOpened():
        ### TODO: Read from the video capture ###
        flag, frame = cap.read() 
        # FPS
        fps = cap.get(cv2.CAP_PROP_FPS)
        # fpss = FPS().start() 
        # frame counter
        counter += 1
        if not flag:
            break
        key_pressed = cv2.waitKey(60) 
        ### TODO: Pre-process the image as needed ###
        p_image = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
        p_image = p_image.transpose((2,0,1))
        p_image = p_image.reshape(1, *p_image.shape) 
        # calculating time for the performance in different models
        infer_timer = time.time()
        ### TODO: Start asynchronous inference for specified request ###
        infer_network.exec_net(p_image)  
        ### TODO: Wait for the result ### 
        if infer_network.wait() == 0:
            # calculating time for the performance in different models
            inferece_time = time.time() - infer_timer  
            ### TODO: Get the results of the inference request ###  
            result = infer_network.get_output() 
            ### TODO: Extract any desired stats from the results ### 
            frame, count = output_box(frame, result, args, width, height)   
            
            ### TODO: Calculate and send relevant information on ###   
            current_count = count    
            if current_count > last_count:
                # if someone enter frame, time start counting
                start_time = time.time() 
                total_count = total_count + current_count - last_count 
                
            # Person duration in the video is calculated
            if current_count < last_count:
                duration = int(time.time() - start_time)
                # if detection failed and double counted, decrease its value and threshold_value is 1 second
                if duration < threshold_value:
                    total_count = total_count - 1  
                if duration >= 4:
                    ### Topic "person/duration": key of "duration" ###
                    client.publish("person/duration", json.dumps({"duration": duration}))
                    ### Topic "person": keys of "count" and "total" ###
                    client.publish("person", json.dumps({"total": total_count})) 
            
            ### current_count, total_count and duration to the MQTT server ###
            client.publish("person", json.dumps({"count": count}))
            last_count = current_count
             
            if key_pressed == 27:
                break 
        # for performance message
        total_inference_time = inferece_time
        msg = "Inference time: %.3f ms" % (total_inference_time * 1000)  
        
        # fpss.update()
        # fpss.stop()
        # fpss = fpss.elapsed()
        msg_fps = "FPS: " + str(int(fps))
        cv2.putText(frame, msg, (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (205, 0, 0), 1)
        cv2.putText(frame, msg_fps, (10, 60), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (205, 0, 0), 1)
        
        txt = "Current count: %d " %current_count
        txtlastcount = "Last count: %d " %last_count
        cv2.putText(frame, txt, (15, 120), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 1)
        #cv2.putText(frame, txtlastcount, (15, 150), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 1)
        ### TODO: Send the frame to the FFMPEG server ###   
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()
        ### TODO: Write an output image if `single_image_mode` ###
        if image_flag:
            cv2.imwrite('output_image.jpg', frame)
            
    # Release the capture and destroy any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    ### Disconnect from MQTT 
    client.disconnect()   

def main():
    """
    Load the network and parse the output.
    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()