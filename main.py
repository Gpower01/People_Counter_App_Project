
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
    parser.add_argument("-pc", "--perf_counts", type=str, default=False,
                        help="Print performance counters")
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client

## First I will set the image pre-processing parameters
def pre_process_image(frame, n, c, h, w):
    image_p = cv2.resize(frame, (w, h))
    # Change data layout from HWC to CHW
    image_p = image_p.transpose((2, 0, 1))
    image_p = image_p.reshape((n, c, h, w))
    return image_p


def draw_boxes(frame, result, width, height, prob_threshold):
    """
    :Draws bounding box when person is detected on video frame 
    :and the probability is more than the specified threshold
    """
    present_count = 0
    for obj in result[0][0]:
        conf = obj[2]
        if conf > prob_threshold:
            xmin = int(obj[3] * width)
            ymin = int(obj[4] * height)
            xmax = int(obj[5] * width)
            ymax = int(obj[6] * height)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
            present_count += 1
    return frame, present_count


def performance_counts(perf_count):
    """
    print information about layers of the model.
    :param perf_count: Dictionary consists of status of the layers.
    :return: None
    """
    print("{:<70} {:<15} {:<15} {:<15} {:<10}".format('name', 'layer_type',
                                                      'exec_type', 'status',
                                                      'real_time, us'))
    for layer, stats in perf_count.items():
        print("{:<70} {:<15} {:<15} {:<15} {:<10}".format(layer,
                                                          stats['layer_type'],
                                                          stats['exec_type'],
                                                          stats['status'],
                                                          stats['real_time']))

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
    
    # Flag for the input image
    single_image_mode = False
    
    # Initialize the variables
    present_request_id = 0
    last_count = 0
    total_count = 0
    start_time = 0
    
    ### TODO: Load the model through `infer_network` ###
    model = args.model
    cpu_extension = args.cpu_extension
    device = args.device
    n, c, h, w = infer_network.load_model(model, device, 1, 1,
                                          present_request_id, cpu_extension)[1]

    ### TODO: Handle the input stream ###
    if args.input == 'CAM':
        input_feed = 0
    
    # Checks for input image
    elif args.input.endswith('.jpg') or args.input.endswith('.bmp') :
        single_image_mode = True
        input_feed = args.input
    
    # Checks for video file
    else:
        input_feed = args.input
        assert os.path.isfile(args.input), "missing file or file does not exist"
    
    cap = cv2.VideoCapture(input_feed)
    cap.open(input_feed)
    
    # Grab the shape of the input
    prob_threshold = args.prob_threshold
    width = int(cap.get(3))
    height = int(cap.get(4))
    lagtime = 0
    
    # Define the codec and create VideoWriter object for the output video
    # 768x432 to match desired resizing
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 24.0, (768,432))

    ### TODO: Loop until stream is over ###
    while cap.isOpened():
        ### TODO: Read from the video capture ###
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)
        ### TODO: Pre-process the image as needed ###
        image_p = pre_process_image(frame, n, c, h, w)
        ### TODO: Start asynchronous inference for specified request ###
        inf_start = time.time()
        infer_network.exec_net(present_request_id, image_p)
        ### TODO: Wait for the result ###
        if infer_network.wait(present_request_id) == 0:
            det_time = time.time() - inf_start
            ### TODO: Get the results of the inference request ###
            result = infer_network.get_output(present_request_id)

            ### TODO: Extract any desired stats from the results ###
            if args.perf_counts:
                perf_count = infer_network.performance_counter(present_request_id)
                performance_counts(perf_count)
            
            frame, present_count = draw_boxes(frame, result, width, height, prob_threshold)
            inf_time_message = "Status, Inference time: {:.3f}ms"\
                                .format(det_time * 1000)
            cv2.putText(frame, inf_time_message, (15, 15),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)

            ### TODO: Calculate and send relevant information on ###
            client.publish(inf_time_message)
            
            # write new frame
            #print("frame size : ", frame.shape[1] ", " frame.shape[0])
            #out.write(frame)
            
            
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###
            ### When person is detected in the video
            if present_count > last_count:
                start_time = time.time()
                total_count += present_count - last_count
                client.publish("person",
                              json.dumps({"total":total_count}))
            
            ## Calculating the duration a person spent on video
            if present_count < last_count:
                duration = int(time.time() - start_time)
                if duration > 0:
                    # Publish messages to the MQTT server
                    client.publish("person/duration",
                                   json.dumps({"duration": duration + lagtime}))
                else:
                    lagtime += 1
                    log.warning(lagtime)
                
            client.publish("person", json.dumps({"count": present_count}))
            last_count = present_count
            
            if key_pressed == 27:
                break
        
        # Send frame to the ffmpeg server
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()
        ### TODO: Send the frame to the FFMPEG server ###

        ### TODO: Write an output image if `single_image_mode` ###
        if single_image_mode:
            cv2.imwrite('output_image.jpg', frame)
    
    #out.release()
    cap.release()
    cv2.destroyAllWindows()
    client.disconnect()
    infer_network.clean()


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
