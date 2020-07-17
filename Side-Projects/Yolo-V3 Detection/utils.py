# TODO : put in external file as utils.py (!)

import cv2, os
import numpy as np

PATH_darknet = '/content/gdrive/My Drive/Colab Notebooks/ARpalus-Challenge/darknet'

scale = 0.00395
classes_file = PATH_darknet + '/data/obj.names'
weights = PATH_darknet + '/backup/yolov3_training_last.weights'
config_file = PATH_darknet + '/cfg/yolov3_training.cfg'

# read class names from text file
classes = None
with open(classes_file, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# generate different colors for different classes
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# read pre-trained model and config file
net = cv2.dnn.readNet(weights, config_file)

def run(frame):
    Width = frame.shape[1]
    Height = frame.shape[0]

    # create input blob
    blob = cv2.dnn.blobFromImage(frame, scale, (416,416), (0, 0, 0), True, crop=False)

    # set input blob for the network
    net.setInput(blob)

    # run inference through the network and gather predictions from output layers
    outs = net.forward(get_output_layers(net))

    # initialization
    class_ids = []
    confidences = []
    boxes = []
    # ----- THRESHOLDS ( confidence level ) ----- #
    conf_threshold = 0.01 # 0.5
    nms_threshold = 0.01 # 0.4
    weak_thershold = 0.01

    # for each detetion from each output layer get the confidence, class id, 
    # bounding box params and ignore weak detections (confidence < 0.5)
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > weak_thershold:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    # apply non-max suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    # go through the detections remaining after nms and draw bounding box
    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]

        draw_bounding_box(frame, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))

    return frame

# function to get the output layer names in the architecture
def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

# function to draw bounding box on the detected object with class name
def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])
    color = COLORS[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 12)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_TRIPLEX, 1, color, 2)
