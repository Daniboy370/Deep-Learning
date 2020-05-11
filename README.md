* MASK R-CNN :  an online implementation of the [MASK R-CNN](https://arxiv.org/abs/1703.06870) paper using Python 3, Keras and TF. My implementation extracts a desired label (out of 80 classes) and emphasize its ROI by converting any other class into B&W [[link](https://github.com/Daniboy370/Deep-Learning/tree/master/Side-Projects/Mask_RCNN)] :

 &nbsp;  &nbsp;  &nbsp;  &nbsp;  &nbsp;  &nbsp;  &nbsp;  &nbsp;  &nbsp;  &nbsp;  &nbsp;  &nbsp;  &nbsp;  &nbsp;  &nbsp;  &nbsp;  &nbsp;  &nbsp;  &nbsp;  &nbsp;  &nbsp;  &nbsp; ![alt text](https://github.com/Daniboy370/Deep-Learning/blob/master/Side-Projects/Mask_RCNN/saved_files/Aladdin_GIF_2.gif)

The model was pretrained on [MS COCO](http://cocodataset.org/#home) dataset for segmented objectes with context. Each frame in the footage undergoes a detection process that returns a python *dictionary* containing bounding boxes, segmentation masks, likeliest detected class and score. Consider the following image of a prime time broadcast, where the reporter was encountered an uninvited visitors :

![alt text](https://github.com/Daniboy370/Deep-Learning/blob/master/Side-Projects/Mask_RCNN/saved_files/Reporter_dogs.png)

The Mask R-CNN returns a label map after after filtering the likeliest classes out of all estimated :

![alt text](https://github.com/Daniboy370/Deep-Learning/blob/master/Side-Projects/Mask_RCNN/saved_files/image_labels_map.png)

An auxiliary function was defined to extract full colors for the 'person' and 'dog' classes, leaving other classes as "0" / background (BG). By default, the Mask R-CNN return any detectable class, shown in color, bounding box, category and confidence level : 

![alt text](https://github.com/Daniboy370/Deep-Learning/blob/master/Side-Projects/Mask_RCNN/saved_files/COCO_test_0.png)


#
Using tensorflow's open-source library for [object detection](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) I implemented two models on frozen images :


* SSD - Single Shot Detector (classification and localization) [[link](https://github.com/Daniboy370/Deep-Learning/blob/master/Side-Projects/SSD_object_detection/img_object_detection.ipynb)] :

![alt text](https://github.com/Daniboy370/Deep-Learning/blob/master/Side-Projects/SSD_object_detection/Images/classified_objects.png)

#
* Mask R-CNN inception resnet v2 (Instance segmentation) :

![alt text](https://github.com/Daniboy370/Deep-Learning/blob/master/Side-Projects/SSD_object_detection/Images/instance_segmentation.png)


#
* DeepLab_v3 implmentation (Instance segmentation) [[link](https://github.com/Daniboy370/Deep-Learning/blob/master/Side-Projects/SSD_object_detection/img_object_detection.ipynb)] :

![alt text](https://github.com/Daniboy370/Deep-Learning/blob/master/Side-Projects/SSD_object_detection/Images/comparison.png)

## Citation
The models are based on the following paper, available [here](https://arxiv.org/abs/1606.00915) :
```
@ARTICLE{7913730,
  author={L. {Chen} and G. {Papandreou} and I. {Kokkinos} and K. {Murphy} and A. L. {Yuille}},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, 
  and Fully Connected CRFs}, year={2018}, volume={40}, number={4}, pages={834-848},}
```
and [Matterport, Inc](https://matterport.com/) Sunnyvale, CA :

```
@misc{matterport_maskrcnn_2017,
  title={Mask R-CNN for object detection and instance segmentation on Keras and TensorFlow},
  author={Waleed Abdulla}, year={2017}, publisher={Github}, journal={GitHub repository},
  howpublished={\url{https://github.com/matterport/Mask_RCNN}},
}
```
#
## Requirements
Python 3.4, TensorFlow 1.3, Keras 2.0.8 and other common packages listed in `requirements.txt`.

### MS COCO Requirements:
To train or test on MS COCO, you'll also need:
* pycocotools (installation instructions below)
* [MS COCO Dataset](http://cocodataset.org/#home)
* Download the 5K [minival](https://dl.dropboxusercontent.com/s/o43o90bna78omob/instances_minival2014.json.zip?dl=0)
  and the 35K [validation-minus-minival](https://dl.dropboxusercontent.com/s/s3tw5zcg7395368/instances_valminusminival2014.json.zip?dl=0)
  subsets. More details in the original [Faster R-CNN implementation](https://github.com/rbgirshick/py-faster-rcnn/blob/master/data/README.md).

If you use Docker, the code has been verified to work on
[this Docker container](https://hub.docker.com/r/waleedka/modern-deep-learning/).


## Installation
1. Clone the desired repository in the root directory
2. Install dependencies 
   ```bash
   pip3 install -r requirements.txt
   ```
3. Run setup from the repository root directory
    ```bash
    python3 setup.py install
    ``` 
3. Download pre-trained COCO weights (mask_rcnn_coco.h5) from the [releases page](https://github.com/matterport/Mask_RCNN/releases).
4. (Optional) To train or test on MS COCO install `pycocotools` from one of these repos. They are forks of the original pycocotools with fixes for Python3 and Windows (the official repo doesn't seem to be active anymore).

    * Linux: https://github.com/waleedka/coco
    * Windows: https://github.com/philferriere/cocoapi.
    You must have the Visual C++ 2015 build tools on your path (see the repo for additional details)


## Technion course homeworks
[Here](https://github.com/Daniboy370/Deep-Learning/tree/master/Homeworks) is a link for several homeworks assignment from the Technion [[crouse](https://www.graduate.technion.ac.il/Subjects.Heb/?Sub=97200)]
