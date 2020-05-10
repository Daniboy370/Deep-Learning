# Deep-Learning

Using tensorflow's open-source library for [object detection](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) I implemented two models :

- **SSD - Single Shot Detector (classification and localization)** [[link](https://github.com/Daniboy370/Deep-Learning/blob/master/Side-Projects/SSD_object_detection/img_object_detection.ipynb)] :

![alt text](https://github.com/Daniboy370/Deep-Learning/blob/master/Side-Projects/SSD_object_detection/Images/classified_objects.png)

- **Mask R-CNN inception resnet v2** (Instance segmentation) :

![alt text](https://github.com/Daniboy370/Deep-Learning/blob/master/Side-Projects/SSD_object_detection/Images/instance_segmentation.png)


- **DeepLab_v3 implmentation** (Instance segmentation) [[link](https://github.com/Daniboy370/Deep-Learning/blob/master/Side-Projects/SSD_object_detection/img_object_detection.ipynb)] :

![alt text](https://github.com/Daniboy370/Deep-Learning/blob/master/Side-Projects/SSD_object_detection/Images/comparison.png)

## Citation
The models are based on the following paper :
```
@ARTICLE{7913730,
  author={L. {Chen} and G. {Papandreou} and I. {Kokkinos} and K. {Murphy} and A. L. {Yuille}},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs}, year={2018}, volume={40}, number={4}, pages={834-848},}
```
and [Matterport, Inc](https://matterport.com/) Sunnyvale, CA :

```
@misc{matterport_maskrcnn_2017,
  title={Mask R-CNN for object detection and instance segmentation on Keras and TensorFlow},
  author={Waleed Abdulla},
  year={2017},
  publisher={Github},
  journal={GitHub repository},
  howpublished={\url{https://github.com/matterport/Mask_RCNN}},
}
```
