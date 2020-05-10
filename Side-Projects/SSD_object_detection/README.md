## DeepLab implementation
Semantic image segmentation with deep convolutional nets, atrous convolution and FCC. Based on Tensorflow's [library](https://github.com/tensorflow/models/tree/master/research/deeplab) available models.

![alt text](https://github.com/Daniboy370/Deep-Learning/blob/master/Side-Projects/SSD_object_detection/Images/SSD_img.png)

## Requirements
Python 3.4, OpenCV (cv2) and other common packages listed in `requirements.txt`.

## Installation
1. Clone this repository
2. Install dependencies
   ```bash
   pip3 install -r requirements.txt
   ```
3. Run setup from the repository root directory
    ```bash
    python3 setup.py install
    ``` 
4. Run main script on 
    * Linux: https://github.com/waleedka/coco
    * Windows: https://github.com/philferriere/cocoapi.
    You must have the Visual C++ 2015 build tools on your path (see the repo for additional details)

## Citation
Use this bibtex to cite this repository:
```
@ARTICLE{7913730,
  author={L. {Chen} and G. {Papandreou} and I. {Kokkinos} and K. {Murphy} and A. L. {Yuille}},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs}, 
  year={2018},
  volume={40},
  number={4},
  pages={834-848},}
```

## Requirements
Python 3.4, TensorFlow 1.3, Keras 2.0.8 and other common packages listed in `requirements.txt`.
