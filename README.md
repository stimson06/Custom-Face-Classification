# Custom-Face-Classification

## Requirements
* [Mediapipe 0.8.6](https://pypi.org/project/mediapipe/)
* [Keras-vggface 0.6](https://pypi.org/project/keras-vggface/)
* [OpenCV 4.5.3](https://pypi.org/project/opencv-python/)
* [Tensorflow 2.5.0](https://pypi.org/project/tensorflow/)
* [Keras 2.4.3](https://pypi.org/project/keras/)

## Sample output
https://user-images.githubusercontent.com/44506282/126061231-79e77c32-dac9-4f62-9990-9c1b0878bb9c.mp4

## Order of execution 
1. **Data_collecion.py** \
Input- User_video \
process - Video --> Image (specific folders) & feature generation \
output - feature.pkl (facial features of each user) 

2. **Recogniser.py** \
Input - feature.pkl \
Output - Real time face classification

## Execution Code
Before execution of the code run Data_collection.py to generate the feature.pickle file
```
from Recogniser import FaceIdentify

face = FaceIdentify(precompute_features_file="./features.pickle")
detected_person = face.detect_face(Instances = 5)
print(detected_person)
```

##### Tested on: Ubuntu 20.04 LTS & Visual Studio
##### Project Status: Under development

