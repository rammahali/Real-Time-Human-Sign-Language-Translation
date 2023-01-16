# Human sign language detection

Real time sign language detector and translator




## About


The Human Sign Language Detection and Translation project is a tool that utilizes advanced deep learning techniques to accurately detect and convert human sign language into text. The goal of this project is to improve communication for the deaf and hard of hearing community. The project is implemented with Python, and uses popular deep learning frameworks including TensorFlow and Keras.  All contributions and feedback are welcome to continue to improve the project.


## Data collection

Human face, pose, right hand , left hand data are collected using mediapipe library and then are stored into numpy arrays , training data consists of 30 frames per video and each frame conists of 30 keypoints 
 
<img src="https://i.ibb.co/HxH2BJB/sign-lang.png" width="400" height="300">

## Currently supported signs

* Hello


<img src="https://res.cloudinary.com/spiralyze/image/upload/f_auto,w_auto/BabySignLanguage/DictionaryPages/hello.svg" width="400" height="300">



* Thanks !

<img src="https://res.cloudinary.com/spiralyze/image/upload/f_auto,w_auto/BabySignLanguage/DictionaryPages/thank_you.svg" width="400" height="300">


* I love you

<img src="https://res.cloudinary.com/spiralyze/image/upload/f_auto,w_auto/BabySignLanguage/DictionaryPages/i_love_you.svg" width="400" height="300">






## Contrubite to this project

Currently this project only consists of 3 signs , and i'm planing to expand it to contain most if not all of the sign languages and i believe this project has a huge potential , to add sign language all you need to do is to

* take a look at 
```
data_collection.py
```
* Add your desired sign to the actions array 
* run the cv2 feed in data_collection.py , a sign recording frame will open
* repeat the sign 30 times 
* open :
```
model_training.py
```
* train the model by just running the code , dont forget to add the new sign name to the actions again

## Used libraries


* Tensorflow
* mediapipe
* Open CV
