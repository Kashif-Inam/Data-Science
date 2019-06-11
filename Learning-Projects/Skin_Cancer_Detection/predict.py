''' Importing Libraries '''
import numpy as np
import pandas as pd
from PIL import Image

import keras
from keras.models import load_model


''' Loading a new image '''
new_image = "C:\\Users\\Kashif\\PycharmProjects\\DeepLearning-Course (Udemy)\\SkinCancerDetectionOfBenign&Melignant\\skin-cancer-malignant-vs-benign\\data\\malignant\\6.jpg"


''' Processes the image '''
new_image = np.asarray(Image.open(new_image).convert("RGB"))
new_image = np.array(new_image, dtype='uint8')
new_image = new_image/255
new_image = new_image.reshape(-1,224, 224, 3)


''' Loading the saved model '''
model = load_model('SkinCancer.h5')


''' Making prediction on that image '''
pred = model.predict(new_image)
#print(pred)

if pred[0][0] > pred[0][1]:
    print('The tumor is benign')
elif pred[0][0] < pred[0][1]:
    print('The tumor is malignant')
