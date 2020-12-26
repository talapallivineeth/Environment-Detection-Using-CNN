#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing modules
from tensorflow.keras.models import model_from_json
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import cv2
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


from tensorflow.keras.models import load_model
model = load_model('narmodel.h5')


# In[3]:


img=cv2.imread('C:/Users/vinu/Pictures/Camera Roll/1111.jpg') #reading the image
resized=cv2.resize(img,(128,128))  #re-sizing the image
new_img=preprocess_input(resized)
reshaped=np.reshape(new_img,(1,128,128,3))
result=model.predict(reshaped)
label=np.argmax(result,axis=1)[0]
if label==0:
    print('Indoor')
else:
    print('Outdoor')


# In[ ]:




