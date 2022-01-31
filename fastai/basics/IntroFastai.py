#!/usr/bin/env python
# coding: utf-8

# In[1]:


## >> pip install ipywidgets
## >> pip install fastai

import numpy as np


# In[2]:


from fastai.vision.all import *


# In[3]:


path = untar_data(URLs.PETS)/'images'

## this function returns True or False


def is_cat(x):
    return x[0].isupper()


dls = ImageDataLoaders.from_name_func(
               path, 
               get_image_files(path),
               seed=42,
               valid_pct=0.2,
               label_func=is_cat,
               item_tfms=Resize(224)
)

learn = cnn_learner(dls, resnet34, metrics=error_rate )

learn.fine_tune(1)


# In[4]:



uploader = widgets.FileUpLoad()

img = PILImage.create(uploader.data[0])

is_cat, _, probs = learn.predict(img)

print(is_cat)

print(probs)


# In[5]:


uploader = widgets.FileUpLoad()


# In[6]:


from ipywidgets import widgets
uploader = widgets.FileUpLoad()


# In[7]:


uploader = widgets.FileUpload()


# In[8]:


uploader


# In[9]:


from ipywidgets import widgets
uploader = widgets.FileUpload()

uploader


# In[ ]:




