#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
torch.cuda.empty_cache()

from fastai.text.all import *


# In[2]:


dls = TextDataLoaders.from_folder(
           untar_data(URLs.IMDB), bs=8, valid='test'
)

learn = text_classifier_learner(dls, AWD_LSTM, drop_mult=0.5, metrics=accuracy)

learn.fine_tune(4, 1e-2)

learn.predict("I really do not like the movie the matrix!")


# In[ ]:




