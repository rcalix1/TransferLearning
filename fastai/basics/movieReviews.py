#!/usr/bin/env python
# coding: utf-8

# In[1]:


from fastai.tabular.all import *
from fastai.collab import *

path = untar_data(   URLs.ML_SAMPLE   )

dls = CollabDataLoaders.from_csv(path/'ratings.csv')

learn = collab_learner(dls, y_range=(0.5, 5.5))

learn.fine_tune(10)    ## used fine_tune here although training from scratch

learn.show_results()


# In[ ]:




