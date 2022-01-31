#!/usr/bin/env python
# coding: utf-8

# In[9]:


from fastai.vision.all import *



learn_inf = load_learner('export.pkl')

learn_inf.predict('fruits/apple/r_99_100.jpg')


# In[10]:


learn_inf.dls.vocab


# In[ ]:




