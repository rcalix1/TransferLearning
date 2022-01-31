#!/usr/bin/env python
# coding: utf-8

# In[1]:


from fastai.tabular.all import *

path = untar_data(URLs.ADULT_SAMPLE)

dls = TabularDataLoaders.from_csv(path/'adult.csv', path=path, y_names='salary',
            cat_names = ['workclass', 'education', 'marital-status', 'occupation',
                        'relationship', 'race'],
            cont_names= ['age', 'fnlwgt', 'education-num'],
            procs= [Categorify, FillMissing, Normalize]
)

learn = tabular_learner(  dls, metrics=accuracy  )

learn.fit_one_cycle(3)     ## no transfer learning here, just learn from scratch 


# In[ ]:




