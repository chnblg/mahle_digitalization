#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pylab as plt 
import warnings
import pandas as pd
import os

from librosa import display
from processing import preprocessing, data_augmentation


# In[2]:


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
warnings.filterwarnings("ignore")


# In[3]:


data_df = pd.read_json('/datasets/UrbanSound8K/processed/mean_mfcc_data.json')
data_df = preprocessing.filter_mfccs(data_df)


# In[4]:


i = 1
print(data_df.label.iloc[i])
display.specshow(data_df.feature.iloc[i], x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.show()


# In[5]:


for i in range(5):
    print(data_df.label.iloc[i])
    display.specshow(data_df.feature.iloc[i], x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.show()
    display.specshow(data_augmentation.time_shifting(data_df.feature.iloc[i]), x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.show()


# In[ ]:




