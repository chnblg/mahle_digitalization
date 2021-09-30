#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import warnings
import os
import cnn_loop

from processing import preprocessing

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# In[2]:


data_df = preprocessing.load_dataset(num_samples=500)


# In[3]:


data_df = pd.read_json('/datasets/UrbanSound8K/processed/mean_mfcc_data.json')
data_df = preprocessing.filter_mfccs(data_df)


# In[4]:


X_train, X_test, y_train, y_test = preprocessing.create_training_data(data_df)


# In[5]:


num_outputs = data_df['label'].unique().shape[0]  # labels = 10

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
print("num_outputs: ", num_outputs)


# In[6]:


model = cnn_loop.CNN(num_outputs, num_models=4, DP_rate=1.0)


# In[7]:


model.build_model()


# In[8]:


model.initialize(X_test, y_test)


# In[9]:


histories, durations = model.train(X_train, X_test, y_train, y_test, num_epochs=500, batch_size=256)


# In[10]:


model.evaluate_model(X_train, X_test, y_train, y_test)


# In[11]:


model.plot_all_histories()


# In[12]:


model.accuracies_vs_models()


# In[13]:


model.duration_vs_models()

