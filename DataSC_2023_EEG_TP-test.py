#!/usr/bin/env python
# coding: utf-8

# 

# <div class="alert-warning">
# 
# # <center>Short Jupyter Script to Test Installation</center>
# 
# This is a very short script to enable you to test your python, mne and Jupyter installation.
# It uploads an EEG dataset (in *python-raw.fif* format) and should plot the electroencephalogram in a separate window, which allows you to scroll through the data, to zoom in and to select individual electrodes.
# You can zoom and dezoom by using the + and - keys, respectively.
# Note the dataset needs to be a folder named, "Data", in your Jupyter "Home" page.
# 
# 
# 
# </div>

# In[1]:


import mne

file2read = 'python-raw.fif'   # Define the path to dataset and the dataset title.

rawIn = mne.io.read_raw_fif(file2read, preload=True) # The rawIn variable is our raw object.


# In[3]:


# Plot the data that you loaded. 

get_ipython().run_line_magic('matplotlib', 'widget')
# We can visualise both the EEG and MEG continuous data.

mne.viz.plot_raw(rawIn, events=None, duration=10.0, n_channels=10, title='Raw EEG Data')

