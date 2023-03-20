#!/usr/bin/env python
# coding: utf-8

#  ## Student Task
#  
#  Using the two previous scriptsn(below) as guides:
#  - DataSC_2023_ExploreEEG.ipynb
#  - DataSC_2023_CleaningPrep.ipynb
#  
# Load in the dataset, sub-004_ses-02_task-ThreeStimAuditoryOddball_eeg.edf and carry out the following steps. Each step can be presented in a different cell, if you like.
#  
# 1. Print the most important information concerning the EEG data to the screen. Important information means that information that you will need to visualize, process or interpret the data.
# 
# 2. Visualize all channels in such a way that you scroll through and explore the data across time. 
#                     - Note down the times (onset and offset) of any very noisy intervals.
#                     - Are there only scalp "eeg" channels? What other channels are present in the data?
# 
# 4. Calculate the mean of the midline channels. If the mean > 0, remove the DC offset.
# 
# 5. Reconstruct the time vector (in seconds) and plot the raw data of a selection of **frontal electrodes**. 
# 
# 6. If required, downsample the data. Presuming that we are interested in frequencies up 40Hz, what is the lowest sampling frequency that we can apply?
# 
# 7. Lowpass filter the data, applying a cutoff of 40Hz. Compare the Cz signal before and after filtering.
# 
# 8. Annotate the continuous data, marking the following:
# - muscular artifacts
# - very large eye blinks 
# 
# 9. Note two time intervals presenting these artifacts and present:
#         - the time course of these artifacts; pick channels that will best present the artifact.
#         - the spectrum of these artifacts
#         - the topography of the artifacts.
# 
# 10. Can you identify any particularly noisy electrodes? You can use the spectrum and visualisations to help. 
# 
# 11. Could you come up with a measure based on **amplitude** to identify bad electrodes? 
# 
# 12. Reference the data to the average reference, leaving out bad electrodes. Compare the midline signals before and after referencing.
# 
# 
#  

# In[ ]:




