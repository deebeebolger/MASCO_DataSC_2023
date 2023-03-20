#!/usr/bin/env python
# coding: utf-8

# ## FIRST STEPS IN EXPLORING EEG DATA
# 
# Now that we have visualized and explored the raw EEG in the temporal, spatial and spectral domains, we will look at the main data cleaning and data preparation techniques used in cognitive science when working with EEG data. 
# 
# In this example, we will use a dataset from the Three-stimulus Auditory Oddball task. 
# This dataset is in a different format to the previous dataset, *.edf* (European Data Format), so we will use a python-based EDF reader to open the dataset. 

# # ******* Preparing the data for analysis **********************
# 
# The following script is an overview of the basic steps applied to prepare the data for analysis.
# This is called *pre-processing* and involves quite simply cleaning the data. 
# The steps applied here are:
# - Downsampling
# - Filtering 
# - Re-referencing
# - Detecting noisy electrodes
# 

# In[ ]:


import os
import matplotlib
matplotlib.use('qtagg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pyedflib
import mne
import ipympl
import pandas as pd

sns.set(rc={'figure.figsize':(8, 6)},
        font_scale=1.5)
sns.set_style('whitegrid')

qt_api = os.environ.get('QT_API')

print(plt.get_backend())

'''
    Load in an EDF dataset from the AuditoryOddball_TBI study folder.
    We will read in the edf file using the pyedflib package.
'''
#get_ipython().run_line_magic('matplotlib', 'qt')
fname = 'sub-004_ses-01_task-ThreeStimAuditoryOddball_eeg.edf'
rawmed1 = pyedflib.EdfReader(fname)


n = rawmed1.signals_in_file                        # Find the data in the file
signal_labels = rawmed1.getSignalLabels()          # Find the channels names
print(f'The physical dimension of signal is {rawmed1.getPhysicalDimension(0)}\n')
print(f'The physical maximum of signal is {rawmed1.getPhysicalMaximum(0)}\n')
print(f'The file header: {rawmed1.getHeader()}\n')
print('The signal labels are: ', signal_labels)
sigbufs = np.zeros((n, rawmed1.getNSamples()[0]))  # Initialize a sigbuf array to receive the data samples
for i in np.arange(n):                             # For each channel add the data to the sigbuf array.
    sigbufs[i, :] = rawmed1.readSignal(i)


# ## TASK 1
# The sampling rate of the current dataset is 500Hz.
# This means that the signal is sampled 500 times in each second.
# Given this information, how can create a time vector that will tell us the time (in seconds) of every sample of data?
# There is some code included below to help you...
# 
# First clue: if the sampling frequency is 500Hz, what is the time interval between each sample?

## -----------------------We know the sampling rate of the data, so can we construct the time vector?----------------------
srate = rawmed1.getSampleFrequency(0)
datasize = sigbufs.shape      # This will give us the size of the sigbufs array:
                              # datasize[0] = number of channels datasize[1]= number of samples
X = datasize[1]
step = 1/srate
time = np.arange(0, X*step, step)

print(f'The sampling frequency is {srate}Hz\n')
print(f'The dimension of the sigbufs object is {datasize}')
print(f'The step in seconds is {step}seconds\n')
print(f'The length of the data in samples is {datasize[1]} samples\n')
print(f'The length of the time vector is: {len(time)}')

# ## Now plot a single channel
# We will plot the Cz channel using the time vector that we just calculated above.
# We use the *index()* method to get the index of the Cz channel.

#get_ipython().run_line_magic('matplotlib', 'inline')
## Now we can plot the data of a single electrode over time.
## We want to plot the Cz electrode...
chanidx = signal_labels.index('Cz')      # Find the index of the Cz electrode.
plt.plot(time,sigbufs[chanidx, :])       # Plot the Cz signal
plt.xlabel('time (seconds)')
plt.show()

# ### Plot an individual channel over a defined time interval
# Here we will just plot the data of the Pz channel over the 60second to 70second time interval.
# This means that we need define a shorter time interval.
### But we may want to visualize individual channel data for a pre-defined time interval.
"""Need to consctruct the new time vector"""
lims_sec    = np.array([60, 70])               # We will define the limits of the time interval, from 60seconds to 70seconds
lim1, lim2  = (lims_sec * srate).astype(int)   # Find the indices of the start and end of chosen time interval
chan2plot   = 'Pz'                             # The index of the channel that you want to plot
chanindx2   = signal_labels.index(chan2plot)
RawIn_sel   = sigbufs[chanindx2, lim1:lim2]     # Extract the raw data of interest

# Now plot the time interval of data.
t = time[lim1:lim2]                             # We define a new time vector, t, as being between lim1 and lim2
plt.plot(t,RawIn_sel)
plt.show()

# ### Plot several channels over a defined time interval
# 
# Here we will plot several channels over a predefined interval.
# We will plot these channels on the same plot, one above the other.

# In[ ]:


chans_sel    = ['C3', 'Cz', 'C4']                                     # Define the channels that you want to plot.
chanidx3 = [signal_labels.index(item2) for item2 in chans_sel ]       # Find the indices of the channels that you want to plot.

RawIn_sel2 = sigbufs[chanidx3, lim1:lim2]   # Extract the data from the
yoffset    = np.array([.001, 0, .001])      # Define a y-offset to seperate the channels
y          = RawIn_sel2.T + yoffset         # Extract the magnitude data for the selected channel
                                        # RawIn_sel2.T finds the transpose of the data array - exchange between columns and rows
pline = plt.plot(t, y)
plt.xlabel('Time (seconds)')
plt.ylabel('Magnitude')
plt.legend(pline, chans_sel)                 # We include a legend to show which signal corresponds to which channel.
plt.show()


# ## Convert the continuous data into an MNE Raw object.
# ![](figures/mne.png)
# 
# The Python-MNE package is a very popular package for the processing and analysis of EEG and MEG data.
# MNE-Python provides many functions to visualize and explore EEG data.
# 
# We create a what is called in MNE-Python a **Raw object** using the data that we loaded above.
# To create this Raw object we will need the following information:
# - the sampling rate (in Hertz) of the data
# - channel labels
# - channel types (EEG, EOG, MEG etc.)
# 
# Of course, to create this raw object and to begin using the MNE functions, we need to have imported mne package.
# You will notice above that we do not include all the channels when creating the MNE Raw object, we exclude the external channels, EXG1 and EG2

# In[ ]:


## ---------------- It is much easier to manipulate and visualize the data using the MNE Package -----------------------.
"""
    So we will create a simple MNE raw object called RawIn
    Initialize an info structure with the following information:
    - sampling rate (srate)
    - channel labels (signal_labels)
    - channel types (eeg) - we need to create this list
"""
# Create the channel type list. All channels are type EEG.

siglabs   = signal_labels
chantypes = ['eeg'] * len(siglabs)
sigIn     = sigbufs[0:len(siglabs), :]
info = mne.create_info(ch_names=siglabs, ch_types=chantypes, sfreq=srate)
RawIn = mne.io.RawArray(sigIn, info)


# ### Drop unwanted channels and mark channels
# We will not use the status channel here, so we will drop it.
# We will define the type of the 'VEOG' channel as an ocular channel (eog).
# This means that we can distinguish it from the scalp electrodes.

# In[ ]:


RawIn.drop_channels('Status')
RawIn.set_channel_types({'VEOG': 'eog'})


# ### The **Raw object**
# If you take a look inside the **RawIn** object, you will see that it has different *attributes* such as:
# - n_times : number of time samples
# - ch_names : the names of the channels
# - times : the time vector
# - an *info* dictionnary with acquisition details such as sampling rate, labels of channels marked as *bad* etc.
# 
# Below we will access this information and print it to screen.

# In[ ]:


T = RawIn.times
Allchans =  RawIn.info['ch_names']
badchans =  RawIn.info['bads']
sampfreq =  RawIn.info['sfreq']
chtypes  = RawIn.get_channel_types()   # Get the channel types

print('The sampling frequency is: ', sampfreq)
print('The first 5 channel names are: {}'.format(', '.join(Allchans[:5])))
print(f'The channel types are {chtypes}\n')


# ### Plotting All EEG Channels
# 
# Now we will plot the raw signals of all channels stacked on above the other over time.
# In the following two cells, we apply two different ways of plotting the EEG signals.
# 
# Note that **remove_dc** is set to **True** or "On". What do you think this means?
# 
# What might we expect if we set **remove_dc** to **False**?

# In[ ]:


##### Visualise all electrode activity #####

scale_dict = dict(mag=1e-12, grad=4e-11, eeg=20e-6, eog=150e-6, ecg=5e-4,
     emg=1e-3, ref_meg=1e-12, misc=1e-3, stim=1,
     resp=1, chpi=1e-4)

#get_ipython().run_line_magic('matplotlib', 'qt')
mne.viz.plot_raw(RawIn, events=None, duration=10, start=0, n_channels= 20, scalings='auto', remove_dc=True)

## We can also plot the data in the RawIn object by using RawIn's '"plot" method

RawIn.plot(duration= 20, start = 60, scalings=scale_dict, remove_dc=True, )


# #### WHAT IF WE WANT TO PLOT ONLY A SPECIFIC TIME INTERVAL?
# 
# In the cell below, you have to plot a single channel, Cz, over the for the 60-70second time window.
# You first need to construct the time vector.
# 
# 
# Some help:
# - You will need to know the sampling frequency (Hz or samples per second) of the data.
# - The time vector and data vector, corresponding to data from Cz electrode, need to have the same length.
# - Need to find the index of Cz electrode.

# In[ ]:


### But we may want to visualize individual channel data for a pre-defined time interval.

lims_sec    = np.array([60, 70])
lim1, lim2  = (lims_sec * sampfreq).astype(int)   # Find the indices of the start and end of chosen time interval
chan_idx    = Allchans.index('Cz')                # The index of the channel that you want to plot
RawIn_sel   = RawIn[chan_idx, lim1:lim2]          # Extract the raw data of interest

#get_ipython().run_line_magic('matplotlib', 'inline')
t = RawIn_sel[1]                             # Extract the time vector
y = RawIn_sel[0].T                           # Extract the magnitude data for the selected channel
plt.plot(t, y)
plt.xlabel('Time (seconds)')
plt.ylabel('Magnitude')
plt.show()


# #### USE OF RAW METHODS TO GET THE INDICES OF TIME POINTS (IN SECONDS)
# The Raw method **time_as_index()** can be used to convert a time, in seconds, into an integer index.
# The times can be presented as a list or an array of times and, in that case, will return an array of indices.
# 
# In addition, we can also index our Raw object, RawIn, using the channel names rather than the indices.
# Here we select 3 central channels to plot in a stacked plot.
# So as to differentiate the signals of each channel, we define an offset for the y axis.

## Use of the Raw method "time_as_index" to find the index
Lims = RawIn.time_as_index(lims_sec)
list_idx = Lims.tolist()
print('The start and end indices of the 60 to 70sec time interval is : ', list_idx)

chan_sel    = ['C3', 'Cz', 'C4']                                     # The index of the channel that you want to plot
RawIn_sel2 = RawIn[chan_sel, Lims[0]:Lims[1]]
yoffset = np.array([.001, 0, .001])
t = RawIn_sel2[1]                             # Extract the time vector
y = RawIn_sel2[0].T + yoffset                         # Extract the magnitude data for the selected channel
pline = plt.plot(t, y)
plt.xlabel('Time (seconds)')
plt.ylabel('Magnitude')
plt.legend(pline, chan_sel)


# ## Assign a montage to our raw object
# 
# This is necessary if we want to plot topographies of our data. 
# Here we are using the standard 10-20 system.

'''
    Add the montage information to the current raw object, RawIn.
    This is required if you want to plot the topography maps.
'''
montage = mne.channels.make_standard_montage('standard_1020')               # Assigning the standard 10-20 montage
mne.viz.plot_montage(mne.channels.make_standard_montage('standard_1020'))   # Visualize the montage
RawIn.set_montage(montage)


# In[ ]:


## To test the effect of the DC offset, we will find the mean of a few electrodes.
"""
    Find the mean of the data from several channels.
    What can we say about the means?
"""
RawIn_temp = RawIn.copy()                     # Create a copy of the RawIn object and name it RawIn_temp
RawIn_temp.pick_channels(['F3', 'Fz', 'F4'])  # We are going to compute of a subset of channels.
dataIn    = RawIn_temp.get_data()             # Extract the data from the RawIn object.
data_mean = np.mean(dataIn, 1)                # We want to find the mean over the time samples, so the 2nd dimension.
Dmean = data_mean.tolist()                    # Converting the array to a list.
print('The mean for each  electrode: {} '.format(Dmean))


# ## 1. DOWNSAMPLING
# 
# Represents the number of times per second that the acquisition system samples the continuous EEG.
# So, given sampling frequency (or sampling rate) of 1024Hz, this means that the system samples the signal every ______ seconds?
# 
# The sampling rate has an effect on the analyses that we can carry out on the EEG.
# For example, if we are interested interested in studying EEG activity around 80Hz, sampling frequency needs to be **at least** twice this frequency of interest - this is the **Nyquist Rule**.
# 
# However, having a high sampling frequency also implies having a greater volume of data. This can mean longer computing times when we are analysing our data.
# Generally, in EEG analysis, we are interested in activity in the 0.1Hz to 80Hz frequency band. This means that we do not necessarily need to have a sampling frequency as high as 1024Hz; a sampling frequency of 512Hz or 250Hz will be sufficient to capture the characteristics of the EEG of interest.
# 
# To reduce the rate at which our EEG is sampled, we can **resample** or **downsample** our data.
# - How does resampling change the EEG signal?
# - What other variable is automatically changed when we resample the EEG data?

rsamp = srate/2                                    # Downsample to half of the original sampling frequency.
RawIn_rs = RawIn.copy().resample(sfreq=rsamp)      # Create a copy of RawIn and apply downsampling to this copy.


# # 2. Filtering
# 
# In EEG, we generally filter to remove high frequency artifacts and low frequency drifts.
# We can filter our time-domain data, our continuous EEG.
# We can also filter our spatial-domain data using spatial filters.
# 
# We begin by filtering our time-domain data:
# - we apply a high-pass filter to remove low frequency drifts
# - we apply a low-pass filter to remove high frequency artifacts.


## Filter the EEG Signal.
#  High-pass filter with limit of 0.1Hz. 
#  Note that we create a copy of the original rawIn object before filtering.

RawIn_hifilt = RawIn.copy().filter(0.1, None, fir_design='firwin')


## Filter the EEG Signal
#. Low-pass filter with a limit of 40Hz
#  Note that we create a copy of the original rawIn object before filtering.

RawIn_lofilt = RawIn_hifilt.copy().filter(None, 40, fir_design='firwin')


# # 3. Re-referencing 
# 
# The potential measured in microVolts is measured in relation to the potential at another point, called the reference.
# 
# This means that the activity at each channel is interpreted relative to the potential at a reference.
# - the reference can be the mean activity of all electrodes.
# - the average of the two mastoids (generally these reference channels are marked as Ref1, Ref2 or EXG1, EXG2)
# The current dataset does not have the external (EXG) channels, so we will apply an average reference.
# 
# However, we cannot include the bad channels or the VEOG when applying the reference.
# We use the *pick_types()* method to exclude these channels when applying the average reference.
# 
# <a href="https://predictablynoisy.com/mne-python/generated/mne.set_eeg_reference.html"> Link to MNE page on **mne.set_eeg.reference()**</a>

# In[ ]:


'''
    Note that we are excluding the eog channel and the bad channels from the average reference calculation.
'''
RawIn_ref = RawIn_lofilt.copy().pick_types(eeg=True, exclude= ['bads','misc', 'stim']).set_eeg_reference()


# # 4. Detecting noisy electrodes 
# 
# Look at the class presentation for an overview of different noise sources in EEG.
# 
# Different approaches can be taken to identify those electrodes to reject from further analysis:
# - Manual detection, manual annotation of the data.
# - Study the spectrum of the data to detect outliers.
# - Automatic detection of outliers based on measures based on amplitude, the predictability of the signal, the presence of energy in certain frequency bands.

# ## 4a. Visual Inspection and Annotation of Data
# 
# Visually inspect the raw data, **RawIn_ref** by calling **RawIn_ref.plot()
# 
# Bad channels are color coded gray. By clicking the lines or channel names on the left, you can mark or unmark a bad channel interactively. You can use +/- keys to adjust the scale (also = works for magnifying the data). Note that the initial scaling factors can be set with parameter scalings. If you don’t know the scaling factor for channels, you can automatically set them by passing scalings=’auto’. With pageup/pagedown and home/end keys you can adjust the amount of data viewed at once.
# 
# You can enter annotation mode by pressing a key. In annotation mode you can mark segments of data (and modify existing annotations) with the left mouse button. You can use the description of any existing annotation or create a new description by typing when the annotation dialog is active. Notice that the description starting with the keyword 'bad' means that the segment will be discarded when epoching the data. Existing annotations can be deleted with the right mouse button. Annotation mode is exited by pressing a again or closing the annotation window.
# 
# This functionality can bug a bit!!

# In[ ]:


### HERE WE WILL MANUALLY ANNOTATE THE CONTINUOUS DATA TO MARK EYE-BLINKS OR BIG ELECTRODE JUMPS
# When you want to annotate a bad section press "a"

fig = RawIn_ref.plot(block=True)              # Open the interactive raw.plot window. This should open a separate window.
fig.canvas.key_press_event('a')

'''
    Alternatively we can visualize the EEG using the mne.viz routine.
'''

#get_ipython().run_line_magic('matplotlib', 'qt')
mne.viz.plot_raw(RawIn_ref, events=None, duration=10, start=0, n_channels= 20, scalings='auto', remove_dc=True)


# # Task 2:
# Manually annotate the continuous by marking examples of the following, if you find them:
# - Eye-blinks
# - Electrode jumps
# - Cardiac artifact (ECG)
# - Muscle artifacts (EMG)

### HERE WE WILL MANUALLY ANNOTATE THE CONTINUOUS DATA TO MARK EYE-BLINKS OR BIG ELECTRODE JUMPS
fig = RawIn_ref.plot(block=True, scalings='auto')              # Open the interactive raw.plot window. This should open a separate window.
fig.canvas.key_press_event('a')

## We will plot our data and annotate.
#get_ipython().run_line_magic('matplotlib', 'qt')
RawIn_ref.plot(duration= 40, start = 60, scalings='auto', remove_dc=False)


# ## Plotting the Frequency Spectrum of EEG Signals
# 
# When trying to detect noisy electrodes it is helpful to look at the frequency spectrum of the electrodes.
# The presence of low frequency or high frequency activity with a lot of energy can indicate a noisy electrode.
# Below we will plot the **Power Spectral Density (PSD)** for frequencies between 0.5Hz and 40Hz.
# The power spectral density will be plotted in dB.
# You can try plotting it again but setting the dB to **False**, can you see a difference?


#get_ipython().run_line_magic('matplotlib', 'widget')
mne.viz.plot_raw_psd(RawIn_ref, fmin=0.5, fmax=40, dB=True)


# ### Mark the Noisy Channels as "Bad"
# 
# Because activity that corresponds to noise is very often of higher amplitude than the EEG activity that interests us.
# We can detect bad electrodes by considering:
# - the time course of the signals.
# - the frequency spectrum of the signals.
# It is important to detect these souces of noise so that we can exclude them from our analysis.
# Here we will mark the noisy channels as **bad** so that we can exclude them from our analysis.
# 
# Note:
# When we select a channel during annotation, it will be added as "bad" to the info attribute of our data object.

# In[ ]:


'''
        We mark a channel as "bad" by adding it to the "bads" attribute of "info".
'''
ChanBad = ['Fp1']
RawIn_ref.info['bads'] = ChanBad

# Plot the PSD again but without the channel marked as "bad".
mne.viz.plot_raw_psd(RawIn_ref, fmin=0.5, fmax=40, dB=True, exclude='bads')


# # Task 3:
# - Can you find any noisy electrodes that we may need to exclude from our data?
# - Plot the time course and the frequency spectra of these electrodes to justify your choice.
# You can do this task in the cell below.

# In[ ]:


'''
   The code for Task 4 can go here.
'''


# ## Automatic Detection of Eye-Blinks
# In MNE there is a function that automatically identifies eye-blinks.
# It allows you to segment the data around the eye-blinks identified and then plot the spatial distribution of the activity corresponding to eye-blinks.
# However, to identify the eye-blinks you need to define a channel on which eye-blinks clearly appear.
# In the code we have set this channel to be **AF8** but it may not be the best choice.

# In[ ]:


eogev_elec = 'AF8'                                #Put the label of your selected electrode here...try different electrodes.
eog_epochs = mne.preprocessing.create_eog_epochs(RawIn_ref, ch_name=eogev_elec, reject_by_annotation=False)
eog_epochs.apply_baseline(baseline=(None, -0.2))  # We go from the start of the interval to the -200ms before 0ms
eog_epochs.average().plot_joint()
eog_epochs.average().plot_topomap()


# ## Automatic Detection of ECG (Cardiac activity)
# In MNE there is a function that automatically identifies cardiac activity (ECG).
# It allows you to segment the data around the eye-blinks identified and then plot the spatial distribution of the activity corresponding to ECG.
# However, to identify the cardiac artifacts you need to define a channel on which they clearly appear.
# Can you identify any channel that clearly displays cardiac artifact?

# In[ ]:


ecg_elec = '';
ecg_epochs = mne.preprocessing.create_ecg_epochs(RawIn_ref, ch_name=ecg_elec, reject_by_annotation=False)
ecg_epochs.apply_baseline(baseline=(, ))              # Can you suggest a baseline interval for ECG??
ecg_epochs.average().plot_joint()
ecg_epochs.average().plot_topomap()


# ## Plot topomaps of Continuous Data
# Generally, when we plot topomaps of continuous data, we plot the topomaps over a defined interval or, more interesting, we plot the spatial distribution corresponding to different frequency activity in the EEG spectrum.
# In the example below, we look at the spatial distribution of activity at 10Hz, this corresponds to alpha frequency band.
# In this example, we calculate the frequency content of the EEG over time.

from matplotlib import cm
### In continuous data, it is more interesting to look at frequency band activity.
refdata = RawIn_ref.get_data()
spectra, freqs = mne.time_frequency.psd_array_welch(refdata, srate, fmin=1, fmax=40, n_fft=256, n_overlap=0, n_per_seg=None, 
                                                        n_jobs=None, average='mean', window='hamming', verbose=None)
print(freqs)       # Print the frequencies to screen.

# Plot the spectra as a function of frequency.
plt.plot(freqs, spectra.T)
plt.ylabel(r'PSD ($\mu$V^2)')

## To start with, lets plot the topography of alpha activity (10Hz) across our continuous data.
layout = mne.find_layout(RawIn_ref.info, ch_type='eeg', exclude='bads')
mne.viz.plot_topomap(spectra[:, 9], RawIn_ref.info, ch_type='eeg', contours=0)


# ## Segmenting the Continuous Data to look at **Evoked** Activity
# 
# We have been looking at the continuous data.
# But, in EEG, we often like to look at the EEG in relation to a stimulus presented during the experiment.
# We are interested in the activity **evoked** by the stimuli.
# 
# To study this **evoked** activity, we segment our data around the stimuli used in our study.
# This means that the chop the data into segments, called **epochs**, by defining a time interval before the stimulus (**baseline**) and a time interval after the stimulus (**post-stimulus interval**).
# 
# In the example below we will load the *.csv file in which the timing of the stimuli are defined.
# Then we can this **event** data to our Raw object and the segment the continuous data.
# 
# ***
# In the data used here, the stimuli are the following:
# - A Standard Tone
# - A Novel Tone
# - A Target Tone

# In[ ]:


event_data = pd.read_csv(
    'sub-004_ses-01_task-ThreeStimAuditoryOddball_events.csv', sep=';', header=None)
annotations = mne.Annotations(event_data[0], event_data[1], event_data[2])
RawIn_ref.set_annotations(annotations)
events, events_id = mne.events_from_annotations(RawIn_ref)
print(events_id)
print(events)


# In[ ]:


'''
   Segmenting the Continuous Data, we will first segment around all the events.
'''

tmin, tmax = [ -0.1,1 ]
reject_criteria = dict(eeg=40e-6)   # Criterion for epoch rejection
event_dict = {'Novel Tone': 1, 'Standard Tone': 2, 'Target Tone': 3}
# Call of function to segment the data into epochs.
epoch_data = mne.Epochs(RawIn_ref, events, event_id=event_dict, tmin=tmin, tmax=tmax, reject=None, reject_by_annotation=False,
                        baseline=(tmin, 0), preload=True,
                        detrend=None, verbose=True)

fig = mne.viz.plot_events(events, event_id=event_dict, sfreq=RawIn_ref.info['sfreq'],
                          first_samp=RawIn_ref.first_samp)


# ## Plotting Epoched Data
# - Here we plot the epoched data of the *Novel Tone* condition only.
# - The frequency spectrum of the *Novel Tone* activity.
# - An ERP-image and mean activity for the *Novel Tone* condition.

#get_ipython().run_line_magic('matplotlib', 'widget')
epoch_data['Novel Tone'].plot(events=events, event_id=event_dict, scalings='auto', butterfly=True)

epoch_data['Novel Tone'].plot_psd(picks='eeg')

epoch_data['Novel Tone'].plot_image(picks='eeg', combine='mean')


# ## Calculate the Evoked Activity for each condition
# We calculate the **evoked** activity for each condition (or stimulus) by averaging over all the epochs corresponding to that stimulus.
# Now we can compare the EEG activity for each experimental condition.
# 
# **Note:** The results here are not very informative as we are looking at the evoked activity of a single subject.
# Normally, we calculate the average activity over several participants.

epochs_novel = epoch_data['Novel Tone']
epochs_standard = epoch_data['Standard Tone']

# Now we will average over the novel and standard trials. This will give us our evoked activity.
evoked_novel = epochs_novel.average()
evoked_standard = epochs_standard.average()

mne.viz.plot_compare_evokeds(dict(novel=evoked_novel, standard=evoked_standard),
                             legend='upper left', show_sensors='upper right')

epoch_fname = 'audoddball_std-epo.fif'
epochs_standard.save(epoch_fname)
