import streamlit as st
import speech_recognition as sr
import glob
import librosa
from matplotlib import pyplot as plt
import numpy as np


st.set_page_config('Voice Analytics', layout='wide')
st.set_option('deprecation.showPyplotGlobalUse', False)


list_of_files = list(map(lambda x:str(x).split('\\')[-1].split('.')[0], glob.glob('./audio-analytics-sample/*')))
file_option = st.sidebar.selectbox("Please select a file", options=list_of_files)

r = sr.Recognizer()

if file_option != None:
  file_text = None
  for i in glob.glob('./audio-analytics-sample/*'):
    if i.find(file_option) > 0:
        file_text = i
        break
  
  with sr.WavFile(file_text) as src:
    audio=r.record(src)
  
  st.header('Speech Extracted')
  with st.spinner('Extracting speech'):
    st.write(r.recognize_whisper(audio))
  
  y, s_r = librosa.load(file_text)

  col1, col2 = st.columns(2)
  with col1:
    fig, ax = plt.subplots()
    ax.set_title('Waveform')
    img = librosa.display.waveshow(y, sr=s_r, x_axis="time", ax=ax)
    st.pyplot(plt.gcf())
  
  with col2:
    D = librosa.stft(y)  # STFT of y
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    fig, ax = plt.subplots()
    ax.set_title('Wave Spectral')
    img = librosa.display.specshow(S_db, x_axis='time', y_axis='linear', ax=ax)
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    st.pyplot(plt.gcf())
