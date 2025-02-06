#streamlit app
import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import librosa
import tensorflow as tf

#main model
autoencoder = tf.keras.models.load_model("model.keras")

def wplot(arr,title="Waveform",xlab='Time (seconds)',ylab='Amplitude',msg="SAMple PLot"):
    """"
        the wave plot of the speech
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    librosa.display.waveshow(arr, sr=16000)
    ax.set_title(title)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    st.pyplot(fig)
    st.subheader(msg)

def spec_plot(spec,title="Spectogram",sr=16000,msg="SAMple PLot"):
    """"
        the spec plot of the speech
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    librosa.display.specshow(spec, sr=sr, x_axis='time', y_axis='log')

    ax.set_title(title)
    ax.set_xlabel('Time')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)
    st.subheader(msg)


def resize_audio(arr, r=16000):
    l = len(arr)
    i = 0
    cnt = 0
    br = []
    brr = []

    while i < l:
        br.append(arr[i])
        i += 1
        cnt += 1
        if cnt == r:
            brr.append(br)
            br = []
            cnt = 0

    if len(br) != 0:
        br.extend([0] * (r - len(br)))
        brr.append(br)

    return brr


def convert(arr, sample_rate=16000, duration_s=1):
    brr = resize_audio(arr)
    brr = np.array(brr)

    audio = []

    # Get duration in samples:
    duration = int(sample_rate * duration_s)

    for data in brr:
        S = np.abs(librosa.stft(data, n_fft=2048))[:-1, :]
        audio.append(S)

    audio = np.array(audio)
    audio = np.expand_dims(audio, -1)

    y_pred = autoencoder.predict(audio)

    res = []
    cnt=0
    audi=[]

    for yp in y_pred:
        crr = np.squeeze(yp, axis=-1)
        aud = librosa.griffinlim(crr)
        if cnt==0:
            audi=aud
            cnt+=1
        res.extend(aud)

    print(len(audi),len(arr))
    return np.array(audi)

nav=st.sidebar.radio("Navigation",["About","Model"],index=1)


if nav=="Model":
    file = st.sidebar.file_uploader("Upload an audio file", type=['wav', 'mp3'])
    st.title("Autoencoder")
    if file is not None:
        data, sr = librosa.load(file)
        S = librosa.stft(data, n_fft=2048)
        st.header("Audio")
        st.write("sample audio file")
        st.audio(file)
        wplot(data, title='Sample Audio',msg="Wave plot")
        spec_plot(S,msg="Spectogram of Audio file")

        pred=convert(data)
        if pred is not None:
            data1, sr1 = librosa.load(pred)
            S = librosa.stft(data1, n_fft=2048)
            st.header("Audio")
            st.write("sample denoised file")
            st.audio(file)
            wplot(data1, title='Predicted Audio',msg="Wave plot")
            spec_plot(S,msg="Spectogram of Predicted file")

elif nav=="About":
    st.title("Speech Enhancement")
    st.write("The project is to improve the quality of speech signal, we are using Autoencoder in it")

    st.image("model.png")
