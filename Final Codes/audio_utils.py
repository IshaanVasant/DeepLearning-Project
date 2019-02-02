#Import all the required libraries
#Basic Math operations
import numpy as np 
import math

#Tensorflow specific operations
import tensorflow as tf 

#Signal proccessing operations
import scipy
import librosa
import librosa.filters

#Some parameter values used throughout the audio util functions
sample_rate = 20000
preemph = 0.97
ref_level_db = 20
power = 1.5
griffin_lim_iters = 60
num_freq = 1025
frame_shift_ms = 12.5
frame_length_ms = 50
min_level_db = -100
num_mels = 80

#Load the input audio file
def load_audio(path):
#Parameters: path: Path to input audio file.
#Return: aud: numpy array of audio file
    aud = librosa.core.load(path, sr = sample_rate)[0]
    return aud

#Save the audio waveform
def wav_save(wav, path):
#Parameters: path: Path to output audio file.
#            wav: numpy array of which waveform has to be saved
  wav *= 32767 / max(0.01, np.max(np.abs(wav)))
  scipy.io.wavfile.write(path, sample_rate, wav.astype(np.int16))

#Preemphasize the input signal
def preemphasis(x):
  return scipy.signal.lfilter([1, - preemph], [1], x)

def inv_preemphasis(x):
  return scipy.signal.lfilter([1], [1, -preemph], x)

#Generate normalized spectrogram of the input signal
def spectrogram(y):
#Parameters: y: numpy array of the input audio
#Return    : Normalized spectrogram
  D = _stft(preemphasis(y))
  S = _amp_to_db(np.abs(D)) -ref_level_db
  return _normalize(S)

#Generate waveform using librosa
def inv_spectrogram(spectrogram):
#Parameter: spectrogram: Linear spectrogram of the input signal
#Return   : The audio waveform of input linear spectrogram using griffin lim algorithm
  S = _db_to_amp(_denormalize(spectrogram) + ref_level_db) 
  return inv_preemphasis(_griffin_lim(S ** power))

#Generate waveform using tensorflow
def inv_spectrogram_tensorflow(spectrogram):
#Parameter: spectrogram: Linear spectrogram of the input signal 
#Return   : The audio waveform of input linear spectrogram using griffin lim algorithm
  S = _db_to_amp_tensorflow(_denormalize_tensorflow(spectrogram) + ref_level_db)
  return _griffin_lim_tensorflow(tf.pow(S, power))

def melspectrogram(y):
#Parameter: y: numpy array of the input audio
# Return  : Normalized mel spectrogram of the audio file  
  D = _stft(preemphasis(y))
  S = _amp_to_db(_linear_to_mel(np.abs(D))) - ref_level_db
  return _normalize(S)


def find_endpoint(wav, threshold_db=-40, min_silence_sec=0.8):
  window_length = int(sample_rate * min_silence_sec)
  hop_length = int(window_length / 4)
  threshold = _db_to_amp(threshold_db)
  for x in range(hop_length, len(wav) - window_length, hop_length):
    if np.max(wav[x:x+window_length]) < threshold:
      return x + hop_length
  return len(wav)

#Griffin Lim algorithm in librosa
#Implementation  based on https://github.com/librosa/librosa/issues/434
def _griffin_lim(S):

  angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
  S_complex = np.abs(S).astype(np.complex)
  y = _istft(S_complex * angles)
  for i in range(griffin_lim_iters):
    angles = np.exp(1j * np.angle(_stft(y)))
    y = _istft(S_complex * angles)
  return y

#Griffin Lim algorithm in tensorflow
#Implemetation based on https://github.com/Kyubyong/tensorflow-exercises/blob/master/Audio_Processing.ipynb
def _griffin_lim_tensorflow(S):

  with tf.variable_scope('griffinlim'):
    S = tf.expand_dims(S, 0)
    S_complex = tf.identity(tf.cast(S, dtype=tf.complex64))
    y = _istft_tensorflow(S_complex)
    for i in range(griffin_lim_iters):
      est = _stft_tensorflow(y)
      angles = est / tf.cast(tf.maximum(1e-8, tf.abs(est)), tf.complex64)
      y = _istft_tensorflow(S_complex * angles)
    return tf.squeeze(y, 0)

#Find Short time Fourier transform in librosa
def _stft(y):
#Parameter: y: numpy array of the input audio
#Return: Short time Fourier transform of y
  n_fft, hop_length, win_length = _stft_parameters()
  return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)

#Find inverse Short time Fourier transform in librosa
def _istft(y):
#Parameter: y: numpy array of the short time fourier transform of the signal
#Return: Inverse Short time fourier transform of the signal
  _, hop_length, win_length = _stft_parameters()
  return librosa.istft(y, hop_length=hop_length, win_length=win_length)

#Find Short time Fourier transform in tensorflow
def _stft_tensorflow(signals):
#Parameter: y: numpy array of the input audio
#Return: Short time Fourier transform of y
  n_fft, hop_length, win_length = _stft_parameters()
  return tf.contrib.signal.stft(signals, win_length, hop_length, n_fft, pad_end=False)

#Find inverse Short time Fourier transform in tensorflow
def _istft_tensorflow(stfts):
#Parameter: y: numpy array of the short time fourier transform of the signal
#Return: Inverse Short time fourier transform of the signal
  n_fft, hop_length, win_length = _stft_parameters()
  return tf.contrib.signal.inverse_stft(stfts, win_length, hop_length, n_fft)

#Compute Short Time Fourier Transform Parameters
def _stft_parameters():
#Return: Short Time Fourier transform parameters
  n_fft = (num_freq - 1) * 2
  hop_length = int(frame_shift_ms / 1000 * sample_rate)
  win_length = int(frame_length_ms / 1000 * sample_rate)
  return n_fft, hop_length, win_length


# Conversions:
_mel_basis = None

#Convert linear to mel scale frequency representation
def _linear_to_mel(spectrogram):
#Parameter: spectrogram: Linear scale spectrogram
#Return: mel scle representation
  global _mel_basis
  if _mel_basis is None:
    _mel_basis = _build_mel_basis()
  return np.dot(_mel_basis, spectrogram)

#Build frequency to mel conversion basis
def _build_mel_basis():
  n_fft = (num_freq - 1) * 2
  return librosa.filters.mel(sample_rate, n_fft, n_mels=num_mels)

#amplitude to db conversion in librosa/numpy
def _amp_to_db(x):
  return 20 * np.log10(np.maximum(1e-5, x))

#db to amplitude conversion in librosa/numpy
def _db_to_amp(x):
  return np.power(10.0, x * 0.05)

#db to amplitude conversion in tensorflow
def _db_to_amp_tensorflow(x):
  return tf.pow(tf.ones(tf.shape(x)) * 10.0, x * 0.05)

def _normalize(S):
  return np.clip((S - min_level_db) / -min_level_db, 0, 1)

def _denormalize(S):
  return (np.clip(S, 0, 1) * -min_level_db) + min_level_db

def _denormalize_tensorflow(S):
  return (tf.clip_by_value(S, 0, 1) * -min_level_db) + min_level_db