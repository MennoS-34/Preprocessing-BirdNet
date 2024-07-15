import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.signal import firwin, lfilter, wiener
import pywt
import soundfile as sf
import noisereduce as nr


# Wavelet-Based Denoising
def wavelet_denoise(input_file, output_file, wavelet='db4', level=4, threshold_factor=1.5):
    audio_data, sample_rate = sf.read(input_file)
    coeffs = pywt.wavedec(audio_data, wavelet, level=level)
    sigma = (np.median(np.abs(coeffs[-1])) / 0.6745)
    threshold = threshold_factor * sigma * np.sqrt(2 * np.log(len(audio_data)))
    coeffs_thresh = [pywt.threshold(c, threshold, mode='soft') if i > 0 else c for i, c in enumerate(coeffs)]
    denoised_audio = pywt.waverec(coeffs_thresh, wavelet)
    denoised_audio = denoised_audio[:len(audio_data)]
    sf.write(output_file, denoised_audio, sample_rate)

"""
# Spectrogram-Based Denoising
def spectrogram_denoise(input_file, output_file):
    audio_data, sample_rate = sf.read(input_file)
    f, t, Sxx = plt.specgram(audio_data, Fs=sample_rate)
    denoised_Sxx = np.median(Sxx, axis=1, keepdims=True)
    reconstructed_audio = np.dot(denoised_Sxx.T, np.exp(1j * 2 * np.pi * np.random.rand(*denoised_Sxx.shape)))
    sf.write(output_file, reconstructed_audio.real, sample_rate)
"""


# FIR Filter
def fir_bandpass_filter(input_file, output_file, lowcut=800, highcut=9000, num_taps=101):
    audio_data, sample_rate = sf.read(input_file)
    b = firwin(num_taps, [lowcut, highcut], fs=sample_rate, pass_zero='bandpass')

    if len(audio_data.shape) == 1: 
        filtered_audio = lfilter(b, 1, audio_data)
    else:  
        filtered_audio = np.array([lfilter(b, 1, channel) for channel in audio_data.T]).T
    
    sf.write(output_file, filtered_audio, sample_rate)


# Wiener Filter
def wiener_filter(input_file, output_file):
    audio_data, sample_rate = sf.read(input_file)
    denoised_audio = wiener(audio_data)
    sf.write(output_file, denoised_audio, sample_rate)

# NoiseReduce
def noise_reduce(input_file, output_file):
    audio_data, sample_rate = sf.read(input_file)
    reduced_noise_audio = nr.reduce_noise(y=audio_data, sr=sample_rate)
    sf.write(output_file, reduced_noise_audio, sample_rate)

def process_folder(input_folder, output_folder, denoise_function, **kwargs):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    i=0
    for filename in os.listdir(input_folder):
        if filename.endswith(".wav"):
            i += 1
            print( f"Progress:{i}  of {str(denoise_function)}")
            input_file = os.path.join(input_folder, filename)
            output_file = os.path.join(output_folder, filename)
            denoise_function(input_file, output_file, **kwargs)

# Example usage
input_folder = 'D:\\Menno\\Thesis\\Expirement\\Data\\North_test\\'
output_folder = 'D:\\Menno\\Thesis\\Expirement\\Data\\Denoised Audio\\'

# Apply each denoising method to all files in the folder
#process_folder(input_folder, os.path.join(output_folder, 'PCA'), pca_denoise)
#process_folder(input_folder, os.path.join(output_folder, 'Wavelet'), wavelet_denoise, wavelet='db4', level=4, threshold_factor=2)
process_folder(input_folder, os.path.join(output_folder, 'FIR'), fir_bandpass_filter)

#process_folder(input_folder, os.path.join(output_folder, 'wiener'), wiener_filter)
#process_folder(input_folder, os.path.join(output_folder, 'noise_reduce'), noise_reduce)
