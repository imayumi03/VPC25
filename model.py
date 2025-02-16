#############################################################################
# YOUR ANONYMIZATION MODEL
# ---------------------
# Should be implemented in the 'anonymize' function
# !! *DO NOT MODIFY THE NAME OF THE FUNCTION* !!
#
# If you trained a machine learning model you can store your parameters in any format you want (npy, h5, json, yaml, ...)
# <!> *SAVE YOUR PARAMETERS IN THE parameters/ DICRECTORY* <!>
############################################################################
import numpy as np 
import librosa

def anonymize(input_audio_path):
    """
    Anonymization algorithm

    Parameters
    ----------
    input_audio_path : str
        Path to the source audio file in ".wav" format.

    Returns
    -------
    audio : numpy.ndarray, shape (samples,), dtype=np.float32
        The anonymized audio signal as a 1D NumPy array of type `np.float32`,
        which ensures compatibility with `soundfile.write()`.
    sr : int
        The sample rate of the processed audio.
    """
    y, sr = librosa.load(input_audio_path, sr=None)

    # Calculer la STFT
    D = librosa.stft(y, n_fft=1024, hop_length=256)
    magnitude, phase = np.abs(D), np.angle(D)

    # Formant shifting : warp fréquentiel
    formant_factor = 1.1
    num_bins = magnitude.shape[0]
    freqs = np.linspace(0, 1, num_bins)
    new_freqs = freqs ** formant_factor
    new_freqs = np.clip(new_freqs, 0, 1)

    # Interpolation sur les magnitudes pour chaque trame
    new_magnitude = np.zeros_like(magnitude)
    for i in range(magnitude.shape[1]):
        new_magnitude[:, i] = np.interp(new_freqs, freqs, magnitude[:, i])

    # Reconstruction du signal anonymisé
    D_anonymized = new_magnitude * np.exp(1j * phase)
    y_anonymized = librosa.istft(D_anonymized, hop_length=256)

    # Assurer que le signal est de type float32 (exigence)
    audio = y_anonymized.astype(np.float32)

    return audio, sr