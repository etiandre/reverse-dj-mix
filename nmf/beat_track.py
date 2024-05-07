import librosa
import numpy as np

def variable_win_stft(y: np.ndarray, beats_idx: np.ndarray, nfft: int):
    spec = np.zeros((1 + nfft // 2, beats_idx.shape[0]-1))
    for i, (left, right) in enumerate(zip(beats_idx, beats_idx[1:])):
        col = np.fft.rfft(y[left:right], nfft)
        spec[:, i] = col
    return spec, beats_idx

def beat_stft(y: np.ndarray, sr, nfft, method="plp"):
    # plt.figure(figsize=(20,6))
    # plt.plot(y)
    if method == "plp":
        hop=384
        win=512
        plp = librosa.beat.plp(y=y, sr=sr, hop_length=hop, win_length=win)
        beats_idx = np.flatnonzero(librosa.util.localmax(plp)) * hop
    elif method == "beat_track":
        _, beats_idx = librosa.beat.beat_track(y=y, sr=sr, units="samples")
    # plt.vlines(beats_idx, -1, 1)
    # max_interval = np.max(np.diff(beats_idx))
    
    spec = np.zeros((1 + nfft // 2, beats_idx.shape[0]-1))
    for i, (left, right) in enumerate(zip(beats_idx, beats_idx[1:])):
        col = np.fft.rfft(y[left:right], nfft)
        spec[:, i] = col
    return spec, beats_idx