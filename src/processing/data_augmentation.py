import numpy as np
import random

from sklearn.utils import shuffle

random.seed(111)

#code from https://www.kaggle.com/davids1992/specaugment-quick-implementation
def apply_time_and_frequency_masking_to_spectrum(spec: np.ndarray, num_freq_mask:int=2, num_time_mask: int=2,
                                                freq_masking_max_percentage: float=0.15,
                                                time_masking_max_percentage: float=0.3) -> np.ndarray:
    spec = spec.copy()
    for i in range(num_time_mask):
        spec = time_masking(spec, time_masking_max_percentage)
    for j in range(num_freq_mask):
        spec = freq_masking(spec, freq_masking_max_percentage)
    
    return spec

def time_masking(spec: np.ndarray, time_masking_max_percentage: float) -> np.ndarray:
    all_frames_num, all_freqs_num = spec.shape
    time_percentage = random.uniform(0.0, time_masking_max_percentage)
    num_frames_to_mask = int(time_percentage * all_frames_num)
    t0 = np.random.uniform(low=0.0, high=all_frames_num - num_frames_to_mask)
    t0 = int(t0)
    spec[t0:t0 + num_frames_to_mask, :] = 0
    
    return spec

def freq_masking(spec: np.ndarray, freq_masking_max_percentage: float) -> np.ndarray:
    
    all_frames_num, all_freqs_num = spec.shape
    freq_percentage = random.uniform(0.0, freq_masking_max_percentage)
    num_freqs_to_mask = int(freq_percentage * all_freqs_num)
    f0 = np.random.uniform(low=0.0, high=all_freqs_num - num_freqs_to_mask)
    f0 = int(f0)
    spec[:, f0:f0 + num_freqs_to_mask] = 0
    
    return spec

def noise(spec: np.ndarray, noise_level: float=0.1) -> np.ndarray:
    noise = np.random.normal(0, noise_level, spec.shape)

    return spec + noise

def time_shift_mfcc(spec: np.ndarray, time_shifting_max_percentage: float=0.3) -> np.ndarray:
    spec = spec.copy()
    all_freqs_num, all_frames_num = spec.shape
    random_shifting_factor = random.uniform(-1.0, 1.0)
    num_frames_to_shift = int(random_shifting_factor * time_shifting_max_percentage * all_frames_num)
    spec = np.roll(spec, num_frames_to_shift, axis=1)
    if num_frames_to_shift > 0:
        spec[:, :num_frames_to_shift] = 0
    else:
        spec[:, num_frames_to_shift:] = 0

    return spec

def apply_noise_batch(x_train: np.ndarray, y_train: np.ndarray, 
                      noise_level: float=0.1, factor: int=2) -> np.ndarray:
    x = x_train.copy()
    y = y_train.copy()
    
    for i in range(factor-1):
        x_augment = np.array([noise(spec=xi, noise_level=noise_level) for xi in x_train.copy()])
        x = np.concatenate((x, x_augment))
        y = np.concatenate((y, y_train.copy()))

    x, y = shuffle(x, y)

    return x, y

def apply_masking_batch(x_train: np.ndarray, y_train: np.ndarray, num_freq_mask:int=2, num_time_mask: int=2,
                                                freq_masking_max_percentage: float=0.15,
                                                time_masking_max_percentage: float=0.3, factor: int=2) -> np.ndarray:
    x = x_train.copy()
    y = y_train.copy()
    
    for i in range(factor-1):
        x_augment = np.array([
            apply_time_and_frequency_masking_to_spectrum(spec=xi, 
                num_freq_mask=num_freq_mask,
                num_time_mask=num_time_mask,
                freq_masking_max_percentage=freq_masking_max_percentage,
                time_masking_max_percentage=time_masking_max_percentage) for xi in x_train.copy()])
        x = np.concatenate((x, x_augment))
        y = np.concatenate((y, y_train.copy()))

    x, y = shuffle(x, y)

    return x, y

def apply_time_shift_batch(x_train: np.ndarray, y_train: np.ndarray, 
                           time_shifting_max_percentage: float=0.3, factor: int=2) -> np.ndarray:
    x = x_train.copy()
    y = y_train.copy()
    
    for i in range(factor-1):
        x_augment = np.array([time_shift_mfcc(spec=xi, 
                                              time_shifting_max_percentage=time_shifting_max_percentage) for xi in x_train.copy()])
        x = np.concatenate((x, x_augment))
        y = np.concatenate((y, y_train.copy()))

    x, y = shuffle(x, y)

    return x, y
