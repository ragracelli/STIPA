import numpy as np
from scipy.signal import butter, filtfilt
from scipy.signal import lfilter

def generate_stipa_signal(duration, fs=96000):
    """
    Generates the STIPA test signal for measurement of the Speech Transmission Index (STI)
    according to the IEC 60268-16:2020 standard.
    
    Parameters:
        duration (float): Duration of the signal in seconds.
        fs (int, optional): Sampling frequency in Hertz. Default is 96 kHz.
    
    Returns:
        np.array: The generated STIPA test signal.
    """
    # Check inputs
    if duration <= 0:
        raise ValueError("Duration must be a positive number.")
    if fs < 22050:
        raise ValueError("Sampling frequency must be at least 22050 Hz.")
    
    # Generate pink noise
    N = int(duration * fs)
    pink_noise = generate_pink_noise(N)
    
    # Filter the pink noise in octave bands
    octave_bands = [125, 250, 500, 1000, 2000, 4000, 8000]
    filter_order = 20
    filtered_pink_noise = np.zeros((N, len(octave_bands)))
    
    for band_idx, center_freq in enumerate(octave_bands):
        filtered_pink_noise[:, band_idx] = apply_octave_filter(pink_noise, center_freq, fs, filter_order)
    
    # Modulate the frequencies
    fm = np.array([[1.6, 1, 0.63, 2, 1.25, 0.8, 2.5],    # modulation frequencies in Hz
                   [8, 5, 3.15, 10, 6.25, 4, 12.5]])
    
    t = np.linspace(0, duration, N)
    modulation = np.zeros((N, len(octave_bands)))
    
    for band_idx in range(len(octave_bands)):
        modulation[:, band_idx] = np.sqrt(0.5 * (1 + 0.55 * (
            np.sin(2 * np.pi * fm[0, band_idx] * t) - np.sin(2 * np.pi * fm[1, band_idx] * t))))
    
    # Set levels of the octave bands
    levels = [-2.5, 0.5, 0, -6, -12, -18, -24]  # revision 5 band levels
    G = 10 ** (np.array(levels) / 20)  # acoustic pressure of bands
    
    # Compute the final STIPA test signal
    signal = np.sum(filtered_pink_noise * modulation * G, axis=1)
    
    # Normalize the RMS of the final signal
    target_rms = 0.1  # empirically derived from the character of the STIPA test signal
    signal = signal * target_rms / np.sqrt(np.mean(signal**2))
    
    return signal


def generate_pink_noise(N):
    """
    Generates pink noise (1/f noise) using the Voss-McCartney algorithm.
    
    Parameters:
        N (int): The number of samples to generate.
    
    Returns:
        np.array: Generated pink noise.
    """
    # Initialize pink noise generation variables
    num_rows = 16
    array = np.zeros((num_rows, N))
    
    # Generate random increments
    for i in range(num_rows):
        step_size = 2**i
        increments = np.random.randn(N // step_size)
        array[i, :N // step_size * step_size] = np.repeat(increments, step_size)
    
    # Sum the rows and normalize
    pink_noise = np.sum(array, axis=0)
    pink_noise = pink_noise / np.max(np.abs(pink_noise))  # Normalize the noise
    
    return pink_noise


def apply_octave_filter(signal, center_freq, fs, order):
    """
    Applies a band-pass filter centered on a given frequency (octave filter).
    
    Parameters:
        signal (np.array): Input signal to filter.
        center_freq (float): Center frequency of the octave filter.
        fs (int): Sampling frequency in Hertz.
        order (int): Filter order.
    
    Returns:
        np.array: Filtered signal.
    """
    low_freq = center_freq / np.sqrt(2)
    high_freq = center_freq * np.sqrt(2)
    
    # Design band-pass Butterworth filter
    nyquist = 0.5 * fs
    low = low_freq / nyquist
    high = high_freq / nyquist
    b, a = butter(order, [low, high], btype='band')
    
    # Apply filter to signal
    filtered_signal = filtfilt(b, a, signal)
    
    return filtered_signal


# Example usage
if __name__ == "__main__":
    duration = 10  # seconds
    fs = 96000  # Hz
    stipa_signal = generate_stipa_signal(duration, fs)
    
    # You can now analyze or save the signal, e.g., using scipy.io.wavfile.write to save as a WAV file.
