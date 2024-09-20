import numpy as np
from scipy.signal import butter, filtfilt, hilbert, lfilter

def stipa(signal, fs, reference=None, fsRef=None, Lsk=None, Lnk=None):
    """
    Computes the Speech Transmission Index (STI) based on the input signal.
    
    Parameters:
        signal (np.array): Input signal.
        fs (int): Sampling frequency of the signal in Hertz.
        reference (np.array, optional): Reference signal. Default is None.
        fsRef (int, optional): Sampling frequency of the reference signal. Default is None.
        Lsk (np.array, optional): Input levels for auditory masking. Default is None.
        Lnk (np.array, optional): Ambient noise levels. Default is None.
    
    Returns:
        tuple: (STI, mk) where STI is the Speech Transmission Index and mk is a 2-by-7 matrix
        of modulation transfer values for each octave band.
    """

    # Default values for parameters
    if fsRef is None:
        fsRef = fs

    if Lsk is None:
        Lsk = np.nan * np.ones(7)

    if Lnk is None:
        Lnk = np.nan * np.ones(7)

    # Flags for ambient noise and auditory masking adjustments
    adjustAmbientNoiseFlag = False
    adjustAuditoryMaskingFlag = False

    if not np.isnan(Lsk).all() and not np.isnan(Lnk).all():
        adjustAmbientNoiseFlag = True
        adjustAuditoryMaskingFlag = True
        Isk = 10 ** (Lsk / 10)
        Ink = 10 ** (Lnk / 10)
    elif not np.isnan(Lsk).all() and np.isnan(Lnk).all():
        adjustAuditoryMaskingFlag = True
        Isk = 10 ** (Lsk / 10)
        Ink = np.zeros(7)
    elif np.isnan(Lsk).all() and not np.isnan(Lnk).all():
        print("Warning: Ambient noise levels alone (Lnk) are insufficient for calculation.")

    # Filter the input signal
    signalFiltered = band_filtering(signal, fs)
    signalFiltered = signalFiltered[int(0.2 * fs):, :]

    # Detect envelope
    signalEnvelope = envelope_detection(signalFiltered, fs)

    # Compute modulation depths
    mk_o = MTF(signalEnvelope, fs)

    # Compute modulation depths for the reference signal
    if reference is not None:
        referenceFiltered = band_filtering(reference, fsRef)
        referenceEnvelope = envelope_detection(referenceFiltered, fsRef)
        mk_i = MTF(referenceEnvelope, fsRef)
        mk = mk_o / mk_i
    else:
        mk = mk_o / 0.55

    # Adjust mk for ambient noise
    if adjustAmbientNoiseFlag:
        mk = adjust_ambient_noise(mk, Isk, Ink)

    # Adjust mk for auditory masking
    if adjustAuditoryMaskingFlag:
        mk = adjust_auditory_masking(mk, Lsk, Isk, Ink)

    # Clip mk values to avoid complex SNR
    mk[mk > 1] = 1

    # Compute SNR and clip it
    SNR = compute_SNR(mk)
    SNR = clip_SNR(SNR)

    # Compute Transmission Index (TI)
    TI = compute_TI(SNR)

    # Compute Modulation Transfer Index (MTI)
    MTI = compute_MTI(TI)

    # Compute Speech Transmission Index (STI)
    STI = compute_STI(MTI)

    return STI, mk


def band_filtering(signal, fs):
    """
    Bandpass filters the input signal in octave bands.
    
    Parameters:
        signal (np.array): Input signal.
        fs (int): Sampling frequency.
    
    Returns:
        np.array: Filtered signal.
    """
    octave_bands = [125, 250, 500, 1000, 2000, 4000, 8000]
    filter_order = 18
    filtered_signal = np.zeros((len(signal), len(octave_bands)))

    for i, center_freq in enumerate(octave_bands):
        filtered_signal[:, i] = apply_octave_filter(signal, center_freq, fs, filter_order)

    return filtered_signal


def envelope_detection(x, fs):
    """
    Detects the intensity envelope of the input signal.
    
    Parameters:
        x (np.array): Filtered signal.
        fs (int): Sampling frequency.
    
    Returns:
        np.array: Detected envelope.
    """
    envelope = x ** 2
    return lowpass_filter(envelope, 100, fs)


def MTF(signal_envelope, fs):
    """
    Computes the Modulation Transfer Function (MTF) from the signal envelope.
    
    Parameters:
        signal_envelope (np.array): Envelope of the filtered signal.
        fs (int): Sampling frequency.
    
    Returns:
        np.array: Modulation depths.
    """
    fm = np.array([[1.6, 1, 0.63, 2, 1.25, 0.8, 2.5], [8, 5, 3.15, 10, 6.25, 4, 12.5]])
    seconds = len(signal_envelope) / fs
    mk = np.zeros((2, 7))

    for k in range(7):
        Ik = signal_envelope[:, k]
        for n in range(2):
            period_duration = np.floor(fm[n, k] * seconds) / fm[n, k]
            index_period = int(period_duration * fs)
            t = np.linspace(0, period_duration, index_period)
            mk[n, k] = 2 * np.sqrt(np.sum(Ik[:index_period] * np.sin(2 * np.pi * fm[n, k] * t))**2 + 
                                   np.sum(Ik[:index_period] * np.cos(2 * np.pi * fm[n, k] * t))**2) / np.sum(Ik[:index_period])
    return mk


def adjust_ambient_noise(mk, Isk, Ink):
    """
    Adjusts modulation depths for ambient noise.
    
    Parameters:
        mk (np.array): Modulation depths.
        Isk (np.array): Signal intensities.
        Ink (np.array): Noise intensities.
    
    Returns:
        np.array: Adjusted modulation depths.
    """
    return mk * (Isk / (Isk + Ink))


def adjust_auditory_masking(mk, Lsk, Isk, Ink):
    """
    Adjusts modulation depths for auditory masking and threshold effects.
    
    Parameters:
        mk (np.array): Modulation depths.
        Lsk (np.array): Signal levels.
        Isk (np.array): Signal intensities.
        Ink (np.array): Noise intensities.
    
    Returns:
        np.array: Adjusted modulation depths.
    """
    Ik = Isk + Ink
    La = np.zeros(6)
    for k in range(6):
        L = Lsk[k]
        if L < 63:
            La[k] = 0.5 * L - 65
        elif 63 <= L < 67:
            La[k] = 1.8 * L - 146.9
        elif 67 <= L < 100:
            La[k] = 0.5 * L - 59.8
        else:
            La[k] = -10

    a = 10 ** (La / 10)
    Iamk = np.hstack(([0], Isk[:-1] * a))
    Ak = np.array([46, 27, 12, 6.5, 7.5, 8, 12])
    Irtk = 10 ** (Ak / 10)

    return mk * (Ik / (Ik + Iamk + Irtk))


def compute_SNR(mk):
    """
    Computes the Signal-to-Noise Ratio (SNR) from modulation depths.
    
    Parameters:
        mk (np.array): Modulation depths.
    
    Returns:
        np.array: SNR values.
    """
    return 10 * np.log10(mk / (1 - mk))


def clip_SNR(SNR):
    """
    Clips the SNR values to the range [-15, 15] dB.
    
    Parameters:
        SNR (np.array): SNR values.
    
    Returns:
        np.array: Clipped SNR values.
    """
    SNR_clipped = np.clip(SNR, -15, 15)
    return SNR_clipped


def compute_TI(SNR):
    """
    Computes the Transmission Index (TI) from SNR values.
    
    Parameters:
        SNR (np.array): SNR values.
    
    Returns:
        np.array: TI values.
    """
    return (SNR + 15) / 30


def compute_MTI(TI):
    """
    Computes the Modulation Transfer Index (MTI) from TI values.
    
    Parameters:
        TI (np.array): TI values.
    
    Returns:
        np.array: MTI values.
    """
    return np.mean(TI, axis=0)


def compute_STI(MTI):
    """
    Computes the Speech Transmission Index (STI) from MTI values.
    
    Parameters:
        MTI (np.array): MTI values.
    
    Returns:
        float: The computed STI.
    """
    alpha_k = np.array([0.085, 0.127, 0.230, 0.233, 0.309, 0.224, 0.173])
    beta_k = np.array([0.085, 0.078, 0.065, 0.011, 0.047, 0.095])
    
    STI = np.sum(alpha_k * MTI) - np.sum(beta_k * np.sqrt(MTI[:-1] * MTI[1:]))
    return min(STI, 1)


def apply_octave_filter(signal, center_freq, fs, order):
    """
    Applies an octave band-pass filter to the input signal.
    
    Parameters:
        signal (np.array): Input signal.
        center_freq (float): Center frequency of the filter.
        fs (int): Sampling frequency.
        order (int): Filter order.
    
    Returns:
        np.array: Filtered signal.
    """
    low = center_freq / np.sqrt(2)
    high = center_freq * np.sqrt(2)
    nyquist = fs / 2
    b, a = butter(order, [low / nyquist, high / nyquist], btype='band')
    return filtfilt(b, a, signal)


from numpy import pad

import numpy as np
from scipy.signal import butter, filtfilt

from scipy.signal import butter, filtfilt, lfilter

def lowpass_filter(signal, cutoff, fs, order=4):
    """
    Applies a low-pass filter to the input signal. Uses lfilter for short signals.
    
    Parameters:
        signal (np.array): Input signal.
        cutoff (float): Cutoff frequency.
        fs (int): Sampling frequency.
        order (int, optional): Filter order. Default is 4.
    
    Returns:
        np.array: Filtered signal.
    """
    nyquist = fs / 2
    b, a = butter(order, cutoff / nyquist, btype='low')

    # Determine the minimum length required for filtfilt
    padlen = 3 * (order * 2)

    print(f"Signal length: {len(signal)}, required padlen: {padlen}")

    if len(signal) >= padlen:
        # Use filtfilt if the signal is long enough
        return lfilter(b, a, signal)
    else:
        # For short signals, use lfilter instead of filtfilt
        print("Signal is too short for filtfilt, using lfilter instead.")
        return lfilter(b, a, signal)


    return filtered_signal




# Example usage:
if __name__ == "__main__":
    signal = np.random.randn(96000)  # Example signal
    fs = 96000  # Sampling frequency
    STI, mk = stipa(signal, fs)
    print(f"STI: {STI}")
