import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from scipy.signal import spectrogram
from stipa import stipa  # Certifique-se de que suas funções estão disponíveis
from generateStipaSignal import generate_stipa_signal
def demonstration():
    # Gerar sinal de teste STIPA
    duration = 25  # segundos
    fs = 48000  # frequência de amostragem
    stipa_signal = generate_stipa_signal(duration, fs)
    
    print(f"STIPA signal generated with shape: {stipa_signal.shape}")
    print(f"STIPA signal stats: min={np.min(stipa_signal)}, max={np.max(stipa_signal)}, mean={np.mean(stipa_signal)}")

    print(f'Generating {duration} seconds of STIPA test signal sampled at {fs} Hz.')

    # Plotar espectrograma do sinal STIPA gerado
    plt.figure()
    f, t, Sxx = spectrogram(stipa_signal, fs, nperseg=1024, noverlap=512, nfft=1024)
    plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud', vmin=-100)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title('Spectrogram of Generated STIPA Test Signal')
    plt.colorbar(label='Intensity [dB]')
    plt.show()
    print('Plotting spectrogram of the generated STIPA test signal.')

    # Salvar sinal STIPA gerado como arquivo WAV
    filename = 'stipaTestSignal.wav'
    wav.write(filename, fs, stipa_signal.astype(np.float32))
    print(f'Saving the generated STIPA test signal as "{filename}".')

    # Carregar sinal STIPA gravado após transmissão pelo canal
    filename_rec = 'stipaMeasurement.wav'
    fs_rec, stipa_rec = wav.read(f'control_measurements/{filename_rec}')
    print(f'Loading measurement "{filename_rec}".')

    # Plotar espectrograma do sinal STIPA gravado
    plt.figure()
    f, t, Sxx = spectrogram(stipa_rec, fs_rec, nperseg=1024, noverlap=512, nfft=1024)
    plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud', vmin=-100)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title('Spectrogram of Recorded STIPA Signal')
    plt.colorbar(label='Intensity [dB]')
    plt.show()
    print('Plotting spectrogram of the measurement signal.')

    # Recortar silêncio do sinal gravado
    start_idx = 22000
    end_idx = 885000
    stipa_rec_cropped = stipa_rec[start_idx:end_idx]
    cropped_duration = (end_idx - start_idx + 1) / fs_rec
    print(f'Cropping silence segments, the final duration is {cropped_duration:.2f} seconds.')

    # Calcular valor do STI
    computed_sti, _ = stipa(stipa_rec_cropped, fs_rec)  # Capture ambos os valores retornados pela função
    print(f'Computed STI value: {computed_sti:.2f}.')  # Imprime apenas o primeiro valor da tupla (STI)

    # Calcular valor do STI ajustado para mascaramento auditivo e ruído ambiente
    Lsk = np.array([72.2, 72.3, 70.1, 59.7, 51.5, 42.8, 36.4])  # níveis medidos do sinal
    Lnk = np.array([41.7, 42.0, 44.3, 38.1, 24.2, 21.0, 19.6])  # níveis medidos de ruído ambiente
    computed_sti_adjusted, _ = stipa(stipa_rec_cropped, fs_rec, Lsk=Lsk, Lnk=Lnk)
    print(f'Computed STI value adjusted for auditory masking and ambient noise: {computed_sti_adjusted:.2f}.')


if __name__ == "__main__":
    demonstration()
