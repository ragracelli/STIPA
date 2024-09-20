import os
import numpy as np
import scipy.io.wavfile as wav
from stipa import stipa  # Certifique-se de que sua função stipa está disponível

def run_verification_tests():
    # Verificação do Teste A.2.2
    print('\nRunning Test A.2.2 - weight factor test')
    print('=======================================')
    
    file_names = [
        'STIPA-sine-pair[125+250]STI=0.13',
        'STIPA-sine-pair[250+500]STI=0.28',
        'STIPA-sine-pair[500+1000]STI=0.4',
        'STIPA-sine-pair[1000+2000]STI=0.53',
        'STIPA-sine-pair[2000+4000]STI=0.49',
        'STIPA-sine-pair[4000+8000]STI=0.3'
    ]
    
    ref_STI = np.array([0.127, 0.279, 0.398, 0.531, 0.486, 0.302])
    test_threshold = 0.001
    
    for i, file_name in enumerate(file_names):
        fs, test_signal = wav.read(os.path.join('verification', 'Annex A.2.2 - weight factor test', f'{file_name}.wav'))
        computed_STI = stipa(test_signal, fs)
        
        print(f'{file_name:<38} Reference STI: {ref_STI[i]:<0.2f} Computed STI: {computed_STI:<0.2f}', end=' ')
        if abs(ref_STI[i] - computed_STI) < test_threshold:
            print('✔')
        else:
            print('✘')

    # Verificação do Teste A.3.1.2
    print('\nRunning Test A.3.1.2 - filter bank phase test')
    print('=============================================')
    
    file_names = [
        'STIPA-sine-edge-carriers-TI=0.1[m=0.059351]',
        'STIPA-sine-edge-carriers-TI=0.2[m=0.11182]',
        'STIPA-sine-edge-carriers-TI=0.3[m=0.20076]',
        'STIPA-sine-edge-carriers-TI=0.4[m=0.33386]',
        'STIPA-sine-edge-carriers-TI=0.5[m=0.5]',
        'STIPA-sine-edge-carriers-TI=0.6[m=0.66614]',
        'STIPA-sine-edge-carriers-TI=0.7[m=0.79924]',
        'STIPA-sine-edge-carriers-TI=0.8[m=0.88818]',
        'STIPA-sine-edge-carriers-TI=0.9[m=0.94065]',
        'STIPA-sine-edge-carriers-TI=0[m=0]',
        'STIPA-sine-edge-carriers-TI=1[m=1]'
    ]
    
    ref_STI = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0, 1])
    test_threshold = 0.01
    
    for i, file_name in enumerate(file_names):
        fs, test_signal = wav.read(os.path.join('verification', 'Annex A.3.1.2 - filter bank phase test', f'{file_name}.wav'))
        computed_STI = stipa(test_signal, fs)
        
        print(f'{file_name:<47} Reference STI: {ref_STI[i]:<0.2f} Computed STI: {computed_STI:<0.2f}', end=' ')
        if abs(ref_STI[i] - computed_STI) < test_threshold:
            print('✔')
        else:
            print('✘')

    # Verificação do Teste C.3.2
    print('\nRunning Test C.3.2 - direct method modulation depth test')
    print('========================================================')
    
    file_names = [
        'STIPA-sinecarrier-M=0',
        'STIPA-sinecarrier-M=0.1',
        'STIPA-sinecarrier-M=0.2',
        'STIPA-sinecarrier-M=0.3',
        'STIPA-sinecarrier-M=0.4',
        'STIPA-sinecarrier-M=0.5',
        'STIPA-sinecarrier-M=0.6',
        'STIPA-sinecarrier-M=0.7',
        'STIPA-sinecarrier-M=0.8',
        'STIPA-sinecarrier-M=0.9',
        'STIPA-sinecarrier-M=1'
    ]
    
    ref_M = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    ref_STI = np.array([0, 0.18, 0.3, 0.38, 0.44, 0.5, 0.56, 0.62, 0.7, 0.82, 1])
    test_threshold = 0.05
    test_overall_offset_threshold = 0.01
    
    for i, file_name in enumerate(file_names):
        fs, test_signal = wav.read(os.path.join('verification', 'Annex C.3.2', f'{file_name}.wav'))
        computed_STI, mk = stipa(test_signal, fs)
        
        print(f'{file_name:<27} Max abs m-value error: {np.max(np.abs(mk - ref_M[i])):<0.3f} Reference STI: {ref_STI[i]:<0.2f} Computed STI: {computed_STI:<0.2f}', end=' ')
        if abs(ref_M[i] - mk) < test_threshold:
            print('✔')
        else:
            print('✘')

    systematic_error = np.sum(np.abs(ref_STI - np.round(computed_STI, 2)))
    print('-----------------------------------------------')
    print(f'Systematic absolute error in the STI: {systematic_error:<0.2f}', end=' ')
    if systematic_error < test_overall_offset_threshold:
        print('✔')
    else:
        print('✘')

    # Verificação do Teste C.4.2
    print('\nRunning Test C.4.2 - direct method filter bank slope test')
    print('=========================================================')
    
    file_names = [
        'Filtertest_lowslope 125',
        'Filtertest_lowslope 250',
        'Filtertest_lowslope 500',
        'Filtertest_lowslope 1000',
        'Filtertest_lowslope 2000',
        'Filtertest_lowslope 4000',
        'Filtertest_lowslope 8000',
        'Filtertest_highslope 125',
        'Filtertest_highslope 250',
        'Filtertest_highslope 500',
        'Filtertest_highslope 1000',
        'Filtertest_highslope 2000',
        'Filtertest_highslope 4000',
        'Filtertest_highslope 8000'
    ]
    
    min_m_value = 0.5
    
    for i, file_name in enumerate(file_names):
        fs, test_signal = wav.read(os.path.join('verification', 'Annex C.4.2', f'{file_name}.wav'))
        _, mk = stipa(test_signal, fs)
        
        mk_f1 = mk[0, i % 7] * 0.55
        mk_f2 = mk[1, i % 7] * 0.55
        
        print(f'{file_name:<29} f1 m-value: {mk_f1:<0.2f} f2 m-value: {mk_f2:<0.2f}', end=' ')
        if all([mk_f1, mk_f2] >= min_m_value):
            print('✔')
        else:
            print('✘')

if __name__ == "__main__":
    run_verification_tests()
