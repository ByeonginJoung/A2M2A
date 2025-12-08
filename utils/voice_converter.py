# audio pre-processing
import noisereduce as nr
import librosa
import subprocess
import soundfile as sf
from scipy.signal import butter, lfilter
from pydub import AudioSegment
import librosa.effects

import os
import numpy as np

def make_tts_like_ver2(input_audio, output_file):

    left_channel_audio = 'demo_items/left_channel.wav'
    sox_reduced_audio = 'demo_items/sox_reduced_left_channel.wav'
    noise_profile = 'demo_items/noise.prof'
    normalized_audio_path = output_file
    #eq_audio_path = 'demo_items/eq_audio.wav'
    
    os.system(f'sox {input_audio} {left_channel_audio} remix 1')
    os.system(f'sox {left_channel_audio} -n noiseprof {noise_profile}')
    os.system(f'sox {left_channel_audio} {sox_reduced_audio} noisered {noise_profile} 0.21')

    data, sr = sf.read(sox_reduced_audio)

    reduced_noise = nr.reduce_noise(y=data, sr=sr)
    sf.write('demo_items/noisereduce_audio.wav', reduced_noise, sr)

    # Step 7: Normalize the audio with a safer target
    def normalize_audio(audio, target_dBFS=-14):
        rms = np.sqrt(np.mean(audio**2))
        target_rms = np.power(10.0, target_dBFS / 20.0)
        normalization_factor = target_rms / rms
        normalized_audio = audio * normalization_factor
        return normalized_audio

    normalized_audio = normalize_audio(reduced_noise, target_dBFS=-14)
    sf.write(normalized_audio_path, normalized_audio, sr)

    """
    # Step 8: Apply Equalization (EQ) with a less aggressive high-pass filter
    def apply_eq(audio, sr):
        # High-pass filter (less aggressive)
        sos = butter(6, 80, btype='highpass', fs=sr, output='sos')
        eq_audio = sosfilt(sos, audio)

        # Mid-frequency boost (1kHz - 3kHz)
        sos = butter(6, [1000, 3000], btype='bandpass', fs=sr, output='sos')
        eq_audio = sosfilt(sos, eq_audio)

        return eq_audio
    eq_audio = apply_eq(normalized_audio, sr)
    sf.write(eq_audio_path, eq_audio, sr)
    """

def make_tts_like(input_file, output_file, lowcut=300.0, highcut=3400.0, target_rms=-20.0, pitch_shift_steps=-2):
    # Load the audio file
    data, rate = librosa.load(input_file, sr=None)

    # Step 1: Noise Reduction
    reduced_noise = nr.reduce_noise(y=data, sr=rate)
    temp_file_1 = "demo_items/temp_reduced_noise.wav"
    sf.write(temp_file_1, reduced_noise, rate)

    # Step 2: Equalization
    def butter_bandpass(lowcut, highcut, fs, order=5):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y

    filtered_audio = butter_bandpass_filter(reduced_noise, lowcut, highcut, rate)
    temp_file_2 = "demo_items/temp_filtered_audio.wav"
    sf.write(temp_file_2, filtered_audio, rate)

    # Load into pydub for further processing
    audio = AudioSegment.from_file(temp_file_2, format="wav")

    # Step 3: Apply Dynamic Range Compression
    compressed_audio = audio.compress_dynamic_range()

    # Step 4: Normalize Loudness
    rms_diff = target_rms - compressed_audio.dBFS
    normalized_audio = compressed_audio.apply_gain(rms_diff)
    temp_file_3 = "demo_items/temp_normalized_audio.wav"
    normalized_audio.export(temp_file_3, format="wav")

    # Step 5: Pitch Correction (Optional)
    y, sr = librosa.load(temp_file_3, sr=None)
    pitch_corrected_audio = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch_shift_steps)

    temp_file_4 = 'demo_items/temp_pitch_corrected_audio.wav'
    # Save the final output
    sf.write(temp_file_4, pitch_corrected_audio, sr)

    process = ['sox', temp_file_4, output_file, 'remix', '1', 'norm']
    subprocess.run(process)

    temp_files = [temp_file_1, temp_file_2, temp_file_3, temp_file_4]


# Example usage
#sub_order = 1
#wav_fname = f'sub{sub_order:03d}/2drt/video/sub{sub_order:03d}_2drt_01_vcv1_r1_video.wav'
#wav_fname = '/hdd4tb_00/project/korean/speech_to_2d_mri/demo_items/test0000.wav'

#make_tts_like(wav_fname, "final_output_tts_like.wav")

#print('complete')
