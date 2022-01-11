# Copyright 2019 RnD at Spoon Radio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""SpecAugment test"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
# import librosa
import numpy as np
import torch
from SpecAugment import spec_augment_pytorch
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

parser = argparse.ArgumentParser(description='Spec Augment')
parser.add_argument('--audio-path', default="/Users/jiyangli/Downloads/train_imu",
                    help='The audio file.')  # '../../stft_label_vIMU/s02_session7_vIMU_stft'
parser.add_argument('--time-warp-para', default=1,
                    help='time warp parameter W')
parser.add_argument('--frequency-mask-para', default=2,
                    help='frequency mask parameter F')
parser.add_argument('--time-mask-para', default=1,
                    help='time mask parameter T')
parser.add_argument('--masking-line-number', default=1,
                    help='masking line number')

args = parser.parse_args()
audio_path = args.audio_path
time_warping_para = args.time_warp_para
time_masking_para = args.frequency_mask_para
frequency_masking_para = args.time_mask_para
masking_line_number = args.masking_line_number

if __name__ == "__main__":

    # Step 0 : load audio file, extract mel spectrogram
    # audio, sampling_rate = librosa.load(audio_path)
    # mel_spectrogram = librosa.feature.melspectrogram(y=audio,
    #                                                  sr=sampling_rate,
    #                                                  n_mels=256,
    #                                                  hop_length=128,
    #                                                  fmax=8000)

    # reshape spectrogram shape to [batch_size, time, frequency]
    # shape = mel_spectrogram.shape
    for root, dirs, files in os.walk(audio_path):
        print("Root = ", root, "dirs = ", dirs, "files = ", files)
    for filename in files:
        if "aug" in filename or "csv" not in filename:
            continue
        print(filename)
        path_one = audio_path + '/' + filename
        spectrogram = pd.read_csv(path_one)
        if spectrogram.shape[1] < 5:
            print("********************")
            print("vIMU is too short!")
            continue
        spectrogram = spectrogram.to_numpy()
        # mel_spectrogram = rgb2gray(mel_spectrogram_raw)

        # reshape spectrogram shape to [batch_size, time, frequency, 1]
        shape = spectrogram.shape
        spectrogram = np.reshape(spectrogram, (-1, shape[0], shape[1]))
        mel_spectrogram = torch.from_numpy(spectrogram)

        # Show Raw mel-spectrogram
        #     spec_augment_pytorch.visualization_spectrogram(mel_spectrogram=mel_spectrogram,
        #                                                   title="Raw Mel Spectrogram")

        # Calculate SpecAugment pytorch
        warped_masked_spectrogram = spec_augment_pytorch.spec_augment(mel_spectrogram=mel_spectrogram)

        new_spectrum = warped_masked_spectrogram.squeeze(2)
        new_spectrum = pd.DataFrame(np.array(new_spectrum)[0])
        save_path = audio_path + "_aug2"
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        filepath = save_path + '/' + filename.split('.')[0] + '_aug_' + \
                   str(time_warping_para) + '_' + str(time_masking_para) + '_' + str(frequency_masking_para)\
                   + '_' + str(masking_line_number) + '.csv'
        new_spectrum.to_csv(filepath, index=False)

        plot = False
        if plot:
            fig, ax = plt.subplots(frameon=False, dpi=100)
            ax.pcolormesh(np.array(mel_spectrogram.squeeze(2)[0]), shading='gouraud', vmax=None)
            plt.title('Raw STFT Magnitude')
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [sec]')
            plt.show()
            fig, ax = plt.subplots(frameon=False, dpi=100)
            ax.pcolormesh(np.array(new_spectrum), shading='gouraud', vmax=None)
            plt.title('STFT Magnitude')
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [sec]')
            plt.show()
    # # Show time warped & masked spectrogram
    #     spec_augment_pytorch.visualization_spectrogram(mel_spectrogram=warped_masked_spectrogram,
    #                                                   title="pytorch Warped & Masked Spectrogram")


