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
import cv2

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

parser = argparse.ArgumentParser(description='Spec Augment')
parser.add_argument('--audio-path', default="/Users/jiyangli/Downloads/train_imu2",
                    help='The audio file.')  # '../../stft_label_vIMU/s02_session7_vIMU_stft'
parser.add_argument('--time-warp-para', default=20,
                    help='time warp parameter W')
parser.add_argument('--frequency-mask-para', default=100,
                    help='frequency mask parameter F')
parser.add_argument('--time-mask-para', default=10,
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
        if "aug" in filename or "jpg" not in filename:
        # if "aug" in filename or "csv" not in filename:
            continue
        print(filename)
        path_one = audio_path + '/' + filename
        spectrogram_ori = cv2.imread(path_one)

        shape = spectrogram_ori.shape
        img_h = shape[0]
        img_w = shape[1]
        # print('&&&&&&&&&&&&&&&')
        # print(img_h, img_w)
        width = img_w/(img_h/4)

        spectrogram = cv2.flip(spectrogram_ori, 0)
        # vmin = min(spectrogram)
        # spectrogram = pd.read_csv(path_one)
        # window_name = 'image'
        # cv2.imshow(window_name, spectrogram)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        if spectrogram.shape[1] < 5:
            print("********************")
            print("vIMU is too short!")
            continue
        # spectrogram = spectrogram.to_numpy()
        spectrogram = cv2.cvtColor(spectrogram, cv2.COLOR_BGR2GRAY)
        # mel_spectrogram = rgb2gray(mel_spectrogram_raw)

        # reshape spectrogram shape to [batch_size, time, frequency, 1]
        shape = spectrogram.shape
        spectrogram = np.reshape(spectrogram, (-1, shape[0], shape[1]))
        spec_tmp = np.array(spectrogram)[0]
        vmin = np.amin(spec_tmp)
        vmax = np.amax(spec_tmp)
        # window_name = 'image'
        # cv2.imshow(window_name, spectrogram)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        mel_spectrogram = torch.from_numpy(spectrogram)

        # Show Raw mel-spectrogram
        #     spec_augment_pytorch.visualization_spectrogram(mel_spectrogram=mel_spectrogram,
        #                                                   title="Raw Mel Spectrogram")

        # Calculate SpecAugment pytorch
        warped_masked_spectrogram = spec_augment_pytorch.spec_augment(mel_spectrogram=mel_spectrogram)

        new_spectrum = warped_masked_spectrogram.squeeze(2)
        new_spectrum = np.array(new_spectrum)[0]
        # new_spectrum = pd.DataFrame(np.array(new_spectrum)[0])
        save_path = audio_path + "_aug"
        if not os.path.isdir(save_path):
            os.mkdir(save_path)

        filepath = save_path + '/' + filename.split('.')[0] + '_aug_' + \
                   str(time_warping_para) + '_' + str(time_masking_para) + '_' + str(frequency_masking_para)\
                   + '_' + str(masking_line_number) + '.jpg'

        # new_spectrum.to_csv(filepath, index=False)
        fig, ax = plt.subplots(frameon=False, figsize=(width, 4), dpi=100)

        # t = np.array(np.max(psig_m_m))
        ax.pcolormesh(new_spectrum, vmax=vmax, vmin=vmin, shading='gouraud')
        # ax.set_title('STFT Magnitude')
        # ax.set_ylabel('Frequency [Hz]')
        # ax.set_xlabel('Time [sec]')
        plt.axis('off')
        # ax.set_aspect(max_f_idx/(2*min_len))
        # ax.set_ylim(0, 6*max_f_idx)
        # plt.show()

        fig.savefig(filepath, transparent=True, bbox_inches='tight', pad_inches=0)

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


