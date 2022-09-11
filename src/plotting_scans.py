import numpy as np
import os
import glob
import sys
import nibabel as nib
import matplotlib.pyplot as plt
import pickle
from matplotlib.patches import Rectangle
import argparse
import seaborn as sns

sys.path.append('../src/')
import metrics

def data_preprocessing(all_data):
    '''normalize and reorient data'''
    all_data_processed = []
    for data in all_data:
        data = np.swapaxes(data, 0, 1)
        data = data / np.abs(data).max()
        all_data_processed.append(data)
    return all_data_processed


previous_path = f'../data/reference_reg_10x-varnet/'
follow_up_path = f'../data/predicted/10x-varnet/nib/'
reference_path = '/home/youssef/Desktop/data/brain-cancer/'
enhanced_path = f'../data/predicted/10x-enhanced-varnet/'

test_files = np.loadtxt('../data/test_long.txt', dtype=str)

#create 3 x 4 figure
# ----------------------------
#| P1 | SV2 | MV2 | RF2 |
#-----------------------------
#| MV2 | SV3 | MV3_MV2 | RF3 |
#-----------------------------
#| P2 | SV3 | MV3_P2 |  RF3|
#-----------------------------

for ii in range(test_files.shape[0]):
    print(test_files[ii])

    #Row #1

    P1 = nib.load(previous_path + 'elastic_' + test_files[ii][0][:-4] + '_' + test_files[ii][1][:-4] + '-0.nii')
    P1 = P1.get_fdata()
    SV2 = nib.load(follow_up_path + test_files[ii][1][:-4] + '_predicted.nii').get_fdata()
    MV2 = nib.load(enhanced_path + test_files[ii][1][:-4] + '_predicted-0.nii').get_fdata()
    RF2 = nib.load(reference_path + test_files[ii][1]).get_fdata()

    #Row 2
    
    MV2 = nib.load(previous_path + 'elastic_' + test_files[ii][1][:-4] + '_' + test_files[ii][2][:-4] + '-2.nii').get_fdata()
    SV3 = nib.load(follow_up_path + test_files[ii][2][:-4] + '_predicted.nii').get_fdata()
    MV3_1 = nib.load(enhanced_path + test_files[ii][2][:-4] + '_predicted-2.nii').get_fdata()
    RF3 = nib.load(reference_path + test_files[ii][2]).get_fdata()

    # Row 3

    P2 = nib.load(previous_path + 'elastic_' + test_files[ii][1][:-4] + '_' + test_files[ii][2][:-4] + '-1.nii').get_fdata()
    SV3 = nib.load(follow_up_path + test_files[ii][2][:-4] + '_predicted.nii').get_fdata()
    MV3 = nib.load(enhanced_path + test_files[ii][2][:-4] + '_predicted-1.nii').get_fdata()
    RF3 = nib.load(reference_path + test_files[ii][2]).get_fdata()

    all_data = [ SV3, P2, MV2, MV3, MV3_1, RF3]
    all_data = data_preprocessing(all_data)
    title = ['Single-visit', 'Fully-Sampled PS', 'Multi-visit PS', 'Multi-visit FS', 'Multi-visit MV', 'Reference']
    
    fig, axes = plt.subplots(nrows=1, ncols=6, figsize=(6,4))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(all_data[i][:,:,110],cmap='gray')
        ax.axis('off')
        ax.set_title(title[i], fontsize=5)

        met = metrics.metrics(all_data[i][:,:,110][np.newaxis], all_data[-1][:,:,110][np.newaxis])

        print(met[0].shape)
        if i != 5:
            met = metrics.metrics(all_data[i][:,:,110][np.newaxis], all_data[-1][:,:,110][np.newaxis])

            ax.text(6,30, f'SSIM: {met[0][0].round(3)}', weight='bold', size=5, color='yellow')
            ax.text(6,70, f'PSNR: {met[1][0].round(3)}', weight='bold', size=5, color='yellow')
    plt.subplots_adjust(wspace=-0.1, hspace=-0.31)
    #plt.tight_layout()
    #plt.show()
    #plt.close()
    fig.savefig(f'../reports/figures/{ii}.png', dpi=300, bbox_inches='tight')





print(test_files.shape)


