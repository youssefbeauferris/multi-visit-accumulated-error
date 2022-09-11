#!/usr/bin/env python
# coding: utf-8

# The following notebook computes the quantitative assessment metrics of the single and multi-visit reconstruction produced in Steps 00-03. The metrics are computed on a third independent test set which does not share any scans from subjects within the train and validation sets. The test set consists of 7 longitudinal pairs of scans. The SSIM and pSNR of the single-visit reconstructions are compared to the multi-visit reconstructions to measure the improvements when incorporating previous subject-specific information. The first and last 20 slices are removed before computing metrics to eliminate slices with little to no anatomical structures. 

# In[123]:


import glob
import nibabel
import sys
import nibabel as nib
import numpy as np
import tqdm
import matplotlib.pyplot as plt
MY_UTILS_PATH = "../src/"
if not MY_UTILS_PATH in sys.path:
    sys.path.append(MY_UTILS_PATH)
import metrics
import seaborn as sns

# In[124]:


test_files = np.loadtxt('../data/test_long.txt',dtype=str)[:,1:]
print(test_files)


# In[125]:


#file pathes
initial_path = '../data/predicted/10x-varnet/nib/'
previous_path = '../data/reference_reg_10x-varnet/'
enhanced_path = '../data/predicted/10x-enhanced-varnet/'
ref_path = '../../../data/brain-cancer/'


# In[126]:


#file ids with pathes
initial_files = [initial_path + file[:-4] + '_predicted.nii' for file in test_files[:,1]]
previous_files = [previous_path + 'elastic_' + file[0][:-4] + '_' + file[1][:-4] + '-1.nii' for file in test_files]
previous_files_multi_visit = [previous_path + 'elastic_' + file[0][:-4] + '_' + file[1][:-4] + '-2.nii' for file in test_files]

enhanced_files = [enhanced_path + file[:-4] + '_predicted-1.nii' for file in test_files[:,1]]
enhanced_files_multi_visit = [enhanced_path + file[:-4] + '_predicted-2.nii' for file in test_files[:,1]]

ref_files = [ref_path + file for file in test_files[:,1]]


# In[127]:


init_mets = []
prev_mets = []
prev1_mets = []
enh_mets = []
enh1_mets = []
for ii in tqdm.tqdm(range(len(test_files))):
    init = nib.load(initial_files[ii]).get_fdata()[:,:,20:-20]
    prev = nib.load(previous_files[ii]).get_fdata()[:,:,20:-20]
    prev1 = nib.load(previous_files_multi_visit[ii]).get_fdata()[:,:,20:-20]
    enh = nib.load(enhanced_files[ii]).get_fdata()[:,:,20:-20]
    enh1 = nib.load(enhanced_files_multi_visit[ii]).get_fdata()[:,:,20:-20]
    ref = nib.load(ref_files[ii]).get_fdata()[:,:,20:-20]
    
    init = np.swapaxes(init,0,2)
    prev = np.swapaxes(prev,0,2)
    prev1 = np.swapaxes(prev1,0,2)
    enh = np.swapaxes(enh,0,2)
    enh1 = np.swapaxes(enh1,0,2)
    ref = np.swapaxes(ref,0,2)
    
    init = init / np.abs(init).max()
    prev = prev / np.abs(prev).max()
    prev1 = prev1 / np.abs(prev1).max()
    enh = enh / np.abs(enh).max()
    enh1 = enh1 / np.abs(enh1).max()    
    ref = ref / np.abs(ref).max()
    
    init_mets.append(metrics.metrics(init, ref))
    prev_mets.append(metrics.metrics(prev, ref))
    prev1_mets.append(metrics.metrics(prev1, ref))
    enh_mets.append(metrics.metrics(enh, ref))
    enh1_mets.append(metrics.metrics(enh1, ref))
    


# In[128]:


init_mets_cat = np.concatenate(init_mets, axis=1)

prev_mets_cat = np.concatenate(prev_mets, axis=1)
prev1_mets_cat = np.concatenate(prev1_mets, axis=1)

enh_mets_cat = np.concatenate(enh_mets, axis=1)
enh1_mets_cat = np.concatenate(enh1_mets, axis=1)

print(enh_mets_cat.shape)




labels = ['previous', 'multi-visit previous', 'single-visit', 'multi-visit generated from f/sampled prior', 'multi-visit generated from multi-visit prior'] 


# In[156]:


mets = np.array([init_mets_cat.mean(axis=1), 
           prev_mets_cat.mean(axis=1), prev_mets_cat.mean(axis=1), 
           enh_mets_cat.mean(axis=1), enh1_mets_cat.mean(axis=1)])


# In[157]:


yerr = np.array([init_mets_cat.std(axis=1), 
           prev_mets_cat.std(axis=1), prev_mets_cat.std(axis=1), 
           enh_mets_cat.std(axis=1), enh1_mets_cat.std(axis=1)])



colors = sns.color_palette('colorblind',3)


# In[163]:


ssim = mets[:,0]
psnr = mets[:,1]


# In[198]:
# In[210]:

width = 0.3
fig = plt.figure(figsize=(12,8))
x = [0, 0.5, .8, 1.3, 1.6]
y = ssim

#plt.bar([0, 1, 1.8, 2.8, 3.6], ssim, yerr=yerr[:,0],  
#        error_kw=dict(ecolor='green', lw=2, capsize=5, capthick=2), 
#        color=[colors[0],colors[1], colors[2], colors[1], colors[2]], 
#        label=([0,1,2], ['VarNet', 'Reference', 'Reconstruction']))

plt.bar(x[0], y[0], yerr=yerr[0,0], width=0.3, error_kw=dict(ecolor='green', lw=2, capsize=5, capthick=2), label='Single-visit 10x')
plt.bar(x[1::2], y[1::2], yerr=yerr[1::2,0],  width=0.3, error_kw=dict(ecolor='green', lw=2, capsize=5, capthick=2), label='Fully-Sampled')
plt.bar(x[2::2], y[2::2], yerr=yerr[2::2,0], width=0.3,  error_kw=dict(ecolor='green', lw=2, capsize=5, capthick=2), label='Under-Sampled')


plt.ylabel('SSIM', weight='bold')
plt.xticks([0,0.65, 1.45], 
           ['Single-visit Reconstruction',
            'Previous Scan',  
            'Multi-visit Reconstruction'], rotation=0)
plt.legend()
plt.show()
plt.close()
plt.tight_layout()

fig.savefig('../reports/figures/metrics_bar_graph_ssim.png', dpi=300)



fig = plt.figure(figsize=(12,8))
x = [0, 0.5, .8, 1.3, 1.6]
y = psnr

#plt.bar([0, 1, 1.8, 2.8, 3.6], ssim, yerr=yerr[:,0],  
#        error_kw=dict(ecolor='green', lw=2, capsize=5, capthick=2), 
#        color=[colors[0],colors[1], colors[2], colors[1], colors[2]], 
#        label=([0,1,2], ['VarNet', 'Reference', 'Reconstruction']))

plt.bar(x[0], y[0], yerr=yerr[0,1], width=0.3, error_kw=dict(ecolor='green', lw=2, capsize=5, capthick=2), label='Single-visit 10x')
plt.bar(x[1::2], y[1::2], yerr=yerr[1::2,1], width=0.3, error_kw=dict(ecolor='green', lw=2, capsize=5, capthick=2), label='Fully-Sampled')
plt.bar(x[2::2], y[2::2], yerr=yerr[2::2,1],  width=0.3, error_kw=dict(ecolor='green', lw=2, capsize=5, capthick=2), label='Under-Sampled')


plt.ylabel('PSNR', weight='bold')
plt.xticks([0,0.65, 1.45], 
           ['Single-visit Reconstruction',
            'Previous Scan',  
            'Multi-visit Reconstruction'], rotation=0)
plt.legend()
plt.show()
plt.close()
plt.tight_layout()

fig.savefig('../reports/figures/metrics_bar_graph_psnr.png', dpi=300)


