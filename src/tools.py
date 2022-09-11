import numpy as np
import nibabel as nib

def load_data(follow_up_files, previous_files, reference_files, norm=512, c1=20,c2=-20):
    kspace = []
    rec = []
    previous = []
    ref = []

    for ii, file in enumerate(follow_up_files):
        img = nib.load(file).get_fdata()[:,:,c1:c2]
        aux_rec  = np.swapaxes(img,0,2)
        #aux_rec = (aux_rec - aux_rec.min()) / (aux_rec.max() - aux_rec.min())
        #aux_rec = (aux_rec - np.mean(aux_rec))/ np.std(aux_rec)
        #aux_rec = aux_rec / norm
        aux_rec = aux_rec / np.abs(aux_rec).max()
        #convert zero filled reconstruction to kspace
        f = np.fft.fft2(aux_rec)
        aux_kspace = np.zeros((*aux_rec.shape,2))
        aux_kspace[:,:,:,0] = f.real
        aux_kspace[:,:,:,1] = f.imag
        kspace.append(aux_kspace)

        #obtain complex reconstruction by taking inverse fft
        complex_rec = np.fft.ifft2(f)
        aux2_rec = np.zeros((*aux_rec.shape,2))
        aux2_rec[:,:,:,0] = complex_rec.real
        aux2_rec[:,:,:,1] = complex_rec.imag
        rec.append(aux2_rec)

        #load previous registered reconstruction
        img2 = nib.load(previous_files[ii]).get_fdata()[:,:,c1:c2]
        previous_rec = np.swapaxes(img2,0,2)[...,np.newaxis]
        #previous_rec = (previous_rec - previous_rec.min()) / (previous_rec.max() - previous_rec.min())
        #previous_rec = (previous_rec - np.mean(previous_rec)) / np.std(previous_rec)
        #previous_rec = previous_rec / norm
        previous_rec = previous_rec / np.abs(previous_rec).max()
        previous.append(previous_rec)

        #load reference reconstruction
        img3 = nib.load(reference_files[ii]).get_fdata()[:,:,c1:c2]
        ref_rec = np.swapaxes(img3,0,2)[:,:,:,np.newaxis]
        #ref_rec = (ref_rec - ref_rec.min()) / (ref_rec.max() - ref_rec.min())
        #ref_rec = (ref_rec - np.mean(ref_rec)) / np.std(ref_rec)
        ref_rec = ref_rec / np.abs(ref_rec).max()
        #ref_rec = ref_rec / norm
        ref.append(ref_rec)
    kspace = np.concatenate(kspace,axis=0)
    rec = np.concatenate(rec)
    previous = np.concatenate(previous)
    ref = np.concatenate(ref)
    return kspace, rec, previous, ref


# On the fly data augmentation
def combine_generator(gen1,gen2,gen3,under_masks):
    while True:
        rec_real = gen1.next()
        rec_imag = gen2.next()
        f = np.fft.fft2(rec_real[:,:,:,0]+1j*rec_imag[:,:,:,0])

        #under_sample k-space
        kspace2 = np.zeros((f.shape[0],f.shape[1],f.shape[2],2))
        kspace2[:,:,:,0] = f.real
        kspace2[:,:,:,1] = f.imag
        indexes = np.random.choice(np.arange(under_masks.shape[0], dtype=int),
                                    rec_real.shape[0], replace=False)
        kspace2[under_masks[indexes]] = 0

        #previous scan and reference
        previous_rec = gen3.next()
        rec = np.abs(rec_real[:,:,:,0] + 1j*rec_imag[:,:,:,0])
        rec = rec[:,:,:,np.newaxis]

        yield([kspace2, previous_rec, under_masks[indexes].astype(np.float32)],[rec, rec])
