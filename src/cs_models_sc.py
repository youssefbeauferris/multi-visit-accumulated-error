import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Lambda, Add, LeakyReLU,  \
                                    MaxPooling2D, concatenate, UpSampling2D,\
                                    Multiply, ZeroPadding2D, Cropping2D,    \
                                    Concatenate

def abs_layer(complex_data):
    """
    Input: 2-channel array representing complex data
    Output: 1-channel array representing magnitude of complex data
    """
    #get real and imaginary portions
    real = Lambda(lambda complex_data : complex_data[:,:,:,0])(complex_data)
    imag = Lambda(lambda complex_data : complex_data[:,:,:,1])(complex_data)

    mag = tf.abs(tf.complex(real,imag))
    mag = tf.expand_dims(mag, -1)
    return mag


def phase_layer(complex_data):
    """
    Input: 2-channel array representing complex data
    Output: 1-channel array representing magnitude of complex data
    """
    #get real and imaginary portions
    real = Lambda(lambda complex_data : complex_data[:,:,:,0])(complex_data)
    imag = Lambda(lambda complex_data : complex_data[:,:,:,1])(complex_data)

    ph = tf.math.atan2(imag,real)
    ph = tf.expand_dims(ph, -1)
    return ph


def polar2cartesian(ph,mag):
    real = mag*tf.math.cos(ph)
    imag = mag*tf.math.sin(ph)
    image_complex_2channel = tf.concat([real, imag], -1)
    return image_complex_2channel

def reduce_dimension(image):
    #convert complex reconstruction into single channel image
    # get real and imaginary portions
    real = Lambda(lambda image: image[:, :, :, 0])(image)
    imag = Lambda(lambda image: image[:, :, :, 1])(image)

    rec = tf.expand_dims(tf.math.abs(tf.math.add(real,imag)),-1)
    return rec

def fft_layer(image):
    # get real and imaginary portions
    real = Lambda(lambda image: image[:, :, :, 0])(image)
    imag = Lambda(lambda image: image[:, :, :, 1])(image)

    image_complex = tf.complex(real, imag)  # Make complex-valued tensor
    kspace_complex = tf.signal.fft2d(image_complex)

    # expand channels to tensorflow/keras format
    real = tf.expand_dims(tf.math.real(kspace_complex), -1)
    imag = tf.expand_dims(tf.math.imag(kspace_complex), -1)
    kspace = tf.concat([real, imag], -1)
    return kspace


def slice_channel(array_2channel):
    #get real and imaginary portions
    array_1channel = Lambda(lambda array_2channel : array_2channel[:,:,:,0])(array_2channel)

    array_1channel = tf.expand_dims(array_1channel, -1)

    return array_1channel


def ifft_layer(kspace_2channel):
    #get real and imaginary portions
    real = Lambda(lambda kspace_2channel : kspace_2channel[:,:,:,0])(kspace_2channel)
    imag = Lambda(lambda kspace_2channel : kspace_2channel[:,:,:,1])(kspace_2channel)

    kspace_complex = tf.complex(real,imag) # Make complex-valued tensor
    image_complex = tf.signal.ifft2d(kspace_complex)

    # expand channels to tensorflow/keras format
    real = tf.expand_dims(tf.math.real(image_complex),-1)
    imag = tf.expand_dims(tf.math.imag(image_complex),-1)
    # generate 2-channel representation of image domain
    image_complex_2channel = tf.concat([real, imag], -1)
    return image_complex_2channel

def cnn_block(cnn_input, depth, nf, kshape,channels):
    """
    :param cnn_input: Input layer to CNN block
    :param depth: Depth of CNN. Disregarding the final convolution block that goes back to
    2 channels
    :param nf: Number of filters of convolutional layers, except for the last
    :param kshape: Shape of the convolutional kernel
    :return: 2-channel, complex reconstruction
    """
    layers = [cnn_input]

    for ii in range(depth):
        # Add convolutional block
        layers.append(Conv2D(nf, kshape, padding='same')(layers[-1]))
        layers.append(LeakyReLU(alpha=0.1)(layers[-1]))
    final_conv = Conv2D(channels, (1, 1), activation='linear')(layers[-1])
    rec1 = Add()([final_conv,cnn_input])
    return rec1


def unet_block(unet_input, kshape=(3, 3),channels = 2):
    """
    :param unet_input: Input layer
    :param kshape: Kernel size
    :return: 2-channel, complex reconstruction
    """

    conv1 = Conv2D(48, kshape, activation='relu', padding='same')(unet_input)
    conv1 = Conv2D(48, kshape, activation='relu', padding='same')(conv1)
    conv1 = Conv2D(48, kshape, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, kshape, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, kshape, activation='relu', padding='same')(conv2)
    conv2 = Conv2D(64, kshape, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, kshape, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, kshape, activation='relu', padding='same')(conv3)
    conv3 = Conv2D(128, kshape, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, kshape, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, kshape, activation='relu', padding='same')(conv4)
    conv4 = Conv2D(256, kshape, activation='relu', padding='same')(conv4)

    up1 = concatenate([UpSampling2D(size=(2, 2))(conv4), conv3], axis=-1)
    conv5 = Conv2D(128, kshape, activation='relu', padding='same')(up1)
    conv5 = Conv2D(128, kshape, activation='relu', padding='same')(conv5)
    conv5 = Conv2D(128, kshape, activation='relu', padding='same')(conv5)

    up2 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv2], axis=-1)
    conv6 = Conv2D(64, kshape, activation='relu', padding='same')(up2)
    conv6 = Conv2D(64, kshape, activation='relu', padding='same')(conv6)
    conv6 = Conv2D(64, kshape, activation='relu', padding='same')(conv6)

    up3 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv1], axis=-1)
    conv7 = Conv2D(48, kshape, activation='relu', padding='same')(up3)
    conv7 = Conv2D(48, kshape, activation='relu', padding='same')(conv7)
    conv7 = Conv2D(48, kshape, activation='relu', padding='same')(conv7)

    conv8 = Conv2D(channels, (1, 1), activation='linear')(conv7)
    out = Add()([conv8, slice_channel(unet_input)])
    return out

def unet_block2(unet_input, kshape=(3, 3),channels = 2):
    """
    :param unet_input: Input layer
    :param kshape: Kernel size
    :return: 2-channel, complex reconstruction
    """

    conv1 = Conv2D(48, kshape, activation='relu', padding='same')(unet_input)
    conv1 = Conv2D(48, kshape, activation='relu', padding='same')(conv1)
    conv1 = Conv2D(48, kshape, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(96, kshape, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(96, kshape, activation='relu', padding='same')(conv2)
    conv2 = Conv2D(96, kshape, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(192, kshape, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(192, kshape, activation='relu', padding='same')(conv3)
    conv3 = Conv2D(192, kshape, activation='relu', padding='same')(conv3)

    up1 = concatenate([UpSampling2D(size=(2, 2))(conv3), conv2], axis=-1)
    conv4 = Conv2D(96, kshape, activation='relu', padding='same')(up1)
    conv4 = Conv2D(96, kshape, activation='relu', padding='same')(conv4)
    conv4 = Conv2D(96, kshape, activation='relu', padding='same')(conv4)

    up2 = concatenate([UpSampling2D(size=(2, 2))(conv4), conv1], axis=-1)
    conv5 = Conv2D(48, kshape, activation='relu', padding='same')(up2)
    conv5 = Conv2D(48, kshape, activation='relu', padding='same')(conv5)
    conv5 = Conv2D(48, kshape, activation='relu', padding='same')(conv5)

    conv6 = Conv2D(channels, (1, 1), activation='linear')(conv5)
    out = Add()([conv6, unet_input])
    return out

def DC_block(rec,mask,sampled_kspace,channels,kspace = False):
    """
    :param rec: Reconstructed data, can be k-space or image domain
    :param mask: undersampling mask
    :param sampled_kspace:
    :param kspace: Boolean, if true, the input is k-space, if false it is image domain
    :return: k-space after data consistency
    """

    if kspace:
        rec_kspace = rec
    else:
        rec_kspace = Lambda(fft_layer)(rec)
    rec_kspace_dc =  Multiply()([rec_kspace,mask])
    rec_kspace_dc = Add()([rec_kspace_dc,sampled_kspace])
    return rec_kspace_dc

def deep_cascade_flat_unrolled(depth_str = 'ikikii', H=256,W=256,depth = 5,kshape = (3,3), nf = 48,channels = 2):
    """
    :param depth_str: string that determines the depth of the cascade and the domain of each
    subnetwork
    :param H: Image heigh
    :param W: Image width
    :param kshape: Kernel size
    :param nf: number of filters in each convolutional layer
    :return: Deep Cascade Flat Unrolled model
    """

    inputs = Input(shape=(H,W,channels))
    mask = Input(shape=(H,W,channels))
    layers = [inputs]
    kspace_flag = True
    for ii in depth_str:

        if ii =='i':
            # Add IFFT
            layers.append(Lambda(ifft_layer)(layers[-1]))
            kspace_flag = False
        # Add CNN block
        layers.append(cnn_block(layers[-1],depth,nf,kshape,channels))

        # Add DC block
        layers.append(DC_block(layers[-1],mask,inputs,channels,kspace=kspace_flag))
        kspace_flag = True
    out = Lambda(ifft_layer)(layers[-1])
    model = Model(inputs=[inputs,mask], outputs=out)
    return model

def deep_cascade_flat_unrolled_end(depth_str = 'ikikii', H=256,W=256,Hpad=3,Wpad=3,depth = 5,kshape = (3,3), nf = 48,channels = 2):
    """
    :param depth_str: string that determines the depth of the cascade and the domain of each
    subnetwork
    :param H: Image heigh
    :param W: Image width
    :param kshape: Kernel size
    :param nf: number of filters in each convolutional layer
    :return: Deep Cascade Flat Unrolled model
    """

    inputs = Input(shape=(H,W,channels)) #kspace zero_flled_rec
    inputs2 = Input(shape=(H,W,1))       #reconstruction for previous scan
    mask = Input(shape=(H,W,channels))   #under sampling masks for zero_filled`
    layers = [inputs]
    kspace_flag = True
    for ii in depth_str:

        if ii =='i':
            # Add IFFT
            layers.append(Lambda(ifft_layer)(layers[-1]))
            kspace_flag = False
        # Add CNN block
        layers.append(cnn_block(layers[-1],depth,nf,kshape,channels))
        # Add DC block
        layers.append(DC_block(layers[-1],mask,inputs,channels,kspace=kspace_flag))
        kspace_flag = True

    layers.append(Lambda(ifft_layer)(layers[-1]))
    mag = Lambda(abs_layer)(layers[-1])
    ph = Lambda(phase_layer)(layers[-1])
    layers.append(tf.concat([mag,inputs2], -1))
    layers.append(unet_block(layers[-1], channels=1))

    out = Add()([layers[-1], mag])
    out1 = polar2cartesian(ph,out)
    out2 = DC_block(out1,mask,inputs,channels,kspace=False)
    out3 = Lambda(ifft_layer)(out2)
    out4 = abs_layer(out3)

    model = Model(inputs=[inputs,inputs2,mask], outputs=[mag,out4])
    return model

def deep_cascade_flat_unrolled_end1(depth_str = 'ikikii', H=256,W=256,Hpad=3,Wpad=3,depth = 5,kshape = (3,3), nf = 48,channels = 2):
    """
    :param depth_str: string that determines the depth of the cascade and the domain of each
    subnetwork
    :param H: Image heigh
    :param W: Image width
    :param kshape: Kernel size
    :param nf: number of filters in each convolutional layer
    :return: Deep Cascade Flat Unrolled model
    """

    inputs = Input(shape=(H,W,channels)) #kspace zero_flled_rec
    inputs2 = Input(shape=(H,W,1))       #reconstruction for previous scan
    mask = Input(shape=(H,W,channels))   #under sampling masks for zero_filled`
    layers = [inputs]
    kspace_flag = True
    for ii in depth_str:

        if ii =='i':
            # Add IFFT
            layers.append(Lambda(ifft_layer)(layers[-1]))
            kspace_flag = False
        # Add CNN block
        layers.append(cnn_block(layers[-1],depth,nf,kshape,channels))
        # Add DC block
        layers.append(DC_block(layers[-1],mask,inputs,channels,kspace=kspace_flag))
        kspace_flag = True

    layers.append(Lambda(ifft_layer)(layers[-1]))
    mag = Lambda(abs_layer)(layers[-1])
    ph = Lambda(phase_layer)(layers[-1])
    layers.append(tf.concat([mag,inputs2], -1))
    layers.append(unet_block(layers[-1], channels=1))

    out = polar2cartesian(ph,layers[-1])
    out1 = DC_block(out,mask,inputs,channels,kspace=False)
    out2 = Lambda(ifft_layer)(out1)
    out3 = abs_layer(out2)
    out4 = Add()([out3, mag])

    model = Model(inputs=[inputs,inputs2,mask], outputs=[mag,out4])
    return model

def deep_cascade_unet(depth_str='ki', H=218, W=170, Hpad = 3, Wpad = 3, kshape=(3, 3),channels = 22):

    inputs = Input(shape=(H,W,channels))
    mask = Input(shape=(H,W,channels))
    layers = [inputs]
    kspace_flag = True
    for ii in depth_str:

        if ii =='i':
            # Add IFFT
            layers.append(Lambda(ifft_layer)(layers[-1]))
            kspace_flag = False
        # Add CNN block
        layers.append(ZeroPadding2D(padding=(Hpad,Wpad))(layers[-1]))
        layers.append(unet_block(layers[-1], kshape,channels))
        layers.append(Cropping2D(cropping=(Hpad,Wpad))(layers[-1]))

        # Add DC block
        layers.append(DC_block(layers[-1],mask,inputs,channels,kspace=kspace_flag))
        kspace_flag = True
    out = Lambda(ifft_layer)(layers[-1])
    model = Model(inputs=[inputs,mask], outputs=out)
    return model
