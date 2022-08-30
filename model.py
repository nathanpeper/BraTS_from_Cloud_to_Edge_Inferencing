from argparser import args
import tensorflow as tf
from tensorflow import keras as K


def dice_coef(target, prediction, axis=(1, 2, 3), smooth=0.0001):
    """
    Sorenson Dice
    \frac{  2 \times \left | T \right | \cap \left | P \right |}{ \left | T \right | +  \left | P \right |  }
    where T is ground truth mask and P is the prediction mask
    """
    prediction = tf.round(prediction)  # Round to 0 or 1

    intersection = tf.reduce_sum(target * prediction, axis=axis)
    union = tf.reduce_sum(target + prediction, axis=axis)
    numerator = tf.constant(2.) * intersection + smooth
    denominator = union + smooth
    coef = numerator / denominator

    return tf.reduce_mean(coef)


def soft_dice_coef(target, prediction, axis=(1, 2, 3), smooth=0.0001):
    """
    Sorenson (Soft) Dice - Don't round predictions
    \frac{  2 \times \left | T \right | \cap \left | P \right |}{ \left | T \right | +  \left | P \right |  }
    where T is ground truth mask and P is the prediction mask
    """
    intersection = tf.reduce_sum(target * prediction, axis=axis)
    union = tf.reduce_sum(target + prediction, axis=axis)
    numerator = tf.constant(2.) * intersection + smooth
    denominator = union + smooth
    coef = numerator / denominator

    return tf.reduce_mean(coef)


def dice_loss(target, prediction, axis=(1, 2, 3), smooth=0.0001):
    """
    Sorenson (Soft) Dice loss
    Using -log(Dice) as the loss since it is better behaved.
    Also, the log allows avoidance of the division which
    can help prevent underflow when the numbers are very small.
    """
    intersection = tf.reduce_sum(prediction * target, axis=axis)
    p = tf.reduce_sum(prediction, axis=axis)
    t = tf.reduce_sum(target, axis=axis)
    numerator = tf.reduce_mean(intersection + smooth)
    denominator = tf.reduce_mean(t + p + smooth)
    dice_loss = -tf.math.log(2.*numerator) + tf.math.log(denominator)

    return dice_loss
                 

def unet_3d(input_dim=(args.tile_height, args.tile_width, args.tile_depth, args.number_input_channels), 
            filters=args.filters,
            number_output_classes=args.number_output_classes,
            use_upsampling=args.use_upsampling,
            concat_axis=-1, 
            model_name=args.saved_model_name):
    """
    3D U-Net
    """

    def ConvolutionBlock(x, name, filters, params):
        """
        Convolutional block of layers
        Per the original paper this is back to back 3D convs
        with batch norm and then ReLU.
        """

        x = K.layers.Conv3D(filters=filters, **params, name=name+"_conv_0")(x)
        x = K.layers.BatchNormalization(name=name+"_bn_0")(x)
        x = K.layers.Activation("relu", name=name+"_relu_0")(x)

        x = K.layers.Conv3D(filters=filters, **params, name=name+"_conv_1")(x)
        x = K.layers.BatchNormalization(name=name+"_bn_1")(x)
        x = K.layers.Activation("relu", name=name)(x)

        return x


    # Convolution parameters
    params = dict(kernel_size=(3, 3, 3), activation=None,
                  padding="same", 
                  kernel_initializer="he_uniform")

    # Transposed convolution parameters
    params_trans = dict(kernel_size=(2, 2, 2), strides=(2, 2, 2),
                        padding="same",
                        kernel_initializer="he_uniform")

    
    inputs = K.layers.Input(shape=input_dim, name="mrimages")

    # BEGIN - Encoding path
    encode_a = ConvolutionBlock(inputs, "encode_a", filters, params)
    pool_a = K.layers.MaxPooling3D(name="pool_a", pool_size=(2, 2, 2))(encode_a)

    encode_b = ConvolutionBlock(pool_a, "encode_b", filters*2, params)
    pool_b = K.layers.MaxPooling3D(name="pool_b", pool_size=(2, 2, 2))(encode_b)

    encode_c = ConvolutionBlock(pool_b, "encode_c", filters*4, params)
    pool_c = K.layers.MaxPooling3D(name="pool_c", pool_size=(2, 2, 2))(encode_c)

    encode_d = ConvolutionBlock(pool_c, "encode_d", filters*8, params)
    pool_d = K.layers.MaxPooling3D(name="pool_d", pool_size=(2, 2, 2))(encode_d)

    encode_e = ConvolutionBlock(pool_d, "encode_e", filters*16, params)
    # END - Encoding path

    
    # BEGIN - Decoding path
    if use_upsampling:
        up = K.layers.UpSampling3D(name="up_e", size=(2, 2, 2),
                                   interpolation="bilinear")(encode_e)
    else:
        up = K.layers.Conv3DTranspose(name="transconv_e", filters=filters*8,
                                      **params_trans)(encode_e)
    concat_d = K.layers.concatenate(
        [up, encode_d], axis=concat_axis, name="concat_d")

    decode_c = ConvolutionBlock(concat_d, "decode_c", filters*8, params)

    if use_upsampling:
        up = K.layers.UpSampling3D(name="up_c", size=(2, 2, 2),
                                   interpolation="bilinear")(decode_c)
    else:
        up = K.layers.Conv3DTranspose(name="transconv_c", filters=filters*4,
                                      **params_trans)(decode_c)
    concat_c = K.layers.concatenate(
        [up, encode_c], axis=concat_axis, name="concat_c")

    decode_b = ConvolutionBlock(concat_c, "decode_b", filters*4, params)

    if use_upsampling:
        up = K.layers.UpSampling3D(name="up_b", size=(2, 2, 2),
                                   interpolation="bilinear")(decode_b)
    else:
        up = K.layers.Conv3DTranspose(name="transconv_b", filters=filters*2,
                                      **params_trans)(decode_b)
    concat_b = K.layers.concatenate(
        [up, encode_b], axis=concat_axis, name="concat_b")

    decode_a = ConvolutionBlock(concat_b, "decode_a", filters*2, params)

    if use_upsampling:
        up = K.layers.UpSampling3D(name="up_a", size=(2, 2, 2),
                                   interpolation="bilinear")(decode_a)
    else:
        up = K.layers.Conv3DTranspose(name="transconv_a", filters=filters,
                                      **params_trans)(decode_a)
    concat_a = K.layers.concatenate(
        [up, encode_a], axis=concat_axis, name="concat_a")
    
    conv_out = ConvolutionBlock(concat_a, "conv_out", filters, params)

    # END - Decoding path    

    
    prediction = K.layers.Conv3D(name="prediction_mask",
                                 filters=number_output_classes, kernel_size=(1, 1, 1),
                                 activation="sigmoid")(conv_out)

    model = K.models.Model(inputs=[inputs], outputs=[prediction], name=model_name)

    model.summary()

    return model
