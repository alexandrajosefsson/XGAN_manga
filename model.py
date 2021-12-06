#Build a Generative Adversarial Network (GAN) to generate new realistic cartoon characters.
import tensorflow as tf
import keras
from keras import Sequential, layers, utils
from keras import Model, losses

#-------------encoders--------------
#encoder1
inLayerEnc_1 = layers.Input(shape=(64, 64, 3)) 
conv1_1 = layers.Conv2D(kernel_size=7, strides=2, filters=32, activation='relu', padding='same', name='encoder1_a')(inLayerEnc_1) #done right????? what should kernel size be? taken from [21] facenet p.4, row1
#output 32x32x32
conv2_1 = layers.Conv2D(kernel_size=7, strides=2, filters=64, activation='relu', padding='same', name='encoder1_b')(conv1_1) #done right????? what should kernel size be? taken from [21] facenet p.4, row1
#output 16x16x64

#encoder2
inLayerEnc_2 = layers.Input(shape=(64, 64, 3)) 
conv1_2 = layers.Conv2D(kernel_size=7, strides=2, filters=32, activation='relu', padding='same',name='encoder2_a')(inLayerEnc_2) #done right????? what should kernel size be? taken from [21] facenet p.4, row1
#output 32x32x32
conv2_2 = layers.Conv2D(kernel_size=7, strides=2, filters=64, activation='relu', padding='same',name='encoder2_b')(conv1_2) #done right????? what should kernel size be? taken from [21] facenet p.4, row1
#output 16x16x64

#shared layer
conv3 = layers.Conv2D(kernel_size=7, strides=2, filters=128, activation='relu', padding='same',name='encoder_a', input_shape=[16,16,64])
conv3_out_1= conv3(conv2_1)
conv3_out_2= conv3(conv2_2)

#shared layer
conv4 = layers.Conv2D(kernel_size=7, strides=2, filters=256, activation='relu', padding='same',name='encoder_b',input_shape=[8,8,128])
conv4_out_1= conv4(conv3_out_1)
conv4_out_2= conv4(conv3_out_2)

#shared layer
conv4_max_pool = layers.MaxPooling2D(pool_size=3, strides=4, padding='same',name='encoder_c',input_shape=[4,4,256]) #get right size with stride=4, not sure about rest though
conv4_max_out_1= conv4_max_pool(conv4_out_1)
conv4_max_out_2= conv4_max_pool(conv4_out_2)

#shared layer
FC1 = layers.Dense(1024, activation='softmax', input_shape=[1,1,256], name='encoder_d')
FC1_1 = FC1(conv4_max_out_1)
FC1_2 = FC1(conv4_max_out_2)
#shared layer
FC2 = layers.Dense(1024, activation='softmax',input_shape=[1,1,1024], name='encoder_e')
FC2_1 = FC2(FC1_1)
FC2_2 = FC2(FC1_2)

model_enc_1 = Model(inLayerEnc_1, FC2_1)
model_enc_2 = Model(inLayerEnc_2, FC2_2)

#utils.plot_model(model_enc_1, 'Model.png', show_shapes=True)


#------------------decoders--------------------
inLayerDec_1 = layers.Input(shape=(1, 1, 1024)) 
inLayerDec_2 = layers.Input(shape=(1, 1, 1024)) 
#shared layer
deconv1 = layers.Conv2DTranspose(kernel_size=7,filters=512, strides=4, activation='relu', padding='same', name='decoder_a', input_shape=[1,1,1024])
deconv1_1 = deconv1(inLayerDec_1)
deconv1_2 = deconv1(inLayerDec_2)

#shared layer
deconv2 = layers.Conv2DTranspose(kernel_size=7,filters=256, strides=2, activation='relu', padding='same', name='decoder_b', input_shape=[4,4,512])
deconv2_1=deconv2(deconv1_1)
deconv2_2=deconv2(deconv1_2)

deconv3_1 =  layers.Conv2DTranspose(kernel_size=7,filters=128, strides=2, activation='relu', padding='same', name='decoder1_a')(deconv2_1)
deconv3_2 =  layers.Conv2DTranspose(kernel_size=7,filters=128, strides=2, activation='relu', padding='same', name='decoder2_a')(deconv2_2)

deconv4_1 =  layers.Conv2DTranspose(kernel_size=7,filters=64, strides=2, activation='relu', padding='same', name='decoder1_b')(deconv3_1)
deconv4_2 =  layers.Conv2DTranspose(kernel_size=7,filters=64, strides=2, activation='relu', padding='same', name='decoder2_b')(deconv3_2)

deconv5_1 =  layers.Conv2DTranspose(kernel_size=7,filters=3, strides=2, activation='relu', padding='same', name='decoder1_c')(deconv4_1)
deconv5_2 =  layers.Conv2DTranspose(kernel_size=7,filters=3, strides=2, activation='relu', padding='same', name='decoder2_c')(deconv4_2)


model_dec_1 = Model(inLayerDec_1, deconv5_1)
model_dec_2 = Model(inLayerDec_2, deconv5_2)

# model.compile('sgd', losses.categorical_crossentropy, metrics=['acc'])
#utils.plot_model(model_dec_1, 'Model.png', show_shapes=True)
print("decoder trainable vars")
print(model_dec_1.trainable_weights)

#---------------discriminator---------------------
#discriminator
inLayerDisc = layers.Input(shape=(64, 64, 3)) 
conv1_disc = layers.Conv2D(kernel_size=7, strides=2, filters=16, activation='relu', padding='same', name='disc_a')(inLayerDisc)
conv2_disc = layers.Conv2D(kernel_size=7, strides=2, filters=32, activation='relu', padding='same',name='disc_b')(conv1_disc)
conv3_disc = layers.Conv2D(kernel_size=7, strides=2, filters=32, activation='relu', padding='same',name='disc_c')(conv2_disc)
conv4_disc = layers.Conv2D(kernel_size=7, strides=2, filters=32, activation='relu', padding='same',name='disc_d')(conv3_disc)

conv4_disc_max_pool = layers.MaxPooling2D(pool_size=3, strides=4, padding='same')(conv4_disc) #get right size with stride=4, not sure about rest though

FC1_disc = layers.Dense(1, activation='sigmoid')(conv4_disc_max_pool)#'softmax')(conv4_disc_max_pool)

model_disc = Model(inLayerDisc, FC1_disc)

# model.compile('sgd', losses.categorical_crossentropy, metrics=['acc'])
#utils.plot_model(model, 'Model.png', show_shapes=True)


#------------dann classifier------------------
inLayerDann = layers.Input(shape=(1, 1, 1024)) 
conv1_dann = layers.Conv2D(kernel_size=1, filters=1024, activation='elu', padding='valid', name='class_dann_a')(inLayerDann) #inputshape should be that of encoded img
conv2_dann = layers.Conv2D(kernel_size=1, filters=1024, activation='elu', padding='valid',name='class_dann_b')(conv1_dann)
conv3_dann = layers.Conv2D(kernel_size=1, filters=1024, activation='elu', padding='valid',name='class_dann_c')(conv2_dann)
conv4_dann = layers.Conv2D(kernel_size=1, filters=1, activation='relu', padding='valid',name='class_dann_d')(conv3_dann)

#want output to be (batch_size x 1 x 1 x 1) (currently it is)
classifier_dann = Model(inLayerDann, conv4_dann)



def mae_criterion(in_, target):
   return tf.reduce_mean((in_ - target)**2)

#should use since labels are not one-hot encoded I think (mae_criterion always spits out 0)
def sce_criterion(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
	
def abs_criterion(in_, target):
    return tf.reduce_mean(tf.abs(in_ - target))
def inverse_transform(images):
    return (images+1.)/2.