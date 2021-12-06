
import tensorflow as tf
import numpy as np
from tensorflow.keras.losses import BinaryCrossentropy
from PIL import Image
from model import *
from keras.preprocessing import image
import os, os.path

#import imageio
import cv2

from glob import glob



class xgan:

		
  def __init__(self, sess, encoder1, encoder2, decoder1, decoder2, discriminator, image_size):
        self.sess = sess
        self.encoder1 = encoder1
        self.encoder2 = encoder2
        self.decoder1 = decoder1
        self.decoder2 = decoder2
        self.discriminator = discriminator

        self.image_size = image_size

        self.mae_criterion = mae_criterion
        self.sce_criterion = sce_criterion
        self.abs_criterion = abs_criterion
        self.build_model()

        self.train()
        self.sample_model()
        #self.pool = ImagePool(args.max_size)


  
  def build_model(self):
      # input : 2 images from domain 1 and 2
      #self.input_data = tf.placeholder(tf.float32,[None, 2, self.image_size, self.image_size, 3], name='real_1_and_2_images')

      self.w_dann = 0.5
      self.w_sem = 0.5
      self.w_gan = 0.5

      self.real_data = tf.placeholder(tf.float32, [None, 64, 64, 3 + 3], name='real_1_and_2_images')
        # Input: image of domain A and B
      self.real_1 = self.real_data[:, :, :, :3]
      self.real_2 = self.real_data[:, :, :, 3:3+3]

      # Generator
      # Encode images in same subspace
      #should have input shape=(64, 64, 3)
      self.encoded_1 = self.encoder1(self.real_1)
      self.encoded_2 = self.encoder2(self.real_2)
      # Recover images (auto-encoders)
      self.restored_1 = self.decoder1(self.encoded_1)
      self.restored_2 = self.decoder2(self.encoded_2)
      # Generate new images
      self.fake_1 = self.decoder1(self.encoded_2)
      self.fake_2 = self.decoder2(self.encoded_1)
	  # Fake image encoder output
      self.encoded_fake_1 = self.encoder1(self.fake_1)
      self.encoded_fake_2 = self.encoder2(self.fake_2)

      # Discriminator
      self.disc_fake1 = self.discriminator(self.fake_1)
      self.disc_fake2 = self.discriminator(self.fake_2)

      #print(self.encoded_1)
	  # Discriminator
      self.fake_1_sample = tf.placeholder(tf.float32,[None, 64, 64,3], name='fake_1_sample')
      self.fake_2_sample = tf.placeholder(tf.float32,[None, 64, 64,3], name='fake_2_sample')
	  # Discriminator output
      self.discriminate_real_1 = self.discriminator(self.real_1)
      self.discriminate_real_2 = self.discriminator(self.real_2)
      self.discriminate_fake_sample_1 = self.discriminator(self.fake_1_sample)
      self.discriminate_fake_sample_2 = self.discriminator(self.fake_2_sample)
      # Gan loss-discriminator part
	  
      #self.dis_gan_loss_real_1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = self.discriminate_real_1, logits = tf.zeros_like(self.discriminate_real_1))) + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.ones_like(self.discriminate_real_1), logits = tf.zeros_like(tf.ones_like(self.discriminate_real_1))))
      self.dis_gan_loss_real_1 = self.sce_criterion(self.discriminate_real_1, tf.ones_like(self.discriminate_real_1))
      #self.dis_gan_loss_fake_1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = self.disc_fake1, logits = tf.zeros_like(self.disc_fake1))) + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.ones_like(self.disc_fake1), logits = tf.zeros_like(tf.ones_like(self.disc_fake1))))
      self.dis_gan_loss_fake_1 = self.sce_criterion(self.disc_fake1, tf.ones_like(self.disc_fake1))
      self.dis_gan_loss_1 = (self.dis_gan_loss_real_1 + self.dis_gan_loss_fake_1) / 2 #ok
      self.dis_gan_loss_real_2 = self.sce_criterion(self.discriminate_real_2, tf.ones_like(self.discriminate_real_2))
      #self.dis_gan_loss_real_2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = self.discriminate_real_2, logits = tf.zeros_like(self.discriminate_real_2))) + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.ones_like(self.discriminate_real_2), logits = tf.zeros_like(tf.ones_like(self.discriminate_real_2))))
      self.dis_gan_loss_fake_2 = self.sce_criterion(self.disc_fake2, tf.ones_like(self.disc_fake2))
      #self.dis_gan_loss_fake_2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = self.disc_fake2, logits = tf.zeros_like(self.disc_fake2))) + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.ones_like(self.disc_fake2), logits = tf.zeros_like(tf.ones_like(self.disc_fake2))))
      self.dis_gan_loss_2 = (self.dis_gan_loss_real_2 + self.dis_gan_loss_fake_2) / 2 #ok
      self.dis_gan_loss = self.dis_gan_loss_1 + self.dis_gan_loss_2 #ok
	  #total discriminator loss
      self.dis_loss = self.w_gan * self.dis_gan_loss
	  
      self.dis_gan_loss_sum = tf.summary.scalar("dis_gan_loss", self.dis_gan_loss)
      self.dis_loss_sum = tf.summary.scalar("dis_loss", self.dis_loss)
      self.dis_sum = tf.summary.merge([self.dis_gan_loss_sum, self.dis_loss_sum])
	  
      # CLassifer needed for cdann
      self.class_dann_1 = classifier_dann(self.encoded_1)
      self.class_dann_2 = classifier_dann(self.encoded_2)
      # Domain-adversarial loss
      self.dann_loss = self.sce_criterion(self.class_dann_1, tf.ones_like(self.class_dann_1)) + self.sce_criterion(self.class_dann_2, tf.ones_like(self.class_dann_2))
      #self.dann_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = self.class_dann_1, logits = tf.zeros_like(self.class_dann_1))) + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = self.class_dann_2, logits = tf.zeros_like(self.class_dann_2)))
     
      # Define different losses
      # reconstruction loss
      self.rec_loss_1 = tf.losses.mean_squared_error(self.restored_1, self.real_1)#self.input_data[:, 0]) #tf.losses.mean_squared_error(target, in_) 
      self.rec_loss_2 = tf.losses.mean_squared_error(self.restored_2, self.real_2)#self.input_data[:, 1])
      self.rec_loss = self.rec_loss_1 + self.rec_loss_2

      print("here2")
      self.sem_loss_1 = self.abs_criterion(self.encoded_1,self.encoded_fake_2)
      self.sem_loss_2 = self.abs_criterion(self.encoded_2,self.encoded_fake_1)
      #self.sem_loss_1 = tf.reduce_mean(tf.abs(self.encoded_1 - self.encoded_fake_2))
      #self.sem_loss_2 = tf.reduce_mean(tf.abs(self.encoded_2 - self.encoded_fake_1)) #NOT SURE HERE, won't sem_loss_1 and sem_loss_2 always be the same? - yes problem
      self.sem_loss = self.sem_loss_1 + self.sem_loss_2
      print("here3")
      #still bit uncertain about this
      #gan loss should be same as that defined in CycleGAN. https://github.com/clvrai/CycleGAN-Tensorflow/blob/master/model.py
      #here we are using least square (seems to be option between least square and mae
      #self.gan_loss_1to2 = tf.reduce_mean(tf.squared_difference(self.fake_2, 0.9))
      #self.gan_loss_2to1 = tf.reduce_mean(tf.squared_difference(self.fake_1, 0.9))
      #self.gan_loss = self.gan_loss_1to2 + self.gan_loss_2to1

      # gan loss-generator part (using sce)
      #self.gen_gan_loss_1 =tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = self.disc_fake1, logits = tf.zeros_like(self.disc_fake1))) + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.ones_like(self.disc_fake1), logits = tf.zeros_like(tf.ones_like(self.disc_fake1))))
      self.gen_gan_loss_1 = self.sce_criterion(self.disc_fake1, tf.ones_like(self.disc_fake1))
      #self.gen_gan_loss_2 =tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = self.disc_fake2, logits = tf.zeros_like(self.disc_fake2))) + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.ones_like(self.disc_fake2), logits = tf.zeros_like(tf.ones_like(self.disc_fake2))))
      self.gen_gan_loss_2 = self.sce_criterion(self.disc_fake2, tf.ones_like(self.disc_fake2))
      self.gen_gan_loss =  self.gen_gan_loss_1 + self.gen_gan_loss_2

      # total gen loss
      self.gen_loss = self.rec_loss \
                      + self.w_dann * self.dann_loss \
                      + self.w_sem * self.sem_loss \
                      + self.w_gan * self.gen_gan_loss


      #for printing values
      self.rec_loss_sum = tf.summary.scalar("rec_loss", self.rec_loss)
      self.dann_loss_sum = tf.summary.scalar("dann_loss", self.dann_loss)
      self.sem_loss_sum = tf.summary.scalar("sem_loss", self.sem_loss)
      self.gen_gan_loss_sum = tf.summary.scalar("gen_gan_loss", self.gen_gan_loss)
      self.gen_loss_sum = tf.summary.scalar("gen_loss", self.gen_loss)
      self.gen_sum = tf.summary.merge([self.rec_loss_sum, self.dann_loss_sum, self.sem_loss_sum, self.gen_gan_loss_sum, self.gen_loss_sum])
		
      # discriminator loss
      # TODO

      #make encoder and decoder variables trainable
      trainable_vars = tf.trainable_variables()
      self.gen_vars = [var for var in trainable_vars if 'encoder' in var.name or 'decoder' in var.name or 'dann' in var.name]
      self.dis_vars = [var for var in trainable_vars if 'disc' in var.name] #need to add name for all discriminator layers for this
      #for var in trainable_vars: print(var.name)
      for var in self.gen_vars: print(var.name)
      for var in self.dis_vars: print(var.name)
      print("model built")




	
  def train(self):
      self.lr = tf.placeholder(tf.float32, None, name='learning_rate')
      lr = 0.001#0.0001
      self.num_epochs = 1#3
      batch_size = 2
      self.train_size = 1000 #1e8 images used to train
      self.gen_optim = tf.train.AdamOptimizer(self.lr, 0.9).minimize(self.gen_loss, var_list=self.gen_vars)
      self.dis_optim = tf.train.AdamOptimizer(self.lr, 0.9).minimize(self.dis_loss, var_list=self.dis_vars)
	  
	  #initialize variables
      init_op = tf.global_variables_initializer()
      self.sess.run(init_op)
	  
      #self.writer = tf.summary.FileWriter("./logs", self.sess.graph)
	  
      for epoch in range(self.num_epochs):
        #load train_1
        imgs_train_1 = glob('./datasets/train1/*')
        #load train_2
        imgs_train_2 = glob('./datasets/train2/*')
        #shuffle data
        np.random.shuffle(imgs_train_1)
        np.random.shuffle(imgs_train_2)
		
        min_len = len(imgs_train_1)
        if (len(imgs_train_2) < min_len):
          min_len = len(imgs_train_2)
        print(min_len)
		
        #path_list = []
        #for filename1 in glob('./datasets/train1/*'):
        #  path_list.append(filename1)
        #for filename2 in glob('./datasets/train2/*'):
        #  path_list.append(filename2)	
        batch_idxs = min(min(len(imgs_train_1), len(imgs_train_1)), self.train_size) // batch_size

        for idx in range(0, 1):#batch_idxs):		  
            #print("idx", idx)
            #combine image from domain 1 and 2
            batch_files = zip(imgs_train_1[idx * batch_size:(idx + 1) * batch_size],imgs_train_2[idx * batch_size:(idx + 1) * batch_size])
            combined_imgs = []
            for batch_file in batch_files:
               img_1 = cv2.imread(batch_file[0])
			   
               img_2 = cv2.imread(batch_file[1])
               img_2 = cv2.resize(img_2, (64,64),interpolation = cv2.INTER_AREA) #resize so img is 64x64 since it wasn't for cartoon
               print(img_2.shape)
			   #normalize img_1
               #print(img_1)
               img_combined = np.concatenate((img_1, img_2), axis=2)
               #print("img_combined")
               #print(img_combined)
               combined_imgs.append(img_combined)
			
            #ra = self.sess.run(self.real_1, feed_dict={self.real_data: combined_imgs}) 
           # Update generator network and record fake outputs
            fake_1, fake_2, rec_loss, dann_loss, sem_loss, gen_gan_loss, gen_loss, _, summary_str = self.sess.run([self.fake_1, self.fake_2, self.rec_loss, self.dann_loss, self.sem_loss,self.gen_gan_loss, self.gen_loss, self.gen_optim, self.gen_sum],feed_dict={self.real_data: combined_imgs, self.lr:lr})
           # Update discriminator network
            dis_loss, _, summary_str = self.sess.run([self.dis_loss, self.dis_optim, self.dis_sum], feed_dict={self.real_data: combined_imgs,self.fake_1_sample: fake_1,self.fake_2_sample: fake_2,self.lr: lr})
            
            print(("Epoch: [%2d] [%4d/%4d] rec_loss: %4.3f|dann_loss: %4.3f|sem_loss: %4.3f|gen_gan_loss: %4.3f|gen_loss: %4.3f|dis_loss: %4.3f" % (epoch, idx, batch_idxs, rec_loss, dann_loss, sem_loss, gen_gan_loss, gen_loss, dis_loss)))
		  
		  

  def sample_model(self):
        sample_dir = './datasets/samples'
        imgs_sample_1 = glob('./datasets/test1/*')
        imgs_sample_2 = glob('./datasets/test2/*')
        np.random.shuffle(imgs_sample_1)
        np.random.shuffle(imgs_sample_2)
        idx = 1
        batch_size = 2
        batch_files = zip(imgs_sample_1[idx * batch_size:(idx + 1) * batch_size],imgs_sample_2[idx * batch_size:(idx + 1) * batch_size]) #list(zip(data2[:self.batch_size], data2[:self.batch_size]))
        #print(batch_files)
        #sample_images = [load_train_data(batch_file, fine_size=64, is_testing=True) for batch_file in batch_files
        #sample_images = np.array(sample_images).astype(np.float32)
        sample_images = []
        for batch_file in batch_files:
            img_1 = cv2.imread(batch_file[0])
            #print(img_1)
            img_2 = cv2.imread(batch_file[1])
            cv2.imwrite('./datasets/samples/test.jpg', img_2)
            #print(img_2)
            img_2 = cv2.resize(img_2, (64,64),interpolation = cv2.INTER_AREA) #resize so img is 64x64 since it wasn't for cartoon
			   #normalize img_1
               #print(img_1)
            img_combined = np.concatenate((img_1, img_2), axis=2)
               #print("img_combined")
               #print(img_combined)
            sample_images.append(img_combined)
		
        fake_1, fake_2 = self.sess.run([self.fake_1, self.fake_2],feed_dict={self.real_data: sample_images}
        )
		
		
        #print(fake_1)
        fake_1 = (fake_1+1.)/2.#inverse_transform(fake_1)
        #print(fake_1)
        print(fake_1[0,:,:,:])
        #fake_1 = merge(fake_1, [batch_size, 1])
        cv2.imwrite('./datasets/samples/1_sample.jpg', fake_1[0,:,:,:]);
        #im = Image.fromarray(fake_1)
        #im.save('./datasets/samples/1_sample.jpg')
        #imsave(inverse_transform(fake_1), [64,64], './datasets/samples/1_sample.jpg')
        #imsave(inverse_transform(fake_1), [64,64], './datasets/samples/1_sample.jpg')
        #fake_1.save('./datasets/samples/1_sample.jpg')
        #fake_2.save('./datasets/samples/2_sample.jpg')
        
with tf.Session() as sess:
	print("hey1")
	model = xgan(sess, model_enc_1, model_enc_2, model_dec_1, model_dec_2, model_disc, 64)
	print("hey")
	
	#model.train(self, lr, num_epochs, batch_size, train_size)



