import en_core_web_lg
import imageio
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle as pkl
import random
import re
import scipy
import string
import tensorflow as tf
import time
import pickle

from myutils import tfsession
from scipy.io import loadmat

import ops
from utils import *

# -------------------------------------------- #

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('mode', type=str, choices=['train', 'infer'], help='"train" or "infer"')
parser.add_argument('--restore', type=int, default=0, help='epoch of checkpoint to restore')
parser.add_argument('--epoch', type=int, default=300, help='epoch to train')
parser.add_argument('--caps', type=int, default=5, help='caps per image to use in trainging set')

args = parser.parse_args()

# -------------------------------------------- #

dictionary_path = './dictionary'
vocab = np.load(dictionary_path + '/vocab.npy')
print('there are {} vocabularies in total'.format(len(vocab)))

word2Id_dict = dict(np.load(dictionary_path + '/word2Id.npy'))
id2word_dict = dict(np.load(dictionary_path + '/id2Word.npy'))
print('Word to id mapping, for example: %s -> %s' % ('flower',
                                                     word2Id_dict['flower']))
print('Id to word mapping, for example: %s -> %s' % ('2428',
                                                     id2word_dict['2428']))
print('Tokens: <PAD>: %s; <RARE>: %s' % (word2Id_dict['<PAD>'],
                                         word2Id_dict['<RARE>']))

#  text = "the flower shown has yellow anther red pistil and bright red petals."
#  print(text)
#  print(sent2IdList(text, word2Id_dict))



# In[2]:


def get_nice_session(fraction=0.25):
    config = tf.ConfigProto()
    if fraction is not None:
        config.gpu_options.per_process_gpu_memory_fraction = fraction
    else:
        config.gpu_options.allow_growth = True
    return tf.Session(config=config)


# In[3]:


data_path = './dataset'
df = pd.read_pickle(data_path + '/text2ImgData.pkl')
num_training_sample = len(df)
n_images_train = num_training_sample
print('There are %d image in training data' % (n_images_train))


# In[4]:


print('Average number of captions of each image: {:.3f}'.format(
    np.mean([len(df['Captions'].iloc[i]) for i in range(len(df))])
))


IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64
IMAGE_DEPTH = 3

def training_data_generator(caption, image_path):
    # load in the image according to image path
    imagefile = tf.read_file(data_path + image_path)
    image = tf.image.decode_image(imagefile, channels=3)
    float_img = tf.image.convert_image_dtype(image, tf.float32)
    float_img.set_shape([None, None, 3])
    image = tf.image.resize_images(float_img, size=[IMAGE_HEIGHT, IMAGE_WIDTH])
    image.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH])
#     image = np.load('./dataset/by_index/%d.npy' % image_path)
    return image, caption

with open('./dataset/thought_front2400.pkl', 'rb') as f:
    thought_caps = pickle.load(f)

def data_iterator(filenames, batch_size, data_generator, shuffle_buffer=None,
                  augmentation=False, number_use_captions=1):
    # Load the training data into two NumPy arrays
    df = pd.read_pickle(filenames)
    caption = []
    image_path = []
    for i, row in enumerate(df.iterrows()):
        index = row[0]
#         caps = row[1]['Captions']
        caps = list(thought_caps[i])
        path = row[1]['ImagePath']
        num_to_choose = min(number_use_captions, len(caps))
        caption.extend(random.sample(caps, num_to_choose))
        image_path.extend([path for _ in range(num_to_choose)])
    image_path = np.asarray(image_path)
    caption = np.asarray(caption)
    
    # shuffle dataset to prevent consective same image due to num_to_choose >= 2
    shuffle_index = np.random.permutation(len(image_path))
    image_path = image_path[shuffle_index]
    caption = caption[shuffle_index]
    

    # Assume that each row of `features` corresponds to the same row as `labels`.
    assert caption.shape[0] == image_path.shape[0]

    dataset = tf.data.Dataset.from_tensor_slices((caption, image_path))
    dataset = dataset.repeat()
    if shuffle_buffer is not None:
        dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.map(data_generator)
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_initializable_iterator()
    output_types = dataset.output_types
    output_shapes = dataset.output_shapes

    return iterator, output_types, output_shapes


# ## Test on data iterator

# In[10]:


# tf.reset_default_graph()
# BATCH_SIZE = 64
# iterator_train, types, shapes = data_iterator(
#     data_path + '/text2ImgData.pkl', BATCH_SIZE, training_data_generator,
#     shuffle_buffer=None, number_use_captions=1
# )
# iter_initializer = iterator_train.initializer
# next_element = iterator_train.get_next()

# from PIL import Image
# from IPython.display import display

# with get_nice_session(fraction=0.1) as sess:
#     sess.run(iter_initializer)
#     image, text = sess.run(next_element)
    
#     for i in range(5):
#         display(Image.fromarray((image[i] * 255).astype('uint8')))
#         print([id2word_dict[word] for word in text[i].astype('str')])

# # Pretrained Word Embedding
# if os.path.isfile('word_embedding_matrix.pkl'):
#     print('Load existed word embedding matrix')
#     with open('word_embedding_matrix.pkl', 'rb') as f:
#         word_embedding_matrix = pkl.load(f)
#     print(word_embedding_matrix.shape)
# else:
#     # Preparing word embedding matrix
#     nlp = en_core_web_lg.load()
#     print(nlp('apple').vector.shape)
#     word_embedding_matrix = [[0 for j in range(300)] for i in range(len(vocab))]
#     for word in word2Id_dict:
#         wordid = int(word2Id_dict[word])
#         wordstr = str(word)
#         if '<PAD>' in wordstr:
#             continue
#         word_vector = nlp(wordstr).vector
#         word_embedding_matrix[wordid] = list(word_vector)
#     word_embedding_matrix = np.array(word_embedding_matrix)
#     print(word_embedding_matrix.shape)
#     with open('word_embedding_matrix.pkl', 'wb') as f:
#         pkl.dump(word_embedding_matrix, f)


# # Text Encoder,
# # G and D

# In[14]:


class TextEncoder:
    """
      Encode text (a caption) into hidden representation
      input: text (a list of id)
      output: hidden representation of input text in dimention of TEXT_DIM
      """

    def __init__(self,
                 text,
                 hparas,
                 training_phase=True,
                 reuse=False,
                 return_embed=False):
#         self.text = text
#         self.hparas = hparas
#         self.train = training_phase
        self.reuse = reuse
        self._build_model()
        self.outputs = text
        print(text)

    def _build_model(self):
        with tf.variable_scope('rnnftxt', reuse=self.reuse):
            garbage = tf.Variable(
                initial_value=0,
                trainable=True,
                name='useless_garbage'
            )
#             # Word embedding
#             word_embed_matrix = tf.Variable(
#                 word_embedding_matrix,
#                 name='wordembed',
#                 dtype=tf.float32
#             )
# #             word_embed_matrix = tf.get_variable(
# #                 'rnn/wordembed',
# #                 shape=(self.hparas['VOCAB_SIZE'], self.hparas['EMBED_DIM']),
# #                 initializer=tf.random_normal_initializer(stddev=0.02),
# #                 dtype=tf.float32)
#             embedded_word_ids = tf.nn.embedding_lookup(
#                 word_embed_matrix, self.text)
#             # RNN encoder
#             LSTMCell = tf.contrib.rnn.BasicLSTMCell(
#                 self.hparas['TEXT_DIM'], reuse=self.reuse)
#             initial_state = LSTMCell.zero_state(
#                 self.hparas['BATCH_SIZE'], dtype=tf.float32)
#             rnn_net = tf.nn.dynamic_rnn(
#                 cell=LSTMCell,
#                 inputs=embedded_word_ids,
#                 initial_state=initial_state,
#                 dtype=np.float32,
#                 time_major=False)
#             self.rnn_net = rnn_net
#             self.outputs = rnn_net[0][:, -1, :]


# In[8]:


class Generator:
    def __init__(self, noise_z, text, training_phase, hparas, reuse):
        self.z = noise_z
        self.text = text
        self.train = training_phase
        self.hparas = hparas
        self.gf_dim = 64
        self.reuse = reuse

        self._build_model()

    def _build_model(self):
        with tf.variable_scope('generator', reuse=self.reuse):
            s = self.hparas['IMAGE_SIZE'][0]
            s2, s4, s8, s16 = int(s / 2), int(s / 4), int(s / 8), int(s / 16)

            gf_dim = self.gf_dim
            
#             print(self.text)
            reduced_text_embedding = ops.lrelu(ops.linear(
                self.text, self.hparas['TEXT_DIM'], 'g_embedding'))
            z_concat = tf.concat([self.z, reduced_text_embedding], 1)
            z_ = ops.linear(z_concat, gf_dim * 8 * s16 * s16, 'g_h0_lin')
            h0 = tf.reshape(z_, [-1, s16, s16, gf_dim * 8])
            h0 = tf.nn.relu(ops.batch_norm(h0))

            h1 = ops.deconv2d(h0, [self.hparas['BATCH_SIZE'],
                               s8, s8, gf_dim * 4], name='g_h1')
            h1 = tf.nn.relu(ops.batch_norm(h1))

            h2 = ops.deconv2d(h1, [self.hparas['BATCH_SIZE'],
                               s4, s4, gf_dim * 2], name='g_h2')
            h2 = tf.nn.relu(ops.batch_norm(h2))

            h3 = ops.deconv2d(h2, [self.hparas['BATCH_SIZE'],
                               s2, s2, gf_dim * 1], name='g_h3')
            h3 = tf.nn.relu(ops.batch_norm(h3))

            h4 = ops.deconv2d(h3, [self.hparas['BATCH_SIZE'], s, s, 3], name='g_h4')

            self.generator_net = tf.tanh(h4) / 2.0 + 0.5
            self.outputs = tf.tanh(h4) / 2.0 + 0.5


# In[9]:


class Discriminator:
    def __init__(self, image, text, training_phase, hparas, reuse):
        self.image = image
        self.text = text
        self.train = training_phase
        self.hparas = hparas
        self.df_dim = 64  # 196 for MSCOCO
        self.reuse = reuse

        self._build_model()

    def _build_model(self):
        with tf.variable_scope('discriminator', reuse=self.reuse):
            df_dim = self.df_dim

            h0 = ops.lrelu(ops.conv2d(
                self.image, df_dim, name='d_h0_conv'))  # 32
            h1 = ops.lrelu(ops.batch_norm(ops.conv2d(
                h0, df_dim * 2, name='d_h1_conv')))  # 16
            h2 = ops.lrelu(ops.batch_norm(ops.conv2d(
                h1, df_dim * 4, name='d_h2_conv')))  # 8
            h3 = ops.lrelu(ops.batch_norm(ops.conv2d(
                h2, df_dim * 8, name='d_h3_conv')))  # 4

            # ADD TEXT EMBEDDING TO THE NETWORK
            reduced_text_embeddings = ops.lrelu(ops.linear(
                self.text, self.hparas['TEXT_DIM'], 'd_embedding'))
            reduced_text_embeddings = tf.expand_dims(reduced_text_embeddings, 1)
            reduced_text_embeddings = tf.expand_dims(reduced_text_embeddings, 2)
            tiled_embeddings = tf.tile(reduced_text_embeddings, [
                                       1, 4, 4, 1], name='tiled_embeddings')

            h3_concat = tf.concat([h3, tiled_embeddings], 3, name='h3_concat')
            h3_new = ops.lrelu(ops.batch_norm(ops.conv2d(
                h3_concat, df_dim * 8, 1, 1, 1, 1, name='d_h3_conv_new')))  # 4

            h4 = ops.linear(tf.reshape(
                h3_new, [self.hparas['BATCH_SIZE'], -1]), 1, 'd_h3_lin')

            self.logits = h4
            self.discriminator_net = tf.nn.sigmoid(h4)
            self.outputs = tf.nn.sigmoid(h4)


# # ~~ GAN ~~

# In[15]:


class WGAN_GP:
    def __init__(self,
                 hparas,
                 training_phase,
                 dataset_path,
                 ckpt_path,
                 inference_path,
                 sample_path,
                 recover=None):
        self.hparas = hparas
        self.train = training_phase
        self.dataset_path = dataset_path  # dataPath+'/text2ImgData.pkl'
        self.ckpt_path = ckpt_path
        self.sample_path = sample_path
        self.inference_path = inference_path

        self._get_session()  # get session
        self._get_train_data_iter()  # initialize and get data iterator
        self._input_layer()  # define input placeholder
        self._get_inference()  # build generator and discriminator
        self._get_loss()  # define gan loss
        self._get_var_with_name()  # get variables for each part of model
        self._optimize()  # define optimizer
        self._init_vars()
        self._get_saver()
        self._base_epoch = 0

        if recover is not None:
            self._load_checkpoint(recover)
            self._base_epoch = int(recover)

    def _get_train_data_iter(self):
        if self.train:  # training data iteratot
            iterator_train, types, shapes = data_iterator(
                self.dataset_path +
                '/text2ImgData.pkl', self.hparas['BATCH_SIZE'],
                training_data_generator,
                number_use_captions=self.hparas['N_CAPS_PER_IMAGE'],
                shuffle_buffer=self.hparas['SHUFFLE_BUFFER']
            )
            iter_initializer = iterator_train.initializer
            next_element = iterator_train.get_next()
            self.sess.run(iterator_train.initializer)
            self.iterator_train = iterator_train
            self.iterator_train_next_element = next_element
        else:  # testing data iterator
            iterator_train, types, shapes = data_iterator_test(
                self.dataset_path + '/testData.pkl', self.hparas['BATCH_SIZE'])
            iter_initializer = iterator_train.initializer
            next_element = iterator_train.get_next()
            self.sess.run(iterator_train.initializer)
            self.iterator_test = iterator_train
            self.iterator_test_next_element = next_element

    def _input_layer(self):
        if self.train:
            self.real_image = tf.placeholder(
                'float32', [
                    self.hparas['BATCH_SIZE'], self.hparas['IMAGE_SIZE'][0],
                    self.hparas['IMAGE_SIZE'][1], self.hparas['IMAGE_SIZE'][2]
                ],
                name='real_image')
#             self.caption = tf.placeholder(
#                 dtype=tf.int64,
#                 shape=[self.hparas['BATCH_SIZE'], self.hparas['MAX_SEQ_LENGTH']],
#                 name='caption')
            self.caption = tf.placeholder(
                dtype=tf.float32,
                shape=[self.hparas['BATCH_SIZE'], 2400],
                name='caption'
            )
            self.z_noise = tf.placeholder(
                tf.float32, [self.hparas['BATCH_SIZE'], self.hparas['Z_DIM']],
                name='z_noise')
        else:
#             self.caption = tf.placeholder(
#                 dtype=tf.int64,
#                 shape=[self.hparas['BATCH_SIZE'], self.hparas['MAX_SEQ_LENGTH']],
#                 name='caption')
            self.caption = tf.placeholder(
                dtype=tf.float32,
                shape=[self.hparas['BATCH_SIZE'], 2400],
                name='caption'
            )
            self.z_noise = tf.placeholder(
                tf.float32, [self.hparas['BATCH_SIZE'], self.hparas['Z_DIM']],
                name='z_noise')

    def _get_inference(self):
        if self.train:
            # GAN training
            # encoding text
#             text_encoder = self.caption
            text_encoder = TextEncoder(
                self.caption, hparas=self.hparas, training_phase=True, reuse=False)
            self.text_encoder = text_encoder
            # generating image
            generator = Generator(
                self.z_noise,
                text_encoder.outputs,
                training_phase=True,
                hparas=self.hparas,
                reuse=False)
            self.generator = generator

            # discriminize
            # fake image
            fake_discriminator = Discriminator(
                generator.outputs,
                text_encoder.outputs,
                training_phase=True,
                hparas=self.hparas,
                reuse=False)
            self.fake_discriminator = fake_discriminator
            # real image
            real_discriminator = Discriminator(
                self.real_image,
                text_encoder.outputs,
                training_phase=True,
                hparas=self.hparas,
                reuse=True)
            self.real_discriminator = real_discriminator

        else:  # inference mode
            self.text_embed = TextEncoder(
                self.caption, hparas=self.hparas, training_phase=False, reuse=False
            )
#             self.text_embed = self.caption
            self.generate_image_net = Generator(
                self.z_noise,
                self.text_embed.outputs,
                training_phase=False,
                hparas=self.hparas,
                reuse=False
            )

    def _get_loss(self):
        if self.train:
            d_loss1 = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.real_discriminator.logits,
                    labels=tf.ones_like(self.real_discriminator.logits),
                    name='d_loss1'))
            d_loss2 = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.fake_discriminator.logits,
                    labels=tf.zeros_like(self.fake_discriminator.logits),
                    name='d_loss2'))
#             d_loss1 = -tf.reduce_mean(self.real_discriminator.logits, name='d_loss1')
#             d_loss2 = tf.reduce_mean(self.fake_discriminator.logits, name='d_loss2')
            self.d_loss = d_loss1 + d_loss2
            self.g_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.fake_discriminator.logits,
                    labels=tf.ones_like(self.fake_discriminator.logits),
                    name='g_loss'))
#             self.g_loss = -tf.reduce_mean(self.fake_discriminator.logits, name='g_loss')
            
    # ---------------------- gradient_penalty ---------------------- #
            self.GP_lambda = self.hparas['GP_lambda']
            epsilon = tf.random_uniform(
                shape=(tf.shape(self.generator.outputs)[0], 1, 1, 1),
                minval=0,
                maxval=1
            )
            x_hat = epsilon * self.generator.outputs + (1.0 - epsilon) * self.real_image
            gradients = tf.gradients(
                Discriminator(
                    x_hat,
                    self.text_encoder.outputs,
                    training_phase=True,
                    hparas=self.hparas,
                    reuse=True
                ).logits,
                [x_hat]
            )
            gradient_penalty = self.GP_lambda * tf.square(tf.norm(gradients[0], ord=2) - 1.0)
            self.d_loss += gradient_penalty
    # ---------------------- gradient_penalty ---------------------- #

    def _optimize(self):
        if self.train:
            with tf.variable_scope('learning_rate'):
                self.lr_var = tf.Variable(self.hparas['LR'], trainable=False)

            discriminator_optimizer = tf.train.AdamOptimizer(
                self.lr_var, beta1=self.hparas['BETA']
            )
            generator_optimizer = tf.train.AdamOptimizer(
                self.lr_var, beta1=self.hparas['BETA']
            )
            self.d_optim = discriminator_optimizer.minimize(
                self.d_loss, var_list=self.discrim_vars
            )
            self.g_optim = generator_optimizer.minimize(
                self.g_loss, var_list=self.generator_vars + self.text_encoder_vars
            )

    def training(self):
        self.n_critic = self.hparas['N_CRITIC']
        save_every = max(1, 15 // self.hparas['N_CAPS_PER_IMAGE'])
        
#         all_trainable_vars = tf.trainable_variables()
#         all_trainable_vars_values = self.sess.run(trainable_vars)
#         import matplotlib.pyplot as plt
#         plt.hist(all_trainable_vars_values)
#         plt.show()
#         exit()
        
        for _epoch in range(self.hparas['N_EPOCH']):
            start_time = time.time()

            if _epoch != 0 and (_epoch % self.hparas['DECAY_EVERY'] == 0):
                new_lr_decay = self.hparas['LR_DECAY']**(
                    _epoch // self.hparas['DECAY_EVERY'])
                self.sess.run(tf.assign(self.lr_var, self.hparas['LR'] * new_lr_decay))
                print("new lr %f" % (self.hparas['LR'] * new_lr_decay))

            n_batch_epoch = int(self.hparas['N_SAMPLE'] / self.hparas['BATCH_SIZE'])
            
            counter = 0
            for _step in range(n_batch_epoch):
                step_time = time.time()
                image_batch, caption_batch = self.sess.run(
                    self.iterator_train_next_element
                )
                b_z = np.random.normal(
                    loc=0.0,
                    scale=1.0,
                    size=(self.hparas['BATCH_SIZE'],
                          self.hparas['Z_DIM'])
                ).astype(np.float32)
                
#                 if counter % self.n_critic:
                if True:
                    # update discriminator
                    self.sess.run(
                        self.d_optim,
                        feed_dict={
                            self.real_image: image_batch,
                            self.caption: caption_batch,
                            self.z_noise: b_z
                        }
                    )
#                 else:
                if True:
                    # update generator
                    self.sess.run(
                        self.g_optim,
                        feed_dict={
                            self.caption: caption_batch,
                            self.z_noise: b_z
                        }
                    )
                if _step % 50 == 0:
                    self.discriminator_error, self.generator_error = self.sess.run(
                        [self.d_loss, self.g_loss],
                        feed_dict={
                            self.real_image: image_batch,
                            self.caption: caption_batch,
                            self.z_noise: b_z
                        }
                    )
                    print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4fs, d_loss: %.3f, g_loss: %.3f"
                          % (_epoch + self._base_epoch + 1,
                             self.hparas['N_EPOCH'] + self._base_epoch,
                             _step, n_batch_epoch,
                             time.time() - step_time,
                             self.discriminator_error,
                             self.generator_error))

            if _epoch != 0 and (_epoch + 1) % save_every == 0:
                self._save_checkpoint(_epoch + self._base_epoch + 1)
                self._sample_visualize(_epoch + self._base_epoch + 1)

    def inference(self):
        for _iters in range(100):
            caption, idx = self.sess.run(self.iterator_test_next_element)
            z_seed = np.random.normal(
                loc=0.0,
                scale=1.0,
                size=(self.hparas['BATCH_SIZE'],
                      self.hparas['Z_DIM'])).astype(np.float32)

            img_gen, rnn_out = self.sess.run(
                [self.generate_image_net.outputs, self.text_embed.outputs],
                feed_dict={self.caption: caption,
                           self.z_noise: z_seed})
            for i in range(self.hparas['BATCH_SIZE']):
                imageio.imwrite(
                    self.inference_path +
                    '/inference_{:04d}.png'.format(idx[i]),
                    img_gen[i])

    def _init_vars(self):
        self.sess.run(tf.global_variables_initializer())

    def _get_session(self):
        self.sess = get_nice_session(fraction=0.2)

    def _get_saver(self):
        max_to_keep=20
        if self.train:
            self.rnn_saver = tf.train.Saver(var_list=self.text_encoder_vars, max_to_keep=max_to_keep)
            self.g_saver = tf.train.Saver(var_list=self.generator_vars, max_to_keep=max_to_keep)
            self.d_saver = tf.train.Saver(var_list=self.discrim_vars, max_to_keep=max_to_keep)
        else:
            self.rnn_saver = tf.train.Saver(var_list=self.text_encoder_vars, max_to_keep=max_to_keep)
            self.g_saver = tf.train.Saver(var_list=self.generator_vars, max_to_keep=max_to_keep)

    def _sample_visualize(self, epoch):
        ni = int(np.ceil(np.sqrt(self.hparas['BATCH_SIZE'])))
        sample_size = self.hparas['BATCH_SIZE']
        max_len = self.hparas['MAX_SEQ_LENGTH']

        sample_seed = np.random.normal(
            loc=0.0, scale=1.0, size=(sample_size,
                                      self.hparas['Z_DIM'])).astype(np.float32)
#         sample_sentence = [
#             "the flower shown has yellow anther red pistil and bright red petals."
#         ] * int(sample_size / ni) + [
#             "this flower has petals that are yellow, white and purple and has dark lines"
#         ] * int(sample_size / ni) + [
#             "the petals on this flower are white with a yellow center"
#         ] * int(sample_size / ni) + [
#             "this flower has a lot of small round pink petals."
#         ] * int(sample_size / ni) + [
#             "this flower is orange in color, and has petals that are ruffled and rounded."
#         ] * int(sample_size / ni) + [
#             "the flower has yellow petals and the center of it is brown."
#         ] * int(sample_size / ni) + [
#             "this flower has petals that are blue and white."
#         ] * int(sample_size / ni) + [
#             "these white flowers have petals that start off white in color and end in a white towards the tips."
#         ] * int(sample_size / ni)

#         for i, sent in enumerate(sample_sentence):
#             sample_sentence[i] = sent2IdList(sent, word2Id_dict, max_len)

        sample_sentence = np.load('./dataset/sample_sentence_4800.npy', encoding='bytes')
        sample_sentence = sample_sentence[:, :2400]

        img_gen, rnn_out = self.sess.run(
            [self.generator.outputs, self.text_encoder.outputs],
            feed_dict={self.caption: sample_sentence,
                       self.z_noise: sample_seed})
        save_images(img_gen, [ni, ni],
                    self.sample_path + '/train_{:02d}.png'.format(epoch))

    def _get_var_with_name(self):
        t_vars = tf.trainable_variables()

        self.text_encoder_vars = [var for var in t_vars if 'rnn' in var.name]
        print('text encoder var to train:', self.text_encoder_vars)
        self.generator_vars = [
            var for var in t_vars if 'generator' in var.name]
        self.discrim_vars = [var for var in t_vars if 'discrim' in var.name]

    def _load_checkpoint(self, recover):
        if self.train:
            self.rnn_saver.restore(
                self.sess, self.ckpt_path + 'rnn_model_' + str(recover) + '.ckpt'
            )
            self.g_saver.restore(
                self.sess, self.ckpt_path + 'g_model_' + str(recover) + '.ckpt'
            )
            self.d_saver.restore(
                self.sess, self.ckpt_path + 'd_model_' + str(recover) + '.ckpt'
            )
        else:
            self.rnn_saver.restore(
                self.sess, self.ckpt_path + 'rnn_model_' + str(recover) + '.ckpt'
            )
            self.g_saver.restore(
                self.sess, self.ckpt_path + 'g_model_' + str(recover) + '.ckpt'
            )
        print('-----success restored checkpoint with epoch {}--------'.format(recover))

    def _save_checkpoint(self, epoch):
        self.rnn_saver.save(self.sess,
                            self.ckpt_path + 'rnn_model_' + str(epoch) + '.ckpt')
        self.g_saver.save(self.sess,
                          self.ckpt_path + 'g_model_' + str(epoch) + '.ckpt')
        self.d_saver.save(self.sess,
                          self.ckpt_path + 'd_model_' + str(epoch) + '.ckpt')
        print('-----success saved checkpoint with epoch {}--------'.format(epoch))
#         if epoch % 100 == 0:
#             os.system('cp -r ./checkpoint ./checkpoint-{}'.format(epoch))


# # Training with 

# In[18]:


if not os.path.exists('./samples'):
    os.mkdir('./samples')
if not os.path.exists('./inference'):
    os.mkdir('./inference')


tf.reset_default_graph()
sub_path = 'WGAN_GP_%d_caps_shuffle_ThoughtVector/' % args.caps
checkpoint_path = './checkpoint/' + sub_path
inference_path = './inference/' + sub_path + 'restore_%d/' % args.restore
sample_path = './samples/' + sub_path

for ppp in ['./inference/' + sub_path, checkpoint_path, inference_path, sample_path]:
    if not os.path.exists(ppp):
        os.mkdir(ppp)

def get_hparas():
    hparas = {
        'MAX_SEQ_LENGTH': 20,
        'EMBED_DIM': 300,  # word embedding dimension
        'VOCAB_SIZE': len(vocab),
        'TEXT_DIM': 64,  # text embrdding dimension
        'GF_DIM': 64,
        'DF_DIM': 64,
        'RNN_HIDDEN_SIZE': 64,
        'Z_DIM': 64,  # random noise z dimension
        'IMAGE_SIZE': [64, 64, 3],  # render image size
        'BATCH_SIZE': 64,
        'LR': 0.002,
        'DECAY_EVERY': 100,
        'LR_DECAY': 0.5,
        'BETA': 0.5,  # AdamOptimizer parameter
        'N_EPOCH': args.epoch,
        'N_SAMPLE': num_training_sample,
        'N_CAPS_PER_IMAGE': args.caps,
        'N_CRITIC': 3,
        'GP_lambda': 1
    }
    hparas['N_SAMPLE'] *= hparas['N_CAPS_PER_IMAGE']
    hparas['SHUFFLE_BUFFER'] = hparas['BATCH_SIZE'] * 2
    return hparas


# In[19]:



def data_iterator_test(filenames, batch_size):
    data = pd.read_pickle(filenames)
#     captions = data['Captions'].values
#     caption = []
#     for i in range(len(captions)):
#         caption.append(captions[i])
    caption = np.load('./dataset/test_captions_4800.npy', encoding='bytes')
    caption = np.asarray(caption)
    caption = caption[:, :2400]
    index = data['ID'].values
    index = np.asarray(index)

    dataset = tf.data.Dataset.from_tensor_slices((caption, index))
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_initializable_iterator()
    output_types = dataset.output_types
    output_shapes = dataset.output_shapes

    return iterator, output_types, output_shapes

n_restore = None if args.restore == 0 else args.restore

if args.mode == 'train':
    gan = WGAN_GP(
        get_hparas(),
        training_phase=True,
        dataset_path=data_path,
        ckpt_path=checkpoint_path,
        inference_path=inference_path,
        sample_path=sample_path,
        recover=n_restore
    )
    gan.training()
else:
    tf.reset_default_graph()
    gan = WGAN_GP(
        get_hparas(),
        training_phase=False,
        dataset_path=data_path,
        ckpt_path=checkpoint_path,
        inference_path=inference_path,
        sample_path=sample_path,
        recover=n_restore)
    img = gan.inference()


# In[18]:


#  myimg = imageio.imread('./inference/inference_0057.png')
#  plt.imshow(myimg)
#  plt.show()


