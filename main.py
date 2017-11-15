#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 16:14:30 2017

@author: PeterTsai
"""
import numpy as np
import tensorflow as tf
#import functools
import itertools
import sys

sys.path.append("./codes")
# LSTM-autoencoder
import LSTMAutoencoder
import miNet
from fixed_flags import FLAGS
import utils
import dataset
from nchoosek_Net import nchoosek_grouping
import net_param

datasets = dataset.dataset(FLAGS.traj_file,FLAGS.fold_file,FLAGS.tactic_file,FLAGS.fold,
                           FLAGS.lstm_max_sequence_length,FLAGS.lstm_input_type)
#utils.showProperties(datasets)

C53_combs = list(itertools.combinations([0,1,2,3,4],3))
C52_combs = list(itertools.combinations([0,1,2,3,4],2))
C55_combs = list(itertools.combinations([0,1,2,3,4],5))
np_nchoosek = [C53_combs,C52_combs,C55_combs]


tf.reset_default_graph()
# placeholder list
p_input = tf.placeholder(tf.float32, [None, FLAGS.lstm_max_sequence_length, datasets.input_feature_dim]) #[batch*5,dynamic step, input_feature]
seqlen = tf.placeholder(tf.int32,[None])
p_inputs= tf.transpose(p_input, perm=[1,0,2])

FLAGS.p_input = p_input
FLAGS.seqlen = seqlen

## cell should be in ae_lstm !!! fix in the future
activation_fn = utils.get_activation_fn(FLAGS.lstm_activation)
lstm_cell_op = utils.get_rnn_type(FLAGS.lstm_type)

if FLAGS.lstm_type is 'LSTM':
    cell = lstm_cell_op(FLAGS.lstm_hidden_dim, activation=activation_fn, use_peepholes=FLAGS.use_peepholes)
else:
    cell = lstm_cell_op(FLAGS.lstm_hidden_dim, activation=activation_fn)

''' add output warping to rnn '''
#cell = tf.contrib.rnn.OutputProjectionWrapper(cell,FLAGS.lstm_hidden_dim,activation=tf.nn.softmax) 
    
optimizer = utils.get_optimizer(FLAGS.optimizer)
with tf.name_scope("ae_lstm"):
    ae = LSTMAutoencoder.LSTMAutoencoder(FLAGS.lstm_hidden_dim, p_inputs, seqlen, #FLAGS.MAX_X, FLAGS.MAX_Y,
                                          cell=cell, decode_without_input=True, optimizer=None)#optimizer(FLAGS.lstm_lr))

tf.add_to_collection('decode_loss',ae.loss)
tf.add_to_collection('ae_lstm/lastoutput',ae.last_output) #'aelstm_lastoutput'
tf.add_to_collection('ae_lstm/dec_output',ae.output)

nchoosek_inputs, tf_C53_combs, tf_C52_combs, tf_C55_combs, C53_input, C52_input, C55_input= nchoosek_grouping(ae,np_nchoosek,datasets.numPlayer)

# =============================================================================
# bb = 4
# padS = datasets.dataTraj
# seqLenMatrix = datasets.seqLenMatrix
# A = padS[0:bb,:,:,:]
# A1 = np.reshape(padS[0:bb,:,:,:],(-1,450,2))
# batch_seqlen = np.reshape(seqLenMatrix[0:bb,:],(-1))
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     last = sess.run(ae.last_output, feed_dict={p_input:A1,seqlen: batch_seqlen})
#     c53,c52,c55 = sess.run([tf_C53_combs,tf_C52_combs,tf_C55_combs]
#         ,feed_dict={p_input:A1,seqlen: batch_seqlen})    
#     c53i,c52i,c55i = sess.run([C53_input, C52_input, C55_input]
#         ,feed_dict={p_input:A1,seqlen: batch_seqlen})
#     c5k_group = sess.run(nchoosek_inputs
#         ,feed_dict={p_input:A1,seqlen: batch_seqlen})    
#     for c in range(bb):
#         if np.sum(c53i[c] - last[c53[c],:]):
#             raise AssertionError("two output are not equal in C53 [batch{0}]".format(c))
#         if np.sum(c52i[c] - last[c52[c],:]):
#             raise AssertionError("two output are not equal in C52 [batch{0}]".format(c))            
#         if np.sum(c55i[c] - last[c55[c],:]):
#             raise AssertionError("two output are not equal in C55 [batch{0}]".format(c))            
# =============================================================================

"""
pre-training cycle
"""
sess = tf.Session()#tf.InteractiveSession()
#==============================================================================
# LSTMAutoencoder.pretraining(ae,sess,datasets,FLAGS)
# 
# """
# generate intermediate feature of training data
# """
# if FLAGS.resetLSTMTempData:
#     dataset.generateLSTMTempData(nchoosek_inputs,sess,datasets,FLAGS,'train')
#     dataset.generateLSTMTempData(nchoosek_inputs,sess,datasets,FLAGS,'test')
# 
#==============================================================================

pretrain_shape = net_param.createPretrainShape(FLAGS.lstm_hidden_dim,
                                               FLAGS.miNet_last_hidden_dim,
                                               FLAGS.miNet_num_hidden_layer)
print(pretrain_shape)  
                    

instNet_shape = np.array([np.append(pretrain_shape,len(datasets.C5k_CLASS[0])),
                          np.append(pretrain_shape,len(datasets.C5k_CLASS[1])),
                          np.append(pretrain_shape,len(datasets.C5k_CLASS[2]))],
                         np.int32)

print(instNet_shape)
num_inst = np.array([10,10,1],np.int32) # 5 choose 3 key players, 5 choose 2 key players, 5 choose 3 key players 
miNet_common_acfun = FLAGS.miNet_common_acfun
acfunList = []
for h in range(FLAGS.miNet_num_hidden_layer):
    acfunList.append(utils.get_activation_fn(miNet_common_acfun))
#acfunList.append(utils.get_activation_fn('linear')) #log_sigmoid
acfunList.append(None)
miList = miNet.main_unsupervised(instNet_shape,acfunList,datasets,FLAGS,sess)
miNet.main_supervised(miList,num_inst,nchoosek_inputs,datasets,FLAGS)
        
sess.close()        





