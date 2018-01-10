#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 01:32:17 2017

@author: PeterTsai
"""

from __future__ import division
from __future__ import print_function
import time
import os
from os.path import join as pjoin

import numpy as np
import tensorflow as tf
import pandas as pd 


import readmat
#from flags import FLAGS
from fixed_flags_miNet import FLAGS
import net_param
import dataset
import utils
import metric
import LSTMAutoencoder


class miNet(object):
    
    _weights_str = "weights{0}"
    _biases_str = "biases{0}"
    _inputs_str = "x{0}"
    _instNets_str = "I{0}"

    def __init__(self, shape, acfunList, batch_norm, sess):
        """Autoencoder initializer

        Args:
            shape: list of ints specifying
                  num input, hidden1 units,...hidden_n units, num logits
        sess: tensorflow session object to use
        """
        self.__shape = shape  # [input_dim,hidden1_dim,...,hidden_n_dim,output_dim]
        self.__num_hidden_layers = len(self.__shape) - 2
        self.acfunList = acfunList
        self.batch_norm = batch_norm

        self.__variables = {}
        self.__sess = sess

        self._setup_instNet_variables()
    
    @property
    def shape(self):
        return self.__shape
    
    @property
    def num_hidden_layers(self):
        return self.__num_hidden_layers
    
    @property
    def session(self):
        return self.__sess
    
    def __getitem__(self, item):
        """Get autoencoder tf variable

        Returns the specified variable created by this object.
        Names are weights#, biases#, biases#_out, weights#_fixed,
        biases#_fixed.

        Args:
            item: string, variables internal name
        Returns:
            Tensorflow variable
        """
        return self.__variables[item]
    
    def __setitem__(self, key, value):
        """Store a tensorflow variable

        NOTE: Don't call this explicity. It should
        be used only internally when setting up
        variables.

        Args:
            key: string, name of variable
            value: tensorflow variable
        """
        self.__variables[key] = value
    
    def _setup_instNet_variables(self):
        with tf.name_scope("InstNet_variables"):
            for i in range(self.__num_hidden_layers + 1):
                # Train weights
                name_w = self._weights_str.format(i + 1)
                w_shape = (self.__shape[i], self.__shape[i + 1])
# =============================================================================
#                 w_init = tf.truncated_normal(w_shape)
                w_init = utils.xavier_initializer(w_shape)
# =============================================================================
                
                self[name_w] = tf.Variable(w_init, name=name_w, trainable=True)
                # Train biases
                name_b = self._biases_str.format(i + 1)
                b_shape = (self.__shape[i + 1],)
# =============================================================================
                b_init = tf.zeros(b_shape) #+ 0.01
                #b_init = utils.xavier_initializer(b_shape,uniform=True)
# =============================================================================
                
                self[name_b] = tf.Variable(b_init, trainable=True, name=name_b)

                if i < self.__num_hidden_layers:
                    # Hidden layer fixed weights (after pretraining before fine tuning)
                    self[name_w + "_fixed"] = tf.Variable(tf.identity(self[name_w]),
                        name=name_w + "_fixed", trainable=False)
                    
                    # Hidden layer fixed biases
                    self[name_b + "_fixed"] = tf.Variable(tf.identity(self[name_b]),
                        name=name_b + "_fixed", trainable=False)
                    
                    # Pretraining output training biases
                    name_b_out = self._biases_str.format(i + 1) + "_out"
                    b_shape = (self.__shape[i],)
# =============================================================================
                    b_init = tf.zeros(b_shape) #+ 0.01
                    #b_init = utils.xavier_initializer(b_shape,uniform=True)
# =============================================================================
                    self[name_b_out] = tf.Variable(b_init, trainable=True, name=name_b_out)
                    
                
    def _w(self, n, suffix=""):
        return self[self._weights_str.format(n) + suffix]
    
    def _b(self, n, suffix=""):
        return self[self._biases_str.format(n) + suffix]
    
    def get_variables_to_init(self, n):
        """Return variables that need initialization

        This method aides in the initialization of variables
        before training begins at step n. The returned
        list should be than used as the input to
        tf.initialize_variables

        Args:
            n: int giving step of training
        """
        assert n > 0
        assert n <= self.__num_hidden_layers + 1

        vars_to_init = [self._w(n), self._b(n)]

        if n <= self.__num_hidden_layers:
            vars_to_init.append(self._b(n, "_out"))
            
        if 1 < n <= self.__num_hidden_layers:
            vars_to_init.append(self._w(n - 1, "_fixed"))
            vars_to_init.append(self._b(n - 1, "_fixed"))
            
        return vars_to_init
    
    @staticmethod
    def _activate(x, w, b, transpose_w=False, acfun=None, keep_prob=1):
        dropout_out = tf.nn.dropout(x,keep_prob)  
        if acfun is not None:
            y = acfun(tf.nn.bias_add(tf.matmul(dropout_out, w, transpose_b=transpose_w), b))
        else:
            y = tf.nn.bias_add(tf.matmul(dropout_out, w, transpose_b=transpose_w), b)
                         
        return y
    
    def pretrain_net(self, input_pl, n, is_target=False):
        """Return net for step n training or target net
        
        Args:
            input_pl:  tensorflow placeholder of AE inputs
            n:         int specifying pretrain step
            is_target: bool specifying if required tensor
                       should be the target tensor
        Returns:
            Tensor giving pretraining net or pretraining target
        """
        assert n > 0
        assert n <= self.__num_hidden_layers

        last_output = input_pl
        for i in range(n - 1):
            acfun = self.acfunList[i]
            w = self._w(i + 1, "_fixed")
            b = self._b(i + 1, "_fixed")
            
            last_output = self._activate(last_output, w, b, acfun=acfun)         
            
        if is_target:
            return last_output
        
        acfun = self.acfunList[n-1]     
        last_output = self._activate(last_output, self._w(n), self._b(n), acfun=acfun)
        
        out = self._activate(last_output, self._w(n), self._b(n, "_out"),
                         transpose_w=True, acfun=acfun)
#==============================================================================
#         #only for sigmoid??
#         out = tf.maximum(out, 1.e-9)
#         out = tf.minimum(out, 1 - 1.e-9)
#==============================================================================
        return out
    
    def single_instNet(self, input_pl, dropout, scope, batch_norm_reuse=False):
        """Get the supervised fine tuning net

        Args:
            input_pl: tf placeholder for ae input data
        Returns:
            Tensor giving full ae net
        """
        last_output = input_pl
        
        for i in range(self.__num_hidden_layers + 1): 
            # Fine tuning will be done on these variables
            acfun = self.acfunList[i]
            w = self._w(i + 1)
            b = self._b(i + 1)
            if self.batch_norm[i]:
                last_output = tf.contrib.layers.batch_norm(last_output,renorm=True,reuse=batch_norm_reuse,scope=scope)
            last_output = self._activate(last_output, w, b, acfun=acfun, keep_prob=dropout)
            
        return last_output
    
    def MIL(self,input_plist, dropout):
        tmpList = list()
        for i in range(int(input_plist.shape[0])):
            name_input = self._inputs_str.format(i + 1)
            #self[name_input] = tf.placeholder(tf.float32,[None, input_dim])
            self[name_input] = input_plist[i]
            
            name_instNet = self._instNets_str.format(i + 1)
            with tf.variable_scope("mil") as scope:
                if i == 0:
                    self[name_instNet] = self.single_instNet(self[name_input], dropout,scope)
                    scope.reuse_variables()
                else:    
                    self[name_instNet] = self.single_instNet(self[name_input], dropout,scope,batch_norm_reuse=True)
            
            tmpList.append(self[name_instNet])
        
            
        self["y"] = tf.stack(tmpList,axis=0) #axis = 0?1?
        self["Y"] =  tf.reduce_max(self["y"],axis=0,name="MILPool")#,keep_dims=True)
        self["maxInst"] = tf.argmax(self["y"],axis=0, name="maxInst")
        
        return self["Y"], self["y"], self["maxInst"]


loss_summaries = {}

def training(loss, learning_rate, loss_key=None, optimMethod=tf.train.AdamOptimizer, var_in_training=None):
  """Sets up the training Ops.

  Creates a summarizer to track the loss over time in TensorBoard.

  Creates an optimizer and applies the gradients to all trainable variables.

  The Op returned by this function is what must be passed to the
  `sess.run()` call to cause the model to train.

  Args:
    loss: Loss tensor, from loss().
    learning_rate: The learning rate to use for gradient descent.
    loss_key: int giving stage of pretraining so we can store
                loss summaries for each pretraining stage

  Returns:
    train_op: The Op for training.
  """
  if loss_key is not None:
    # Add a scalar summary for the snapshot loss.
    loss_summaries[loss_key] = tf.summary.scalar(loss.op.name, loss)
  else:
    tf.summary.scalar(loss.op.name, loss)
    for var in tf.trainable_variables():
      tf.summary.histogram(var.op.name, var)
  # Create the gradient descent optimizer with the given learning rate.
  #optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  #optimizer = tf.train.AdamOptimizer(learning_rate)
  optimizer = optimMethod(learning_rate)
  # Create a variable to track the global step.
  global_step = tf.Variable(0, name='global_step', trainable=False)
  # Use the optimizer to apply the gradients that minimize the loss
  # (and also increment the global step counter) as a single training step.
  if var_in_training is not None:
      train_op = optimizer.minimize(loss, global_step=global_step, var_list=var_in_training)
  else:
      train_op = optimizer.minimize(loss, global_step=global_step)
  return train_op, global_step



def main_unsupervised(ae_shape,acfunList,batch_norm,dataset,FLAGS,sess=None):
    
    if sess is None:
        sess = tf.Session()
   
    aeList = list()
    for a in range(len(ae_shape)):
        aeList.append(miNet(ae_shape[a],acfunList, batch_norm, sess))
    
                                      
    return aeList


def main_supervised(instNetList,num_inst,inputs,dataset,FLAGS):
    with instNetList[0].session.graph.as_default():
        sess = instNetList[0].session
        
        bagOuts = []
        instOuts = []
        maxInstOuts = []
        playerOuts = []
        instIdx = np.insert(np.cumsum(num_inst),0,0)
        keep_prob_ = tf.placeholder(dtype=tf.float32,
                                           name='dropout_ratio')

        def TF_InstToPlayerIndex(out_maxInst,num_inst,playerMap):
            bool_instMask = tf.one_hot(out_maxInst,num_inst,axis=-1,dtype=tf.int32)
            num_class = int(bool_instMask.shape[1])
            bool_instMask = tf.reshape(bool_instMask,[-1, num_inst])
            tf_playerMap = tf.constant(playerMap,dtype=tf.int32)
            num_player = int(tf_playerMap.shape[1])
            playerIdx = tf.matmul(bool_instMask,tf_playerMap)
            playerIdx = tf.reshape(playerIdx,[-1,num_class,num_player])
            return playerIdx
            
        offset = tf.constant(instIdx)
        for k in range(len(instNetList)):
            #with tf.name_scope('C5{0}'.format(dataset.k[k])):
            with tf.variable_scope('C5{0}'.format(dataset.k[k])):
                out_Y, out_y, out_maxInst = instNetList[k].MIL(tf.transpose(inputs[k],perm=(1,0,2)),k,keep_prob_)
                
            bagOuts.append(out_Y)
            instOuts.append(out_y)
            maxInstOuts.append(out_maxInst+offset[k])
            
            #convert instance index to player index
            playerIdx = TF_InstToPlayerIndex(out_maxInst,num_inst[k],dataset.playerMap[k])
# =============================================================================
#             bool_instMask = tf.one_hot(out_maxInst,num_inst[k],axis=-1,dtype=tf.int32)
#             num_class = int(bool_instMask.shape[1])
#             bool_instMask = tf.reshape(bool_instMask,[-1, num_inst[k]])
#             tf_playerMap = tf.constant(dataset.playerMap[k],dtype=tf.int32)
#             num_player = int(tf_playerMap.shape[1])
#             playerIdx = tf.matmul(bool_instMask,tf_playerMap)
#             playerIdx = tf.reshape(playerIdx,[-1,num_class,num_player])
# =============================================================================
            playerOuts.append(playerIdx)
            

        hist_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        hist_summaries = [tf.summary.histogram(v.op.name + "_fine_tuning", v) for v in hist_variables]
        trainable_vars_summary_op = tf.summary.merge(hist_summaries)            
        


                    
        NUM_CLASS = len(dataset.tacticName)
        NUM_PLAYER= dataset.numPlayer
        Y_placeholder = tf.placeholder(tf.float32,
                                        shape=(None,NUM_CLASS),
                                        name='targetB_pl')

        y_placeholder = tf.placeholder(tf.float32,
                                        shape=(None,NUM_PLAYER),
                                        name='targetP_pl')
        
        Y_val_pred = tf.concat(bagOuts,1,name='output')
        y_maxInsts = tf.concat(maxInstOuts,1, name='maxInsts')
        common_loss_sw = tf.placeholder(dtype=tf.float32,shape=(),name='common_loss')
        inst_loss_sw = tf.placeholder(dtype=tf.float32,shape=(),name='inst_loss')
        
        def indexPlayerMap(t):
            ''' t [num_player] key player configuartion'''
            playerMap = tf.cast(tf.concat(dataset.playerMap,0),tf.bool) #[totalNumInst x num_player]
            correct_map = tf.reduce_prod(tf.cast(tf.logical_not(tf.logical_xor(playerMap,tf.cast(t,tf.bool))),tf.uint8),axis=1)
            correct_idx = tf.argmax(correct_map)
            return correct_idx
        
        y_idxGT = tf.map_fn(indexPlayerMap,y_placeholder,dtype=tf.int64)
        y_onehotGT = tf.one_hot(y_idxGT,instIdx[-1])
        y_maxInstsGT = tf.multiply(tf.tile(tf.expand_dims(tf.cast(y_idxGT,tf.float32),axis=1),[1,NUM_CLASS]) + 1,Y_placeholder) - 1
        
        bb = [len(dataset.C5k_CLASS[i]) for i in range(len(dataset.C5k_CLASS))]
        ###bagIdx = np.insert(np.cumsum(bb),0,0)
        ###def retrieveY_from_y_gt(m):
        ###    b = m[0]
        ###    y_idx = m[1]
        ###    Y_idx = tf.argmax(m[2])
        ###    #i = tf.where(tf.greater(y_idx,instIdx))
        ###    #subnet_idx = i[-1]
        ###    gi = tf.cast(tf.greater_equal(y_idx,instIdx),tf.float32)
        ###    li = tf.cast(tf.less(y_idx,np.roll(instIdx,-1)),tf.float32)
        ###    subnet_idx = tf.argmax(gi*li)
        ###    
        ###    subnet_y_idx = y_idx - tf.gather(instIdx,subnet_idx) #instIdx[subnet_idx]
        ###    subnet_Y_idx = Y_idx - tf.gather(bagIdx,subnet_idx) #bagIdx[subnet_idx]
        ###    
        ###    Y_retrieved = instOuts[subnet_idx][subnet_y_idx,b,subnet_Y_idx]
        ###    local_vars = {'b':b,'y_idx':y_idx,'Y_idx':Y_idx,
        ###                  'subnet_idx':subnet_idx,
        ###                  'subnet_y_idx':subnet_y_idx,'subnet_Y_idx':subnet_Y_idx}
        ###    
        ###    return Y_retrieved,local_vars 
        ###mm = (tf.range(tf.shape(Y_placeholder)[0]),y_idxGT,Y_placeholder)
        ###Y_predGT = tf.map_fn(retrieveY_from_y_gt,mm,dtype=tf.float32)        
        ###Y_predGTmap = Y_placeholder * Y_predGT
        def retrieveY_from_y_gt(m):
            instOut = m[0]
            Y_idx = tf.argmax(m[1])
            y_idx = tf.argmax(m[2])
            #y_max = tf.max(m[1])
            #Y_max = tf.max(m[2])
            
            y_val_fromYgt = tf.gather(instOut,Y_idx,axis=1)
            top_y_val = tf.gather(y_val_fromYgt,y_idx,axis=0)
            Y_pred_gt = m[1] * top_y_val
            '''Y_pred_gt [0 0 0 X ..0] or [0 ... 0] [num_tactics]'''
            return Y_pred_gt
            
        tactic_sep = tf.split(Y_placeholder,bb,axis=1)
        inst_sep = tf.split(y_onehotGT,num_inst,axis=1)
        def retrieveValue(instOut,tactic_sep,inst_sep):
            mm = (tf.transpose(instOut,perm=(1,0,2)),tactic_sep,inst_sep)
            rr = tf.map_fn(retrieveY_from_y_gt,mm,dtype=tf.float32)
            return rr
        subs_Y_val = []    
        for k in range(len(instNetList)):
            subnet_Y_val = retrieveValue(instOuts[k],tactic_sep[k],inst_sep[k])
            subs_Y_val.append(subnet_Y_val)
        Y_predGTmap = tf.concat(subs_Y_val,1)  #[batch x NUM_CLASS]
        
        if FLAGS.similarity_y_source == 'gt':
            Y = tf.where(tf.cast(Y_placeholder,tf.bool),Y_predGTmap,Y_val_pred)
        else:
            Y = Y_val_pred

        Y_predmap = tf.one_hot(tf.argmax(tf.nn.softmax(Y),axis=1),NUM_CLASS)
        y_playerPool = tf.concat(playerOuts,1,name='key_player')
        y_accu, y = metric.calulcutePAccuTF(Y,y_playerPool,Y_placeholder,y_placeholder)


# =============================================================================
#         bool_YMask = tf.one_hot(tf.argmax(Y,axis=-1),int(Y.shape[1]),axis=-1)
#         bool_YMask = tf.cast(bool_YMask,tf.bool)
#         y = tf.boolean_mask(y_playerPool,bool_YMask)        
#         correct_prediction = tf.equal(tf.argmax(Y,axis=1), tf.argmax(Y_placeholder,axis=1))
#         y_correctY = tf.boolean_mask(y,correct_prediction)
#         y_placeholder_correctY = tf.cast(tf.boolean_mask(y_placeholder,correct_prediction),tf.int32)
#         y_correct = tf.equal(y_correctY,y_placeholder_correctY)
#         y_accu = tf.reduce_sum(tf.cast(y_correct,tf.float32))/(tf.reduce_sum(tf.cast(correct_prediction,tf.float32))*NUM_PLAYER)
# =============================================================================
        with tf.name_scope('softmax_cross_entory_with_logit'):
            x_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Y,
                                labels=Y_placeholder,name='softmax_cross_entropy'))            
            train_tactic_prediction_op = tf.summary.histogram('train/tactic_prediction',tf.nn.softmax(Y))
            train_tactic_score_op = tf.summary.histogram('train/tactic_prediction',Y)
            
            test_tactic_prediction_op = tf.summary.histogram('test/tactic_prediction',tf.nn.softmax(Y))
            test_tactic_score_op = tf.summary.histogram('test/tactic_prediction',Y)            

        lstm_last_output = tf.get_collection('ae_lstm/lastoutput') #'aelstm_lastoutput'
        train_lstm_last_output_op = tf.summary.histogram('ae_lstm/train/lastoutput',lstm_last_output)
        test_lstm_last_output_op = tf.summary.histogram('ae_lstm/test/lastoutput',lstm_last_output)
        train_lstm_last_output_sparity = tf.summary.scalar('ae_lstm/train/last_output_sparsity', tf.nn.zero_fraction(lstm_last_output))
        test_lstm_last_output_sparity = tf.summary.scalar('ae_lstm/test/last_output_sparsity', tf.nn.zero_fraction(lstm_last_output))
        
        if FLAGS.similarity_y_source == 'pred':
            print('load y from Y prediction!')
            mean_sim_variance, sim_local_vars = metric.inst_similarity(Y_predmap,Y_placeholder,y_maxInsts,inputs)
            bag_sim, bag_local_vars = metric.withBag_similarity(Y_predmap,Y_placeholder,y_maxInsts,inputs,FLAGS.bag_similarity_type)
        elif FLAGS.similarity_y_source == 'gt':
            print('load y from y ground truth!')
            mean_sim_variance, sim_local_vars = metric.inst_similarity(Y_predmap,Y_placeholder,y_maxInstsGT,inputs)
            bag_sim, bag_local_vars = metric.withBag_similarity(Y_predmap,Y_placeholder,y_maxInstsGT,inputs,FLAGS.bag_similarity_type)
            
        ##norm_sim_variance = sim_variance/tf.cast(tf.count_nonzero(sim_variance),tf.float32)
        ##norm_sim_variance = tf.where(tf.is_nan(norm_sim_variance),tf.zeros_like(norm_sim_variance),norm_sim_variance)
        ##mean_sim_variance = tf.reduce_sum(norm_sim_variance)
        #mean_sim_variance = tf.reduce_sum(sim_variance)/tf.cast(tf.count_nonzero(sim_variance),tf.float32)
        #mean_sim_variance = tf.where(tf.is_nan(mean_sim_variance),tf.zeros_like(mean_sim_variance),mean_sim_variance)
        sim_beta = FLAGS.sim_beta
        bag_sim_beta = FLAGS.bag_sim_beta
        
        num_train = len(dataset.trainIdx)
        regularization = tf.get_collection('decode_loss')
        decode_beta = FLAGS.decode_beta#tf.cast(FLAGS.decode_beta,tf.float32)
        normalized_regularization = regularization
# =============================================================================
#         normalized_regularization = tf.sqrt(regularization)*tf.constant(dataset.MAX_Y,dtype=tf.float32) 
# =============================================================================
        loss = common_loss_sw * (x_entropy + decode_beta * tf.squeeze(normalized_regularization,[0])) + inst_loss_sw * (sim_beta * tf.reduce_sum(mean_sim_variance) - bag_sim_beta * tf.reduce_sum(bag_sim))
   
        train_x_entropy_op = tf.summary.scalar('train/x_entropy_loss',x_entropy)
        train_aelstm_op = tf.summary.scalar('train/aelstm_loss', tf.squeeze(normalized_regularization,[0]))
        train_sim_op    = tf.summary.scalar('train/top_inst_sim', tf.reduce_sum(mean_sim_variance))
        train_bag_sim_op= tf.summary.scalar('train/bag_sim', tf.reduce_sum(bag_sim))
        train_sim_epoch_op    = tf.summary.scalar('train/top_inst_sim_epoch', tf.reduce_sum(mean_sim_variance))
        train_bag_sim_epoch_op= tf.summary.scalar('train/bag_sim_epoch', tf.reduce_sum(bag_sim))        
        
# =============================================================================
#         train_aelstm_op = tf.summary.scalar('train/aelstm_loss=x{}'.format(decode_beta),decode_beta * tf.squeeze(normalized_regularization,[0]))
#         train_sim_op    = tf.summary.scalar('train/top_inst_sim=x{}'.format(sim_beta), sim_beta * tf.reduce_sum(mean_sim_variance))
#         train_bag_sim_op= tf.summary.scalar('train/bag_sim=x{}'.format(bag_sim_beta), bag_sim_beta * tf.reduce_sum(bag_sim))
# =============================================================================
        train_loss_op = tf.summary.scalar('train/total_loss',loss)        
        
        test_x_entropy_op = tf.summary.scalar('test/x_entropy_loss',x_entropy)
        test_aelstm_op = tf.summary.scalar('test/aelstm_loss', tf.squeeze(normalized_regularization,[0]))
        test_sim_op    = tf.summary.scalar('test/top_inst_sim', tf.reduce_sum(mean_sim_variance))
        test_bag_sim_op= tf.summary.scalar('test/bag_sim', tf.reduce_sum(bag_sim))
# =============================================================================
#         test_aelstm_op = tf.summary.scalar('test/aelstm_lossx{}'.format(decode_beta),decode_beta * tf.squeeze(normalized_regularization,[0]))
#         test_sim_op    = tf.summary.scalar('test/top_inst_simx{}'.format(sim_beta), sim_beta * tf.reduce_sum(mean_sim_variance))
#         test_bag_sim_op= tf.summary.scalar('test/bag_simx{}'.format(bag_sim_beta), bag_sim_beta * tf.reduce_sum(bag_sim))
# =============================================================================
        test_loss_op = tf.summary.scalar('test/total_loss',loss)
        
        with tf.name_scope('MultiClassEvaluation'):
            accu, _ = metric.multiClassEvaluation(Y, Y_placeholder)

        train_accu_op = tf.summary.scalar('train/accuracy',accu)        
        test_accu_op = tf.summary.scalar('test/accuracy',accu)
        
        train_pAccu_op = tf.summary.scalar('train/p_accu',y_accu)
        test_pAccu_op = tf.summary.scalar('test/p_accu', y_accu)
# =============================================================================
#         Y_labels_image = label2image(Y_placeholder)
#         #label_op = tf.summary.image('tactic_labels',Y_label_image)
#         Y_logits_image = label2image(tf.nn.softmax(Y))
#         #label_op = tf.summary.histogram('tactic_labels',Y_placeholder)
# =============================================================================
        train_epochs_merged = tf.summary.merge([train_accu_op,train_pAccu_op,train_sim_epoch_op,train_bag_sim_epoch_op])
        train_steps_merged = tf.summary.merge([train_lstm_last_output_op,train_lstm_last_output_sparity,
                                   train_tactic_prediction_op,
                                   train_tactic_score_op,
                                   train_x_entropy_op,train_aelstm_op,#train_sim_op,train_bag_sim_op,
                                   train_loss_op,#train_accu_op,
                                   trainable_vars_summary_op])#,output_op,label_op])
        test_epochs_merged = tf.summary.merge([test_lstm_last_output_op,test_lstm_last_output_sparity,
                                   test_tactic_prediction_op,
                                   test_tactic_score_op,
                                   test_x_entropy_op,test_aelstm_op,test_sim_op,test_bag_sim_op,
                                   test_loss_op,test_accu_op,test_pAccu_op])#    
        
        summary_writer = tf.summary.FileWriter(FLAGS.miNet_train_summary_dir,
                                           tf.get_default_graph(),flush_secs=FLAGS.flush_secs)
        vars_to_init = []
        # initialize lstm variables         
        vars_to_init.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='encoder'))
                                
        for k in range(len(instNetList)):
            instNet = instNetList[k]
            vars_to_init.extend(instNet.get_variables_to_init(instNet.num_hidden_layers + 1))
        
        if not FLAGS.save_gradints:
            train_op, global_step = training(loss, FLAGS.supervised_learning_rate, None, 
                                             optimMethod=utils.get_optimizer(FLAGS.optimizer),
                                             var_in_training=vars_to_init)
        else:
            global_step = tf.Variable(tf.zeros([],dtype=tf.int32), name='global_step', trainable=False)
            train_op = tf.contrib.layers.optimize_loss(loss, global_step, 
                                                       learning_rate=FLAGS.supervised_learning_rate, 
                                                       optimizer=utils.get_optimizer(FLAGS.optimizer),
                                                       summaries=["loss","gradients","global_gradient_norm","gradient_norm"])
            
            summaries_vars_optimize_loss = [var for var in tf.get_collection(tf.GraphKeys.SUMMARIES) if ('OptimizeLoss' in var.name)]
            optimize_loss_op = tf.summary.merge(summaries_vars_optimize_loss)
        
# =============================================================================
#         vars_to_init.append(global_step)
# =============================================================================
        #optim_vars = [var for var in tf.global_variables() if (FLAGS.optimizer in var.name)]
        #learning_rate_var = [var for var in tf.global_variables() if ('learning_rate' in var.name)]
        
        log_path = FLAGS.miNet_train_model_dir #+ '/fine_tune/'
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        model_ckpt= tf.train.latest_checkpoint(log_path)
        saver = tf.train.Saver(tf.trainable_variables())#vars_to_init)
        
        sess.run(tf.global_variables_initializer()) 
        #for v in tf.trainable_variables():
        #            print("%s with value %s" % (v.name, sess.run(tf.is_variable_initialized(v))))        
                    
        if not model_ckpt : #brand new
            max_epochs = FLAGS.finetuning_epochs
            resume_step = 0
            model_ckpt = log_path + '/model.ckpt'
            print('New training from epochs {} to {}'.format(resume_step,max_epochs))
            saver._max_to_keep = FLAGS.finetuning_epochs / FLAGS.finetuning_saving_epochs
            saver.saver_def.max_to_keep = int(saver._max_to_keep)
        else :
            split = model_ckpt.split("-")
            resume_step = int(split[-1])
            saver.restore(sess, model_ckpt)
            model_ckpt = split[0]
            checkpoint_paths = [(log_path + x[:-6]) for x in os.listdir(log_path) if x.endswith(".index")]
            saver.recover_last_checkpoints(checkpoint_paths)
            if FLAGS.fine_tune_resume : #Resume to designed epochs (Designed - last checkpoint)
                #for v in tf.trainable_variables():
                #    print("%s with value %s" % (v.name, sess.run(tf.is_variable_initialized(v))))
                 
                #need error detection (now resume must be multiple of finetuning_epochs %we )
                max_epochs = (FLAGS.finetuning_epochs - resume_step%FLAGS.finetuning_epochs) %FLAGS.finetuning_epochs
                print('Resume training from epochs {} to {}'.format(resume_step,resume_step+max_epochs))
                saver._max_to_keep = FLAGS.finetuning_epochs / FLAGS.finetuning_saving_epochs
                saver.saver_def.max_to_keep = int(saver._max_to_keep)
            else : # Continue Train
                max_epochs = FLAGS.finetuning_epochs
                print('Continue training from epochs {} to {}'.format(resume_step,max_epochs+resume_step))
                saver._max_to_keep = (resume_step + FLAGS.finetuning_epochs) / FLAGS.finetuning_saving_epochs
                saver.saver_def.max_to_keep = int(saver._max_to_keep)

        print('max_to_keep:',saver._max_to_keep)
        print('max_to_keep in saver_def:',saver.saver_def.max_to_keep)

        ''' feed data '''
        if not os.path.exists(FLAGS._confusion_dir):
            os.makedirs(FLAGS._confusion_dir)
        
        if not os.path.exists(FLAGS._logit_txt):
            os.makedirs(FLAGS._train_logit_txt)
            os.makedirs(FLAGS._test_logit_txt)
        
        if not os.path.exists(FLAGS._dec_output_dir):
            os.makedirs(FLAGS._dec_output_train_dir)
            os.makedirs(FLAGS._dec_output_test_dir)     

        if not os.path.exists(FLAGS._key_player_dir):
            os.makedirs(FLAGS._key_player_dir)
              
            
        #text_file = open(FLAGS._ipython_console_txt,"w")
        #os.remove(FLAGS._ipython_console_txt)
        timeStamp = str(int(time.time()))
        FLAGS._ipython_console_txt = '{0}-{1}.log'.format(FLAGS._ipython_console_txt,timeStamp)
        open(FLAGS._ipython_console_txt,"w")
        fold = FLAGS.fold 
        trainIdx = dataset.trainIdx
        testIdx = dataset.testIdx
        seqLenMatrix = dataset.seqLenMatrix

        ''' generate batch data for training '''        
        print("")
        print('fold %d' %(fold+1))
        datadir = 'dataGood/multiPlayers/syncLargeZoneVelocitySoftAssign(R=16,s=10)/train/fold%d' %(fold+1)
        file_str= '{0}ZoneVelocitySoftAssign(R=16,s=10){1}_training%d.mat' %(fold+1)

        _, batch_multi_Y, batch_multi_KPlabel = readmat.multi_class_read(datadir,file_str,num_inst,dataset)
        num_train = len(batch_multi_Y)
        if FLAGS.shrink_label:
            batch_multi_Y = 0.3 * batch_multi_Y + 0.2
            print('positive label: %f, negative label: %f' %(batch_multi_Y.max(),batch_multi_Y.min()))
        strBagShape = "the shape of bags is ({0},{1})".format(batch_multi_Y.shape[0],batch_multi_Y.shape[1])
        print(strBagShape)
        batch_multi_X = dataset.dataTraj[trainIdx,:]
        train_data = {'sequences':batch_multi_X,'seqlen':seqLenMatrix[trainIdx,:],'label':batch_multi_Y,'KPLabel':batch_multi_KPlabel}

        testdir = 'dataGood/multiPlayers/syncLargeZoneVelocitySoftAssign(R=16,s=10)/test/fold%d' %(fold+1)
        test_file_str= '{0}ZoneVelocitySoftAssign(R=16,s=10){1}_test%d.mat' %(fold+1) 
        _, test_multi_Y, test_multi_label = readmat.multi_class_read(testdir,test_file_str,num_inst,dataset)       
        num_test = len(test_multi_Y)
        if FLAGS.shrink_label:
            test_multi_Y = 0.3 * test_multi_Y + 0.2
            print('positive label: %f, negative label: %f' %(test_multi_Y.max(),test_multi_Y.min()))

        strBagShape = "the shape of bags is ({0},{1})".format(test_multi_Y.shape[0],test_multi_Y.shape[1])
        print(strBagShape)
        test_multi_X = dataset.dataTraj[testIdx,:]
        test_data = {'sequences':test_multi_X,'seqlen':seqLenMatrix[testIdx,:],'label':test_multi_Y,'KPLabel':test_multi_label}
      
        if FLAGS.finetune_batch_size == 0:
            FLAGS.finetune_batch_size = len(batch_multi_Y)#len(test_multi_Y)
        

        def fetch_data(data,selectIndex,dropout,loss_mode='mixed'):
            random_sequences = np.reshape(data['sequences'][selectIndex,:],(-1,data['sequences'].shape[2],data['sequences'].shape[3]))
            batch_seqlen = np.reshape(data['seqlen'][selectIndex,:],(-1))
            target_feed = data['label'][selectIndex].astype(np.int32)            
            targetP_feed= data['KPLabel'][selectIndex].astype(np.int32)
            
            if dropout:
                keep_prob = FLAGS.keep_prob
            else:
                keep_prob = 1.0
                
            if loss_mode == 'mixed':
                common_loss = 1.0
                inst_loss   = 1.0
            elif loss_mode == 'common':
                common_loss = 1.0
                inst_loss   = 0.0
            elif loss_mode == 'inst':
                common_loss = 0.0
                inst_loss   = 1.0                
            
                
            feed_dict={FLAGS.p_input:random_sequences,
                       FLAGS.seqlen: batch_seqlen,
                       Y_placeholder: target_feed,
                       y_placeholder: targetP_feed,
                       keep_prob_: keep_prob,
                       common_loss_sw: common_loss,
                       inst_loss_sw: inst_loss}
            return feed_dict
        def generate_result(data,num_data,accus_merged,phase,count):
            # similarity debugging
            selectIndex = np.arange(num_data) 
            feed_dict = fetch_data(data,selectIndex,False)
            #maxInsts_val = run_step(sess,y_maxInsts,feed_dict)
            #sim_vars = run_step(sess,sim_variance,feed_dict)
            #mean_sim_var = run_step(sess,sim_beta * mean_sim_variance,feed_dict)
            #num_tactic_inovolved = run_step(sess,tf.count_nonzero(sim_variance),feed_dict)
            sim_vals,bag_sim_vals = run_step(sess,[sim_local_vars,bag_local_vars],feed_dict)
            ##e, m, t, t1 = run_step(sess,[idx,y_maxInsts,Y_placeholder,Y_predmap],feed_dict)
            ##gt = run_step(sess,y_maxInstsGT,feed_dict)
            y_gt,y_max,Y11,Y12 = run_step(sess,[Y_predGTmap,Y_val_pred,Y,Y_placeholder,],feed_dict)
            
            selectIndex = np.arange(num_data) 
            feed_dict = fetch_data(data,selectIndex,False)
            inst_vals = run_step(sess,instOuts,feed_dict)
            instProbDir = '{}/inst_prob{}-{}'.format(FLAGS._test_logit_txt,actual_epochs,phase)
            if not os.path.exists(instProbDir):
                os.makedirs(instProbDir)            
            for k in range(len(inst_vals)):
                for t in range(inst_vals[k].shape[2]):
                    np.savetxt('{}/inst_prob{}-{}/C5{}-t{}.txt'.format(
                            FLAGS._test_logit_txt,actual_epochs,phase,dataset.k[k],dataset.C5k_CLASS[k][t]),
                            np.transpose(inst_vals[k][:,:,t]),fmt='%f', delimiter=' ')
                        
            loss_value, bagAccu, pAccu, Y_pred, player_pred = run_step(sess,[loss, accu, y_accu, Y_predmap, y],feed_dict)
            utils.printLog(FLAGS._ipython_console_txt,'')
            utils.printLog(FLAGS._ipython_console_txt,'Epochs %d: %s loss = %.5f ' % (actual_epochs, phase, loss_value))
            utils.printLog(FLAGS._ipython_console_txt,'Epochs %d: %s accuracy = %.5f '  % (actual_epochs, phase, bagAccu))
            #bagAccu,pAccu = metric.calculateAccu(Y_pred,inst_pred,batch_multi_Y,batch_multi_KPlabel,dataset)
            utils.printLog(FLAGS._ipython_console_txt,'bag accuracy %.5f, inst accuracy %.5f' %(bagAccu, pAccu))

            Y_scaled, Y_unscaled = run_step(sess,[tf.nn.softmax(Y),Y],feed_dict) 
            np.savetxt('{}/logit{}-{}-scaled.txt'.format(FLAGS._test_logit_txt,actual_epochs,phase),Y_scaled,fmt='%.4f', delimiter=' ')
            np.savetxt('{}/logit{}-{}-unscaled.txt'.format(FLAGS._test_logit_txt,actual_epochs,phase),Y_unscaled,fmt='%.4f', delimiter=' ')            
 
            summary_str = run_step(sess,accus_merged,feed_dict)
            summary_writer.add_summary(summary_str, count)                 
            
            filename = FLAGS._confusion_dir + '/Fold{0}_Epoch{1}_{2}.csv'.format(fold,actual_epochs,phase)
            Y_label = data['label']
            if FLAGS.shrink_label:
                Y_label = (Y_label - 0.2 ) / 0.3
            metric.ConfusionMatrix(Y_pred,Y_label,dataset,filename,FLAGS._ipython_console_txt)
            
            
            filename = FLAGS._key_player_dir + '/Fold{0}_Epoch{1}_{2}.csv'.format(fold,actual_epochs,phase)
            info_filename = FLAGS._key_player_dir + '/Fold{0}_Epoch{1}_{2}_stat.csv'.format(fold,actual_epochs,phase)
            metric.keyPlayerResult(Y_pred,Y_label,player_pred,dataset,phase,filename,FLAGS._ipython_console_txt,info_filename)
            
            y_label = data['KPLabel']
            y_accu_Y_correct_per_tactic, _ = metric.calculatePAccu(player_pred,y_label,Y_pred,Y_label)
            reorder = np.concatenate(dataset.C5k_CLASS).ravel().tolist()
            orderTactic = [dataset.tacticName[i] for i in reorder]
            df = pd.DataFrame(y_accu_Y_correct_per_tactic,index=orderTactic,columns=['player accuracy'])
            yaccu_filename = FLAGS._key_player_dir + '/Fold{0}_Epoch{1}_{2}_yaccu.csv'.format(fold,actual_epochs,phase)
            df.transpose().to_csv(yaccu_filename,na_rep='NaN')            
            utils.printLog(FLAGS._ipython_console_txt," ")
            utils.printLog(FLAGS._ipython_console_txt,df.transpose())
            utils.printLog(FLAGS._ipython_console_txt," ")

        #count = 0
        for epochs in range(max_epochs):
            actual_epochs = epochs+resume_step+1
            perm = np.arange(num_train)
            np.random.shuffle(perm)
            utils.printLog(FLAGS._ipython_console_txt,"|-------------|-----------|-------------|----------|")
            
            start_time = time.time()
            ''' training of one epoch '''
            for step in range(int(num_train/FLAGS.finetune_batch_size)):                
                    
                selectIndex = perm[FLAGS.finetune_batch_size*step:FLAGS.finetune_batch_size*step+FLAGS.finetune_batch_size]
                
                ''' get train data and feed into graph in training stage '''
                feed_dict = fetch_data(train_data,selectIndex,True,loss_mode = 'common')
                #sim_vals_b = run_step(sess,sim_local_vars,feed_dict)
                #bag_vals_b = run_step(sess,bag_local_vars,feed_dict)
                #print(sim_vals_b['sim_vals'])
                #print('tactic correct:\n', sim_vals_b['tactic_correct'])
                _, loss_value = run_step(sess,[train_op, loss],feed_dict)
                
                #nchoosek = run_step(sess,tf.concat(inputs,1),feed_dict)
                #decloss,last_output = run_step(sess,[regularization,lstm_last_output],feed_dict)
                #sim_vals_a = run_step(sess,sim_local_vars,feed_dict)
                #print(sim_vals_a['sim_vals'])
                #print('tactic correct:\n', sim_vals_a['tactic_correct'])
                # Write the summaries and print an overview fairly often.
                #count = count + 1
                count = actual_epochs * int(num_train/FLAGS.finetune_batch_size) + step
                #print('count:', count)
                ''' save training data result after n training steps '''
                if step % FLAGS.finetuning_summary_step == 0:
                    duration = time.time() - start_time
                    
                    utils.printLog(FLAGS._ipython_console_txt,'|   Epoch %d  |  Step %d  | train loss = %.3f | (%.3f sec)' % (actual_epochs, step, loss_value, duration))

                    feed_dict = fetch_data(train_data,selectIndex,True)
                    summary_str = run_step(sess,train_steps_merged,feed_dict)
                    summary_writer.add_summary(summary_str, count)

                    if FLAGS.save_gradints:
                        summary_str = run_step(sess,optimize_loss_op,feed_dict)               
                        summary_writer.add_summary(summary_str, count)
                
                    feed_dict = fetch_data(train_data,selectIndex,False)
                    Y_scaled, Y_unscaled = run_step(sess,[tf.nn.softmax(Y),Y],feed_dict)                         
                    np.savetxt('{}/logit{}_{}.txt'.format(FLAGS._train_logit_txt,actual_epochs,step),Y_scaled,fmt='%.4f', delimiter=' ')
                    
                    start_time = time.time()
                
            selectIndex = np.arange(num_train) 
            feed_dict = fetch_data(train_data,selectIndex,True,loss_mode='inst')
            sim_vars,sim_vals = run_step(sess,[mean_sim_variance,sim_local_vars],feed_dict)
            _, loss_value = run_step(sess,[train_op, loss],feed_dict)
            utils.printLog(FLAGS._ipython_console_txt,'|   Epoch %d  |  inst. involved  | train loss = %.3f |' % (actual_epochs, loss_value))

            if actual_epochs % FLAGS.finetuning_saving_epochs == 0:
                save_path = saver.save(sess, model_ckpt, global_step=actual_epochs)
                utils.printLog(FLAGS._ipython_console_txt,"Model saved in file: %s" % save_path)

            #count = actual_epochs * int(num_train/FLAGS.finetune_batch_size) + step
            ''' evaluate test performance after one epoch (need add training performance)'''
            generate_result(train_data,num_train,train_epochs_merged,'train',actual_epochs)#count)
            generate_result(test_data,num_test,test_epochs_merged,'test',actual_epochs)#count)
            # train data result 
# =============================================================================
#             selectIndex = np.arange(num_train) 
#             feed_dict = fetch_data(train_data,selectIndex,False)            
#             loss_value,bagAccu, pAccu = run_step(sess,[loss, accu, y_accu],feed_dict)
#             utils.printLog(FLAGS._ipython_console_txt,'Epochs %d: train loss = %.5f ' % (actual_epochs, loss_value))
#             utils.printLog(FLAGS._ipython_console_txt,'Epochs %d: train accuracy = %.5f '  % (actual_epochs, bagAccu))
#             #bagAccu,pAccu = metric.calculateAccu(Y_pred,inst_pred,batch_multi_Y,batch_multi_KPlabel,dataset)
#             utils.printLog(FLAGS._ipython_console_txt,'bag accuracy %.5f, inst accuracy %.5f' %(bagAccu, pAccu))
# 
#             Y_scaled, Y_unscaled = run_step(sess,[tf.nn.softmax(Y),Y],feed_dict) 
#             np.savetxt('{}/logit{}-train-scaled.txt'.format(FLAGS._test_logit_txt,actual_epochs),Y_scaled,fmt='%.4f', delimiter=' ')
#             np.savetxt('{}/logit{}-train-unscaled.txt'.format(FLAGS._test_logit_txt,actual_epochs),Y_unscaled,fmt='%.4f', delimiter=' ')            
#             
#             summary_str = run_step(sess,train_epochs_merged,feed_dict)
#             summary_writer.add_summary(summary_str, count)
#         
# 
#             # test data result
#             selectIndex = np.arange(num_test)
#             feed_dict = fetch_data(test_data,selectIndex,False)
#             loss_value,bagAccu, Y_pred, inst_pred = run_step(sess,[loss, accu, Y_predmap, instOuts],feed_dict)
#             utils.printLog(FLAGS._ipython_console_txt,'')
#             utils.printLog(FLAGS._ipython_console_txt,'Epochs %d: test loss = %.5f '  % (actual_epochs, loss_value))
#             utils.printLog(FLAGS._ipython_console_txt,'Epochs %d: test accuracy = %.5f '  % (actual_epochs, bagAccu))
#         
#             ''' save summaries of test data '''
#             summary_str = run_step(sess,test_epochs_merged,feed_dict)
#             summary_writer.add_summary(summary_str, count)     
#         
#             Y_scaled, Y_unscaled= run_step(sess,[tf.nn.softmax(Y),Y],feed_dict)
#             np.savetxt('{}/logit{}-test-scaled.txt'.format(FLAGS._test_logit_txt,actual_epochs),Y_scaled,fmt='%.4f', delimiter=' ')
#             np.savetxt('{}/logit{}-test-unscaled.txt'.format(FLAGS._test_logit_txt,actual_epochs),Y_unscaled,fmt='%.4f', delimiter=' ')
#                                 
#             bagAccu,pAccu = metric.calculateAccu(Y_pred,inst_pred,test_multi_Y,test_multi_label,dataset)
#             utils.printLog(FLAGS._ipython_console_txt,'bag accuracy %.5f, inst accuracy %.5f' %(bagAccu, pAccu))
# 
#             tf_pAccu = run_step(sess,y_accu,feed_dict)
#             if np.abs(pAccu - tf_pAccu) > 1e-6: #or tf_pAccu == 1.0:
#                 print('tf test inst accuracy %.5f' %(tf_pAccu))
#                 print('tf and np version of pAccu is conflicted!')
#                 return
# =============================================================================
        

            
            
            ''' save decode result '''
            dec_output = tf.get_collection('ae_lstm/dec_output')
            if actual_epochs % FLAGS.save_dec_epochs == 0:
                selectIndex = np.arange(num_train)
                feed_dict = fetch_data(train_data,selectIndex,False)
                dec_val = run_step(sess,dec_output,feed_dict)
                dec_val = dec_val[0]
                LSTMAutoencoder.plot_traj_3d(dec_val,feed_dict[FLAGS.p_input],feed_dict[FLAGS.seqlen],
                                             dataset.MAX_X,dataset.MAX_Y,actual_epochs,FLAGS._dec_output_train_dir,
                                             dataset.trainIdx)

                selectIndex = np.arange(num_test)
                feed_dict = fetch_data(test_data,selectIndex,False)
                dec_val = run_step(sess,dec_output,feed_dict)
                dec_val = dec_val[0]
            
                LSTMAutoencoder.plot_traj_3d(dec_val,feed_dict[FLAGS.p_input],feed_dict[FLAGS.seqlen],
                                             dataset.MAX_X,dataset.MAX_Y,actual_epochs,FLAGS._dec_output_test_dir,
                                             dataset.testIdx)
                print('finish saving decode result!!')

         
        ''' for final epochs files doesn't exist '''           
        
        ''' evaluate test performance after fininshing training (need add training performance)'''   
        selectIndex = np.arange(num_test)
        feed_dict = fetch_data(test_data,selectIndex,False)
        bagAccu, Y_pred, inst_pred, y_accu_tf, player_pred = run_step(sess,[accu, Y_predmap, instOuts,y_accu,y],feed_dict)

        if max_epochs is not 0:
            print('\nAfter %d Epochs: accuracy = %.5f'  % (actual_epochs, bagAccu))
        else:
            print('\nLoad/Resume Model: accuracy = %.5f'  % (bagAccu))
        #metric.calculateAccu(Y_pred,inst_pred,test_multi_Y,test_multi_label,dataset)
        
        phase = 'test'
        if max_epochs is not 0:
            filename = FLAGS._key_player_dir + '/Fold{0}_Epoch{1}_{2}_final.csv'.format(fold,actual_epochs,phase)
            info_filename = FLAGS._key_player_dir + '/Fold{0}_Epoch{1}_{2}_final_stat.csv'.format(fold,actual_epochs,phase)
        else:
            filename = None
            info_filename = None
        
        metric.keyPlayerResult(Y_pred,test_multi_Y,player_pred,dataset,phase,filename,FLAGS._ipython_console_txt,info_filename)
        utils.printLog(FLAGS._ipython_console_txt," ")
        #y_accu_Y_correct_per_tactic, avg_y_accu_Y_correct_per_tactic = metric.calculatePAccu(player_pred,test_multi_label,Y_pred,test_multi_Y)
        #print('y accu per tactic:', y_accu_Y_correct_per_tactic)
        #print('avg y accu per tactic:', avg_y_accu_Y_correct_per_tactic)
        #print('avg y accu:', y_accu_tf)
        time.sleep(0.5)
        
        
        # generate confunsion matrix
        if max_epochs is not 0:
            filename = FLAGS._confusion_dir + '/Fold{0}_Epoch{1}_test_final.csv'.format(fold,actual_epochs)
        else:
            filename = None
        metric.ConfusionMatrix(Y_pred,test_multi_Y,dataset,filename,FLAGS._ipython_console_txt)        
 
        summary_writer.close()           

        ''' for final epochs files doesn't exist '''
# =============================================================================
#         ''' save decode result '''
#         selectIndex = np.arange(num_train)
#         feed_dict = fetch_data(train_data,selectIndex,False)
#         dec_val = run_step(sess,dec_output,feed_dict)
#         dec_val = dec_val[0]
#         LSTMAutoencoder.plot_traj_3d(dec_val,feed_dict[FLAGS.p_input],feed_dict[FLAGS.seqlen],
#                                      FLAGS.MAX_X,FLAGS.MAX_Y,actual_epochs,FLAGS._dec_output_train_dir,
#                                      dataset.trainIdx)
#  
#         selectIndex = np.arange(num_test)
#         feed_dict = fetch_data(test_data,selectIndex,False)
#         dec_val = run_step(sess,dec_output,feed_dict)
#         dec_val = dec_val[0]
#         LSTMAutoencoder.plot_traj_3d(dec_val,feed_dict[FLAGS.p_input],feed_dict[FLAGS.seqlen],
#                                      FLAGS.MAX_X,FLAGS.MAX_Y,actual_epochs,FLAGS._dec_output_test_dir,
#                                      dataset.testIdx)
#         print('finish saving decode result!!')
# =============================================================================

    

def run_step(sess, op_list, feed_dict):
    ''' unified sess.run op '''   
    sess_result_list = sess.run(op_list,feed_dict=feed_dict)
    return sess_result_list
    
def GenerateSummaryStr(tag,summary_op,tf_op,input_data,label_data,sess,input_pl,target_pl,keep_prob):
    input_feed = np.transpose(input_data, (1,0,2))
    target_feed = label_data.astype(np.int32)
    summary_str = sess.run(summary_op(tag,tf_op),
                                    feed_dict={input_pl: input_feed,
                                               target_pl: target_feed,
                                               keep_prob: 1.0
                                               })
    return summary_str

def label2image(label_vec):
    try: 
        if len(label_vec.shape) == 2:
            labelImage = tf.expand_dims(tf.expand_dims(label_vec,axis = 2),axis=0)
            labelImage = tf.transpose(labelImage,perm=[0,2,1,3])
            return labelImage
    except:
        print('label dim must be 2!')
    
if __name__ == '__main__':
    tf.reset_default_graph()
    datasets = dataset.dataset(FLAGS.traj_file,FLAGS.fold_file,FLAGS.fold,
                          FLAGS.lstm_max_sequence_length,FLAGS.MAX_X,FLAGS.MAX_Y)
    pretrain_shape = net_param.createPretrainShape(FLAGS.lstm_hidden_dim,
                                               FLAGS.miNet_last_hidden_dim,
                                               FLAGS.miNet_num_hidden_layer)
    print(pretrain_shape)  
                    
    for h in range(FLAGS.miNet_num_hidden_layer):
        FLAGS.pre_layer_learning_rate.extend([0.001])#GD[0.01,0.01]
    
    instNet_shape = np.array([np.append(pretrain_shape,len(datasets.C5k_CLASS[0])),
                              np.append(pretrain_shape,len(datasets.C5k_CLASS[1])),
                              np.append(pretrain_shape,len(datasets.C5k_CLASS[2]))],
                             np.int32)
    
    print(instNet_shape)
    num_inst = np.array([10,10,1],np.int32) # 5 choose 3 key players, 5 choose 2 key players, 5 choose 3 key players 
    miNet_common_acfun = 'relu'
    acfunList = []
    for h in range(FLAGS.miNet_num_hidden_layer):
        acfunList.append(utils.get_activation_fn(miNet_common_acfun))
    acfunList.append(utils.get_activation_fn('sigmoid'))
    miList = main_unsupervised(instNet_shape,acfunList,datasets,FLAGS)
    