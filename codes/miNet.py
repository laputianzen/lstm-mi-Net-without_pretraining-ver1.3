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



import readmat
#from flags import FLAGS
from fixed_flags_miNet import FLAGS
import net_param
import dataset
import utils
import metric

class miNet(object):
    
    _weights_str = "weights{0}"
    _biases_str = "biases{0}"
    _inputs_str = "x{0}"
    _instNets_str = "I{0}"

    def __init__(self, shape, acfunList, sess):
        """Autoencoder initializer

        Args:
            shape: list of ints specifying
                  num input, hidden1 units,...hidden_n units, num logits
        sess: tensorflow session object to use
        """
        self.__shape = shape  # [input_dim,hidden1_dim,...,hidden_n_dim,output_dim]
        self.__num_hidden_layers = len(self.__shape) - 2
        self.acfunList = acfunList

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
                #a = tf.multiply(4.0, tf.sqrt(6.0 / (w_shape[0] + w_shape[1])))
                #w_init = tf.random_uniform(w_shape, -1 * a, a)
# =============================================================================
#                 w_init = tf.truncated_normal(w_shape)
                w_init = utils.xavier_initializer(w_shape)
# =============================================================================
                
                self[name_w] = tf.Variable(w_init, name=name_w, trainable=True)
                # Train biases
                name_b = self._biases_str.format(i + 1)
                b_shape = (self.__shape[i + 1],)
# =============================================================================
#                 b_init = tf.zeros(b_shape) + 0.01
                b_init = utils.xavier_initializer(b_shape,uniform=True)
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
#                     b_init = tf.zeros(b_shape) + 0.01
                    b_init = utils.xavier_initializer(b_shape,uniform=True)
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
#            if i == self.__num_hidden_layers+1:
#                acfun = tf.sigmoid
#            else:
#                acfun = tf.nn.relu
            acfun = self.acfunList[i]
            w = self._w(i + 1, "_fixed")
            b = self._b(i + 1, "_fixed")
            
            last_output = self._activate(last_output, w, b, acfun=acfun)         
            
        if is_target:
            return last_output
        
#        if n == self.__num_hidden_layers:
#            acfun = tf.sigmoid
#        else:
#            acfun = tf.nn.relu
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
    
    def single_instNet(self, input_pl, dropout):
        """Get the supervised fine tuning net

        Args:
            input_pl: tf placeholder for ae input data
        Returns:
            Tensor giving full ae net
        """
        last_output = input_pl
        
        for i in range(self.__num_hidden_layers + 1):
#            if i+1 == self.__num_hidden_layers+1:
#                acfun = tf.log_sigmoid
#                dropout = 1.0
#            else:
#                acfun = tf.nn.relu
            #acfun = "sigmoid" acfun = "relu"            
            # Fine tuning will be done on these variables
            dropout = 1.0
            #acfun = tf.sigmoid
            acfun = self.acfunList[i]
            w = self._w(i + 1)
            b = self._b(i + 1)
            
            last_output = self._activate(last_output, w, b, acfun=acfun, keep_prob=dropout)
            
        return last_output
    
    def MIL(self,input_plist, dropout):
        #input_dim = self.shape[0]
        tmpList = list()
        for i in range(int(input_plist.shape[0])):
            name_input = self._inputs_str.format(i + 1)
            #self[name_input] = tf.placeholder(tf.float32,[None, input_dim])
            self[name_input] = input_plist[i]
            
            name_instNet = self._instNets_str.format(i + 1)
            with tf.variable_scope("mil") as scope:
                if i == 0:
                #if scope.reuse == False:
                    self[name_instNet] = self.single_instNet(self[name_input], dropout)
                    #scope.reuse = True
                    scope.reuse_variables()
                else:    
                    self[name_instNet] = self.single_instNet(self[name_input], dropout)
            
            tmpList.append(self[name_instNet])
            #if not i == 0:
                #self["y"]  = tf.concat([self["y"],self[name_instNet]],1)
            #    self["y"]  = [self["y"],self[name_instNet]]
            #else:
            #    self["y"] = self[name_instNet]
        
        self["y"] = tf.stack(tmpList,axis=1)
        self["Y"] =  tf.reduce_max(self["y"],axis=1,name="MILPool")#,keep_dims=True)
        self["maxInst"] = tf.argmax(self["y"],axis=1, name="maxInst")
        
        #batch_size = int(self["y"].shape[0])
        #topInstIdx = tf.reshape(tf.argmax(self["y"],axis=1),[batch_size,1])
        #self["kinst"] = tf.multiply(tf.round(self["Y"]),
        #    tf.cast(tf.argmax(self["y"],axis=1)+1,tf.float32),name='key_instance')
        
        #topInstIdx = tf.argmax(self["y"],axis=1)
        #self["kinst"] = tf.multiply(tf.round(self["Y"]),
        #    tf.cast(topInstIdx+1,tf.float32),name='key_instance')
        # consider tf.expand_dims to support tf.argmax
        
        #return self["Y"], self["kinst"]
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



def main_unsupervised(ae_shape,acfunList,dataset,FLAGS,sess=None):
    #tf.reset_default_graph()
    if sess is None:
        sess = tf.Session()
   
    aeList = list()
    for a in range(len(ae_shape)):
        aeList.append(miNet(ae_shape[a],acfunList, sess))
    
#==============================================================================
#     fold = FLAGS.fold
#     learning_rates = FLAGS.pre_layer_learning_rate
#     
# #for fold in range(5):
#     print('fold %d' %(fold+1))
#     for k in range(len(ae_shape)):
#         file_str = FLAGS._intermediate_feature_dir + "/C5{0}_fold{1}"
#         batch_X = np.load(file_str.format(dataset.k[k],fold+1)+'_train.npy')
#         test_X = np.load(file_str.format(dataset.k[k],fold+1)+'_test.npy')
#         num_train = len(batch_X)
#         
#         print("\nae_shape has %s pretarined layer" %(len(ae_shape[k])-2))
#         for i in range(len(ae_shape[k]) - 2):
#             n = i + 1
#             _pretrain_model_dir = '{0}/C5{1}/pretrain{2}/'.format(FLAGS.miNet_pretrain_model_dir,dataset.k[k],n)
#             if not os.path.exists(_pretrain_model_dir):
#                 os.makedirs(_pretrain_model_dir)
#             
#             with tf.variable_scope("pretrain_{0}_mi{1}".format(n,k+1)):
#                 input_ = tf.placeholder(dtype=tf.float32,
#                                         shape=(None, ae_shape[k][0]),
#                                         name='ae_input_pl')
#                 target_ = tf.placeholder(dtype=tf.float32,
#                                          shape=(None, ae_shape[k][0]),
#                                          name='ae_target_pl')
# 
#                 layer = aeList[k].pretrain_net(input_, n)
# 
# 
# 
#                 with tf.name_scope("target"):
#                     target_for_loss = aeList[k].pretrain_net(target_, n, is_target=True)
#                     
#                 if acfunList[i] is utils.get_activation_fn('sigmoid'):#if n == aeList[k].num_hidden_layers:
#                     loss = loss_x_entropy(layer, target_for_loss)
#                 elif acfunList[i] is utils.get_activation_fn('relu'):
#                     loss  = tf.reduce_mean(tf.square(layer - target_for_loss))
#                     #loss  = tf.sqrt(tf.reduce_mean(tf.square(layer - target_for_loss)))
#                         
#                 vars_to_init = aeList[k].get_variables_to_init(n)
#                 
# 
#                 train_op, global_step = training(loss, learning_rates[i], i, 
#                                         optimMethod=utils.get_optimizer(FLAGS.optimizer),
#                                         var_in_training=vars_to_init)
# 
#                 #vars_to_init.append(global_step)    
# #==============================================================================
# #                 writer = tf.summary.FileWriter(pjoin(FLAGS.miNet_pretrain_summary_dir,
# #                                                   'instNet_pre_training'),tf.get_default_graph())
# #                 writer.close()  
# #==============================================================================
#                 
#                 # adam special parameter beta1, beta2
#                 pretrain_vars =  tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="pretrain_{0}_mi{1}".format(n,k+1))
#                 #optim_vars = [var for var in pretrain_vars if ('beta' in var.name or 'Adam' in var.name)]
# #                for var in adam_vars:
# #                    vars_to_init.append(var)
#                         
#                 pretrain_test_loss  = tf.summary.scalar('pretrain_test_loss',loss)
#                 
#                 saver = tf.train.Saver(vars_to_init)
#                 model_ckpt = _pretrain_model_dir+ 'model.ckpt'    
#             
#                 if os.path.isfile(model_ckpt+'.meta'):
#                     #tf.reset_default_graph()
#                     print("|---------------|---------------|---------|----------|")
#                     saver.restore(sess, model_ckpt)
#                     for v in vars_to_init:
#                         print("%s with value %s" % (v.name, sess.run(tf.is_variable_initialized(v))))
#                 
#                 else:
#                     summary_dir = pjoin(FLAGS.miNet_pretrain_summary_dir, 'C5{0}/pretrain{1}/'.format(dataset.k[k],n))
#                     summary_writer = tf.summary.FileWriter(summary_dir,
#                                                            graph=sess.graph,
#                                                            flush_secs=FLAGS.flush_secs)
#                     #summary_vars = [aeList[k]["biases{0}".format(n)], aeList[k]["weights{0}".format(n)]]
#     
#                     hist_summarries = [tf.summary.histogram(v.op.name, v)
#                                    for v in vars_to_init]#summary_vars]
#                     hist_summarries.append(loss_summaries[i])
#                     summary_op = tf.summary.merge(hist_summarries)                    
#                     
#                     if FLAGS.pretrain_batch_size is None:
#                         FLAGS.pretrain_batch_size = batch_X.shape[0]
#                     sess.run(tf.variables_initializer(vars_to_init))
#                     #sess.run(tf.variables_initializer(optim_vars))
#                     sess.run(tf.variables_initializer(pretrain_vars))
#                     print("\n\n")
#                     print("| Training Step | Cross Entropy |  Layer  |   Epoch  |")
#                     print("|---------------|---------------|---------|----------|")
#         
#                     count = 0
#                     for epochs in range(FLAGS.pretraining_epochs):
#                         perm = np.arange(num_train)
#                         np.random.shuffle(perm)
#                         
#                         train_loss = 0.0
#                         for step in range(int(num_train/FLAGS.pretrain_batch_size)):
#                             selectIndex = perm[FLAGS.pretrain_batch_size*step:FLAGS.pretrain_batch_size*step+FLAGS.pretrain_batch_size]
#                             input_feed = np.reshape(batch_X[selectIndex,:,:],
#                                                     (batch_X[selectIndex,:,:].shape[0]*batch_X[selectIndex,:,:].shape[1],batch_X[selectIndex,:,:].shape[2]))
#                             target_feed = input_feed
#                             loss_summary, loss_value = sess.run([train_op, loss],
#                                                             feed_dict={
#                                                                 input_: input_feed,
#                                                                 target_: target_feed,
#                                                                 })
# 
#                             count = count + 1
#                             train_loss += loss_value
#                             #if (count-1)*FLAGS.pretrain_batch_size*batch_X.shape[1] % (10*batch_X.shape[1]) ==0:
#                             if step % 10 ==0 or step == int(num_train/FLAGS.pretrain_batch_size)-1:
#                                 summary_str = sess.run(summary_op, feed_dict={
#                                                                 input_: input_feed,
#                                                                 target_: target_feed,
#                                                                 })
#                                 summary_writer.add_summary(summary_str, count)
#         
#                                 output = "| {0:>13} | {1:13.4f} | Layer {2} | Epoch {3}  |"\
#                                         .format(step, loss_value, n, epochs + 1)
#     
#                                 print(output)
#                         print ('epoch %d: mean train loss = %.3f' %(epochs,train_loss/(num_train/FLAGS.pretrain_batch_size)))        
#                         test_input_feed = np.reshape(test_X,(test_X.shape[0]*test_X.shape[1],test_X.shape[2]))
#                         test_target_feed = np.reshape(test_X,(test_X.shape[0]*test_X.shape[1],test_X.shape[2]))
#                         #test_target_feed = test_Y.astype(np.int32)     
#                         loss_summary, loss_value = sess.run([train_op, loss],
#                                                             feed_dict={
#                                                                     input_: test_input_feed,
#                                                                     target_: test_target_feed,
#                                                                     })
#     
#                         pretrain_test_loss_str = sess.run(pretrain_test_loss,
#                                                   feed_dict={input_: test_input_feed,
#                                                              target_: test_target_feed,
#                                                      })                                          
#                         summary_writer.add_summary(pretrain_test_loss_str, epochs)
#                         print ('epoch %d: test loss = %.3f' %(epochs,loss_value))    
#                         time.sleep(3)
#                     summary_writer.close()         
# #                text_file = open("Output.txt", "a")
# #                for b in range(len(ae_shape) - 2):
# #                    if sess.run(tf.is_variable_initialized(ae._b(b+1))):
# #                        #print("%s with value in [pretrain %s]\n %s" % (ae._b(b+1).name, n, ae._b(b+1).eval(sess)))
# #                        text_file.write("%s with value in [pretrain %s]\n %s\n" % (ae._b(b+1).name, n, ae._b(b+1).eval(sess)))
# #                text_file.close()
#                     save_path = saver.save(sess, model_ckpt)
#                     print("Model saved in file: %s" % save_path)
#                                       
#                 #input("\nPress ENTER to CONTINUE\n")  
#     
#         time.sleep(0.5)
#==============================================================================
                                      
    return aeList





#text_file = open("final_result.txt", "w")

def main_supervised(instNetList,num_inst,inputs,dataset,FLAGS):
    
    fold = FLAGS.fold
    if not os.path.exists(FLAGS._confusion_dir):
        os.makedirs(FLAGS._confusion_dir)
    
    if not os.path.exists(FLAGS._logit_txt):
        os.makedirs(FLAGS._logit_txt)
        
    text_file = open(FLAGS._result_txt,"w")
    with instNetList[0].session.graph.as_default():
        sess = instNetList[0].session
        
        bagOuts = []
        instOuts = []
        maxInstOuts = []
        instIdx = np.insert(np.cumsum(num_inst),0,0)
        keep_prob_ = tf.placeholder(dtype=tf.float32,
                                           name='dropout_ratio')

        offset = tf.constant(instIdx)
        for k in range(len(instNetList)):
            with tf.name_scope('C5{0}'.format(dataset.k[k])):            
                out_Y, out_y, out_maxInst = instNetList[k].MIL(tf.transpose(inputs[k],perm=(1,0,2)),keep_prob_)
            bagOuts.append(out_Y)
            instOuts.append(out_y)
            maxInstOuts.append(out_maxInst+offset[k])
            

        hist_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        hist_summaries = [tf.summary.histogram(v.op.name + "_fine_tuning", v)for v in hist_variables if not ('decoder' in v.name)]    
#        hist_summaries = [tf.summary.histogram(v.op.name + "_fine_tuning", v)
#                              for v in hist_summaries]
        summary_op = tf.summary.merge(hist_summaries)            
        
        #Y = tf.dynamic_stitch(FLAGS.C5k_CLASS,bagOuts)
        Y = tf.concat(bagOuts,1,name='output')
        
# =============================================================================
#         y_maxInsts = tf.concat(maxInstOuts,1, name='maxInsts')
# =============================================================================
                    
        NUM_CLASS = len(dataset.tacticName)
        Y_placeholder = tf.placeholder(tf.float32,
                                        shape=(None,NUM_CLASS),
                                        name='target_pl')
        #loss = loss_x_entropy(tf.nn.softmax(Y), tf.cast(Y_placeholder, tf.float32))
        with tf.name_scope('softmax_cross_entory_with_logit'):
            x_entropy = metric.loss_x_entropy(tf.nn.softmax(Y), Y_placeholder)
            tactic_prediction_op = tf.summary.histogram('tactic prediction',tf.nn.softmax(Y))
            tactic_score_op = tf.summary.histogram('tactic_prediction',Y)
# =============================================================================
#             loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=Y,
#                                 labels=tf.argmax(Y_placeholder,axis=1),name='softmax_cross_entropy'))
#         
# =============================================================================
        lstm_last_output = tf.get_collection('aelstm_lastoutput')
        lstm_last_output_op = tf.summary.histogram('aelstm_lastoutput',lstm_last_output)
        num_train = len(dataset.trainIdx)
        regularization = tf.get_collection('decode_loss')
        decode_beta = tf.cast(FLAGS.decode_beta,tf.float32)
        normalized_regularization = tf.sqrt(regularization)*tf.constant(FLAGS.MAX_Y,dtype=tf.float32) 
        loss = x_entropy + decode_beta * tf.squeeze(normalized_regularization,[0])

#        loss_xs = loss
#        beta = FLAGS.beta
#        loss = loss + tf.reduce_sum(variance) * beta
        x_entropy_op = tf.summary.scalar('x_entropy_loss',x_entropy)
        aelstm_op = tf.summary.scalar('aelstm_loss',tf.squeeze(normalized_regularization,[0]))
        loss_op = tf.summary.scalar('total_loss',loss)
        #loss = loss_supervised(logits, labels_placeholder)
        
        with tf.name_scope('MultiClassEvaluation'):
            accu, error = metric.multiClassEvaluation(Y, Y_placeholder)
        #train_op, global_step = training(error, FLAGS.supervised_learning_rate, None, optimMethod=FLAGS.optim_method)
#        with tf.name_scope('correctness'):
#            correct =tf.equal(tf.argmax(Y,1),tf.argmax(Y_placeholder,1))
#            error = 1 - tf.reduce_mean(tf.cast(correct, tf.float32))
        
        error_op = tf.summary.scalar('test_error',error)
        accu_op = tf.summary.scalar('test_accuracy',accu)

# =============================================================================
#         Y_labels_image = label2image(Y_placeholder)
#         #label_op = tf.summary.image('tactic_labels',Y_label_image)
#         Y_logits_image = label2image(tf.nn.softmax(Y))
#         #label_op = tf.summary.histogram('tactic_labels',Y_placeholder)
# =============================================================================
        merged = tf.summary.merge([lstm_last_output_op,
                                   tactic_prediction_op,
                                   tactic_score_op,
                                   x_entropy_op,aelstm_op,
                                   loss_op,error_op,
                                   accu_op,summary_op])#,output_op,label_op])
        summary_writer = tf.summary.FileWriter(pjoin(FLAGS.miNet_pretrain_summary_dir,
                                                      'fine_tuning_iter{0}'.format(FLAGS.finetuning_epochs)),tf.get_default_graph())
                                                #graph_def=sess.graph_def,
                                                #flush_secs=FLAGS.flush_secs)
        vars_to_init = []
        # initialize lstm variabel         
        vars_to_init.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='encoder'))
                                
        for k in range(len(instNetList)):
            instNet = instNetList[k]
            vars_to_init.extend(instNet.get_variables_to_init(instNet.num_hidden_layers + 1))
#        vars_to_init = tf.trainable_variables()
        
#==============================================================================
#         train_op, global_step = training(loss, FLAGS.supervised_learning_rate, None, 
#                                          optimMethod=utils.get_optimizer(FLAGS.optimizer),
#                                          var_in_training=vars_to_init)
#==============================================================================
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = tf.contrib.layers.optimize_loss(loss, global_step, 
                                                   learning_rate=FLAGS.supervised_learning_rate, 
                                                   optimizer=utils.get_optimizer(FLAGS.optimizer),
                                                   summaries=["loss","gradients","global_gradient_norm","gradient_norm"])
        
        summaries_vars_optimize_loss = [var for var in tf.get_collection(tf.GraphKeys.SUMMARIES) if ('OptimizeLoss' in var.name)]
        optimize_loss_op = tf.summary.merge(summaries_vars_optimize_loss)
        
        vars_to_init.append(global_step)
        # adam special parameter beta1, beta2
        #optim_vars = [var for var in tf.global_variables() if ('beta' in var.name or 'Adam' in var.name)] 
        optim_vars = [var for var in tf.global_variables() if (FLAGS.optimizer in var.name)]
        learning_rate_var = [var for var in tf.global_variables() if ('learning_rate' in var.name)]
        #'OptimizeLoss'
        sess.run(tf.variables_initializer(vars_to_init))
        sess.run(tf.variables_initializer(optim_vars))
        sess.run(tf.variables_initializer(learning_rate_var))
        sess.run(tf.variables_initializer(tf.trainable_variables()))
# =============================================================================
#         train_loss  = tf.summary.scalar('train_loss',loss)
# =============================================================================
        #steps = FLAGS.finetuning_epochs * num_train
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
        strBagShape = "the shape of bags is ({0},{1})".format(batch_multi_Y.shape[0],batch_multi_Y.shape[1])
        print(strBagShape)
        batch_multi_X = dataset.dataTraj[dataset.trainIdx,:]
        train_data = {'sequences':batch_multi_X,'seqlen':seqLenMatrix[trainIdx,:],'label':batch_multi_Y}

        testdir = 'dataGood/multiPlayers/syncLargeZoneVelocitySoftAssign(R=16,s=10)/test/fold%d' %(fold+1)
        test_file_str= '{0}ZoneVelocitySoftAssign(R=16,s=10){1}_test%d.mat' %(fold+1) 
        _, test_multi_Y, test_multi_label = readmat.multi_class_read(testdir,test_file_str,num_inst,dataset)       
        num_test = len(test_multi_Y)
        strBagShape = "the shape of bags is ({0},{1})".format(test_multi_Y.shape[0],test_multi_Y.shape[1])
        print(strBagShape)
        test_multi_X = dataset.dataTraj[testIdx,:]
        test_data = {'sequences':test_multi_X,'seqlen':seqLenMatrix[testIdx,:],'label':test_multi_Y}
      
        if FLAGS.finetune_batch_size is None:
            FLAGS.finetune_batch_size = len(test_multi_Y)
        

        def fetch_data(data,selectIndex,dropout):
            random_sequences = np.reshape(data['sequences'][selectIndex,:],(-1,data['sequences'].shape[2],data['sequences'].shape[3]))
            batch_seqlen = np.reshape(data['seqlen'][selectIndex,:],(-1))
            target_feed = data['label'][selectIndex].astype(np.int32)            
            
            if dropout:
                keep_prob = FLAGS.keep_prob
            else:
                keep_prob = 1.0
                
            feed_dict={FLAGS.p_input:random_sequences,
                       FLAGS.seqlen: batch_seqlen,
                       Y_placeholder: target_feed,
                       keep_prob_: keep_prob}
            return feed_dict
            
        count = 0
        for epochs in range(FLAGS.finetuning_epochs):
            perm = np.arange(num_train)
            np.random.shuffle(perm)
            #numPlayer = len(batch_multi_KPlabel[0])
            print("|-------------|-----------|-------------|----------|")
            text_file.write("|-------------|-----------|-------------|----------|\n")
            
            ''' training of one epoch '''
            for step in range(int(num_train/FLAGS.finetune_batch_size)):
                start_time = time.time()
                    
                selectIndex = perm[FLAGS.finetune_batch_size*step:FLAGS.finetune_batch_size*step+FLAGS.finetune_batch_size]
                
                ''' get train data and feed into graph in training stage '''

                feed_dict = fetch_data(train_data,selectIndex,True)
                _, loss_value, logit, label = run_step(sess,[train_op, loss, Y, Y_placeholder],feed_dict)
                ''' more details for debugging '''
#                los, m, var, mf, xpmf, maxInst, ta_pred,num_ta, related_inst_idx, relate_inst_idx = sess.run([loss_xs, mean, variance, mask_feature, expand_mask_feature, y_maxInsts, tactic_pred, numEachTactic, pred_related_inst_idx, pred_relate_inst_idx],
#                                                          feed_dict={
#                                                                  FLAGS.lstm.p_input:random_sequences,
#                                                                  FLAGS.lstm.seqlen: batch_seqlen,
#                                                                  Y_placeholder: target_feed,
#                                                                  keep_prob_: FLAGS.keep_prob
#                                                })
                ''' model outputs for tensorboarad image summaries and debugging '''
                feed_dict = fetch_data(train_data,selectIndex,False)
                Y_scaled, Y_unscaled = run_step(sess,[tf.nn.softmax(Y),Y],feed_dict)      
                duration = time.time() - start_time
                
                # Write the summaries and print an overview fairly often.
                count = count + 1
                ''' save training data result after n training steps '''
                if step % 10 == 0:
                    # Print status to stdout.
                    #print('Step %d: loss = %.2f (%.3f sec)' % (count, loss_value, duration))
                    print('|   Epoch %d  |  Step %d  |  loss = %.3f | (%.3f sec)' % (epochs+1, step, loss_value, duration))
                    text_file.write('|   Epoch %d  |  Step %d  |  loss = %.3f | (%.3f sec)\n' % (epochs+1, step, loss_value, duration))
                    feed_dict = fetch_data(train_data,selectIndex,True)
                    summary_str = run_step(sess,merged,feed_dict)
                    summary_writer.add_summary(summary_str, count)
                    
                    summary_str = run_step(sess,optimize_loss_op,feed_dict)               
                    summary_writer.add_summary(summary_str, count)
                    
                    np.savetxt('{}/logit{}_{}.txt'.format(FLAGS._logit_txt,epochs,step),Y_scaled,fmt='%.4f', delimiter=' ')
                    #np.savetxt('{}/logit{}_{}.txt'.format(FLAGS._logit_txt,epochs,step),logit,fmt='%.4f', delimiter=' ')               
                    


            ''' evaluate test performance after one epoch (need add training performance)'''
            # test data result
            selectIndex = np.arange(num_test)
            feed_dict = fetch_data(test_data,selectIndex,False)
            loss_value,bagAccu, Y_pred, inst_pred = run_step(sess,[loss, accu, tf.argmax(tf.nn.softmax(Y),axis=1), instOuts],feed_dict)
            print('Epochs %d: test loss = %.5f '  % (epochs+1, loss_value)) 
            print('Epochs %d: accuracy = %.5f '  % (epochs+1, bagAccu)) 
            text_file.write('Epochs %d: accuracy = %.5f\n\n'  % (epochs+1, bagAccu))
            
            
            
            bagAccu,pAccu = metric.calculateAccu(Y_pred,inst_pred,test_multi_Y,test_multi_label,dataset)
            text_file.write('bag accuracy %.5f, inst accuracy %.5f\n' %(bagAccu, pAccu))
            
            filename = FLAGS._confusion_dir + '/Fold{0}_Epoch{1}_test.csv'.format(fold,epochs)
            metric.ConfusionMatrix(Y_pred,test_multi_Y,dataset,filename)
            #print("")

                    #summary_str = sess.run(summary_op, feed_dict=feed_dict)
                    #summary_writer.add_summary(summary_str, step)
                    #summary_img_str = sess.run(
                    #    tf.image_summary("training_images",
                    #                tf.reshape(input_pl,
                    #                        (FLAGS.batch_size,
                    #                         FLAGS.image_size,
                    #                         FLAGS.image_size, 1)),
                    #             max_images=FLAGS.batch_size),
                    #    feed_dict=feed_dict
                    #)
                    #summary_writer.add_summary(summary_img_str)
                    
#            for b in range(instNet.num_hidden_layers + 1):
#                if sess.run(tf.is_variable_initialized(instNet._b(b+1))):
#                    #print("%s with value in [pretrain %s]\n %s" % (ae._b(b+1).name, n, ae._b(b+1).eval(sess)))
#                    text_file.write("%s with value after fine-tuning\n %s\n" % (instNet._b(b+1).name, instNet._b(b+1).eval(sess)))
#            text_file.close()
#            save_path = saver.save(sess, model_ckpt)
#            print("Model saved in file: %s" % save_path)    
#        else:
#            saver = tf.train.import_meta_graph(model_ckpt+'.meta')
#            saver.restore(sess, model_ckpt)                    
        

        ''' evaluate test performance after fininshing training (need add training performance)'''   
        selectIndex = np.arange(num_test)
        feed_dict = fetch_data(test_data,selectIndex,False)
        bagAccu, Y_pred, inst_pred = run_step(sess,[accu, tf.argmax(tf.nn.softmax(Y),axis=1), instOuts],feed_dict)

        Y_scaled, Y_unscaled = run_step(sess,[tf.nn.softmax(Y),Y],feed_dict)

        ''' calculate instance accuracy (not used afterward)'''
        test_target_feed = test_data['label']
        inst_pred_matrix = np.empty([test_target_feed.shape[0],max(num_inst),test_target_feed.shape[1]])
        inst_pred_matrix.fill(np.nan)
        for test_id in range(test_target_feed.shape[0]):
            for k in range(len(dataset.C5k_CLASS)):
                for c in range(len(dataset.C5k_CLASS[k])):
                    realTacticID = dataset.C5k_CLASS[k][c]
                    inst_pred_matrix[test_id,:,realTacticID] = np.exp(inst_pred[k][test_id,:,c])
                    
        test_inst_label = np.empty([test_target_feed.shape[0],max(num_inst)])
        test_inst_label.fill(np.nan)
        for test_id in range(len(test_multi_label)):
            k = np.sum(test_multi_label[test_id,:])
            k_idx = dataset.k.index(k)
            inst_gt = dataset.playerMap[k_idx].index(test_multi_label[test_id,:].tolist())
            test_inst_label[test_id,inst_gt] = 1.0
            
        
        print('\nAfter %d Epochs: accuracy = %.5f'  % (epochs+1, bagAccu))
        metric.calculateAccu(Y_pred,inst_pred,test_multi_Y,test_multi_label,dataset)
        time.sleep(0.5)
        
        
        # generate confunsion matrix
        filename = FLAGS._confusion_dir + '/Fold{0}_Epoch{1}_test_final.csv'.format(fold,epochs)
        metric.ConfusionMatrix(Y_pred,test_multi_Y,dataset,filename)        
 
        summary_writer.close()           


    

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
    