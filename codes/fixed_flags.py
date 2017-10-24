#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 18:04:57 2017

@author: PeterTsai
"""
# load base setting, mainly folder name
FLAGS = lambda: None
FLAGS.optimizer = 'RMSProp'
FLAGS.fold = 0
FLAGS.root_dir = './'
FLAGS.exp_dir = FLAGS.root_dir + 'experiment/' + 'fold{0}/'.format(FLAGS.fold+1) + FLAGS.optimizer
#FLAGS.summary_dir = FLAGS.exp_dir + '/summaries'
#FLAGS._ckpt_dir   = FLAGS.exp_dir + '/model'
#FLAGS._confusion_dir = FLAGS.exp_dir + '/confusionMatrix'
#FLAGS._result_txt = FLAGS.exp_dir + '/final_result.txt'    
FLAGS.flush_secs  = 120

FLAGS.auto_load_ckpt = True

# input dataset
FLAGS.input_data_dir = FLAGS.root_dir + '/raw'
FLAGS.traj_file = FLAGS.input_data_dir + '/S_fixed.mat'
FLAGS.fold_file = FLAGS.input_data_dir + '/split/tactic_bagIdxSplit5(1).mat'
FLAGS.pretrain_batch_size = 2
FLAGS.flush_secs  = 120

FLAGS.save_gradints = True

FLAGS.miNet_last_hidden_dim = 16
FLAGS.miNet_num_hidden_layer = 1
FLAGS.pretraining_epochs = 600#600
FLAGS.finetune_batch_size = 2
FLAGS.finetuning_epochs = 1000
FLAGS.finetuning_summary_step = 1
FLAGS.finetuning_saving_epochs = 10
FLAGS.fine_tune_resume = True
FLAGS.miNet_common_acfun = 'sigmoid'

FLAGS.pre_layer_learning_rate = []   
FLAGS.supervised_learning_rate = 0.01
FLAGS.keep_prob = 1.0    
FLAGS.decode_beta = 0.1

FLAGS.lstm_type = 'BasicLSTM'
FLAGS.use_peepholes=True
FLAGS.lstm_hidden_dim = 256
FLAGS.lstm_max_sequence_length = 450
FLAGS.lstm_input_dim = 2
FLAGS.lstm_pretrain_iteration = 500 #5
FLAGS.lstm_pretrain_batch_size = 2
FLAGS.lstm_activation = 'softmax'#'relu6' # default tanh
FLAGS.lstm_lr = 0.001

FLAGS.resetLSTMTempData = False
#FLAGS.lstm.activation_lstm = tf.nn.softmax#tf.nn.relu6
#FLAGS.lstm.optimizer = tf.train.RMSPropOptimizer(0.003)

FLAGS.MAX_X = 326
FLAGS.MAX_Y = 348

FLAGS.ae_lstm_pretrain_model_dir = '{0}/{4}/hidden{1}_{2}_iter{3}/ae_lstm/model'.format(
        FLAGS.exp_dir,FLAGS.lstm_hidden_dim,FLAGS.lstm_activation,
        FLAGS.lstm_pretrain_iteration, FLAGS.lstm_type)

FLAGS.ae_lstm_summary_dir = '{0}/{4}/hidden{1}_{2}_iter{3}/ae_lstm/summaries'.format(
        FLAGS.exp_dir,FLAGS.lstm_hidden_dim,FLAGS.lstm_activation,
        FLAGS.lstm_pretrain_iteration, FLAGS.lstm_type)

FLAGS.ae_lstm_dec_dir = '{0}/{4}/hidden{1}_{2}_iter{3}/ae_lstm//decode'.format(
        FLAGS.exp_dir,FLAGS.lstm_hidden_dim,FLAGS.lstm_activation,
        FLAGS.lstm_pretrain_iteration, FLAGS.lstm_type)

FLAGS._intermediate_feature_dir = FLAGS.ae_lstm_pretrain_model_dir   + '/tempData' 

FLAGS.miNet_pretrain_dir = '{0}/{4}/hidden{1}_{2}_iter{3}/miNet_h{5}L{6}_iter{7}_{8}'.format(
        FLAGS.exp_dir,FLAGS.lstm_hidden_dim,FLAGS.lstm_activation,
        FLAGS.lstm_pretrain_iteration, FLAGS.lstm_type,
        FLAGS.miNet_last_hidden_dim,FLAGS.miNet_num_hidden_layer,
        FLAGS.pretraining_epochs,FLAGS.miNet_common_acfun)

#==============================================================================
# FLAGS.miNet_fine_tune_dir = '{0}/{4}/hidden{1}_{2}_iter{3}/miNet_h{5}L{6}_iter{7}_{8}/fine{9}'.format(
#         FLAGS.exp_dir,FLAGS.lstm_hidden_dim,FLAGS.lstm_activation,
#         FLAGS.lstm_pretrain_iteration, FLAGS.lstm_type,
#         FLAGS.miNet_last_hidden_dim,FLAGS.miNet_num_hidden_layer,
#         FLAGS.pretraining_epochs,FLAGS.miNet_common_acfun,
#         FLAGS.finetuning_epochs)
#==============================================================================

FLAGS.miNet_pretrain_model_dir = FLAGS.miNet_pretrain_dir + '/model'
FLAGS.miNet_pretrain_summary_dir = FLAGS.miNet_pretrain_dir + '/summaries'
FLAGS._confusion_dir = FLAGS.miNet_pretrain_dir + '/confusionMatrix'
FLAGS._result_txt = FLAGS.miNet_pretrain_dir + '/final_result.txt'
FLAGS._logit_txt = FLAGS.miNet_pretrain_dir + '/logits'

FLAGS.ae_lstm_debug = False#True
FLAGS.ae_lstm_save_summary_step = 10
FLAGS.ae_lstm_save_ckpt_step    = 10
FLAGS.ae_lstm_save_dec_step     = 20