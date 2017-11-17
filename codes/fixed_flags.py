#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 18:04:57 2017

@author: PeterTsai
"""
# load base setting, mainly folder name
FLAGS = lambda: None
FLAGS.optimizer = 'RMSProp'
FLAGS.supervised_learning_rate = 0.01
FLAGS.supervised_weight_decay = 'constant' #exponential, piecewise, polynomial


FLAGS.fold = 0
FLAGS.root_dir = './'
FLAGS.exp_dir = FLAGS.root_dir + 'experiment/' + FLAGS.optimizer  




# input dataset
FLAGS.input_data_dir = FLAGS.root_dir + '/raw'
FLAGS.traj_file = FLAGS.input_data_dir + '/S_fixed.mat'
FLAGS.fold_file = FLAGS.input_data_dir + '/split/tactic_bagIdxSplit5(1).mat'
FLAGS.tactic_file = FLAGS.input_data_dir + '/tacticsInfo.mat'
#FLAGS.pretrain_batch_size = 2

FLAGS.miNet_last_hidden_dim = 16
FLAGS.miNet_num_hidden_layer = 1
FLAGS.miNet_common_acfun = 'sigmoid'
FLAGS.keep_prob = 1.0    


FLAGS.finetune_batch_size = 0 #2 0 means all training data
FLAGS.finetuning_epochs = 6
FLAGS.finetuning_summary_step = 10
FLAGS.finetuning_saving_epochs = 1
FLAGS.save_dec_epochs = 50
FLAGS.fine_tune_resume = True

FLAGS.decode_beta = 0.1

FLAGS.lstm_type = 'BasicLSTM'
FLAGS.use_peepholes=True
FLAGS.lstm_hidden_dim = 256
FLAGS.lstm_max_sequence_length = 450
FLAGS.lstm_input_type = 'P'
#FLAGS.lstm_input_dim = 2
FLAGS.lstm_activation = 'softmax'#'relu6' # default tanh



FLAGS.flush_secs  = 120
FLAGS.auto_load_ckpt = True
FLAGS.save_gradints = True
#FLAGS.MAX_X = 326
#FLAGS.MAX_Y = 348
#FLAGS.frameRate = 30


FLAGS.miNet_train_dir = '{0}_lr{1}_{2}_batch{14}/{3}_hidden{4}_{5}_{6}/miNet_h{7}L{8}_iter{9}_{10}_keepprob{11}/decode_beta{12}/fold{13}'.format(
        FLAGS.exp_dir,FLAGS.supervised_learning_rate,FLAGS.supervised_weight_decay,
        FLAGS.lstm_type, FLAGS.lstm_hidden_dim,FLAGS.lstm_activation,FLAGS.lstm_input_type,
        FLAGS.miNet_last_hidden_dim,FLAGS.miNet_num_hidden_layer,
        FLAGS.finetuning_epochs,FLAGS.miNet_common_acfun,FLAGS.keep_prob,FLAGS.decode_beta,FLAGS.fold+1,FLAGS.finetune_batch_size)


FLAGS.miNet_train_model_dir = FLAGS.miNet_train_dir + '/model'
FLAGS.miNet_train_summary_dir = FLAGS.miNet_train_dir + '/summaries'
FLAGS._confusion_dir = FLAGS.miNet_train_dir + '/confusionMatrix'
FLAGS._ipython_console_txt = FLAGS.miNet_train_dir + '/log.txt'
FLAGS._logit_txt = FLAGS.miNet_train_dir + '/logits'
FLAGS._train_logit_txt = FLAGS._logit_txt + '/train'
FLAGS._test_logit_txt = FLAGS._logit_txt + '/test'
FLAGS._dec_output_dir = FLAGS.miNet_train_dir + '/decode'
FLAGS._dec_output_train_dir = FLAGS._dec_output_dir + '/train'
FLAGS._dec_output_test_dir = FLAGS._dec_output_dir + '/test'
FLAGS._key_player_dir = FLAGS.miNet_train_dir + '/keyPlayerDetection'


# =============================================================================
# # lstm pretraining settings
# FLAGS.resetLSTMTempData = False
# FLAGS.ae_lstm_debug = False#True
# FLAGS.ae_lstm_save_summary_step = 10
# FLAGS.ae_lstm_save_ckpt_step    = 10
# FLAGS.ae_lstm_save_dec_step     = 20
# =============================================================================

FLAGS1 = FLAGS