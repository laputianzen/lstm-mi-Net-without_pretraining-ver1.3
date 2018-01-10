#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 16:34:33 2017

@author: PeterTsai
"""
import tensorflow as tf
import os
from os.path import join as pjoin

def data_out(FLAGS,path):
    return 'a'



def search_file(searching_for,start_path):
    last_root    = start_path
    current_root = start_path
    found_path   = None
    while found_path is None and current_root:
        pruned = False
        for root, dirs, files in os.walk(current_root):
            if not pruned:
               try:
                  # Remove the part of the tree we already searched
                  del dirs[dirs.index(os.path.basename(last_root))]
                  pruned = True
               except ValueError:
                  pass
            if searching_for in files:
               # found the file, stop
               found_path = os.path.join(root)#, searching_for)
               break
        # Otherwise, pop up a level, search again
        last_root    = current_root
        current_root = os.path.dirname(last_root)
    return found_path

def find_code_root_dir():
    mypath = os.path.dirname(os.path.abspath(__file__))
    #print('mypath:',mypath)
    root_dir = search_file('main.py',mypath)
    return root_dir

# =============================================================================
# # single out    
# def raw_out(input_data_dir,file_name):
#     #print(os.getcwd())
#     root_dir = find_code_root_dir()    
#     print('parentpath:',root_dir)
#     return pjoin(root_dir,input_data_dir,file_name)
# =============================================================================



flags = tf.app.flags
FLAGS = flags.FLAGS

# dataset setttings
flags.DEFINE_integer("fold", 0, "fold")
#@FLAGS.fold = 0
flags.DEFINE_integer("gpu_device", 0, "gpu_device")
#FLAGS.gpu_device = 0
tf.app.flags.DEFINE_float('gpu_memory_fraction',0.8,'gpu_memory_fraction')
#FLAGS.gpu_memory_fraction = 0.8

# Optimizer settings
flags.DEFINE_string('optimizer','RMSProp','optimizer')
#@FLAGS.optimizer = 'RMSProp'
tf.app.flags.DEFINE_float("supervised_learning_rate", 0.001, "learning rate") 
#@FLAGS.supervised_learning_rate = 0.01
flags.DEFINE_string('supervised_weight_decay','constant','supervised_weight_decay')
#@FLAGS.supervised_weight_decay = 'constant' #exponential, piecewise, polynomial

##flags.DEFINE_string('root_dir',os.getcwd(),'root_dir'')
#@FLAGS.root_dir = './'


# fine tune settings
#@FLAGS.finetune_batch_size = 0
flags.DEFINE_integer("finetuning_epochs", 500, "miNet_last_hidden_dim")
#@FLAGS.finetuning_epochs = 500
flags.DEFINE_integer("finetuning_summary_step", 10, "finetuning_summary_step")
#@FLAGS.finetuning_summary_step = 10
flags.DEFINE_integer("finetuning_saving_epochs", 1, "finetuning_saving_epochs")
#@FLAGS.finetuning_saving_epochs = 1
flags.DEFINE_integer("save_dec_epochs", 0, "save_dec_epochs")
#@FLAGS.save_dec_epochs = 50
flags.DEFINE_boolean('fine_tune_resume',True,'fine_tune_resume')
#@FLAGS.fine_tune_resume = True

# regularization 
tf.app.flags.DEFINE_float('decode_beta',1,'decode_beta')
#@FLAGS.decode_beta = 0.1

# Directories
#flags.DEFINE_string('exp_dir',


# LSTM Architecture Specific Flags
flags.DEFINE_string('lstm_type', 'GRU','lstm_type')
#@FLAGS.lstm_type = 'BasicLSTM'
flags.DEFINE_boolean('use_peepholes',True,'use_peepholes')
#@FLAGS.use_peepholes=True
flags.DEFINE_integer("lstm_hidden_dim", 256, "LSTM hidden state dimension")
#@FLAGS.lstm_hidden_dim = 256
flags.DEFINE_integer("lstm_max_sequence_length", 450, "LSTM max sequence length")
#@FLAGS.lstm_max_sequence_length = 450
flags.DEFINE_string("lstm_input_type", 'P', "LSTM input dimension P=2, PV=4")
#@FLAGS.lstm_input_type = 'P'
flags.DEFINE_string('lstm_activation', 'tanh','lstm activation function')
#@FLAGS.lstm_activation = 'softmax'

# miNet Architecture Specific Flags
flags.DEFINE_integer("miNet_last_hidden_dim", 16, "miNet_last_hidden_dim")
#@FLAGS.miNet_last_hidden_dim = 16
flags.DEFINE_integer("miNet_num_hidden_layer", 1, "miNet_num_hidden_layer")
#@FLAGS.miNet_num_hidden_layer = 1
flags.DEFINE_integer("finetune_batch_size", 2, "finetune_batch_size")
#@FLAGS.finetune_batch_size = 0 #2 0 means all training data
flags.DEFINE_string('miNet_common_acfun', 'sigmoid','miNet_common_acfun')
#@FLAGS.miNet_common_acfun = 'sigmoid'
tf.app.flags.DEFINE_float("keep_prob", 1.0, "keep_prob")
#@FLAGS.keep_prob = 1.0
tf.app.flags.DEFINE_integer("batch_norm_layer",-1,"batch_norm_layer")
#FLAGS.batch_norm_layer = -1

# tensorflow settings
flags.DEFINE_integer('flush_secs',120,'flush_secs')
#@FLAGS.flush_secs  = 120
flags.DEFINE_boolean('auto_load_ckpt',True,'auto_load_ckpt')
#@FLAGS.auto_load_ckpt = True
flags.DEFINE_boolean('save_gradints',True,'save_gradints')
#@FLAGS.save_gradints = True
 

flags.DEFINE_string('exp_dir', 'experiment',
                    'Directory to put the experiment data')

# file settings (dataGood)

flags.DEFINE_string('input_data_dir', 'raw',
                    'Directory to read the input data')
#@@FLAGS.input_data_dir = FLAGS.root_dir + '/raw'
# raw datapath
flags.DEFINE_string('traj_file', 'S_fixed.mat',
                    'Directory to read the trajectories data')
#@FLAGS.traj_file = FLAGS.input_data_dir + '/S_fixed.mat'
flags.DEFINE_string('fold_file', 'split/tactic_bagIdxSplit5(1).mat',
                    'Directory to read the fold data')
#@FLAGS.fold_file = FLAGS.input_data_dir + '/split/tactic_bagIdxSplit5(1).mat'
flags.DEFINE_string('tactic_file', 'tacticsInfo.mat',
                    'Directory to read the tacticsInfo data')
#@FLAGS.tactic_file = FLAGS.input_data_dir + '/tacticsInfo.mat'

# label datapath

flags.DEFINE_string('miNet_train_model_dir', 'model',
                    'Directory to put the model data')
#@FLAGS.miNet_train_model_dir = FLAGS.miNet_train_dir + '/model'
flags.DEFINE_string('miNet_train_summary_dir', 'summaries',
                    'Directory to put the summary data')
#@FLAGS.miNet_train_summary_dir = FLAGS.miNet_train_dir + '/summaries'
flags.DEFINE_string('_confusion_dir', 'confusionMatrix',
                    'Directory to put the confusionMatrix data')
#@FLAGS._confusion_dir = FLAGS.miNet_train_dir + '/confusionMatrix'
flags.DEFINE_string('_ipython_console_txt', 'console',
                    'Directory to put the log data')
#@FLAGS._ipython_console_txt = FLAGS.miNet_train_dir + '/log'

flags.DEFINE_string('_logit_txt', 'logits',
                    'Directory to put the logits data')
#@@FLAGS._logit_txt = FLAGS.miNet_train_dir + '/logits'
flags.DEFINE_string('_train_logit_txt', 'logits/train',
                    'Directory to put the logits data of training set')
#@FLAGS._train_logit_txt = FLAGS._logit_txt + '/train'
flags.DEFINE_string('_test_logit_txt', 'logits/test',
                    'Directory to put the logits data of test set')
#@FLAGS._test_logit_txt = FLAGS._logit_txt + '/test'



flags.DEFINE_string('_dec_output_dir', 'decode',
                    'Directory to put the logits data')
#@@FLAGS._dec_output_dir = FLAGS.miNet_train_dir + '/decode'

flags.DEFINE_string('_dec_output_train_dir', 'decode/train',
                    'Directory to put the decode data of training set')
#@FLAGS._dec_output_train_dir = FLAGS._dec_output_dir + '/train'
flags.DEFINE_string('_dec_output_test_dir', 'decode/test',
                    'Directory to put the decode data of test set')
#@FLAGS._dec_output_test_dir = FLAGS._dec_output_dir + '/test'

flags.DEFINE_string('_key_player_dir', 'keyPlayerDetection',
                    'Directory to put the keyPlayerDetection data')
#@FLAGS._key_player_dir = FLAGS.miNet_train_dir + '/keyPlayerDetection'




def raw_out(FLAGS):
    #print(os.getcwd())
    root_dir = find_code_root_dir()    
    #print('parentpath:',root_dir)
    FLAGS.traj_file   = pjoin(root_dir,FLAGS.input_data_dir,FLAGS.traj_file)
    #print('traj_file:', FLAGS.traj_file)
    FLAGS.fold_file   = pjoin(root_dir,FLAGS.input_data_dir,FLAGS.fold_file)
    FLAGS.tactic_file = pjoin(root_dir,FLAGS.input_data_dir,FLAGS.tactic_file)

    #return FLAGS

def experiment_out(FLAGS):
    root_dir = find_code_root_dir()  
    FLAGS.exp_dir = pjoin(root_dir, FLAGS.exp_dir, FLAGS.optimizer)
    #print('exp dir:', FLAGS.exp_dir)
    FLAGS.miNet_train_dir = '{0}_lr{1}_{2}_batch{14}/{3}_hidden{4}_{5}_{6}/miNet_h{7}L{8}_iter{9}_{10}_keepprob{11}/decode_beta{12}/fold{13}'.format(
        FLAGS.exp_dir,FLAGS.supervised_learning_rate,FLAGS.supervised_weight_decay,
        FLAGS.lstm_type, FLAGS.lstm_hidden_dim,FLAGS.lstm_activation,FLAGS.lstm_input_type,
        FLAGS.miNet_last_hidden_dim,FLAGS.miNet_num_hidden_layer,
        FLAGS.finetuning_epochs,FLAGS.miNet_common_acfun,FLAGS.keep_prob,FLAGS.decode_beta,FLAGS.fold+1,FLAGS.finetune_batch_size)
    
    FLAGS.miNet_train_model_dir = pjoin(FLAGS.miNet_train_dir, FLAGS.miNet_train_model_dir)
    #print('model dir:', FLAGS.model_dir)
    FLAGS.miNet_train_summary_dir = pjoin(FLAGS.miNet_train_dir, FLAGS.miNet_train_summary_dir)
    FLAGS._confusion_dir = pjoin(FLAGS.miNet_train_dir,FLAGS._confusion_dir)
    #FLAGS.log_dir = pjoin(FLAGS.miNet_train_dir,FLAGS.log_dir)
    FLAGS._ipython_console_txt = pjoin(FLAGS.miNet_train_dir,FLAGS._ipython_console_txt)
    
    FLAGS._logit_txt = pjoin(FLAGS.miNet_train_dir,FLAGS._logit_txt)
    FLAGS._train_logit_txt = pjoin(FLAGS.miNet_train_dir,FLAGS._train_logit_txt)
    FLAGS._test_logit_txt  = pjoin(FLAGS.miNet_train_dir,FLAGS._test_logit_txt) 

    FLAGS._dec_output_dir = pjoin(FLAGS.miNet_train_dir,FLAGS._dec_output_dir)
    FLAGS._dec_output_train_dir = pjoin(FLAGS.miNet_train_dir,FLAGS._dec_output_train_dir)
    FLAGS._dec_output_test_dir = pjoin(FLAGS.miNet_train_dir,FLAGS._dec_output_test_dir)
 
    FLAGS._key_player_dir = pjoin(FLAGS.miNet_train_dir,FLAGS._key_player_dir)


#print(FLAGS.__dict__)
raw_out(FLAGS)
experiment_out(FLAGS)

def main(unused_argv):
    fold = FLAGS.fold
    print("fold:", fold)
    optimizer = FLAGS.optimizer
    print("optimizer:", optimizer) 
    supervised_learning_rate = FLAGS.supervised_learning_rate
    print("supervised_learning_rate:", supervised_learning_rate)
    supervised_weight_decay = FLAGS.supervised_weight_decay
    print("supervised_weight_decay:", supervised_weight_decay)
    
# =============================================================================
#     print('before raw_out:', FLAGS.traj_file)
#     FLAGS.traj_file = raw_out(FLAGS.input_data_dir,FLAGS.traj_file)
#     print('after raw_out:', FLAGS.traj_file)
#     
#     print('before raw_out:', FLAGS.fold_file)
#     FLAGS.fold_file = raw_out(FLAGS.input_data_dir,FLAGS.fold_file)
#     print('after raw_out:', FLAGS.fold_file)
# =============================================================================
    ''' FLAGS is passed by reference '''
    raw_out(FLAGS)
    print(FLAGS.__dict__)
    experiment_out(FLAGS)
    print(FLAGS.__dict__)

# 使用这种方式保证了，如果此文件被其他文件 import的时候，不会执行main 函数
if __name__ == '__main__':
    tf.app.run()   # 解析命令行参数，调用main 函数 main(sys.argv) 