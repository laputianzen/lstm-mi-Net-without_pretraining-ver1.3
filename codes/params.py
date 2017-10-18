#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 23:58:13 2017

@author: PeterTsai
"""
import tensorflow as tf

def base(optimizer='Adam'):
    # load base setting, mainly folder name
    FLAGS = lambda: None
    FLAGS.exp_dir = 'experiment/'+optimizer
    FLAGS.summary_dir = FLAGS.exp_dir + '/summaries'
    FLAGS._ckpt_dir   = FLAGS.exp_dir + '/model'
    FLAGS._confusion_dir = FLAGS.exp_dir + '/confusionMatrix'
    FLAGS._result_txt = FLAGS.exp_dir + '/final_result.txt'    
    FLAGS.flush_secs  = 120
    
    # input dataset
    FLAGS.input_data_dir = 'raw'
    FLAGS.traj_file = FLAGS.input_data_dir + '/S_fixed.mat'
    FLAGS.fold_file = FLAGS.input_data_dir + '/split/tactic_bagIdxSplit5(1).mat'
    return FLAGS

#def load_dataset_info(FLAGS):
    
def load_lstm_info(FLAGS):
    FLAGS.lstm = lambda:None
    FLAGS.lstm.hidden_num = 256
    FLAGS.lstm.max_step_num = 450
    FLAGS.lstm.input_elem_num = 2
    FLAGS.lstm.iteration = 100 #5
    FLAGS.lstm.activation_lstm_name = 'softmax'#'relu6' # default tanh
    FLAGS.lstm.activation_lstm = tf.nn.softmax#tf.nn.relu6
    FLAGS.lstm.optimizer = tf.train.RMSPropOptimizer(0.003)

def load_tacticInfo(FLAGS):
    FLAGS.tacticName =['F23','EV','HK','PD','PT','RB','SP','WS','WV','WW']
    FLAGS.C5k_CLASS = [[0,1,2,3,5,7],[6,9],[4,8]]
    FLAGS.k = [3,2,5]
    FLAGS.playerMap = [[[1,1,1,0,0],[1,1,0,1,0],[1,1,0,0,1],[1,0,1,1,0],[1,0,1,0,1],
                           [1,0,0,1,1],[0,1,1,1,0],[0,1,1,0,1],[0,1,0,1,1],[0,0,1,1,1]],
                       [[1,1,0,0,0],[1,0,1,0,0],[1,0,0,1,0],[1,0,0,0,1],[0,1,1,0,0],
                           [0,1,0,1,0],[0,1,0,0,1],[0,0,1,1,0],[0,0,1,0,1],[0,0,0,1,1]],
                       [[1,1,1,1,1]]];
                        
    FLAGS.MAX_X = 326
    FLAGS.MAX_Y = 348
    
def normal():
    FLAGS = base()
    FLAGS.pretrain_batch_size = 2
    FLAGS.pretraining_epochs = 600
    FLAGS.finetune_batch_size = 2
    FLAGS.finetuning_epochs_epochs = 200
    FLAGS.pre_layer_learning_rate = []   
    FLAGS.keep_prob = 1.0
    
    return FLAGS


def develop():   
    FLAGS = base()
    FLAGS.pretrain_batch_size = 2
    FLAGS.flush_secs  = 120
    FLAGS.pretraining_epochs = 6#600
    FLAGS.finetune_batch_size = 2
    FLAGS.finetuning_epochs_epochs = 10
    
    FLAGS.pre_layer_learning_rate = []   
    FLAGS.keep_prob = 1.0
    
    return FLAGS

def showProperties(obj):
    properties = obj.__dict__
    for i in properties:
        print('{0}: {1}'.format(i, properties[i]))     