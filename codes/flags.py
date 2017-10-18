#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 16:34:33 2017

@author: PeterTsai
"""
import tensorflow as tf

def home_out(path):
  return pjoin(os.environ['HOME'], 'tmp', 'mnist', path)


flags = tf.app.flags
FLAGS = flags.FLAGS

# Directories
#flags.DEFINE_string('exp_dir',

flags.DEFINE_string('summary_dir', home_out('summaries'),
                    'Directory to put the summary data')

flags.DEFINE_string('_ckpt_dir', home_out('model'),
                    'Directory to put the model checkpoints')
        
flags.DEFINE_string('_confusion_dir', home_out('confusionMatrix'),
                    'Directory to put the confusionMatrix')

flags.DEFINE_string('_result_txt', home_out('final_result.txt'),
                    'final result text ')

# LSTM Architecture Specific Flags
flags.DEFINE_integer("lstm_hidden_dim", 256, "LSTM hidden state dimension")

flags.DEFINE_integer("lstm_max_sequence_length", 450, "LSTM max sequence length")

flags.DEFINE_integer("lstm_input_dim", 2, "LSTM input dimension")

flags.DEFINE_integer("lstm_pretrain_iteration", 100, "LSTM max sequence length")

flags.DEFINE_string('lstm_activation', tf.nn.softmax,'lstm activation function')