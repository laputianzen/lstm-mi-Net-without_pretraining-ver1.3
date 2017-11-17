#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 14:49:27 2017

@author: PeterTsai
"""
#for name in dir():
#    if not name.startswith('_'):
#        del globals()[name]
import sys
#import tensorflow as tf
#from utils.flags import FLAGS

sys.path.append("./codes")
import flags
from fixed_flags import FLAGS1
from flags import FLAGS


def main():
    vars_FLAGS1 = vars(FLAGS1)
    #print(vars_FLAGS1)
# =============================================================================
#     if not 'FLAGS' in globals():
#         from flags import FLAGS
#     else:
#         vars_FLAGS  = vars(FLAGS)
#         if not vars_FLAGS['__flags']:
#             from flags import FLAGS
# =============================================================================
    flags.raw_out(FLAGS)
    flags.experiment_out(FLAGS)
    vars_FLAGS = vars(FLAGS)['__flags']
    #print(vars_FLAGS)


    diff_key_FLAGS = set(vars_FLAGS) - set(vars_FLAGS1)
    print('Exist in flags.py, lack in fixed.flag.py: %s' % str(diff_key_FLAGS) )
    print(' ')
    diff_key_FLAGS1= set(vars_FLAGS1) - set(vars_FLAGS)
    print('vice versa: %s' % str(diff_key_FLAGS1) )
    
    intersect_key = set.intersection(set(vars_FLAGS),set(vars_FLAGS1))
    print(' ')
    print('checking intersection vars value...')
    for key in intersect_key:
        try:
            vars_FLAGS[key] is vars_FLAGS1[key]
            #print('%s content is different in flags.py and fixed_flags.py' %key)
        except:
            print('%s in flags.py: %s' %(key,vars_FLAGS[key]))
            print('%s in fixed_flags.py: %s' %(key,vars_FLAGS1[key]))
    print('Done!!')
    #compare flags.py to fixed_flags.py
# =============================================================================
#     for key, val in vars_FLAGS1.items():
#         try:
#             vars_FLAGS[key]
#         except:
#             print('No %s exists in flags.py!!' %key)
#         else:
#             if vars_FLAGS[key] is vars_FLAGS1[key]:
#                 print('%s content is different in flags.py and fixed_flags.py' %key)
#             else:
#                 print('has same value')
# =============================================================================
        #print('key: %s,val: %s' %(key, val))
    
    #num_hidden_layers = FLAGS.num_hidden_layers
    #print("num_hidden_layers", num_hidden_layers)
    #hidden1_units = FLAGS.hidden1_units 
    #print("hidden1_units", hidden1_units)
    #activation = FLAGS.lstm_activation
    #print("activation op", activation)

# 使用这种方式保证了，如果此文件被其他文件 import的时候，不会执行main 函数
#if __name__ == '__main__':
#    #main()
#    tf.app.run()   # 解析命令行参数，调用main 函数 main(sys.argv) 
    
main()