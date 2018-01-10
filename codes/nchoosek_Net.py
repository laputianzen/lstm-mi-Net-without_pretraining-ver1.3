#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 01:20:28 2017

@author: PeterTsai
"""
import tensorflow as tf

def genearateTFCombs(tf_output,np_combs,player_num):
    tf_batch_size = tf.divide(tf.shape(tf_output)[0],tf.constant(player_num))
    index = tf.range(0, tf_batch_size) * player_num
    CC = []
    k = len(np_combs[0])
    for c in range(len(np_combs)*k):
        CC.append(index)
    CC = tf.stack(CC,axis=0)
    CC = tf.transpose(tf.reshape(CC,[len(np_combs),k,-1]),perm=(2,0,1))
    t = tf.stack([tf.cast(tf_batch_size,tf.int32), 1,1])
    tf_np_combs = tf.tile(tf.expand_dims(tf.constant(np_combs),axis=0),t)
#    np_cc = np.stack(np_combs,axis=0)
#    arrays = [np_cc  for _ in range(np_batch_size)]
#    np_cc = np.stack(arrays, axis=0).astype(np.int32)
    combs = tf.add(tf_np_combs,tf.cast(CC,tf.int32))
    return combs

def nchoosek_grouping(ae,np_nchoosek,numPlayer,nk_pooling):
    C53_combs = np_nchoosek[0]
    C52_combs = np_nchoosek[1]
    C55_combs = np_nchoosek[2]
    with tf.name_scope("nchoosek_group_inputs"):
        tf_C53_combs = genearateTFCombs(ae.last_output,C53_combs,numPlayer)
        tf_C52_combs = genearateTFCombs(ae.last_output,C52_combs,numPlayer)
        tf_C55_combs = genearateTFCombs(ae.last_output,C55_combs,numPlayer)

        C53_input = tf.gather(ae.last_output,tf_C53_combs)
        C52_input = tf.gather(ae.last_output,tf_C52_combs)
        C55_input = tf.gather(ae.last_output,tf_C55_combs)
    
        # [batch_num, nchoosek_num, k_num, feature_dim]
        C53_input_merge = pooling(C53_input, nk_pooling.split('_'),name='C53_{}'.format(nk_pooling))
        C52_input_merge = pooling(C52_input, nk_pooling.split('_'),name='C52_{}'.format(nk_pooling))
        C55_input_merge = pooling(C55_input, nk_pooling.split('_'),name='C55_{}'.format(nk_pooling))

        
# =============================================================================
#         C53_input_merge = tf.reduce_mean(C53_input,axis=2,name='C53')
#         C52_input_merge = tf.reduce_mean(C52_input,axis=2,name='C52')
#         C55_input_merge = tf.reduce_mean(C55_input,axis=2,name='C55')
# =============================================================================
        nchoosek_inputs = [C53_input_merge, C52_input_merge, C55_input_merge]
        local_vars = {'tf_C53_combs':tf_C53_combs,'tf_C52_combs':tf_C52_combs,
                      'tf_C55_combs':tf_C55_combs,'C53_input':C53_input,
                      'C52_input':C52_input,'C55_input':C55_input}
        
    return nchoosek_inputs, local_vars

def pooling(inputs,pool_type,name=None):
    inputs_dict = {'avg':tf.reduce_mean(inputs,axis=2),
                   'max':tf.reduce_max(inputs,axis=2),
                   'min':tf.reduce_min(inputs,axis=2)}
    input_select=[inputs_dict[x] for x in pool_type]
    input_merge = tf.concat(input_select,axis=2,name=name)
    
    
    return input_merge