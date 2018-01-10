#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 09:50:32 2017

@author: PeterTsai
"""

import time
import tensorflow as tf
import numpy as np
import pandas as pd 

import utils


def block_diagonal(matrices, dtype=tf.float32):
  r"""Constructs block-diagonal matrices from a list of batched 2D tensors.

  Args:
    matrices: A list of Tensors with shape [..., N_i, M_i] (i.e. a list of
      matrices with the same batch dimension).
    dtype: Data type to use. The Tensors in `matrices` must match this dtype.
  Returns:
    A matrix with the input matrices stacked along its main diagonal, having
    shape [..., \sum_i N_i, \sum_i M_i].

  """
  matrices = [tf.convert_to_tensor(matrix, dtype=dtype) for matrix in matrices]
  blocked_rows = tf.Dimension(0)
  blocked_cols = tf.Dimension(0)
  batch_shape = tf.TensorShape(None)
  for matrix in matrices:
    full_matrix_shape = matrix.get_shape().with_rank_at_least(2)
    batch_shape = batch_shape.merge_with(full_matrix_shape[:-2])
    blocked_rows += full_matrix_shape[-2]
    blocked_cols += full_matrix_shape[-1]
  ret_columns_list = []
  for matrix in matrices:
    matrix_shape = tf.shape(matrix)
    ret_columns_list.append(matrix_shape[-1])
  ret_columns = tf.add_n(ret_columns_list)
  row_blocks = []
  current_column = 0
  for matrix in matrices:
    matrix_shape = tf.shape(matrix)
    row_before_length = current_column
    current_column += matrix_shape[-1]
    row_after_length = ret_columns - current_column
    row_blocks.append(tf.pad(
        tensor=matrix,
        paddings=tf.concat(
            [tf.zeros([tf.rank(matrix) - 1, 2], dtype=tf.int32),
             [(row_before_length, row_after_length)]],
            axis=0)))
  blocked = tf.concat(row_blocks, -2)
  blocked.set_shape(batch_shape.concatenate((blocked_rows, blocked_cols)))
  return blocked

def withBag_similarity(tactic_pred,tactic_gt,y_maxInsts,inputs,similarity_type):
    #pairwise_l2_norm(inputs[0])
    #batch_size = tf.cast(tf.shape(Y_placeholder)[0],tf.int32)
    batch_size = tf.shape(tactic_pred)[0]
    feature_dim = int(inputs[0].shape[2])
    num_tactics = int(tactic_pred.shape[1])
    pad_inputs = inputs[:]
    pad_inputs.insert(0,tf.zeros([batch_size,1,feature_dim]))
    tactic_correct = tf.multiply(tactic_pred, tactic_gt)
    totalNumInst = 0
    for n in range(len(inputs)):
        totalNumInst = totalNumInst + int(inputs[n].shape[1])
    pred_relate_inst_idx = tf.reduce_sum(tf.multiply(tf.cast(y_maxInsts+1,tf.float32),tactic_correct),axis=1)
    pred_relate_inst_mask =  tf.one_hot(tf.cast(pred_relate_inst_idx,tf.int32),totalNumInst+1)
    
    matrices = []
    '''
    for n in range(len(pad_inputs)):
        alternates = tf.map_fn(pairwise_l2_norm,pad_inputs[n])
        matrices.append(alternates)
        #i = tf.constant(0)
        #c = lambda i: tf.less(i,batch_size)
        #b = lambda i: pairwise_l2_norm(inputs[n][i])
        #r = tf.while_loop(c,b,[i])
        #for b in range(batch_size):
        #    m = pairwise_l2_norm(inputs[n][b])
    bag_sim = block_diagonal(matrices)
    total_bag_sim = tf.reduce_mean(bag_sim)
    sum_bag_sim_along_one_inst = tf.reduce_sum(bag_sim,axis=2) /totalNumInst
    within_bag_sim_along_top_inst = tf.reduce_mean(tf.reduce_sum(sum_bag_sim_along_one_inst*pred_relate_inst_mask,axis=1))
    '''
    for n in range(len(pad_inputs)):
        alternates = tf.map_fn(pairwise_l2_norm,pad_inputs[n])   
        
        A_width = tf.cast(tf.shape(alternates)[0],tf.float32)
        A_size = A_width*A_width - A_width  
        matrices.append(alternates/A_size)
    bag_sim = block_diagonal(matrices) 
    total_bag_sim = tf.reduce_mean(bag_sim)
    sum_bag_sim_along_one_inst = tf.reduce_sum(bag_sim,axis=2)
    within_bag_sim_along_top_inst = tf.reduce_mean(tf.reduce_sum(sum_bag_sim_along_one_inst*pred_relate_inst_mask,axis=1))
    #m = pairwise_l2_norm(inputs[n])
    #matrices.append(m)
    local_vars = {'bag_sim':bag_sim,'y_maxInsts':y_maxInsts,
                  'pred_relate_inst_idx':pred_relate_inst_idx,
                  'pred_relate_inst_mask':pred_relate_inst_mask,
                  'sum_bag_sim_along_one_inst':sum_bag_sim_along_one_inst,
                  'within_bag_sim_along_top_inst':within_bag_sim_along_top_inst,
                  'tactic_correct':tactic_correct,
                  'total_bag_sim':total_bag_sim,'pad_inputs':pad_inputs}
    #block_diag = block_diagonal(matrices)
    
    bag_sim_val = tf.where(tf.equal(str(similarity_type),str('top')),within_bag_sim_along_top_inst, total_bag_sim)
    #if similarity_type == 'top':
    #    bag_sim_val = within_bag_sim_along_top_inst
    #elif similarity_type == 'all':
    #    bag_sim_val = total_bag_sim
        
    return bag_sim_val,local_vars

def pairwise_l2_norm(A):
    r = tf.reduce_sum(A*A,axis=1)
    r = tf.reshape(r,[-1,1])
    D = r - 2*tf.matmul(A, tf.transpose(A)) + tf.transpose(r)
    
    class_size = tf.shape(A)[0]
    diag_mask = tf.ones_like(D) - tf.diag(tf.ones([class_size]))
    D = D * diag_mask
    return D / tf.cast(class_size,tf.float32)

def pairwise_l2_norm2(A,B):
    rA = tf.reduce_sum(A**2, axis=1, keep_dims = True)
    rB = tf.reduce_sum(B**2, axis=1, keep_dims = True)
    
    D = rA - 2*A*tf.transpose(B) -tf.transpose(rB)
    
    return D

#def generate_correct_mask(tactic_pred,tactic_gt):
#    true_label = tf.reduce_max(tactic_gt)
#    false_label= tf.reduce_min(tactic_gt)
    
    
    

def generate_mask_feature(tactic_pred,tactic_gt,y_maxInsts,inputs):
    tactic_correct = tf.multiply(tactic_pred, tactic_gt)
    
    batch_size = tf.shape(tactic_pred)[0]
    feature_dim = int(inputs[0].shape[2])
    num_tactics = int(tactic_pred.shape[1])
    totalNumInst = 0
    for n in range(len(inputs)):
        totalNumInst = totalNumInst + int(inputs[n].shape[1])

    pred_relate_inst_idx = tf.reduce_sum(tf.multiply(tf.cast(y_maxInsts+1,tf.float32),tactic_correct),axis=1)
    pred_relate_inst_mask =  tf.one_hot(tf.cast(pred_relate_inst_idx,tf.int32),totalNumInst+1)
    feature_mask = tf.tile(tf.expand_dims(pred_relate_inst_mask,axis=2),[1,1,feature_dim])
    pad_inputs = inputs[:]
    pad_inputs.insert(0,tf.zeros([batch_size,1,feature_dim]))
    input_feature = tf.concat(pad_inputs,1)

    tactic_pred_feature_mask = tf.tile(tf.expand_dims(tactic_correct,axis=2),[1,1,feature_dim])
    tactic_pred_feature_mask = tf.transpose(tactic_pred_feature_mask,perm=[1,0,2])
    
    mask_feature = tf.reduce_sum(input_feature * feature_mask,axis=1)
    tile_mask_feature = tf.tile(tf.expand_dims(mask_feature,axis=0),[num_tactics,1,1])
    expand_mask_feature = tf.multiply(tile_mask_feature,tactic_pred_feature_mask)

    return expand_mask_feature    

def inst_similarity(tactic_pred,tactic_gt,y_maxInsts,inputs):

    num_tactics = int(tactic_pred.shape[1])
    tactic_correct = tf.multiply(tactic_pred, tactic_gt)
    numCorrectEachTactic = tf.reduce_sum(tactic_correct,axis=0)
    expand_mask_feature = generate_mask_feature(tactic_pred,tactic_gt,y_maxInsts,inputs)
    
    feature_list = tf.split(expand_mask_feature,num_tactics)
    
    batch_mask = tf.matmul(tactic_correct,tf.transpose(tactic_correct))
    tmp_result = []
    tmp_unmaskD =[]
    tmp_D       =[]
    for l in range(num_tactics):
        A = tf.squeeze(feature_list[l],axis=0)        
        #tf.boolean_mask(A,tf.cast(tactic_correct,tf.bool))
        unmaskD = pairwise_l2_norm(A)
        tmp_unmaskD.append(unmaskD)
        D = batch_mask * unmaskD
        tmp_D.append(D)
        
        D_width = tf.cast(tf.shape(D)[0],tf.float32)
        D_size = D_width*D_width - D_width        
        tmp_result.append(tf.reduce_sum(D)/D_size)
        
    sim_vals = tf.stack(tmp_result,axis=0)
    sim_unmaskD= tf.stack(tmp_unmaskD,axis=0)
    sim_D    = tf.stack(tmp_D,axis=0)
    local_vars = {'sim_vals':sim_vals,'tactic_correct':tactic_correct,'sim_unmaskD':sim_unmaskD,'sim_D':sim_D,'batch_mask':batch_mask,'numCorrectEachTactic':numCorrectEachTactic,'expand_mask_feature':expand_mask_feature}
    
    return sim_vals,local_vars

#def inst_similarity(tactic_pred,tactic_gt,y_maxInsts,inputs):
#    batch_size = tf.shape(tactic_pred)[0]
#    feature_dim = int(inputs[0].shape[2])
#    num_tactics = int(tactic_pred.shape[1])
#    totalNumInst = 0
#    for n in range(len(inputs)):
#        totalNumInst = totalNumInst + int(inputs[n].shape[1])
#        
#    tactic_correct = tf.multiply(tactic_pred, tactic_gt)
#    inst_idx = tf.boolean_mask(y_maxInsts,tf.cast(tactic_correct,tf.bool))

def default():
    mean_sim_variance = 0.0
    local_vars = {'mean_sim_variance':mean_sim_variance} 
    return local_vars

def inst_similarity_unit(tactic_pred,tactic_gt,y_maxInsts,inputs):
    tactic_correct = tf.multiply(tactic_pred, tactic_gt)
    numCorrectEachTactic = tf.reduce_sum(tactic_correct,axis=0) 

    batch_size = tf.shape(tactic_pred)[0]
    feature_dim = int(inputs[0].shape[2])
    num_tactics = int(tactic_pred.shape[1])
    totalNumInst = 0
    for n in range(len(inputs)):
        totalNumInst = totalNumInst + int(inputs[n].shape[1])

    pred_relate_inst_idx = tf.reduce_sum(tf.multiply(tf.cast(y_maxInsts+1,tf.float32),tactic_correct),axis=1)
    pred_relate_inst_mask =  tf.one_hot(tf.cast(pred_relate_inst_idx,tf.int32),totalNumInst+1)
    feature_mask = tf.tile(tf.expand_dims(pred_relate_inst_mask,axis=2),[1,1,feature_dim])
    inputs.insert(0,tf.zeros([batch_size,1,feature_dim]))
    input_feature = tf.concat(inputs,1)

    tactic_pred_feature_mask = tf.tile(tf.expand_dims(tactic_correct,axis=2),[1,1,feature_dim])
    tactic_pred_feature_mask = tf.transpose(tactic_pred_feature_mask,perm=[1,0,2])
    
    mask_feature = tf.reduce_sum(input_feature * feature_mask,axis=1)
    tile_mask_feature = tf.tile(tf.expand_dims(mask_feature,axis=0),[num_tactics,1,1])
    expand_mask_feature = tf.multiply(tile_mask_feature,tactic_pred_feature_mask)
    
    ##temp_factor = tf.where(tf.is_inf(tf.cast(batch_size,tf.float32) / (numCorrectEachTactic)),
    ##                       tf.zeros_like(numCorrectEachTactic),tf.cast(batch_size,tf.float32) / (numCorrectEachTactic))
    ##factor = tf.tile(tf.expand_dims(temp_factor,axis=1) ,[1,feature_dim])
    #factor = tf.tile(tf.expand_dims(tf.cast(batch_size,tf.float32) / (numCorrectEachTactic),axis=1) ,[1,feature_dim])
    
    mean, variances = tf.nn.moments(expand_mask_feature,[1])
    
    ##variance = tf.reduce_sum(factor * variances + factor*(1-factor)*tf.square(mean),axis=1) 
    variance = tf.reduce_sum(variances,axis=1)
    
    numTacticsCorrect =tf.cast(tf.count_nonzero(variance),tf.float32)
    
    mean_sim_variance = tf.where(tf.cast(numTacticsCorrect,tf.bool),tf.reduce_sum(variance)/numTacticsCorrect,0.0)
    #variance = tf.where(tf.less_equal(numCorrectEachTactic,tf.ones_like(numCorrectEachTactic)),
    #                   tf.zeros_like(numCorrectEachTactic) , temp_variance)
                        
    
    #variance = tf.where(tf.less_equal(numCorrectEachTactic,tf.ones_like(numCorrectEachTactic)),
    #                   tf.zeros_like(numCorrectEachTactic) , 2.0*tf.ones_like(numCorrectEachTactic))
    ##local_vars = {'tactic_correct':tactic_correct,'pred_relate_inst_idx':pred_relate_inst_idx,
    ##              'pred_relate_inst_mask':pred_relate_inst_mask,'input_feature':input_feature,
    ##              'feature_mask':feature_mask,'mask_feature':mask_feature,
    ##              'tactic_pred_feature_mask':tactic_pred_feature_mask,
    ##              'tile_mask_feature':tile_mask_feature,'expand_mask_feature':expand_mask_feature,
    ##              'numCorrectEachTactic':numCorrectEachTactic,
                  #'factor':factor,
    ##              'mean':mean, 'variances':variances,
                  #'temp_variance':temp_variance,
                  #'temp_factor':temp_factor,
    ##              'variance':variance,
    ##              'numTacticsCorrect':numTacticsCorrect, 'mean_sim_variance':mean_sim_variance} 
    local_vars = {'mean_sim_variance':mean_sim_variance} 
    return local_vars

def inst_similarity_cond(tactic_pred,tactic_gt,y_maxInst,inputs,isTraining=False):
    result = tf.cond(tf.equal(isTraining, tf.constant(True)),lambda: default(),lambda: inst_similarity_unit(tactic_pred,tactic_gt,y_maxInst,inputs))
    mean_sim_variance = result['mean_sim_variance']
    local_vars = result
    return mean_sim_variance, local_vars

def inst_similarity_unstable(tactic_pred,tactic_gt,y_maxInsts,inputs):
    # regularization (instance intra-similarity)
    ''' tactic_pred [batch, num_tactics]: tactic prediction, in one-hot format
        tactic_gt   [batch, num_tactics]: tactic ground truth, in one-hot format    
        y_maxInsts  [batch, num_tactics]: max insts of each tactics, save its inst's index 
        inputs list of mi-subnet [batch, num_insts, feature_dim] '''
    batch_size = tf.shape(tactic_pred)[0]
    feature_dim = int(inputs[0].shape[2])
    num_tactics = int(tactic_pred.shape[1])
    totalNumInst = 0
    for n in range(len(inputs)):
        totalNumInst = totalNumInst + int(inputs[n].shape[1])
    #pred_related_inst_idx = tf.multiply(tf.cast(y_maxInsts,tf.float32), tactic_pred)
    
    ''' tactic_correct [batch, num_tactics]: correct when prediction matches ground truth
        pred_relate_inst_idx [batch] : max instance index in correct tactic prediction
        pred_relate_inst_mask [batch, num_tactics, totalNumInst+1]: transform y_maxInsts to one-hot format, one inst is negative
        feature_mask [batch, totalNumInst+1, feature_dim]: if instIdx is gt, mask is 1 vector; otherwise, 0 vector
        input_feature [batch, totalNumInst, feature_dim]: concat subnet instance feature
        mask_feature [batch, feature_dim] : correct prediction instance feature
    '''
    tactic_correct = tf.multiply(tactic_pred, tactic_gt)
    pred_relate_inst_idx = tf.reduce_sum(tf.multiply(tf.cast(y_maxInsts+1,tf.float32),tactic_correct),axis=1)
    pred_relate_inst_mask =  tf.one_hot(tf.cast(pred_relate_inst_idx,tf.int32),totalNumInst+1)
    feature_mask = tf.tile(tf.expand_dims(pred_relate_inst_mask,axis=2),[1,1,feature_dim])
    inputs.insert(0,tf.zeros([batch_size,1,feature_dim]))
    input_feature = tf.concat(inputs,1)
    mask_feature = tf.reduce_sum(input_feature * feature_mask,axis=1)
    
    ''' tactic_pred_feature_mask [num_tactics, batch, feature_dims]
        : copy correct prediction along feature dim. '''
    tactic_pred_feature_mask = tf.tile(tf.expand_dims(tactic_correct,axis=2),[1,1,feature_dim])
    tactic_pred_feature_mask = tf.transpose(tactic_pred_feature_mask,perm=[1,0,2])
    #matrix = []
    #for d in range(feature_dim):
    #    matrix.append(tactic_pred)

    #mmatrix = tf.transpose(tf.stack(matrix,axis=2),perm=[1,0,2])
    #XX = []
    #for t in range(num_tactics):
    #    XX.append(tf.multiply(mmatrix[t],mask_feature))
    #expand_mask_feature = tf.stack(XX,axis=0)
    ''' expand_mask_feature [num_tactics, batch_size, feature_dim]: 
        if tactic_idx == correct, feature is max instance input feature, otherwise is 0 vector
    '''
    tile_mask_feature = tf.tile(tf.expand_dims(mask_feature,axis=0),[num_tactics,1,1])
    expand_mask_feature = tf.multiply(tile_mask_feature,tactic_pred_feature_mask )
    
    '''    
        mean [tactic, feature_dim]: mean vector for each tactic (only correct prediction)
        variance [tactic] : scalar value for each tactic ie. sum((x-mean)^2)/(correctEachTactic*feature_dims)
    ''' 
    numCorrectEachTactic = tf.reduce_sum(tactic_correct,axis=0) # [num_tactics]
    expand_numCorrectEachTactic = tf.tile(tf.expand_dims(numCorrectEachTactic,axis=1),[1,feature_dim]) #[num_tactics,feature_dim]
    accum = tf.reduce_sum(expand_mask_feature,axis=1)
    meanNan = accum/ expand_numCorrectEachTactic #[num_tactics, feature_dim]
    mean = tf.where(tf.is_nan(meanNan),tf.zeros_like(meanNan),meanNan)
    expand_mean = tf.tile(tf.expand_dims(mean,axis=1),[1,batch_size,1]) #[num_tactics,batch,feature_dim]
    
    squared_diff = tf.squared_difference(expand_mask_feature,expand_mean) #[num_tactics,batch,feature_dim]
    #sum_squared_diff = tf.reduce_sum(squared_diff,axis=2)
    hits = tf.cast(tf.count_nonzero(expand_mask_feature,axis=2),tf.bool) #[num_tactics,batch]
    hits = tf.tile(tf.expand_dims(hits,axis=2),[1,1,feature_dim])
    #squared_diff [batch,feature]
    actual_squared_diff = tf.where(hits,squared_diff,tf.zeros([num_tactics,batch_size,feature_dim]))
    sum_actual_squared_diff = tf.reduce_sum(actual_squared_diff,axis=2)
    varianceNan = tf.divide(tf.reduce_sum(sum_actual_squared_diff,axis=1), numCorrectEachTactic) # can also divide by feature_dim
    variance = tf.where(tf.is_nan(varianceNan),tf.zeros_like(varianceNan),varianceNan)


    local_vars = {'tactic_correct':tactic_correct,'pred_relate_inst_idx':pred_relate_inst_idx,
                  'pred_relate_inst_mask':pred_relate_inst_mask,'input_feature':input_feature,
                  'feature_mask':feature_mask,'mask_feature':mask_feature,
                  'tactic_pred_feature_mask':tactic_pred_feature_mask,
                  'tile_mask_feature':tile_mask_feature,'expand_mask_feature':expand_mask_feature,
                  'numCorrectEachTactic':numCorrectEachTactic,'expand_numCorrectEachTactic':expand_numCorrectEachTactic,
                  'accum':accum,'meanNan':meanNan,'mean':mean,'expand_mean':expand_mean,'squared_diff':squared_diff,
                  #'sum_squared_diff':sum_squared_diff,
                  'hits':hits,'actual_squared_diff':actual_squared_diff,'sum_actual_squared_diff':sum_actual_squared_diff,
                  'varianceNan':varianceNan,'variance':variance}
    
    #local_vars = tactic_correct,pred_relate_inst_mask,input_feature,feature_mask,mask_feature,
    #              pred_relate_inst_idx,tactic_pred_feature_mask,tile_mask_feature,expand_mask_feature,
    #              numCorrectEachTactic,expand_numCorrectEachTactic,accum,meanNan,mean,
    #              expand_mean,squared_diff,hits,actual_squared_diff,varianceNan,variance]

    return variance, local_vars

def loss_x_entropy(output, target):
    """Cross entropy loss
    
    See https://en.wikipedia.org/wiki/Cross_entropy

    Args:
        output: tensor of net output
        target: tensor of net we are trying to reconstruct
    Returns:
        Scalar tensor of cross entropy
        
"""
    with tf.name_scope("xentropy_loss"):
        net_output_tf = tf.convert_to_tensor(output, name='input')
        target_tf = tf.convert_to_tensor(target, name='target')
        cross_entropy = tf.add(tf.multiply(tf.log(net_output_tf, name='log_output'),
                                           target_tf),
                             tf.multiply(tf.log(1 - net_output_tf),
                                    (1 - target_tf)))
        return -1 * tf.reduce_mean(tf.reduce_sum(cross_entropy, 1),
                                   name='xentropy_mean')

def multiClassEvaluation(logits, labels):
    """Evaluate the quality of the logits at predicting the label.
    
    Args:
        logits: Logits tensor, float - [batch_size, NUM_CLASSES].
        labels: Labels tensor, int32 - [batch_size], with values in the
            range [0, NUM_CLASSES).

    Returns:
        A scalar int32 tensor with the number of examples (out of batch_size)
        that were predicted correctly.
        """    
    correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(labels,1))
    accu  = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    error = tf.subtract(1.0,accu)
    return accu, error

def calulcutePAccuTF(Y,y_playerPool,Y_placeholder,y_placeholder):
    bool_YMask = tf.one_hot(tf.argmax(Y,axis=-1),int(Y.shape[1]),axis=-1)
    bool_YMask = tf.cast(bool_YMask,tf.bool)
    y = tf.boolean_mask(y_playerPool,bool_YMask)        
    correct_prediction = tf.equal(tf.argmax(Y,axis=1), tf.argmax(Y_placeholder,axis=1))
    y_correctY = tf.boolean_mask(y,correct_prediction)
    y_placeholder_correctY = tf.cast(tf.boolean_mask(y_placeholder,correct_prediction),tf.int32)
    y_correct = tf.equal(y_correctY,y_placeholder_correctY)
    NUM_PLAYER = int(y_placeholder.shape[1])
    y_accu = tf.reduce_sum(tf.cast(y_correct,tf.float32))/(tf.reduce_sum(tf.cast(correct_prediction,tf.float32))*NUM_PLAYER)
    return y_accu, y
      
def calculatePAccu(y_pred,y_label,Y_pred,Y_label):
    NUM_PLAYER = y_pred.shape[1]
    Y_correct_map = Y_pred * Y_label
    y_correct_map = np.equal(y_pred, y_label)
    y_correct_per_tactic = np.matmul(np.transpose(Y_correct_map),y_correct_map)
    Y_correct_per_tactic = np.sum(Y_correct_map,axis=0)
    y_accu_Y_correct_per_tactic = np.nan_to_num(np.sum(y_correct_per_tactic,axis=1) / (Y_correct_per_tactic * NUM_PLAYER))
    avg_y_accu_Y_correct_per_tactic = np.mean(y_accu_Y_correct_per_tactic)
    return y_accu_Y_correct_per_tactic, avg_y_accu_Y_correct_per_tactic

def calculateAccu(Y_pred,inst_pred,test_Y,test_label,dataset):
    
    num_k = np.zeros(len(dataset.tacticName),dtype=np.int8)
    IdxMap = []
    tmp = []
    k = 0
    boundary = len(dataset.C5k_CLASS[0])
    for idx in range(len(dataset.tacticName)):
        if idx >= boundary:
            k = k + 1            
            boundary = boundary + len(dataset.C5k_CLASS[k])
            IdxMap.append(tmp)
            tmp = []
        num_k[idx] = k
        tmp.append(idx)
    IdxMap.append(tmp)
    
    KP_pred = np.zeros((len(Y_pred),5))
    for bagIdx in range(len(Y_pred)):
        k = num_k[Y_pred[bagIdx]]
        c = IdxMap[k].index(Y_pred[bagIdx])
        kinst = np.argmax(inst_pred[k][:,bagIdx,c])
        KP_pred[bagIdx] = dataset.playerMap[k][kinst]
# =============================================================================
#     for bagIdx in range(len(Y_pred)):
#         for k in range(len(dataset.k)):
#             if Y_pred[bagIdx] in dataset.C5k_CLASS[k]:
#                 c = dataset.C5k_CLASS[k].index(Y_pred[bagIdx])
#                 kinst = np.argmax(inst_pred[k][:,bagIdx,c])
#                 #kinst = np.argmax(inst_pred[k][bagIdx,:,c])
#                 KP_pred[bagIdx] = dataset.playerMap[k][kinst]
# =============================================================================
                
    Y_correct = np.equal(Y_pred,np.argmax(test_Y,1))
    bagAccu = np.sum(Y_correct) / Y_pred.size
    
    y_correct = np.equal(KP_pred[Y_correct,:],test_label[Y_correct,:])
    
    pAccu = np.sum(y_correct) / KP_pred[Y_correct,:].size
    #print(y_correct)
    #print('bag accuracy %.5f, inst accuracy %.5f' %(bagAccu, pAccu))
    time.sleep(1)
    return bagAccu, pAccu   
   
        
def ConfusionMatrix(logits,labels,dataset,filename,text_file):
    C = np.zeros((len(dataset.tacticName),len(dataset.tacticName)))
    CM = np.zeros((len(dataset.tacticName)+1,len(dataset.tacticName)+1))
    ''' convert tactic from network order to dataset order '''
    reorder = np.concatenate(dataset.C5k_CLASS).ravel().tolist()
    orderTactic = [dataset.tacticName[i] for i in reorder]
    #orderTactic = ['F23','EV','HK','PD','RB','WS','SP','WW','PT','WV']
    tacticNum = len(orderTactic)
    for bagIdx in range(len(labels)):
        gt = np.argmax(labels[bagIdx])
        pred = np.argmax(logits[bagIdx])
        #pred = logits[bagIdx]
        C[gt,pred] = C[gt,pred] + 1    

    gtC = np.sum(C,axis=1)
    predC=np.sum(C,axis=0)
    TP = np.diag(C)
    precision = TP/predC  
    
    recall = TP/gtC
    CM[0:tacticNum,0:tacticNum] = C
    CM[tacticNum,0:tacticNum] = precision
    CM[0:tacticNum,tacticNum] = recall.T
    CM[tacticNum,tacticNum] = np.sum(TP)/np.sum(C)*100
    rowIdx = orderTactic + ['Precision']
    colIdx = orderTactic + ['Recall']
    df = pd.DataFrame(CM,index=rowIdx,columns=colIdx)
    
    utils.printLog(text_file,df.round(3).to_string())
    #print(df.round(3))
    #text_file.write(df.round(3).to_string())
    #print(C)
    #text_file.write(np.array2string(C))
    if filename is not None:
        df.round(3).to_csv(filename,na_rep='NaN')     

def keyPlayerResult(Y_pred,Y_label,y_pred,dataset,phase,text_file,log_file,info_file):
    if phase == 'train':
        selectIndex = dataset.trainIdx
    else:
        selectIndex = dataset.testIdx
    NAME =[]
    for i in range(len(selectIndex)):
        hit = [(selectIndex[i] in dataset.videoIndex[r]) for r in range(len(dataset.videoIndex))]
        loc = hit.index(True)
        tacticName = dataset.tacticName[loc]
        string = tacticName + "-" + selectIndex[i].astype(str) + ":"
        NAME.append(string)
    #nv_reorder = [0,1,2,3,8,4,6,5,9,5,7]     
    #[[0,1,2,3,5,7],[6,9],[4,8]]
    
    reorder = np.concatenate(dataset.C5k_CLASS).ravel().tolist()
    inv_reorder = [reorder.index(i) for i in range(len(dataset.tacticName))]
    PRED_NAME =[]    
    
    for i in range(len(selectIndex)):
        pred_loc = Y_pred[i].tolist().index(1)
        tacticName = dataset.tacticName[inv_reorder[pred_loc]]
        PRED_NAME.append(tacticName) 
       
    rolePlayerMap = dataset.gtRoleOrder[selectIndex]
    Y_correct_map = Y_pred * Y_label
    Y_correct = 2 * np.sum(Y_correct_map,axis=-1,keepdims=True) - 1 #correct=1, error=-1
    NUM_PLAYER = rolePlayerMap.shape[1]
    keyPlayers = y_pred * (rolePlayerMap+1) * np.tile(Y_correct,[1,NUM_PLAYER])
    DAT = np.column_stack((PRED_NAME, keyPlayers.astype(int)))
    df = pd.DataFrame(DAT,index=NAME,columns=['Y_pred','p1','p2','p3','p4','p5'])
    if text_file is not None:
        #np.savetxt(text_file, DAT, delimiter=" ", fmt="%s")
        df.to_csv(text_file,na_rep='NaN') 
    #np.savetxt(text_file,keyPlayers,fmt='%d', delimiter=' ')
    #with open(text_file,'w') as file:
        #file.write(np.array2string(keyPlayers))
    rolePlayerAccumMap = np.zeros((len(dataset.tacticName),dataset.numPlayer))
    for i in range(len(selectIndex)):
        hit = [(selectIndex[i] in dataset.videoIndex[r]) for r in range(len(dataset.videoIndex))]
        loc = hit.index(True)
        for p in range(dataset.numPlayer):
            if keyPlayers[i][p] > 0:
                roleIndex = int(keyPlayers[i][p])-1
                rolePlayerAccumMap[loc,roleIndex] = rolePlayerAccumMap[loc,roleIndex]+1
        
    num_k = np.zeros(len(dataset.tacticName),dtype=np.int8)
    k = 0
    boundary = len(dataset.C5k_CLASS[0])
    for idx in range(len(dataset.tacticName)):
        if idx >= boundary:
            k = k + 1            
            boundary = boundary + len(dataset.C5k_CLASS[k])
        num_k[idx] = dataset.k[k]
           
    reorderMapTemp = rolePlayerAccumMap[reorder]
    reorderMapSum = np.sum(reorderMapTemp,axis=1,keepdims=True)
    num_vid = reorderMapSum/np.expand_dims(num_k,axis=1)
    reorderMap = np.concatenate((reorderMapTemp,num_vid),axis=1)
    orderTactic = [dataset.tacticName[i] for i in reorder]
    orderTacticInfo = [orderTactic[t]+'('+ num_k[t].astype(str)+')' for t in range(len(dataset.tacticName))]
    role= ['r1','r2','r3','r4','r5','num_vid']
    df = pd.DataFrame(reorderMap,index=orderTacticInfo,columns=role).astype(int)
    #print(df)
    utils.printLog(log_file," ")
    utils.printLog(log_file,df.to_string())
    if info_file is not None:
        df.to_csv(info_file,na_rep='NaN') 