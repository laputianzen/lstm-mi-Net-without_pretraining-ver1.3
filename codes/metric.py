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
        
    rolePlayerMap = dataset.gtRoleOrder[selectIndex]
    Y_correct_map = Y_pred * Y_label
    Y_correct = np.sum(Y_correct_map,axis=-1,keepdims=True)
    NUM_PLAYER = rolePlayerMap.shape[1]
    keyPlayers = y_pred * (rolePlayerMap+1) * np.tile(Y_correct,[1,NUM_PLAYER])
    DAT = np.column_stack((NAME, keyPlayers.astype(int)))
    if text_file is not None:
        np.savetxt(text_file, DAT, delimiter=" ", fmt="%s")
    #np.savetxt(text_file,keyPlayers,fmt='%d', delimiter=' ')
    #with open(text_file,'w') as file:
        #file.write(np.array2string(keyPlayers))
    rolePlayerAccumMap = np.zeros((len(dataset.tacticName),dataset.numPlayer))
    for i in range(len(selectIndex)):
        hit = [(selectIndex[i] in dataset.videoIndex[r]) for r in range(len(dataset.videoIndex))]
        loc = hit.index(True)
        for p in range(dataset.numPlayer):
            if keyPlayers[i][p] != 0:
                roleIndex = int(keyPlayers[i][p])-1
                rolePlayerAccumMap[loc,roleIndex] = rolePlayerAccumMap[loc,roleIndex]+1
        
    #nv_reorder = [0,1,2,3,8,4,6,5,9,5,7]     
    #[[0,1,2,3,5,7],[6,9],[4,8]]
    
    reorder = np.concatenate(dataset.C5k_CLASS).ravel().tolist()
    num_k = np.zeros(len(dataset.tacticName),dtype=np.int8)
    k = 0
    boundary = len(dataset.C5k_CLASS[0])
    for idx in range(len(dataset.tacticName)):
        if idx >= boundary:
            k = k + 1            
            boundary = boundary + len(dataset.C5k_CLASS[k])
        num_k[idx] = dataset.k[k]
    #inv_reorder = [reorder.index(i) for i in range(len(dataset.tacticName))]
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