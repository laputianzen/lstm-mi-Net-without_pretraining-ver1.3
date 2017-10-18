#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 22:25:10 2017

@author: PeterTsai
"""
import numpy as np
import scipy.io
import os

class dataset(object):
    "Merge dataset functions and properties into single class object "
    def __init__(self, traj_file, fold_file, 
                 fold, max_sequence_length, MAX_X, MAX_Y):
        self.traj_file = traj_file
        
        'read datasets'
        S, numVid, numPlayer = read_data_sets(traj_file)
        self.numPlayer= numPlayer
        self.numVid = numVid
        self.dataTrajOrigin = S

        'load 5-fold cross-validation'
        testIdx_fold, num_test, trainIdx_fold, num_train = load_fold(fold_file,fold,numVid)
        self.testIdx = testIdx_fold
        self.num_test = num_test
        self.trainIdx = trainIdx_fold
        self.num_train = num_train
        
        'create seqlenMatrix'
        self.seqLenMatrix = generate_seqLenMatrix(S,numPlayer)
        
        padS = paddingZeroToTraj(S,max_sequence_length)
        padS = normalize_data(padS,MAX_X,MAX_Y)
        self.dataTraj = padS
        
        self = load_tacticInfo(self)
    
def load_fold(fold_file, fold, numVid):
    cross_fold = scipy.io.loadmat(fold_file)
    testIdx_fold = cross_fold['test_bagIdx']
    testIdx_fold = testIdx_fold[0][fold][0]
    num_test = len(testIdx_fold)
    
    trainIdx_fold = np.arange(numVid)
    trainIdx_fold = np.setdiff1d(trainIdx_fold,testIdx_fold)
    num_train = len(trainIdx_fold)
    
    return testIdx_fold, num_test, trainIdx_fold, num_train

def read_data_sets(traj_file):
    trajs = scipy.io.loadmat(traj_file)
    S = trajs['S']
#    print('video number:{0}'.format(len(S)))
    numVid, numPlayer = S.shape

    return S, numVid, numPlayer

def generate_seqLenMatrix(S,numPlayer):
    seqLen = np.array([S[seq,0].shape[0] for seq in range(len(S))])
    seqLenMatrix= np.stack([seqLen,]*numPlayer,axis=1)
    return seqLenMatrix

def normalize_data(padS,MAX_X,MAX_Y):
    nPadS = np.zeros_like(padS,dtype=np.float32) #force cast padS into np.float32
    nPadS[:,:,:,0] = padS[:,:,:,0]/MAX_X
    nPadS[:,:,:,1] = padS[:,:,:,1]/MAX_Y
    padS = nPadS    
    return padS

def paddingZeroToTraj(S,max_step_num):
    padS = []
    
    for v in range(S.shape[0]):
        #print("video:",v)
        vs = np.stack(S[v,:],axis=0)
        #print("before padding:",vs.shape) 
        npad = ((0,0),(0,max_step_num-S[v,0].shape[0]),(0,0))
        pad = np.pad(vs, pad_width = npad, mode='constant', constant_values=0)
        #print("before padding:",pad.shape)
        padS.append(pad)
        
    padS = np.stack(padS,axis=0)     
    return padS

def load_tacticInfo(dataset):
    dataset.tacticName =['F23','EV','HK','PD','PT','RB','SP','WS','WV','WW']
    dataset.C5k_CLASS = [[0,1,2,3,5,7],[6,9],[4,8]]
    dataset.k = [3,2,5]
    dataset.playerMap = [[[1,1,1,0,0],[1,1,0,1,0],[1,1,0,0,1],[1,0,1,1,0],[1,0,1,0,1],
                           [1,0,0,1,1],[0,1,1,1,0],[0,1,1,0,1],[0,1,0,1,1],[0,0,1,1,1]],
                       [[1,1,0,0,0],[1,0,1,0,0],[1,0,0,1,0],[1,0,0,0,1],[0,1,1,0,0],
                           [0,1,0,1,0],[0,1,0,0,1],[0,0,1,1,0],[0,0,1,0,1],[0,0,0,1,1]],
                       [[1,1,1,1,1]]]
    return dataset

def generateLSTMTempData(nchoosek_inputs,sess,dataset,FLAGS,file_suffix):
    C53_data = []
    C52_data = []
    C55_data = []
    seqLenMatrix = dataset.seqLenMatrix
    padS = dataset.dataTraj
    if file_suffix is 'train':
        fold_idx = dataset.trainIdx
    elif file_suffix is 'test':
        fold_idx = dataset.testIdx
    fold = FLAGS.fold
    p_input = FLAGS.p_input
    seqlen = FLAGS.seqlen
    numVid = len(fold_idx)
    for v in range(numVid):
        random_sequences = padS[fold_idx[v],:]
        batch_seqlen = np.reshape(seqLenMatrix[fold_idx[v],:],(-1))
    #    C53_merge = sess.run(C53_input_merge, 
    #             feed_dict={p_input:random_sequences,seqlen: batch_seqlen})
    #    C52_merge = sess.run(C52_input_merge, 
    #             feed_dict={p_input:random_sequences,seqlen: batch_seqlen})    
    #    C55_merge = sess.run(C55_input_merge, 
    #             feed_dict={p_input:random_sequences,seqlen: batch_seqlen})
        ncc = sess.run(nchoosek_inputs, feed_dict={p_input:random_sequences,seqlen: batch_seqlen})
        C53_data.append(ncc[0])
        C52_data.append(ncc[1])
        C55_data.append(ncc[2])
        
    
    _intermediate_feature_dir = FLAGS.ae_lstm_pretrain_model_dir   + '/tempData'
    if not os.path.exists(_intermediate_feature_dir):
        os.makedirs(_intermediate_feature_dir)
     
    C53_data = np.concatenate(C53_data,axis=0)
    C52_data = np.concatenate(C52_data,axis=0)
    C55_data = np.concatenate(C55_data,axis=0)
    np.save(_intermediate_feature_dir + '/C53_fold{0}_{1}.npy'.format(fold+1,file_suffix),C53_data)
    np.save(_intermediate_feature_dir + '/C52_fold{0}_{1}.npy'.format(fold+1,file_suffix),C52_data)
    np.save(_intermediate_feature_dir + '/C55_fold{0}_{1}.npy'.format(fold+1,file_suffix),C55_data)
    
    return _intermediate_feature_dir   

  