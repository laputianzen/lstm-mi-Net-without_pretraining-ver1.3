#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 22:25:10 2017

@author: PeterTsai
"""
import numpy as np
import scipy.io
import os
import matplotlib.pyplot as plt

class dataset(object):
    "Merge dataset functions and properties into single class object "
    def __init__(self, traj_file, fold_file, tactic_file,
                 fold, max_sequence_length,input_mode):
        self.traj_file = traj_file
        
        'read datasets'
        S, numVid, numPlayer = read_data_sets(traj_file)
        self.numPlayer= numPlayer
        self.numVid = numVid
        self.dataTrajOrigin = S

        self.MAX_X = 326
        self.MAX_Y = 348
        self.frameRate = 30
        
        'load 5-fold cross-validation'
        testIdx_fold, num_test, trainIdx_fold, num_train = load_fold(fold_file,fold,numVid)
        self.testIdx = testIdx_fold
        self.num_test = num_test
        self.trainIdx = trainIdx_fold
        self.num_train = num_train
        
        'create seqlenMatrix'
        self.seqLenMatrix = generate_seqLenMatrix(S,numPlayer)
        
        padS = paddingZeroToTraj(S,max_sequence_length)
        padS = rescale_position(padS,self.MAX_X,self.MAX_Y,self.seqLenMatrix)
        #padS = normalize_data(padS,MAX_X,MAX_Y)
        padSV= generatePV(padS,self.frameRate,self.seqLenMatrix)
        if input_mode == 'P':
            self.dataTraj = padS
            self.input_feature_dim = 2
        elif input_mode == 'P+V':
            self.dataTraj = padSV
            self.input_feature_dim = 4
        
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

def rescale_position(padS,MAX_X,MAX_Y,seqLenMatrix,inv=False):
    nPadS = np.zeros_like(padS,dtype=np.float32)
    for b in range(len(padS)):
        if not inv:
            nX = padS[b,:,:,0]/MAX_X
            nY = padS[b,:,:,1]/MAX_Y
            nX = 2*nX-1
            nY = 2*nY-1
        else:
            nX = (padS[b,:,:,0] + 1)/2
            nY = (padS[b,:,:,1] + 1)/2
            nX = nX*MAX_X
            nY = nY*MAX_Y
        nPadS[b,:,0:seqLenMatrix[b,0],0] = nX[:,0:seqLenMatrix[b,0]]
        nPadS[b,:,0:seqLenMatrix[b,0],1] = nY[:,0:seqLenMatrix[b,0]]
    return nPadS    

def normalize_data(padS,MAX_X,MAX_Y):
    nPadS = np.zeros_like(padS,dtype=np.float32) #force cast padS into np.float32
    nPadS[:,:,:,0] = padS[:,:,:,0]/MAX_X
    nPadS[:,:,:,1] = padS[:,:,:,1]/MAX_Y
    # from [0,1] to [-1,1]
    padS = nPadS   
    return padS

def generatePV(padS,frameRate,seqLenMatrix):
    padSV = np.zeros([padS.shape[0],padS.shape[1],padS.shape[2],padS.shape[3]*2],dtype=np.float32)
    padSV[:,:,:,0] = padS[:,:,:,0]
    padSV[:,:,:,1] = padS[:,:,:,1]
    showHistogram(padS,50,seqLenMatrix)
    
    rawV = (np.roll(padS,-1,axis=2) - padS)*frameRate
    # set final step to 0
    rawV[:,:,-1,:] = 0
    for v in range(len(seqLenMatrix)):
        last_frame_idx = seqLenMatrix[v,0] - 1
        rawV[:,:,last_frame_idx,:] = rawV[:,:,last_frame_idx-1,:]

    padSV[:,:,:,2] = rawV[:,:,:,0]
    padSV[:,:,:,3] = rawV[:,:,:,1]
    showHistogram(rawV,50,seqLenMatrix)
    #nV = standardize_velocity(rawV,seqLenMatrix)
    #showHistogram(nV,50,seqLenMatrix)
    #padSV[:,:,:,2] = nV[:,:,:,0]
    #padSV[:,:,:,3] = nV[:,:,:,1]  

    return padSV    
    
def standardize_velocity(rawV,seqLenMatrix): #mean is 0
    nPadV= np.zeros_like(rawV,dtype=np.float32)
    num_valid_step = np.sum(seqLenMatrix)
    Vx_std = np.sqrt(np.sum(abs(rawV[:,:,:,0])**2)/num_valid_step)
    Vy_std = np.sqrt(np.sum(abs(rawV[:,:,:,1])**2)/num_valid_step)
    #Vx_std = np.sqrt(np.mean(abs(rawV[:,:,:,0])**2))
    #Vy_std = np.sqrt(np.mean(abs(rawV[:,:,:,1])**2))
    nPadV[:,:,:,0] = rawV[:,:,:,0]/Vx_std
    nPadV[:,:,:,1] = rawV[:,:,:,1]/Vy_std
    
    return nPadV

def showHistogram(x,num_bins,seqLenMatrix):
    total_step = x.shape[0]*x.shape[1]*x.shape[2]*x.shape[3]
    num_valid_step = np.sum(seqLenMatrix)*x.shape[3]
    padding_step = total_step - num_valid_step
    hist, bins = np.histogram(x, bins=num_bins)
    index0 = np.argmax(hist)
    hist[index0] = hist[index0] - padding_step
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.show()

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

  