#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 01:58:40 2017

@author: PeterTsai
"""
import numpy as np

def createPretrainShape(num_input,num_output,num_hidden_layer):
    reduce_dim = round((num_input-num_output)/(num_hidden_layer))
    shape = np.zeros(num_hidden_layer+1,dtype=np.int32)
    shape[0] = num_input
    shape[num_hidden_layer] = num_output
    for l in range(num_hidden_layer-1):
        shape[l+1] = shape[l] - reduce_dim
        
    return shape