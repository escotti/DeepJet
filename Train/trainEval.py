
# coding: utf-8

import sys
import os
import keras
import tensorflow as tf

from keras.losses import kullback_leibler_divergence, categorical_crossentropy
from keras.models import load_model, Model
from testing import testDescriptor
from argparse import ArgumentParser
from keras import backend as K
from Losses import * #needed!                                                                                                                                                                                                                                                   
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from root_numpy import array2root
import pandas as pd
import h5py
from eval_funcs import loadModel, makeRoc, _byteify, makeLossPlot

#keras.backend.set_image_data_format('channels_last')

class MyClass:
    """A simple example class"""
    def __init__(self):
        self.inputDataCollection = ''
        self.outputDir = ''

#import setGPU
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from training_base import training_base
from Losses import loss_NLL
import sys

'''
Data Sets
'''
trainDataCollection_cpf_sv = '/afs/cern.ch/work/e/erscotti/Data/convert_20170717_ak8_deepDoubleB_db_cpf_sv_train_val/dataCollection.dc'
trainDataCollection_sv='/afs/cern.ch/work/e/erscotti/Data/convert_20170717_ak8_deepDoubleB_db_sv_train_val/dataCollection.dc'
trainDataCollection_db='/afs/cern.ch/work/e/erscotti/Data/convert_20170717_ak8_deepDoubleB_db_train_val/dataCollection.dc'

testDataCollection_cpf_sv = trainDataCollection_cpf_sv.replace("train_val","test")
testDataCollection_sv = trainDataCollection_sv.replace("train_val","test")
testDataCollection_db = trainDataCollection_db.replace("train_val","test")

sampleDatasets_cpf_sv = ["db","cpf","sv"]
sampleDatasets_sv = ["db","sv"]
sampleDatasets_db = ["db"]


'''
Settings Variables
'''

#removed variables from each variable set.
removedVars = None

#Toggle training and eval
TrainBool = True
EvalBool= True

#Toggle to load model directly (True) or load weights (False)
LoadModel = False # false should always work, true is faster but can't be used with Lambda Layers (removals)

#select model
from DeepJet_models_removals import deep_model_removals as trainingModel

#select DataColletions train and test data should have the same sets of variables (ie. db, or db+sv)
inputDatasets = sampleDatasets_db
inputTrainDataCollection = trainDataCollection_db
inputTestDataCollection = testDataCollection_db

#choose loss function;  standard is categorical_crossentropy
lossFunction = 'categorical_crossentropy'

#choose output directory
trainDir = 'train_testing/'

if TrainBool:
    args = MyClass()
    args.inputDataCollection = inputTrainDataCollection
    args.outputDir = trainDir

    #also does all the parsing
    train=training_base(testrun=False,args=args)

    if not train.modelSet():

        train.setModel(trainingModel,inputDatasets,removedVars)
    
        train.compileModel(learningrate=0.001,
                           loss=lossFunction,
                           metrics=['accuracy'])
    
        model,history,callbacks = train.trainModel(nepochs=100, 
                                                   batchsize=1024, 
                                                   stop_patience=100, 
                                                   lr_factor=0.5, 
                                                   lr_patience=20, 
                                                   lr_epsilon=0.00000001, 
                                                   lr_cooldown=2, 
                                                   lr_minimum=0.00000001, 
                                                   maxqsize=100)

if EvalBool:
	
	if TrainBool:
		evalModel = model
	else:
		evalModel = loadModel(trainDir,inputTrainDataCollection,trainingModel,LoadModel,inputDatasets,removedVars)
    
    evalDir = trainDir.replace('train','out')
    
    from DataCollection import DataCollection
    testd=DataCollection()
    testd.readFromFile(inputTestDataCollection)

    if os.path.isdir(evalDir):
        raise Exception('output directory: %s must not exists yet' %evalDir)
    else:
        os.mkdir(evalDir)

    df, features_val = makeRoc(testd, evalModel, evalDir)

    makeLossPlot(trainDir,evalDir)