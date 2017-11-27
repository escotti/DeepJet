
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

#keras.backend.set_image_data_format('channels_last')

class MyClass:
    """A simple example class"""
    def __init__(self):
        self.inputDataCollection = ''
        self.outputDir = ''

import setGPU
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from training_base import training_base
from Losses import loss_NLL
import sys

trainDataCollection_pf_cpf_sv = '/cms-sc17/convert_20170717_ak8_deepDoubleB_db_pf_cpf_sv_train_val/dataCollection.dc'
trainDataCollection_sv='/cms-sc17/convert_20170717_ak8_deepDoubleB_db_sv_train_val/dataCollection.dc'

testDataCollection_pf_cpf_sv = trainDataCollection_pf_cpf_sv.replace("train_val","test")
testDataCollection_sv = trainDataCollection_sv.replace("train_val","test")

sampleDatasets_pf_cpf_sv = ["db","pf","cpf","sv"]
sampleDatasets_sv = ["db","sv"]

removedVars = [[],[0,1,2,3,4,5,6,7,8,9,10,13]]


#Toggle training or eval
TrainBool = True
EvalBool= True
CompareBool = False

#Toggle to load model directly (True) or load weights (False)
LoadModel = False

#select model and eval functions
from DeepJet_models_removals import deep_model_removals as trainingModel
from eval_funcs import loadModel, makeRoc, _byteify, makeLossPlot, makeComparisonPlots
trainDir = 'train_deep_sv_removals_d3d_d3dsig_loss/'
inputTrainDataCollection = trainDataCollection_sv
inputTestDataCollection = testDataCollection_sv
lossFunction = loss_kldiv


#for Comparisons
from DeepJet_models_removals import deep_model_removal_sv, deep_model_removals
from DeepJet_models_ResNet import deep_model_doubleb_sv
compModels = [trainingModel, deep_model_removals, deep_model_removal_sv, deep_model_removal_sv, deep_model_doubleb_sv]
compNames = ["trackVars","d3d+d3dsig","SV-ptrel_erel_pt_mass","SV-pt,e,etaRel_deltaR_pt_mass","SV"]
compRemovals = (removedVars,[[],[0,1,2,3,4,5,6,7,8,9,10,13]],[0,1,5,6],[0,1,3,4,5,6],[])
compLoadModels = [LoadModel,False,False,False,True]
compTrainDirs = [trainDir,'train_deep_sv_removals_d3d_d3dsig_only/',"train_deep_sv_removals_ptrel_erel_pt_mass/","train_deep_sv_removals_ptrel_erel_etarel_deltaR_pt_mass/","train_deep_init_64_32_32_b1024/"]
compareDir = "comparedROCSTrack/"
compDatasets = [sampleDatasets_pf_cpf_sv,["db","sv"],["db","sv"],["db","sv"],["db","sv"]]
compTrainDataCollections = [trainDataCollection_pf_cpf_sv,trainDataCollection_sv,trainDataCollection_sv,trainDataCollection_sv,trainDataCollection_sv]
compTestDataCollections = [testDataCollection_pf_cpf_sv, testDataCollection_sv,testDataCollection_sv,testDataCollection_sv,testDataCollection_sv]

if TrainBool:
    args = MyClass()
    args.inputDataCollection = inputTrainDataCollection
    args.outputDir = trainDir

    #also does all the parsing
    train=training_base(testrun=False,args=args)

    if not train.modelSet():

        train.setModel(trainingModel,sampleDatasets_sv,removedVars)
    
        train.compileModel(learningrate=0.001,
                           loss=lossFunction,
                           metrics=['accuracy'])
    
        model,history,callbacks = train.trainModel(nepochs=300, 
                                                   batchsize=1024, 
                                                   stop_patience=1000, 
                                                   lr_factor=0.7, 
                                                   lr_patience=10, 
                                                   lr_epsilon=0.00000001, 
                                                   lr_cooldown=2, 
                                                   lr_minimum=0.00000001, 
                                                   maxqsize=100)

if EvalBool:

    evalModel = loadModel(trainDir,inputTrainDataCollection,trainingModel,LoadModel,sampleDatasets,removedVars)
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
    
if CompareBool:
    
    from DataCollection import DataCollection

    if os.path.isdir(compareDir):
        raise Exception('output directory: %s must not exists yet' %compareDir)
    else:
        os.mkdir(compareDir)
        

    models = []
    testds = []
    for i in range(len(compModels)):
        testd=DataCollection()
        testd.readFromFile(compTestDataCollections[i])
        curModel = loadModel(compTrainDirs[i],compTrainDataCollections[i],compModels[i],compLoadModels[i],compDatasets[i],compRemovals[i])
        models.append(curModel)
        testds.append(testd)

    makeComparisonPlots(testds,models,compNames,compareDir)
