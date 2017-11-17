
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

trainDataCollection='/cms-sc17/convert_20170717_ak8_deepDoubleB_db_sv_train_val/dataCollection.dc'
testDataCollection = trainDataCollection.replace("train_val","test")

print testDataCollection

trainDir = 'train_deep_sv_removals_ptrel_erel_etarel_deltaR_pt_mass/'

#Toggle training or eval
TrainBool = True
EvalBool= True

#Toggle to load model directly or load weights
LoadModel = False

#select model and makeRoc
from DeepJet_models_removals import deep_model_removal_sv as model
from eval_funcs import makeRoc, _byteify, makeLossPlot



if TrainBool:
    args = MyClass()
    args.inputDataCollection = trainDataCollection
    args.outputDir = trainDir

    #also does all the parsing
    train=training_base(testrun=False,args=args)

    if not train.modelSet():

        train.setModel(model)
    
        train.compileModel(learningrate=0.001,
                           loss=['categorical_crossentropy'],
                           metrics=['accuracy'])
    
        model,history,callbacks = train.trainModel(nepochs=500, 
                                                   batchsize=1024, 
                                                   stop_patience=1000, 
                                                   lr_factor=0.7, 
                                                   lr_patience=10, 
                                                   lr_epsilon=0.00000001, 
                                                   lr_cooldown=2, 
                                                   lr_minimum=0.00000001, 
                                                   maxqsize=100)

if EvalBool:
    sess = tf.InteractiveSession()

    inputModel = '%s/KERAS_check_best_model.h5'%trainDir
    inputWeights = '%s/KERAS_check_best_model_weights.h5' %trainDir
    evalDir = trainDir.replace('train','out')
    
    from DeepJet_models_removals import deep_model_removal_sv
    
    from DataCollection import DataCollection
    
    traind=DataCollection()
    traind.readFromFile(trainDataCollection)
    #train_data.useweights=useweights                                                                                                                                                                                                                                               
    testd=DataCollection()
    testd.readFromFile(testDataCollection)

    if os.path.isdir(evalDir):
        raise Exception('output directory: %s must not exists yet' %evalDir)
    else:
        os.mkdir(evalDir)
    
    if(LoadModel):
        evalModel = load_model(inputModel, custom_objects = global_loss_list)

    else:
        shapes=traind.getInputShapes()
        train_inputs = []
        for s in shapes:
            train_inputs.append(keras.layers.Input(shape=s))
        evalModel = model(train_inputs,traind.getNClassificationTargets(),traind.getNRegressionTargets())
        evalModel.load_weights(inputWeights)


    df, features_val = makeRoc(testd, evalModel, evalDir)

    makeLossPlot(trainDir,evalDir)
    
