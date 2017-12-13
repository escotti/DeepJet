
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

#import setGPU
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from training_base import training_base
from Losses import loss_NLL
import sys

trainDataCollection_cpf_sv = '/afs/cern.ch/work/e/erscotti/Data/convert_20170717_ak8_deepDoubleB_db_cpf_sv_train_val/dataCollection.dc'#
#trainDataCollection_sv='/afs/cern.ch/work/e/erscotti/Data/convert_20170717_ak8_deepDoubleB_db_sv_train_val/dataCollection.dc'

testDataCollection_cpf_sv = trainDataCollection_cpf_sv.replace("train_val","test")
#testDataCollection_sv = trainDataCollection_sv.replace("train_val","test")

sampleDatasets_cpf_sv = ["db","cpf","sv"]
sampleDatasets_sv = ["db","sv"]

#removedVars = [[],[-1],[0,1,2,3,4,5,6,7,8,9,10,13]]
removedVars = None
inputDatasets = sampleDatasets_cpf_sv

#Toggle training or eval

#Toggle to load model directly (True) or load weights (False)
LoadModel = False

#select model and eval functions
from DeepJet_models_removals import deep_model_removals as trainingModel
from eval_funcs import loadModel, makeRoc, _byteify, makeLossPlot, makeComparisonPlots
inputTrainDataCollection = trainDataCollection_cpf_sv
inputTestDataCollection = testDataCollection_cpf_sv
lossFunction = 'categorical_crossentropy'

trainDir = 'train_deep_sv_removals_d3d_d3dsig_loss/'

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
