import keras
import numpy as np
import tensorflow as tf
from keras import backend as K


kernel_initializer = 'he_normal'
kernel_initializer_fc = 'lecun_uniform'



def FC(data, num_hidden, act='relu', p=None, name=''):
    if act=='leakyrelu':
        fc = keras.layers.Dense(num_hidden, activation='linear', name='%s_%s' % (name,act), kernel_initializer=kernel_initializer_fc)(data) # Add any layer, with the default of a linear squashing function                                                                                                                                                                                                           
        fc = keras.layers.advanced_activations.LeakyReLU(alpha=.001)(fc)   # add an advanced activation                                                                                                     
    else:
        fc = keras.layers.Dense(num_hidden, activation=act, name='%s_%s' % (name,act), kernel_initializer=kernel_initializer_fc)(data)
    if not p:
        return fc
    else:
        dropout = keras.layers.Dropout(rate=p, name='%s_dropout' % name)(fc)
        return dropout


def crop(start, end):
    def slicer(x):
        return x[:,:,start:end]
        
    return keras.layers.Lambda(slicer)


def cropInputs(inputs, datasets, removedVars):
    '''
    Arguments
    inputs: array of inputs, in same order as mentioned declared in datasets
    datasets: array of string labels for inputs; "db", "sv", "pf", "cpf"
    removedVars: array of arrays of ints, with the indices of variables to be removed in each data set, if given "-1" removes all

    Returns
    array of Lambda layers the same as input layers without removed variables
    '''

    croppedLayers = []
    for i in range(len(datasets)):
        inputSet = inputs[i]
        dataset = datasets[i]
        removals = removedVars[i]
        if len(removals)==0:
            croppedLayers.append(inputSet)
        elif removals[0] == -1:
            continue
        else:
            passedVars = []
            start = 0
            end = int(inputSet.shape[-1])
            print type(inputSet.shape[-1])
            print end
            print removals
            #if dataset is "db":
            #    end = 28
            #elif dataset is "sv":
            #    end = 14
            #elif dataset is "pf":
            #    end = 10
            #elif dataset is "cpf":
            #    end = 30
                            
            for j in removals:
                if j == start:
                    start +=1
                if j > start:
                    sliced = crop(start,j)(inputSet)
                    passedVars.append(sliced)
                    start = j+1

            sliced = crop(start,end)(inputSet)
            print sliced.shape
            if sliced.shape[-1]>0:
                passedVars.append(sliced)
               # print passedVars
            if len(passedVars)> 1:
                cut_layer = keras.layers.concatenate(passedVars, axis = 2, name = 'cut_%s'%dataset )
            else:
                cut_layer = passedVars
            croppedLayers.append(cut_layer)

    return croppedLayers


def deep_model_removals(inputs, num_classes, num_regclasses, datasets, removedVars = None, **kwargs):
    
    cutInputs = inputs

    if removedVars is not None:
        cutInputs = cropInputs(inputs, datasets, removedVars)

    
    flattenLayers =[]
    for i in range(len(cutInputs)):
        flattenLayers.append(keras.layers.Flatten()(cutInputs[i]))

    concat = keras.layers.concatenate(flattenLayers,name="concat")

    fc = FC(concat, 64, p=0.1, name='fc1')
    fc = FC(fc, 32, p=0.1, name='fc2')
    fc = FC(fc, 32, p=0.1, name='fc3')
    output = keras.layers.Dense(num_classes, activation='softmax', name='softmax', kernel_initializer=kernel_initializer_fc)(fc)

    model = keras.models.Model(inputs=inputs, outputs=output)

    print model.summary()
    return model



def deep_model_removal_sv(inputs, num_classes,num_regclasses,datasets, removedVars = None, **kwargs):


#ordering of sv variables
                         #0 'sv_ptrel',
                         #1 'sv_erel',
                         #2 'sv_phirel',
                         #3 'sv_etarel',
                         #4 'sv_deltaR',
                         #5 'sv_pt',
                         #6 'sv_mass',
                         #7 'sv_ntracks',
                         #8 'sv_normchi2',
                         #9 'sv_dxy',
                         #10 'sv_dxysig',
                         #11 'sv_d3d',
                         #12 'sv_d3dsig',
                         #13 'sv_costhetasvpv'
    

    input_db = inputs[0]
    input_sv = inputs[1]

    sv_shape = input_sv.shape

    passedVars = []
    start = 0
    end = 14
    index =0
    for i in removedVars:
        if i == start:
            start +=1
        if i > start:
            sliced = crop(start,i)(input_sv)
            print sliced.shape
            passedVars.append(sliced)
            start = i+1
    sliced = crop(start,end)(input_sv)
    passedVars.append(sliced)
    print passedVars
   
    cut_sv = keras.layers.concatenate(passedVars, axis = 2, name = 'cut_sv')
    
    x = keras.layers.Flatten()(input_db)

    sv = keras.layers.Flatten()(cut_sv)
    
    concat = keras.layers.concatenate([x, sv], name='concat')
    
    fc = FC(concat, 64, p=0.1, name='fc1')
    fc = FC(fc, 32, p=0.1, name='fc2')
    fc = FC(fc, 32, p=0.1, name='fc3')
    output = keras.layers.Dense(num_classes, activation='softmax', name='softmax', kernel_initializer=kernel_initializer_fc)(fc)

    model = keras.models.Model(inputs=inputs, outputs=output)

    print model.summary()
    return model
