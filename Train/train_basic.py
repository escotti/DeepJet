from training_base import training_base


class MyClass:
    """A simple example class"""
    def __init__(self):
        self.inputDataCollection = ''
        self.outputDir = ''



from DeepJet_models_removals import deep_model_removals  as trainingModel 

inputTrainDataCollection = '/afs/cern.ch/work/e/erscotti/Data/convert_20170717_ak8_deepDoubleB_db_train_val/dataCollection.dc'
trainDir = "trainOut_basic/"

args = MyClass()
args.inputDataCollection = inputTrainDataCollection
args.outputDir = trainDir



#also does all the parsing
train=training_base(testrun=False, args=args)

if not train.modelSet():
	
    train.setModel(trainingModel,dropoutRate=0.1)
    
    train.compileModel(learningrate=0.005,
                       loss=['categorical_crossentropy'],
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