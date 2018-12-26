from keras import backend as K
from keras.optimizers import Adadelta,SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint
from Image_Generator import TextImageGenerator
from model_crnn import get_model
from parameter import *
import  os
K.set_learning_phase(0)
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# # Model description and training

model = get_model(training=True)

try:
    model.load_weights('LSTM+BN4--2116--0.011.hdf5')
    print("...Previous weight data...")
except:
    print("...New weight data...")
    pass
'''
train_dir_path=''
train_label_path = 'train.txt'
tiger_train = TextImageGenerator(train_dir_path,train_label_path, img_w, img_h, batch_size, downsample_factor)
tiger_train.build_data()

valid_dir_path=''
valid_label_path='valid.txt'
tiger_val = TextImageGenerator(valid_dir_path,valid_label_path, img_w, img_h, val_batch_size, downsample_factor)
tiger_val.build_data()

'''


sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
early_stop = EarlyStopping(monitor='loss', min_delta=0.001, patience=4, mode='min', verbose=1)
checkpoint = ModelCheckpoint(filepath='LSTM+BN5--{epoch:02d}--{val_loss:.3f}.hdf5', monitor='loss', verbose=1, mode='min', period=1)
# the loss calc occurs elsewhere, so use a dummy lambda func for the loss
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)
model.summary()
# captures output of softmax so we can decode the output during visualization
model.fit_generator(generator=tiger_train.next_batch(),
                    steps_per_epoch=int(tiger_train.n / batch_size),
                    epochs=30,
                    callbacks=[checkpoint],
                    validation_data=tiger_val.next_batch(),
                    validation_steps=int(tiger_val.n / val_batch_size))
