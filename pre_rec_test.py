from PIL import Image
from CRNN_Keras.model_crnn import get_model
from config import crnn_modelpath
from CRNN_Keras.parameter import letters,img_w, img_h
import cv2
import numpy as np
import keras.backend as K
import datetime
import tensorflow as tf


def decode(pred):
    charactersS = letters
    t = pred.argmax(axis=2)[0]
    length = len(t)
    char_list = []
    n = len(letters)
    for i in range(length):
        if t[i] != n and (not (i > 0 and t[i - 1] == t[i])):
            char_list.append(charactersS[t[i]])
    return u''.join(char_list)
def decode_blank(pred):
    charactersS = letters+' '
    t = pred.argmax(axis=2)[0]
    length = len(t)
    char_list = []
    n = len(letters)
    for i in range(length):
        #if t[i] != n and (not (i > 0 and t[i - 1] == t[i])):
            char_list.append(charactersS[t[i]])
    return u''.join(char_list)

def load_model():
    modelPath = crnn_modelpath

    model = get_model(training=False)

    try:
        model.load_weights(modelPath)
        print("...Previous weight data...")
    except:
        raise Exception("No weight file!")
    return model

def predict(save_file,keras_model):
    """
    输入图片，输出keras模型的识别结果
    """
    img = cv2.imread(save_file, cv2.IMREAD_GRAYSCALE)

    scale = img.shape[0] * 1.0 / img_h
    w = img.shape[1] / scale
    w = int(w)
    img = cv2.resize(img, (w, img_h))
    img_pred = img.astype(np.float32)
    img_pred = (img_pred / 255.0) * 2.0 - 1.0
    # img_pred = img_pred.T
    img_pred = np.expand_dims(img_pred, axis=-1)
    img_pred = np.expand_dims(img_pred, axis=0)

    net_out_value = keras_model.predict(img_pred)

    y_pred = net_out_value[:, 2:, :]

    pred_texts = decode(y_pred)
    pred_text=decode_blank(y_pred)
    print(pred_text)
    starttime = datetime.datetime.now()
    #a=K.ctc_decode(y_pred, input_length=np.ones(y_pred.shape[0]) * y_pred.shape[1], )[0][0]


    #out = K.get_value(K.ctc_decode(y_pred, input_length=np.ones(y_pred.shape[0]) * y_pred.shape[1], )[0][0])[:, :]
    endtime = datetime.datetime.now()
    print(endtime - starttime)
    return pred_texts
model=load_model()

save_file='upload/result/1.png'
starttime = datetime.datetime.now()
sim_pred = predict(save_file,model)
endtime = datetime.datetime.now()
print(endtime - starttime)
print(sim_pred)