import numpy as np
import pandas as pd
import cv2
import os, shutil
import matplotlib.pyplot as plt

import tensorflow as tf
import keras
from keras import layers, models


def preprocessImage(path, shape):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (shape))
    img = (img/255).astype(np.float32)
    img = img.T
    img = np.expand_dims(img, axis=-1)
    return img

label2char ={0: ' ',
 1: "'",
 2: '-',
 3: 'A',
 4: 'B',
 5: 'C',
 6: 'D',
 7: 'E',
 8: 'F',
 9: 'G',
 10: 'H',
 11: 'I',
 12: 'J',
 13: 'K',
 14: 'L',
 15: 'M',
 16: 'N',
 17: 'O',
 18: 'P',
 19: 'Q',
 20: 'R',
 21: 'S',
 22: 'T',
 23: 'U',
 24: 'V',
 25: 'W',
 26: 'X',
 27: 'Y',
 28: 'Z',
 29: '`'}

def getStringFromEncode(lst :list):
    return ''.join([label2char[i] if i in label2char else '' for i in lst])

# model = tf.keras.models.load_model(r'C:\Users\User\Desktop\github\tuftsProject\ocr_model.h5')

# print("model loaded")

# data_path = r"C:\Users\User\Desktop\github\tuftsProject\test_v2_test"
# data = pd.read_csv(r"C:\Users\User\Desktop\github\tuftsProject\written_name_test_v2.csv")
# print(data)



def decode_batch_predictions(pred):
    pred = pred[:, :-2] # first two layers of ctc garbage
    input_len = np.ones(pred.shape[0])*pred.shape[1]
    
    results = keras.backend.ctc_decode(pred, 
                                        input_length=input_len,
                                        greedy=True)[0][0]
    
    output_text = []
    for res in results.numpy():
        outstr = getStringFromEncode(res)
        output_text.append(outstr)
    
    # return final text results
    return output_text

# reading image and label
# index = 10
# imgPath = os.path.join(data_path, data["FILENAME"][index])
# label = data["IDENTITY"][index]


# indx = 0
folder = "upload_images"

def delete_imgs():
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try: 
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except:
            print('Failed to delete ')

def predicting(path, model):
    # showing the image
    img = preprocessImage(path, (256,64))
    plt.imshow(img)
    plt.title('Image')

    #predicing 
    img = np.expand_dims(img, axis=0)
    print(img.shape)

    preds = model.predict(img)
    pred_texts = decode_batch_predictions(preds)
    # print(pred_texts)
    return pred_texts

img_files = os.listdir(folder)
img_path = os.path.join(folder, img_files[-1])

# print(predicting(img_path))










