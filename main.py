import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Reshape, Bidirectional, LSTM, Dense, Lambda, Activation, BatchNormalization, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


tf.config.run_functions_eagerly(True)

# CONTAIN ALL THE IMAGE NAMES 
print("[INFO]. Reading the CSV file.")
train = pd.read_csv('written_name_train_v2.csv')
valid = pd.read_csv('written_name_validation_v2.csv')

#Showing the sample images
# for i in range(6):
#     ax = plt.subplot(2, 3, i+1)
#     print(train.loc[i, 'FILENAME'])
#     img_dir = os.sep.join([r"C:\Users\User\Desktop\github\tuftsProject\train_v2_test", train.loc[i, 'FILENAME']])
#     image = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
#     if image is not None:
#         cv2.imshow("Img", image)
#         cv2.waitKey(100)
#     else:
#         print(f"Failed to load image: {img_dir}")

# Check for NaNs in our label 
# print("Number of NaNs in train set      : ", train['IDENTITY'].isnull().sum())
# print("Number of NaNs in validation set : ", valid['IDENTITY'].isnull().sum())

# Removing any NaNs
train.dropna(axis=0, inplace=True)
valid.dropna(axis=0, inplace=True)

#CLEANING PROCESS 
# 1: Unreadable Images 
unreadable = train[train['IDENTITY'] == 'UNREADABLE']
unreadable.reset_index(inplace = True, drop=True)
train = train[train['IDENTITY'] != 'UNREADABLE']
valid = valid[valid['IDENTITY'] != 'UNREADABLE']


# 2: Convert all labels to lowercase 
train['IDENTITY'] = train['IDENTITY'].str.upper()
valid['IDENTITY'] = valid['IDENTITY'].str.upper()
train.reset_index(inplace = True, drop=True) 
valid.reset_index(inplace = True, drop=True)



# PREPROCESSING IMAGES FOR TRAINING
def preprocess(img):
    (h, w) = img.shape    
    final_img = np.ones([64, 256])*255 # blank white image
    # crop
    if w > 256:
        img = img[:, :256]    
    if h > 64:
        img = img[:64, :]
    
    final_img[:h, :w] = img
    return cv2.rotate(final_img, cv2.ROTATE_90_CLOCKWISE)

# ESTABLISH TRAINING AND VALIDATION SAMPLE SIZE 
train_size = 10000
valid_size= 1000

# Function to read the images 
def read_img(img_path, csv_file):
    img_dir = os.sep.join([img_path, csv_file.loc[i, 'FILENAME']])
    image = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
    image = preprocess(image)
    image = image/255.
    return image
    
# Add the processed images to the training sample array 
train_x = []
for i in range(train_size):
    image = read_img(r"C:\Users\User\Desktop\github\tuftsProject\train_v2_test", train)
    train_x.append(image)

# Add the processed images to the validation sample array 
valid_x = []
for i in range(valid_size):
    image = read_img(r"C:\Users\User\Desktop\github\tuftsProject\validation_v2_test", valid)
    valid_x.append(image)


# CONVERT TO NUMPY ARRAYS 
train_x = np.array(train_x).reshape(-1, 256, 64, 1)
valid_x = np.array(valid_x).reshape(-1, 256, 64, 1)

# PREPARING LABELS FOR CTC LOSS 
alphabets = u"ABCDEFGHIJKLMNOPQRSTUVWXYZ-' "
max_str_len = 24 # max length of input labels
num_of_characters = len(alphabets) + 1 # +1 for ctc pseudo blank
num_of_timestamps = 64 # max length of predicted labels

# These two functions will convert each letter to a number that computer can read 
def label_to_num(label):
    label_num = []
    for ch in label:
        label_num.append(alphabets.find(ch))
        
    return np.array(label_num)

def num_to_label(num):
    ret = ""
    for ch in num:
        if ch == -1:  # CTC Blank
            break
        else:
            ret+=alphabets[ch]
    return ret

# TEST A NAME 
# name = 'JEBASTIN'
# print(name, '\n',label_to_num(name))


# Preparing the images for training 
train_y = np.ones([train_size, max_str_len]) * -1
train_label_len = np.zeros([train_size, 1])
train_input_len = np.ones([train_size, 1]) * (num_of_timestamps-2)
train_output = np.zeros([train_size])

for i in range(train_size):
    train_label_len[i] = len(train.loc[i, 'IDENTITY'])
    train_y[i, 0:len(train.loc[i, 'IDENTITY'])]= label_to_num(train.loc[i, 'IDENTITY'])    

valid_y = np.ones([valid_size, max_str_len]) * -1
valid_label_len = np.zeros([valid_size, 1])
valid_input_len = np.ones([valid_size, 1]) * (num_of_timestamps-2)
valid_output = np.zeros([valid_size])

for i in range(valid_size):
    valid_label_len[i] = len(valid.loc[i, 'IDENTITY'])
    valid_y[i, 0:len(valid.loc[i, 'IDENTITY'])]= label_to_num(valid.loc[i, 'IDENTITY'])    


# TEST ONE IMAGE 
# print('True label : ',train.loc[0, 'IDENTITY'] , '\ntrain_y : ',train_y[0],'\ntrain_label_len : ',train_label_len[0], 
#       '\ntrain_input_len : ', train_input_len[0])


# BUILDING MODEL 
input_data = Input(shape=(256, 64, 1), name='input')

inner = Conv2D(32, (3, 3), padding='same', name='conv1', kernel_initializer='he_normal')(input_data)  
inner = BatchNormalization()(inner)
inner = Activation('relu')(inner)
inner = MaxPooling2D(pool_size=(2, 2), name='max1')(inner)


inner = Conv2D(64, (3, 3), padding='same', name='conv2', kernel_initializer='he_normal')(inner)
inner = BatchNormalization()(inner)
inner = Activation('relu')(inner)
inner = MaxPooling2D(pool_size=(2, 2), name='max2')(inner)
inner = Dropout(0.3)(inner)


inner = Conv2D(128, (3, 3), padding='same', name='conv3', kernel_initializer='he_normal')(inner)
inner = BatchNormalization()(inner)
inner = Activation('relu')(inner)
inner = MaxPooling2D(pool_size=(1, 2), name='max3')(inner)
inner = Dropout(0.3)(inner)


# CNN to RNN
inner = Reshape(target_shape=((64, 1024)), name='reshape')(inner)
inner = Dense(64, activation='relu', kernel_initializer='he_normal', name='dense1')(inner)

## RNN
inner = Bidirectional(LSTM(256, return_sequences=True), name = 'lstm1')(inner)
inner = Bidirectional(LSTM(256, return_sequences=True), name = 'lstm2')(inner)


## OUTPUT
inner = Dense(num_of_characters, kernel_initializer='he_normal',name='dense2')(inner)
y_pred = Activation('softmax', name='softmax')(inner)

model = Model(inputs=input_data, outputs=y_pred)
model.summary()

# CTC loss function
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

labels = Input(name='gtruth_labels', shape=[max_str_len], dtype='float32')
input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')

ctc_loss = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
model_final = Model(inputs=[input_data, labels, input_length, label_length], outputs=ctc_loss)


# TRAINING MODEL!!!!!
# the loss calculation occurs elsewhere, so we use a dummy lambda function for the loss
model_final.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=Adam(lr = 0.0001), metrics=['accuracy'])

history = model_final.fit(x=[train_x, train_y, train_input_len, train_label_len], y=train_output, 
                validation_data=([valid_x, valid_y, valid_input_len, valid_label_len], valid_output),
                epochs=1, batch_size=128)

# Save and load the model
model_final.save('model2.h5')

loaded_model = tf.keras.models.load_model('model2.h5')
print("hello")


#CHECK PERFORMANCE ON MODEL
print("accuracy: ")
print(history.history['accuracy'])

preds = model.predict(valid_x)
print("preds:")
print(preds)

decoded = K.get_value(K.ctc_decode(preds, input_length=np.ones(preds.shape[0])*preds.shape[1], 
                                   greedy=True)[0][0])

prediction = []
for i in range(valid_size):
    prediction.append(num_to_label(decoded[i]))

y_true = valid.loc[0:valid_size, 'IDENTITY']
correct_char = 0
total_char = 0
correct = 0

# Loops through validation data and checks for each character and word for the accuracy 
for i in range(valid_size):
    # print("checkingTotalChar")
    pr = prediction[i]
    tr = y_true[i]
    total_char += len(tr)
    # print("total char: ")
    print(total_char)
    print("pr and tr: ")
    print(pr)
    print(tr)

    for j in range(min(len(tr), len(pr))):
        #print("checkingCorrectChar")
        if tr[j] == pr[j]:
            #print("checkingCorrectChar2")
            correct_char += 1
            
    if pr == tr :
        #print("checkingCorrect")
        correct += 1 

print("final pr tr")
print(pr)
print(tr)

print('Correct characters predicted : %.2f%%' %(correct_char*100/total_char))
print('Correct words predicted      : %.2f%%' %(correct*100/valid_size))


# PREDICTIONS using test dataset 
test = pd.read_csv('written_name_test_v2.csv')
plt.figure(figsize=(15, 10))
for i in range(6):
    ax = plt.subplot(2, 3, i+1)
    print(test.loc[i, 'FILENAME'])

    image = read_img(r"C:\Users\User\Desktop\github\tuftsProject\test_v2_test", test)
    plt.imshow(image, cmap='gray')
    pred = model.predict(image.reshape(1, 256, 64, 1))
    decoded = K.get_value(K.ctc_decode(pred, input_length=np.ones(pred.shape[0])*pred.shape[1], 
                                       greedy=True)[0][0])

    
    plt.title(num_to_label(decoded[0]), fontsize=12)
    plt.axis('off')

    
plt.show()
plt.subplots_adjust(wspace=0.2, hspace=-0.8)