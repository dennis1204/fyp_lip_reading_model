#!/usr/bin/env python
# coding: utf-8

# In[147]:


import os
import cv2
import tensorflow as tf
import numpy as np
from typing import List
from matplotlib import pyplot as plt
import imageio


# In[148]:


tf.config.list_physical_devices('GPU')


# In[149]:


physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass


# 1.

# In[150]:


def load_video(path:str) -> List[float]: 

    cap = cv2.VideoCapture(path)
    frames = []
    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))): 
        ret, frame = cap.read()
        frame = tf.image.rgb_to_grayscale(frame)
        frames.append(frame[ 315:385,300:440,:])
    cap.release()
    
    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, tf.float32))
    return tf.cast((frames - mean), tf.float32) / std


# In[151]:


vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]


# In[152]:


char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)

print(
    f"The vocabulary is: {char_to_num.get_vocabulary()} "
    f"(size ={char_to_num.vocabulary_size()})"
)


# In[153]:


char_to_num.get_vocabulary()


# In[154]:


char_to_num(['n','e','i','1'])


# In[155]:


num_to_char([14,  5,  9, 30])


# In[156]:


def load_alignments(path:str) -> List[str]: 
    with open(path, 'r') as f: 
        lines = f.readlines() 
    tokens = []
    for line in lines:
        line = line.split()
        if line[2] != 'sil': 
            tokens = [*tokens,' ',line[2]]
    return char_to_num(tf.reshape(tf.strings.unicode_split(tokens, input_encoding='UTF-8'), (-1)))[1:]


# In[157]:


def load_data(path: str): 
    path = bytes.decode(path.numpy())
    file_name = path.split('/')[-1].split('.')[0]
    # File name splitting for windows
    # file_name = path.split('\\')[-1].split('.')[0]
    video_path = os.path.join('data_cantonese','s1',f'{file_name}.mp4')
    alignment_path = os.path.join('data_cantonese','alignment','s1',f'{file_name}.align')
    frames = load_video(video_path) 
    alignments = load_alignments(alignment_path)
    
    return frames, alignments


# In[158]:


test_path = './data_cantonese/s1/IMG_8806.mp4'


# In[159]:


tf.convert_to_tensor(test_path).numpy().decode('utf-8').split('/')[-1].split('.')[0]


# In[160]:


tf.convert_to_tensor(test_path)


# In[161]:


frames, alignments = load_data(tf.convert_to_tensor(test_path))


# In[162]:


load_data(tf.convert_to_tensor(test_path))


# In[163]:


plt.imshow(frames[70])


# In[164]:


alignments


# In[165]:


tf.strings.reduce_join([bytes.decode(x) for x in num_to_char(alignments.numpy()).numpy()])


# In[166]:


def mappable_function(path:str) ->List[str]:
    result = tf.py_function(load_data, [path], (tf.float32, tf.int64))
    return result


# 2.

# In[167]:


from matplotlib import pyplot as plt


# In[204]:


data = tf.data.Dataset.list_files('./data_cantonese/s1/*.mp4')
data = data.shuffle(20, reshuffle_each_iteration=False)
data = data.map(mappable_function)
data = data.padded_batch(2, padded_shapes=([90,None,None,None],[20]))
data = data.prefetch(tf.data.AUTOTUNE)
# Added for split 
# Determine the sizes for training and testing sets
total_samples = 10
train_size = int(0.8 * total_samples)

# Split the data
train_data = data.take(train_size)
test_data = data.skip(train_size)


# In[169]:


# for frames, target_sequence in data:
#     print("Target Sequence Length:", tf.shape(target_sequence)[1])


# In[170]:


for sample in data.take(5):  # Adjust the number of samples to print
    video_frames, label = sample
    print("Video Frames Shape:", video_frames.shape)
    print("Label:", label.numpy())
    print("~" * 50)


# In[171]:


len(data)


# In[172]:


# Assuming `data` is your dataset
print("Element Spec:")
print(data.element_spec)

# Access the padded_shapes attribute for each element
for element_spec in data.element_spec:
    if isinstance(element_spec, tuple) and len(element_spec) == 2:
        print("Padded Shape:", element_spec[1])


# In[173]:


frames, alignments = data.as_numpy_iterator().next()


# In[174]:


data.as_numpy_iterator().next()


# In[175]:


len(frames)


# In[176]:


sample = data.as_numpy_iterator()


# In[177]:


val = sample.next(); val[0]


# In[178]:


len(val[0])


# In[179]:


imageio.mimsave('./animation.gif', val[0][0], fps=10)


# In[180]:


# 0:videos, 0: 1st video out of the batch,  0: return the first frame in the video 
plt.imshow(val[0][0][30])


# In[181]:


tf.strings.reduce_join([num_to_char(word) for word in val[1][0]])


# 3.

# In[182]:


from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv3D, LSTM, Dense, Dropout, Bidirectional, MaxPool3D, Activation, Reshape, SpatialDropout3D, BatchNormalization, TimeDistributed, Flatten, Masking
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import legacy
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler


# In[183]:


data.as_numpy_iterator().next()[0][0].shape


# In[184]:


# model = Sequential()
# model.add(Conv3D( 128, 3, input_shape=(100,100,720,1), padding='same'))
# model.add(Activation('relu'))
# model.add(MaxPool3D((1,2,2)))

# model.add(Conv3D(128, 3, padding='same'))
# model.add(Activation('relu'))
# model.add(MaxPool3D((1,2,2)))

# model.add(Conv3D(64, 3, padding='same'))
# model.add(Activation('relu'))
# model.add(MaxPool3D((1,2,2)))

# model.add(TimeDistributed(Flatten()))

# model.add(Bidirectional(LSTM(64, kernel_initializer='Orthogonal', return_sequences=True)))
# model.add(Dropout(.5))

# model.add(Bidirectional(LSTM(64, kernel_initializer='Orthogonal', return_sequences=True)))
# model.add(Dropout(.5))

# model.add(Dense(char_to_num.vocabulary_size()+1, kernel_initializer='he_normal', activation='softmax'))


# In[185]:


def create_lip_reading_model(input_shape, num_classes):
    model = Sequential()
    
    # model.add(Masking(mask_value=0.0, input_shape=input_shape))
    model.add(Conv3D(128, 3, input_shape=input_shape, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1, 2, 2)))

    model.add(Conv3D(256, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1, 2, 2)))

    model.add(Conv3D(70, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1, 2, 2)))

    model.add(TimeDistributed(Flatten()))

    model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
    model.add(Dropout(0.5))

    model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
    model.add(Dropout(0.5))

    model.add(Dense(num_classes, kernel_initializer='he_normal', activation='softmax'))

    return model


# In[ ]:





# In[186]:


input_shape = (90,70,140,1)  # Modify based on your actual input shape
num_classes =  char_to_num.vocabulary_size()+1 # Replace with the actual number of classes


# In[187]:


def scheduler(epoch, lr):
    if epoch < 30:
        return lr
    else:
        return lr * tf.math.exp(-0.1)


# In[188]:


def CTCLoss(y_true, y_pred):
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")
# Print shapes for debugging
    # tf.print("y_true shape:", batch_len)
    # tf.print("y_pred shape:", input_length)

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss


# In[189]:


class ProduceExample(tf.keras.callbacks.Callback): 
    def __init__(self, dataset) -> None: 
        self.dataset = dataset.as_numpy_iterator()
    
    def on_epoch_end(self, epoch, logs=None) -> None:
        # Get the next batch of data from the dataset
        data = self.dataset.next()
        # Make predictions using the trained model
        yhat = self.model.predict(data[0])
        # Decode the predicted sequences using CTC decoding
        decoded = tf.keras.backend.ctc_decode(yhat, [90,90], greedy=False)[0][0].numpy()
        for x in range(len(yhat)):           
            print('Original:', tf.strings.reduce_join(num_to_char(data[1][x])).numpy().decode('utf-8'))
            print('Prediction:', tf.strings.reduce_join(num_to_char(decoded[x])).numpy().decode('utf-8'))
            print('~'*100)


# In[205]:


from tensorflow.keras.models import load_model
custom_objects = {'CTCLoss': CTCLoss}
checkpoint_directory = './saved_models'
# Ensure the checkpoint directory exists; if not, create it
os.makedirs(checkpoint_directory, exist_ok=True)
# Specify the checkpoint file path
checkpoint_path = os.path.join(checkpoint_directory, 'checkpoint')
# Check if there is a checkpoint file
if os.path.exists(checkpoint_path):
    print("Checkpoint exists in models.")
    # model = create_lip_reading_model(input_shape, num_classes)
    model = load_model(checkpoint_path,  custom_objects=custom_objects)
     # Retrieve the last epoch from the training history
    last_epoch = max(model.history.epoch) + 1 if hasattr(model.history, 'epoch') else 1
    # model.load_weights(checkpoint_path)
    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0001),
                    loss=CTCLoss)
    checkpoint_callback = ModelCheckpoint(
    os.path.join('saved_models','checkpoint'),
    monitor='loss', 
    save_weights_only=False, 
    save_freq='epoch', 
    save_format='h5') 
    print(f"Model loaded from checkpoint. Continue training from epoch {last_epoch}.")
    schedule_callback = LearningRateScheduler(scheduler)
    example_callback = ProduceExample(test_data)
    model.fit(train_data, validation_data=test_data, epochs=50, 
              callbacks=[schedule_callback, example_callback],
              initial_epoch=last_epoch)
else:
    # If there is no checkpoint, create a new model
    # input_shape = (100, 100, 720, 1)  # Modify based on your actual input shape
    # num_classes =  char_to_num.vocabulary_size()+1 # Replace with the actual number of classes
    model = create_lip_reading_model(input_shape, num_classes)
    print("No checkpoint found. Creating a new model.")


# In[206]:


model.summary()


# In[207]:


yhat = model.predict(val[0])


# In[208]:


tf.strings.reduce_join([num_to_char(x) for x in tf.argmax(yhat[0],axis=1)])


# In[209]:


tf.strings.reduce_join([num_to_char(tf.argmax(x)) for x in yhat[0]])


# In[210]:


model.input_shape


# In[211]:


model.output_shape


# In[212]:


# from tensorflow.keras.utils import plot_model

# # Assuming 'model' is your neural network
# plot_model(model, to_file='model_network.png', show_shapes=True, show_layer_names=True)


# 4.

# In[213]:


model.compile(optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.0001)
, loss=CTCLoss)


# In[214]:


# print("Number of sequences in the batch:", batch_len.numpy())


# In[215]:


checkpoint_callback = ModelCheckpoint(
    os.path.join('saved_models','checkpoint'),
    monitor='loss', 
    save_weights_only=False, 
    save_freq='epoch', 
    save_format='h5') 


# In[216]:


schedule_callback = LearningRateScheduler(scheduler)


# In[217]:


example_callback = ProduceExample(test_data)


# In[ ]:


model.fit(train_data, validation_data=test_data, epochs=50, callbacks=[schedule_callback, example_callback])


# In[ ]:




