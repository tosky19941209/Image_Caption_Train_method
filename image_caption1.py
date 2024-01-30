import tensorflow as tf
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input, RepeatVector, Dense, LSTM
from keras.layers import Embedding, Dropout, TimeDistributed, Concatenate
from keras.layers import Activation
from keras.optimizers import Adam
from keras.layers import add
from keras.callbacks import ModelCheckpoint
import os
import numpy as np
from PIL import Image

model_1 = tf.keras.models.load_model('Image_Caption_Trained_model.h5')

predict = model_1.predict('')