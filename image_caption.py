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

max_length = 34

fn = "Flickr_Data/Flickr_Data/Flickr_TextData/Flickr8k.token.txt"
f = open(fn, 'r')
capts = f.read()
#Group all captions by filename, for references
captions = dict()
i = 0

try:
    for line in capts.split("\n"):
        txt = line.split('\t')
        fn = txt[0].split('#')[0]
        if fn not in captions.keys():
            captions[fn] = [txt[1]]
        else:
            captions[fn].append(txt[1])
        i += 1
except:
    pass #pass Model
    


def getCaptions(path):
    
    f = open(path, 'r')
    capts = f.read()
    desc = dict()

    try:
        for line in capts.split("\n"):
            image_id = line
            image_descs = captions[image_id]

            for des in image_descs:
                ws = des.split(" ")
                w = [word for word in ws if word.isalpha()]
                des = "startseq " + " ".join(w) + " endseq"
                if image_id not in desc:
                    desc[image_id] = list()
                desc[image_id].append(des)
    except:
        pass
    
    return desc


train_caps = getCaptions("Flickr_Data/Flickr_Data/Flickr_TextData/Flickr_8k.trainImages.txt")
val_caps = getCaptions("Flickr_Data/Flickr_Data/Flickr_TextData/Flickr_8k.devImages.txt")


train_captions = []
for key, desc_list in train_caps.items():
    for i in range(len(desc_list)):
        train_captions.append(desc_list[i])

# Tokenize top 5000 words in Train Captions
vocab_size = 5000
tokenizer = Tokenizer(num_words=vocab_size,
                      oov_token="<unk>",
                      filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
# tokenizer.fit_on_texts(train_captions)
word_index = tokenizer.word_index
index_word = tokenizer.index_word


def word_for_id(integer, tokenizer):
    return tokenizer.index_word.get(integer)

# generate a description for an image
def generate_desc(model, tokenizer, photo, max_length):
    # seed the generation process
    in_text = 'startseq'
    # iterate over the whole length of the sequence
    for i in range(max_length):
        # integer encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        #print("sequence after tok: ", sequence)
        # pad input
        sequence = pad_sequences([sequence], maxlen=max_length)
        # predict next word
        if i==0:
            photo = np.expand_dims(photo, axis=0)
        #print("photo: ", photo)
        #print("sequence: ", sequence)
        yhat = model.predict([photo, sequence], verbose=0)
        # convert probability to integer
        yhat = np.argmax(yhat)
        # map integer to word
        word = word_for_id(yhat, tokenizer)
        # stop if we cannot map the word
        if word is None:
            break
        # append as input for generating the next word
        in_text += ' ' + word
        # stop if we predict the end of the sequence
        if word == 'endseq':
            break
    return in_text


def image_to_feat_vec(imagePath):
    img1 = image.load_img(imagePath, target_size=(224, 224))
    x = image.img_to_array(img1)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    fea_x = model_1.predict(x)
    fea_x1 = np.reshape(fea_x , fea_x.shape[1])
    return fea_x1

imagePath = "GarageImages/GarageImages/image1086.jpg"
photo = image_to_feat_vec(imagePath)
print("Predicted Caption:", generate_desc(model_1, tokenizer, photo, max_length))
Image.open(imagePath)