#Import libraries
from flask import Flask, request, jsonify
import pickle

import warnings
warnings.filterwarnings("ignore")
import os, argparse
import keras
import cv2, spacy, numpy as np
from keras.models import model_from_json
from keras.optimizers import SGD
import joblib
from keras import backend as K
from keras.utils.vis_utils import plot_model
K.set_image_data_format('channels_first')
from keras.models import Model
from keras.applications.vgg16 import VGG16


app = Flask(__name__)

#load the Models

VQA_model_file_name      = 'VQA/VQA_MODEL.json'
VQA_weights_file_name   = 'VQA/VQA_MODEL_WEIGHTS.hdf5'
label_encoder_file_name  = 'VQA/FULL_labelencoder_trainval.pkl'

#Image Model
model = VGG16(weights='imagenet')

new_input = model.input
hidden_layer = model.layers[-2].output

image_model = Model(new_input, hidden_layer)
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
image_model.compile(optimizer=sgd, loss='categorical_crossentropy')
image_model.summary()

#VQA model

def get_VQA_model(VQA_model_file_name, VQA_weights_file_name):
    vqa_model = model_from_json(open(VQA_model_file_name).read())
    vqa_model.load_weights(VQA_weights_file_name)
    vqa_model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return vqa_model


#Feature extraction
def get_image_features(image_file_name):
    ''' Runs the given image_file to VGG 16 model and returns the
    weights (filters) as a 1, 4096 dimension vector '''
    image_features = np.zeros((1, 4096))
    # Magic_Number = 4096  > Comes from last layer of VGG Model

    # Since VGG was trained as a image of 224x224, every new image
    # is required to go through the same transformation
    im = cv2.resize(cv2.imread(image_file_name), (224, 224))
    im = im.transpose((2,0,1)) # convert the image to RGBA


    # this axis dimension is required because VGG was trained on a dimension
    # of 1, 3, 224, 224 (first axis is for the batch size
    # even though we are using only one image, we have to keep the dimensions consistent
    im = np.expand_dims(im, axis=0)

    image_features[0,:] = image_model.predict(im)[0]
    return image_features

def_im_features = get_image_features("deforestation.jpg")
print(1)

ship_im_features = get_image_features("cargoship.jpg")
print(2)

fishingship_im_features = get_image_features("fishingship.jpg")
print(3)

garbage_im_features = get_image_features("garbage.jpg")
print(4)

print("before word embeddings")
word_embeddings = spacy.load('en_core_web_lg')

def get_question_features(question):
    ''' For a given question, a unicode string, returns the time series vector
    with each word (token) transformed into a 300 dimension representation
    calculated using Glove Vector '''
    tokens = word_embeddings(question)
    question_tensor = np.zeros((1, 30, 300))
    for j in range(len(tokens)):
        question_tensor[0,j,:] = tokens[j].vector
    return question_tensor

# testquestion = "what is in the image?"

# testquestion_feats = get_question_features(testquestion)

# print(testquestion_feats)

# testquestion1 = "is there a woman in the picture?"

# testquestion_feats1 = get_question_features(testquestion1)

# print(testquestion_feats1)

confLevels = {90: "The answer you are seeking for is _.", 80: "I am pretty sure that the answer is _.", 60: "The answer is possibly _.", 40:"It might be _, but I am not sure.", 0: "Can't be sure, it's probably better if you ask another question."}

@app.route('/', methods=['GET'])
def home():
    return '''<h1>VQA Model Service</h1>
<p>A prototype API for a VQA model testing</p>'''


@app.route('/api/def',methods=['POST'])
def predict():
    K.clear_session()
    # Get the data from the POST request.
    data = request.form['question']
    question_feats = get_question_features(data)

    print(data)


    vqa_model = get_VQA_model(VQA_model_file_name, VQA_weights_file_name)
    vqa_model.summary()
    
    #predict
    y_output = vqa_model.predict([question_feats, def_im_features])

    print("Y OUTPUT IS GENERATED")

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    labelencoder = joblib.load(label_encoder_file_name)
    
    i = 0
    for label in reversed(np.argsort(y_output)[0,-5:]):
        if(i == 0):
            result =labelencoder.inverse_transform([label])[0]
            conf = y_output[0, label]*100
            confidence = str('{:0.2f}'.format(y_output[0,label]*100)).zfill(5)

            finalresult = result + str(conf) + confidence
            break
    
    keys = list(confLevels.keys())
    for i in range(len(keys)):
        if (conf>keys[i]):
            print(conf, keys[i])
            answer = confLevels[keys[i]].replace('_', result)
            break


    return answer

@app.route('/api/ship',methods=['POST'])
def predict1():
    K.clear_session()
    # Get the data from the POST request.
    data = request.form['question']
    question_feats = get_question_features(data)

    print(data)


    vqa_model = get_VQA_model(VQA_model_file_name, VQA_weights_file_name)
    vqa_model.summary()
    
    #predict
    y_output = vqa_model.predict([question_feats, ship_im_features])

    print("Y OUTPUT IS GENERATED")

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    labelencoder = joblib.load(label_encoder_file_name)
    
    i = 0
    for label in reversed(np.argsort(y_output)[0,-5:]):
        if(i == 0):
            result =labelencoder.inverse_transform([label])[0]
            conf = y_output[0, label]*100
            confidence = str('{:0.2f}'.format(y_output[0,label]*100)).zfill(5)

            finalresult = result + str(conf) + confidence
            break
    
    keys = list(confLevels.keys())
    for i in range(len(keys)):
        if (conf>keys[i]):
            print(conf, keys[i])
            answer = confLevels[keys[i]].replace('_', result)
            break


    return answer

@app.route('/api/fship',methods=['POST'])
def predict2():
    K.clear_session()
    # Get the data from the POST request.
    data = request.form['question']
    question_feats = get_question_features(data)

    print(data)


    vqa_model = get_VQA_model(VQA_model_file_name, VQA_weights_file_name)
    vqa_model.summary()
    
    #predict
    y_output = vqa_model.predict([question_feats, fishingship_im_features])

    print("Y OUTPUT IS GENERATED")

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    labelencoder = joblib.load(label_encoder_file_name)
    
    i = 0
    for label in reversed(np.argsort(y_output)[0,-5:]):
        if(i == 0):
            result =labelencoder.inverse_transform([label])[0]
            conf = y_output[0, label]*100
            confidence = str('{:0.2f}'.format(y_output[0,label]*100)).zfill(5)

            finalresult = result + str(conf) + confidence
            break
    
    keys = list(confLevels.keys())
    for i in range(len(keys)):
        if (conf>keys[i]):
            print(conf, keys[i])
            answer = confLevels[keys[i]].replace('_', result)
            break


    return answer

@app.route('/api/garbage',methods=['POST'])
def predict3():
    K.clear_session()
    # Get the data from the POST request.
    data = request.form['question']
    question_feats = get_question_features(data)

    print(data)


    vqa_model = get_VQA_model(VQA_model_file_name, VQA_weights_file_name)
    vqa_model.summary()
    
    #predict
    y_output = vqa_model.predict([question_feats, garbage_im_features])

    print("Y OUTPUT IS GENERATED")

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    labelencoder = joblib.load(label_encoder_file_name)
    
    i = 0
    for label in reversed(np.argsort(y_output)[0,-5:]):
        if(i == 0):
            result =labelencoder.inverse_transform([label])[0]
            conf = y_output[0, label]*100
            confidence = str('{:0.2f}'.format(y_output[0,label]*100)).zfill(5)

            finalresult = result + str(conf) + confidence
            break
    
    keys = list(confLevels.keys())
    for i in range(len(keys)):
        if (conf>keys[i]):
            print(conf, keys[i])
            answer = confLevels[keys[i]].replace('_', result)
            break


    return answer

if __name__ == '__main__':
    app.run(port=5000, debug=True)