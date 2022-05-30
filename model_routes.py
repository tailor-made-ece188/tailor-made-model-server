
# for path related functionalities
import os
# for array operations
import numpy as np
# tensorflow framework
import tensorflow as tf
# keras API for deep learning
from tensorflow import keras
from route_config import *
from auth_routes import auth_required
from flask import json, jsonify, make_response
import pickle
from bson.objectid import ObjectId

#firebase imports
import firebase_admin
from firebase_admin import credentials, storage, initialize_app
cred= credentials.Certificate("/Users/matthewdev/CodingProjects/ECE188/tailor-made-model-server/firebaseCred.json")
default_app = initialize_app(cred, {'databaseURL': 'tailor-made-ece188.appspot.com'})
print(default_app.name)

import cv2
import PIL
from PIL import Image 
import requests
from torchvision import transforms 

#helper fns
def resize_image(image):
    # scale the image
    image = tf.cast(image, tf.float32)
    image = image/255.0
    # resize image
    image = tf.image.resize(image, (128,128))
    return image



@app.route("/classifyImage", methods=['POST'])
@auth_required
def classify(uid):
    url = ""
    image_name = ""
    if request.is_json:
        try:
            json_data = request.get_json()
            url = json_data['image_url']
            image_name = json_data['image_name']

        except:    
            return make_response(jsonify({"message": "Error, must include image url and name"}), 400)
        
    url2 = url

    url_img = Image.open(requests.get(url2, stream=True).raw)
    convert_im=transforms.ToTensor()
    # img = convert_im(url_img)

    # file_name='/content/test1.jpg'
    # im = Image.open(file_name, mode='r')
    # im
    # file = tf.io.read_file(file_name)
    #    decode png file into a tensor

    img=tf.image.decode_jpeg(requests.get(url2).content, channels=3, name="jpeg_reader")
    # img = tf.image.decode_png(file, channels=3, dtype=tf.uint8)
    print(img.shape)
    x,y,z=img.shape
    x_scale = x/128
    y_scale = y/128
    new_img = resize_image(img)
    # plt.imshow(new_img)
    # print(new_img.shape)
    pred = unet(new_img[None])
    temp=0
    for i in pred:
        temp = np.squeeze(pred[0])
        temp = tf.argmax(pred[0], axis=-1)
        temp=np.squeeze(temp)
        # plt.imshow(temp)
        cv2.imwrite("output.png", temp)
    
    # pred = tf.image.convert_image_dtype(i/255.0, dtype=tf.uint8)

    # im = Image.fromarray(pred)
    # im.save("your_file.jpeg")
    bucket = storage.bucket("tailor-made-ece188.appspot.com")
    objId = ObjectId(uid)
    user = db.users.find_one_or_404({"_id" : objId })
    username = user['username']
    blob = bucket.blob("images/" + username + "/" + image_name +"_classified")
    blob.upload_from_filename("output.png")
    # Opt : if you want to make public access from the URL
    blob.make_public()

    print("your file url", blob.public_url)
    public_url = blob.public_url
    # requests.post('https://tailor-made-ece188.herokuapp.com/addSegmentedImage', headers=)
    return make_response(jsonify({"image_url": blob.public_url}), 200)

    # k = 0
    # colors=[]
    # min_x=np.ones(60)*128
    # max_x=np.zeros(60)
    # min_y=np.ones(60)*128
    # max_y=np.zeros(60)
    # color_count=np.zeros(60)

    # for i in pred:

        #save image
        
        # plot the predicted mask
        # plt.subplot(4,3,1+k*3)
        # i = tf.argmax(i, axis=-1)
        # plt.imshow(np.squeeze(i),cmap='jet', norm=NORM)
        
        # plt.axis('off')
        # plt.title('Prediction')
        # print(i)
        # arr = i.numpy()
        # x,y=arr.shape
        # for i in range(x):
        # for j in range(y):
        #     k = arr[i,j]
        #     if (k != 0):
        #     color_count[k]+=1
        #     if k not in colors:
        #     colors.append(k)
        #     if(i>max_x[k]):
        #     max_x[k]=i
        #     if(j>max_y[k]):
        #     max_y[k]=j
        #     if(i<min_x[k]):
        #     min_x[k]=i
        #     if(j<min_y[k]):
        #     min_y[k]=j  
        # print(colors)       
        # # plot the actual image
        # plt.subplot(4,3,2+k*3)
        # plt.imshow(np.squeeze(new_img))
        # plt.axis('off')
        # plt.title('Actual Image')
    #     k += 1
    #     if k == 4: break
    # plt.suptitle('Predition After 150 Epochs (By Fine-tuning from 51th Epoch)', color='red', size=20)  
    # plt.show()        
