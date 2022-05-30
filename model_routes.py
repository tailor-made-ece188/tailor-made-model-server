
# for path related functionalities
from firebase_admin import credentials, storage, initialize_app
import firebase_admin
from bson.objectid import ObjectId
import pickle
from flask import json, jsonify, make_response
from auth_routes import auth_required
from route_config import *
import urllib.request
from skimage import io
import keras
import random
import shutil
import numpy as np
import tensorflow as tf
import colorsys
import skimage
import argparse
import uuid
from mrcnn.model import MaskRCNN
import mrcnn.model as modellib
from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import utils
import cv2
import os
import sys


ROOT_DIR = os.path.relpath('./Mask_RCNN-master/')

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library


# firebase imports
cred = credentials.Certificate(
    "/Users/matthewdev/CodingProjects/ECE188/tailor-made-model-server/firebaseCred.json")
default_app = initialize_app(
    cred, {'databaseURL': 'tailor-made-ece188.appspot.com'})
print(default_app.name)


# helper fns


def resize_image(image):
    # scale the image
    image = tf.cast(image, tf.float32)
    image = image/255.0
    # resize image
    image = tf.image.resize(image, (128, 128))
    return image


class TestConfig(Config):
    NAME = "Deepfashion2"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 13


class_names = ['short_sleeved_shirt', 'long_sleeved_shirt', 'short_sleeved_outwear', 'long_sleeved_outwear', 'vest', 'sling',
               'shorts', 'pants', 'skirt', 'short_sleeved_dress', 'long_sleeved_dress',
               'vest_dress', 'sling_dress', '']


def random_colors(N):
    np.random.seed(1)
    colors = [tuple(255 * np.random.rand(3)) for _ in range(N)]
    return colors


colors = random_colors(len(class_names))
class_dict = {
    name: color for name, color in zip(class_names, colors)
}


def apply_mask(image, mask, color, alpha=0.5):
    """apply mask to image"""
    for n, c in enumerate(color):
        image[:, :, n] = np.where(
            mask == 1,
            image[:, :, n] * (1 - alpha) + alpha * c,
            image[:, :, n]
        )
    return image


def display_instances(image, boxes, masks, ids, names, scores):
    cropped_images = []
    image_to_crop = image.copy()
    n_instances = boxes.shape[0]
    print("no of potholes in frame :", n_instances)
    if not n_instances:
        print('NO INSTANCES TO DISPLAY')
    else:
        assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]

    for i in range(n_instances):
        if not np.any(boxes[i]):
            continue
        y1, x1, y2, x2 = boxes[i]
        label = names[ids[i]]
        color = class_dict[label]
        score = scores[i] if scores is not None else None
        caption = '{} {:.2f}'.format(label, score) if score else label
        random_name = str(uuid.uuid4())
        mask = masks[:, :, i]

        cropped_images.append(image_to_crop[y1:y2, x1:x2])

        image = apply_mask(image, mask, color)
        image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        image = cv2.putText(image, caption, (x1, y1),
                            cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)

    return image, cropped_images


@app.route("/classifyImage", methods=['POST'])
@auth_required
def predict(uid):
    image_url = ""
    image_name = ""
    if request.is_json:
        try:
            json_data = request.get_json()
            image_url = json_data['image_url']
            image_name = json_data['image_name']
        except:
            return make_response(jsonify({"message": "Error, must include image url and name"}), 400)
    objID = ObjectId(uid)
    if not objID:
        return make_response(jsonify({'message': 'missing uid'}), 400)

    urllib.request.urlretrieve(image_url, "transferImage.jpg")
    frame = cv2.imread("/content/transferImage.jpg")
    results = model.detect([frame], verbose=0)
    r = results[0]
    masked_image = display_instances(
        frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
    cv2.imwrite("output.png", masked_image)
    bucket = storage.bucket("tailor-made-ece188.appspot.com")
    objId = ObjectId(uid)
    user = db.users.find_one_or_404({"_id": objId})
    username = user['username']
    blob = bucket.blob("images/" + username + "/" + image_name + "_classified")
    blob.upload_from_filename("output.png")
    # Opt : if you want to make public access from the URL
    blob.make_public()

    print("your file url", blob.public_url)
    public_url = blob.public_url
    # requests.post('https://tailor-made-ece188.herokuapp.com/addSegmentedImage', headers=)
    return make_response(jsonify({"image_url": blob.public_url}), 200)

# def classify(uid):
#     url = ""
#     image_name = ""
#     if request.is_json:
#         try:
#             json_data = request.get_json()
#             url = json_data['image_url']
#             image_name = json_data['image_name']

#         except:
#             return make_response(jsonify({"message": "Error, must include image url and name"}), 400)

#     url2 = url

#     url_img = Image.open(requests.get(url2, stream=True).raw)
#     convert_im=transforms.ToTensor()
#     # img = convert_im(url_img)

#     # file_name='/content/test1.jpg'
#     # im = Image.open(file_name, mode='r')
#     # im
#     # file = tf.io.read_file(file_name)
#     #    decode png file into a tensor

#     img=tf.image.decode_jpeg(requests.get(url2).content, channels=3, name="jpeg_reader")
#     # img = tf.image.decode_png(file, channels=3, dtype=tf.uint8)
#     print(img.shape)
#     x,y,z=img.shape
#     x_scale = x/128
#     y_scale = y/128
#     new_img = resize_image(img)
#     # plt.imshow(new_img)
#     # print(new_img.shape)
#     pred = unet(new_img[None])
#     temp=0
#     for i in pred:
#         temp = np.squeeze(pred[0])
#         temp = tf.argmax(pred[0], axis=-1)
#         temp=np.squeeze(temp)
#         # plt.imshow(temp)
#         cv2.imwrite("output.png", temp)

#     # pred = tf.image.convert_image_dtype(i/255.0, dtype=tf.uint8)

#     # im = Image.fromarray(pred)
#     # im.save("your_file.jpeg")
#     bucket = storage.bucket("tailor-made-ece188.appspot.com")
#     objId = ObjectId(uid)
#     user = db.users.find_one_or_404({"_id" : objId })
#     username = user['username']
#     blob = bucket.blob("images/" + username + "/" + image_name +"_classified")
#     blob.upload_from_filename("output.png")
#     # Opt : if you want to make public access from the URL
#     blob.make_public()

#     print("your file url", blob.public_url)
#     public_url = blob.public_url
#     # requests.post('https://tailor-made-ece188.herokuapp.com/addSegmentedImage', headers=)
#     return make_response(jsonify({"image_url": blob.public_url}), 200)

    # k = 0
    # colors=[]
    # min_x=np.ones(60)*128
    # max_x=np.zeros(60)
    # min_y=np.ones(60)*128
    # max_y=np.zeros(60)
    # color_count=np.zeros(60)

    # for i in pred:

    # save image

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
