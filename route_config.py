from flask import Flask, request, jsonify
from dotenv import load_dotenv
from flask_pymongo import PyMongo
import os
import pickle
load_dotenv()

app = Flask(__name__)
app.config["MONGO_URI"] = os.environ.get("MONGO_URI", "")
mongo = PyMongo(app)
db = mongo.db
unet = pickle.load(open('model2.sav', 'rb'))


@app.route("/", methods=['GET'])
def our_server():
    return "Our backend server!"
