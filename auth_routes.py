# from datetime import date
import os
import datetime
# from posix import environ
from route_config import *
from flask import jsonify, make_response, request
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from bson.objectid import ObjectId
import jwt
import datetime

# TODO: USE IS_JSON and GETJSON


@app.route('/loginUsername', methods=['POST'])
def login_user():
    auth = request.authorization
    if request.is_json:
        print(request.get_json())
        auth = request.get_json()['authorization']
        # auth = request.get_json()['authorization']
    if not auth or not auth.username or not auth.password:
        return make_response(jsonify({'message': 'Missing authorization credentials'}),  401)
    print("auth.username is: " + auth.username)
    user = db.users.find_one_or_404({"username": auth.username})
    print(user)
    if check_password_hash(user["password"], auth.password):
        jwt_token = jwt.encode({
            '_id': str(user['_id']),
            'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=4)
        },
            os.environ.get("PASSWORD_SALT"), "HS256")
        return jsonify({'jwt_token': jwt_token})
    return make_response(jsonify({'message': 'Invalid username or password'}),  401)



def auth_required(f):
    @wraps(f)
    def decorator(*args, **kwargs):
        auth_token = None
        if 'auth-token' in request.headers:
            auth_token = request.headers['auth-token']
        if not auth_token:
            return make_response(jsonify({'message': 'no auth token'}), 401)
        try:
            jwt_data = jwt.decode(auth_token, os.environ.get(
                "PASSWORD_SALT"), algorithms=["HS256"])
            uid = ObjectId(jwt_data['_id'])
            db.users.find_one_or_404({"_id": uid})
        except:
            return make_response(jsonify({"message": 'token is invalid'}), 401)

        print('authenticated')
        return f(uid, *args, **kwargs)
    return decorator
