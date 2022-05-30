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
