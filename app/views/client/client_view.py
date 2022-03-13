import base64
import json

import flask_login
from flask import request, jsonify

from app import login_manager
from app.models import Client, get_client, add_client
from app.blueprints import client_app
from detector import loader


def message(error, msg):
    data = {
        'error': error,
        'msg': msg
    }
    return jsonify(data)


@client_app.route('/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        user = get_client(data['username'])
        if user:
            return message(1, 'the username has been occupied')
        else:
            add_client(data['username'], data['password'])
            return message(0, "registration succeeded")
    except Exception as e:
        print(e)
        return message(1, "registration failed")


@client_app.route('/predict', methods=['GET', 'POST'])
@flask_login.login_required
def predict():
    if request.method == 'POST':
        data = request.get_json()
        # data["image"] 是str类型
        data = json.loads(data, strict=False)
        img_raw = base64.b64decode(data["image"])
        faces, scores = loader.predict_raw(img_raw)

        response = {
            "faceNum": len(faces)
        }
        faceList = []
        if len(faces) != 0:
            for i in range(len(faces)):
                face = faces[i]
                faceList.append({
                    # item() 取出张量具体位置的元素元素值，并且返回的是该位置元素值的高精度值，保持原元素类型不变
                    "x1": int(face[0].item()),
                    "y1": int(face[1].item()),
                    "x2": int(face[2].item()),
                    "y2": int(face[3].item()),
                    "score": scores[i].item()
                })

        response["faces"] = faceList
        print("The server is still running...")
        return jsonify(response)


# login_manager configuration
@login_manager.user_loader
def load_client(email):
    client = get_client(email)
    return client


@login_manager.request_loader
def request_loader(request):
    try:
        cID = request.get_json()['cID']
        client = get_client(cID)
        if client:
            return client
    except Exception as e:
        print(e)
        return


@login_manager.unauthorized_handler
def unauthorized_handler():
    return message(1, 'Unauthorized attempt')
