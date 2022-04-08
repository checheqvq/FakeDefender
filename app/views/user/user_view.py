import base64
import flask_login

from detector import loader
from flask import jsonify, request, current_app
from app.extensions import login_manager, send_verif_code, phone_code_map, del_verif_code
from app.models.user import User, get_user, add_user
from app.blueprints import user_app

login_manager.login_view = 'login'
login_manager.login_message = '需要登陆后才能访问本页'


def message(status, msg, data):
    jsonObj = {
        'code': status,
        'msg': msg,
        'data': data
    }
    return jsonify(jsonObj)


@user_app.route('/', methods=['GET', 'POST'])
def hello():
    return message(0, "hello", None)


@user_app.route('/get_verif_code', methods=['POST'])
def get_verif_code():
    try:
        data = request.get_json()
        user = get_user(data['phone'])
        if user:
            current_app.logger.info(f"{data['phone']}已经被注册了")
            return message(1, '该手机已经被注册了', None)
        else:
            current_app.logger.info(f"{data['phone']}已经发送验证码")
            send_verif_code(data['phone'], phone_code_map)
            return message(0, "已发送短信验证码，请填入验证码", None)
    except Exception as e:
        current_app.logger.info("获取验证码失败")
        # return message('bad register')
        return message(1, "获取验证码失败", None)


@user_app.route('/register', methods=['GET', 'POST'])
def register():
    try:
        data = request.get_json()
        current_app.logger.info(f"手机号验证码映射：{phone_code_map}")
        print(phone_code_map)
        if data['code'] == phone_code_map[data['phone']]:
            add_user(data['phone'], data['password'])
            del_verif_code(data['phone'], phone_code_map)
            current_app.logger.info(f"{data['phone']}注册成功")
            return message(0, "注册成功", None)
        else:
            current_app.logger.info(f"{data['phone']}验证码错误")
            return message(1, "验证码错误", None)
    except Exception as e:
        print(e)
        current_app.logger.info("注册失败")
        return message(1, "注册失败", None)


@user_app.route('/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        user = User.query.get(data['phone'])
        if user.check_password(data['password']):
            flask_login.login_user(user, remember=True)
            current_app.logger.info(f"{data['phone']}登录成功")
            return message(0, "登陆成功", None)
        else:
            current_app.logger.info(f"{data['phone']}登录失败")
            return message(1, '账号或密码错误', None)
    except Exception as e:
        print(e)
        return message(1, '没有此用户', None)
        # return message(1)


@user_app.route('/logout', methods=['GET', 'POST'])
@flask_login.login_required
def logout():
    flask_login.logout_user()
    return message(0, '退出成功', None)


@user_app.route('/predict', methods=['GET', 'POST'])
@flask_login.login_required
def predict():
    if request.method == 'POST':
        data = request.get_json()
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
        return message(0, "识别成功", response)


@login_manager.user_loader
def load_user(phone):
    user = User.query.get(phone)
    return user


@login_manager.request_loader
def request_loader(request):
    try:
        phone = request.get_json()['phone']
        user = get_user(phone)
        if user:
            return user
    except Exception:
        return


@login_manager.unauthorized_handler
def unauthorized_handler():
    return message(0, '未授权访问', None)

