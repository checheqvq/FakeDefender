from threading import Thread
from flask import current_app
from flask_mail import Message, Mail

mail = Mail()


def send_async_email(app, msg):
    with app.app_context():
        mail.send(msg)


def send_link(to, token):
    app = current_app._get_current_object()
    msg = Message(
        subject='激活',
        recipients=[to],
        body=f'''
            请点击下面的链接进行激活，有效期一天：\n
            http://10.136.126.13:5000/client/confirm/{token}\n
            '''
    )
    thr = Thread(target=send_async_email, args=[app, msg])
    thr.start()
    return thr
