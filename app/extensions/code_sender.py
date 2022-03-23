import random
from flask import current_app
from urllib import request, parse
from hashlib import md5


def MD5(str):
    m = md5()
    m.update(str.encode("utf8"))
    return m.hexdigest()


statusStr = {
    '0': '短信发送成功',
    '-1': '参数不全',
    '-2': '服务器空间不支持,请确认支持curl或者fsocket,联系您的空间商解决或者更换空间',
    '30': '密码错误',
    '40': '账号不存在',
    '41': '余额不足',
    '42': '账户已过期',
    '43': 'IP地址限制',
    '50': '内容含有敏感词'
}

smsapi = "http://api.smsbao.com/"
# 短信平台账号
user = 'gopher'
# 短信平台密码
password = MD5('NE@5t4LniTJzmh')

phone_code_map = {}


def send_verif_code(phone, phone_code_map):
    code = random.randint(1000, 9999)
    data = parse.urlencode({'u': user, 'p': password, 'm': phone,
                            'c': f'【FakeDefender】您的验证码是{code}。如非本人操作，请忽略本短信。'})
    send_url = smsapi + 'sms?' + data
    response = request.urlopen(send_url)
    the_page = response.read().decode('utf-8')
    if the_page == '0':
        phone_code_map[phone] = str(code)
    current_app.logger.info(f"{phone} 的验证码发送状态{statusStr[the_page]}")


def del_verif_code(phone, phone_code_map):
    phone_code_map.pop(phone)
