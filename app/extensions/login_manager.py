# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     login_manager.py
   Description :
   Author :       unstoppable
   date：          13/11/2021
-------------------------------------------------
   Change Activity:
                   13/11/2021:
-------------------------------------------------
"""
from flask_login import LoginManager

login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.login_message = 'Login required.'
