# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     logger.py
   Description :
   Author :       unstoppable
   date：          21/11/2021
-------------------------------------------------
   Change Activity:
                   21/11/2021:
-------------------------------------------------
"""
import logging
from logging.handlers import TimedRotatingFileHandler


def logger_init_app(app):
    handler = TimedRotatingFileHandler("app/logs/flask.log", when="D", interval=1, backupCount=30, encoding="UTF-8",
                                       delay=False, utc=True)
    formatter = logging.Formatter("[%(asctime)s][%(filename)s:%(lineno)d][%(levelname)s] - %(message)s")
    handler.setFormatter(formatter)
    app.logger.addHandler(handler)
    app.logger.setLevel(logging.DEBUG)

