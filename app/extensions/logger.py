import logging
from logging.handlers import TimedRotatingFileHandler


def logger_init_app(app):
    handler = TimedRotatingFileHandler("app/logs/flask.log", when="D", interval=1, backupCount=30, encoding="UTF-8",
                                       delay=False, utc=True)
    formatter = logging.Formatter("[%(asctime)s][%(filename)s:%(lineno)d][%(levelname)s] - %(message)s")
    handler.setFormatter(formatter)
    app.logger.addHandler(handler)
    logging.getLogger('werkzeug').addHandler(handler)
    app.logger.setLevel(logging.DEBUG)
    app.logger.info("-------------------------------New Running--------------------------------")

