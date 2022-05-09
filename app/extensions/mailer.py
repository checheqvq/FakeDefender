import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.utils import parseaddr, formataddr


class Mailer:
    def __init__(self):
        self.sender = "1092003187@qq.com"
        self.smtpserver = "smtp.qq.com:465" # 邮件服务器，如果是qq邮箱那就是这个了，其他的可以自行查找
        self.username = '1092003187@qq.com' # 这里还是你的邮箱
        self.password = 'kmerudljmtphicje' # 上面获取的SMTP授权码，相当于是一个密码验证

    def mail(self, receiver, subject, body, image):
        msgRoot = MIMEMultipart('related') # 邮件类型，如果要加图片等附件，就得是这个
        msgRoot['Subject'] = subject # 邮件标题，以下设置项都很明了
        msgRoot['From'] = self.sender
        msgRoot['To'] = receiver # 发给单人
        # 以下为邮件正文内容，含有一个居中的标题和一张图片
        content = MIMEText('''<html>
                                <head>
                                    <style>#string{text-align:center;font-size:25px;}</style>
                                </head>
                                <body>
                                    <div id="string">
                                        <p>%s</p>
                                        <div>
                                            <img src="data:image/png;base64,%s" alt="image">
                                        </div>
                                    </div>
                                </body>
                            </html>'''%(body, image),'html','utf-8')
        # 如果有编码格式问题导致乱码，可以进行格式转换：
        # content = content.decode('utf-8').encode('gbk')
        msgRoot.attach(content)
        # 连接邮件服务器，因为使用SMTP授权码的方式登录，必须是465端口
        smtp = smtplib.SMTP_SSL(self.smtpserver)

        smtp.login(self.username, self.password)
        smtp.sendmail(self.sender, receiver, msgRoot.as_string())
        smtp.quit()


# if __name__ == '__main__':
#     mailer = Mailer()
#     mailer.mail("crackedpoly@outlook.com", "FakeDefender告警", f"您监护的用户当前处于危险当中，用户手机号为{phone}","iVBORw0KGgoAAAANSUhEUgAAABgAAAAVCAYAAABc6S4mAAAAJUlEQVQ4jWNUUFT5z0BDwERLw0ctGLVg1IJRC0YtGLVg1AKiAQC7nQGO0W75+wAAAABJRU5ErkJggg==")