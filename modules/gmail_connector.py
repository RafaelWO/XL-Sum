#@author: rweingartner-ortner

import smtplib
import imghdr
from email.mime.text import MIMEText
from email.message import EmailMessage
import logging
import os


class GmailConnector:
    """
    A class to connect to a gmail account and send an email.
    
    Attributes
    ----------
    gmailAccount : str
        The name of a gmail (or google) account, e.g. the email adress
    gmailPassword : str
        The password to a gmail (or google) account
    emailToSend : MIMEText
        An instance of MIMEText which defines the contents of the email which was build using the method 'buildEmail()'
    receivers : str array
        An array of email adresses to which the email should be sent

    Methods
    -------
    buildEmail(receivers, subject, body, body_type)
        Creates an email object using the given parameters and stores it
    sendEmail()
        Sends the stored email object using the stored gmail (or google) account

    Example
    -------
    The following code sends an email from the default account to a receiver mail:
    import gmail_connector as gc

    gmail = gc.GmailConnector()
    gmail.buildEmail(receivers= ['abcd@gmail.com'], subject= 'ConnectorTest', body= 'Plain text', body_type= 'plain')
    gmail.sendEmail()
    """

    def __init__(self, account, password, loggername=None):
        """
        Parameters
        ----------
        account : str
            The account name of a gmail (or google) account which will be the email sender
        password : str
            The password to the account
        loggername : str
            The name of an registered logger
        """

        self.gmailAccount = account
        self.gmailPassword = password
        if loggername:
            self.logger = logging.getLogger(loggername)


    def buildEmail(self, receivers, subject, body, body_type='plain', img_attachments=None):
        """Creates an email object using the given parameters and stores it
        
        Parameters
        ----------
        receivers : str array
            An array of email adresses to which the email should be sent
        subject : str
            The email subject
        body : str
            The body or content of the email
        body_type : str
            The type of the body or content (MIME type), e.g. 'plain', 'html'
        img_attachments : str array
            An array of image paths which should be attached to the email.
        """
        
        #email = MIMEText(body, body_type)
        email = EmailMessage()
        email.set_content(body, subtype= body_type)
        email['Subject'] = subject
        email['From'] = self.gmailAccount
        email['To'] = ','.join(receivers)

        # Open the files in binary mode.  Use imghdr to figure out the
        # MIME subtype for each specific image.
        if img_attachments:
            for file in img_attachments:
                if os.path.exists(file):
                    with open(file, 'rb') as fp:
                        img_data = fp.read()
                    email.add_attachment(img_data, maintype='image',
                                         subtype=imghdr.what(None, img_data))

        self.receivers = receivers
        self.emailToSend = email

    def sendEmail(self):
        """Sends the stored email object using the stored gmail (or google) account

        Connects to gmail via SMTP (SSL) with the host= 'smtp.gmail.com' and port= '465'
        """
        
        s = smtplib.SMTP_SSL(host= 'smtp.gmail.com', port= 465)
        s.login(user = self.gmailAccount, password = self.gmailPassword)
        s.sendmail(self.gmailAccount, self.receivers, self.emailToSend.as_string())
        s.quit()
        self.logger.info("[GmailConnector] Email with subject '%s' sent to [%s]" % (self.emailToSend['Subject'], self.emailToSend['To']))
