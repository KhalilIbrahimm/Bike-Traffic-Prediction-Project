
# Required libraries are imported.
#import telegram 
import requests  # Requests library is used for making HTTP calls to Telegram API.


# Define a Bot class.
class Bot:
    # Constructor for the Bot class.
    def __init__(self, MESSSAGE):
        # Telegram bot token. Be careful! This should not be exposed, especially in shared or public code.
        self.TOKEN = ""  
        # Chat ID where the message will be sent.
        self.CHAT_ID = ""
        # Message to be sent.
        self.MESSAGE = MESSSAGE

    # Method to send the message.
    def send_message(self):
        url = f"https://api.telegram.org/bot{self.TOKEN}/sendMessage?chat_id={self.CHAT_ID}&text={self.MESSAGE}"
        return requests.get(url).json() # this sends the message
