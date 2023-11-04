
# Required libraries are imported.
#import telegram 
import requests  # Requests library is used for making HTTP calls to Telegram API.

## Getting Started with Telegram Integration:
# To integrate with Telegram and obtain your token and chat ID, follow the steps outlined in this guide: https://docs.influxdata.com/kapacitor/v1/reference/event_handlers/telegram


# Define a Bot class.
class Bot:
    # Constructor for the Bot class.
    def __init__(self, MESSSAGE):
        # Telegram bot token. 
        self.TOKEN = "WRITE YOUR TOKEN HERE"  
        # Chat ID where the message will be sent.
        self.CHAT_ID = "WRITE YOUR CHAT ID HERE"
        # Message to be sent.
        self.MESSAGE = MESSSAGE

    # Method to send the message.
    def send_message(self):
        url = f"https://api.telegram.org/bot{self.TOKEN}/sendMessage?chat_id={self.CHAT_ID}&text={self.MESSAGE}"
        return requests.get(url).json() # this sends the message
