from telethon.sync import TelegramClient, events
import os
from dotenv import load_dotenv
import logging
from pathlib import Path
import queue
import grpc
from concurrent import futures


"""
Telegram Messaging Helper

This module provides functions to send messages, photos, documents, 
and videos to a Telegram channel using the Telethon library.

Background:
Telegram is a popular messaging app that is widely used for communication 
and sharing media. In this tutorial, we will learn how to use Python to send 
messages to a Telegram channel using the Telethon library.
"""

# load api keys
dotenv_path =  Path(__file__).resolve().parent.parent.parent / '.env'
load_dotenv(dotenv_path=dotenv_path)

# Access the API key
api_id = os.getenv('API_ID')
api_hash = os.getenv('API_HASH')
chat_id = os.getenv('CHAT_ID')
bot_token = os.getenv('BOT_TOKEN')

# Create a new client instance for the bot
client = TelegramClient('bot', api_id, api_hash).start(bot_token=bot_token)

# Create a queue for messages
message_queue = queue.Queue()

@client.on(events.NewMessage(pattern='/start'))
async def start(event):
    await event.respond('Hello! I am a Telethon bot. How can I assist you today?')
    logging.info(f'Start command received from {event.sender_id}')

@client.on(events.NewMessage)
async def keyword_responder(event):
    message = event.text.lower()

    # Enqueue the message for processing by the bot/agent
    message_queue.put(message)

    responses = {
        'hello': 'Hi there! How can I help you today?',
        'how are you': 'I am just a bot, but I am here to assist you!',
        'what is your name': 'I am MyAwesomeBot, your friendly Telegram assistant.',
        'bye': 'Goodbye! Have a great day!',
        'time': 'I cannot tell the current time, but you can check your device!',
        'date': 'I cannot provide the current date, but your device surely can!',
        'weather': 'I cannot check the weather, but there are many apps that can help you with that!',
        'thank you': 'You are welcome!',
        'help me': 'Sure! What do you need help with?',
        'good morning': 'Good morning! I hope you have a great day!',
        'good night': 'Good night! Sweet dreams!',
        'who created you': 'I was created by a developer using the Telethon library in Python.',
    }

    response = responses.get(message, None)

    if response:
        await event.respond(response)
    else:
        # Default response
        default_response = (
            "I didn't understand that command. Here are some commands you can try:\n"
            "/start - Start the bot\n"
            "/help - Get help information\n"
            "/info - Get information about the bot\n"
            "/echo <message> - Echo back the message\n"
        )
        await event.respond(default_response)
    logging.info(f'Message received from {event.sender_id}: {event.text}')

# Start the client
client.start()
client.run_until_disconnected()