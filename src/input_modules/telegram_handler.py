from telethon.sync import TelegramClient, events
import os
from dotenv import load_dotenv
import logging
from pathlib import Path
import queue
import grpc
from concurrent import futures
import grpc
from llm_pb2 import QueryRequest  # Import the generated classes from your .proto file
from llm_pb2_grpc import LLMStub  # Import the generated gRPC client stub

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

# Set up gRPC client
llm_channel = grpc.insecure_channel('localhost:50051')  # Replace with your server's address
llm_client = LLMStub(llm_channel)

@client.on(events.NewMessage)
async def keyword_responder(event):
    message = event.text.lower()

    # Enqueue the message for processing by the bot/agent
    message_queue.put(message)

    # Send the message to the LLM server
    try:
        request = QueryRequest(user_id=str(event.sender_id), query=message)
        response = llm_client.Query(request)
        llm_response = response.answer
    except grpc.RpcError as e:
        llm_response = "Sorry, I couldn't process your request at the moment."

    # Respond with the LLM's answer
    await event.respond(llm_response)
    logging.info(f'Message received from {event.sender_id}: {event.text}')

# Start the client
client.start()
client.run_until_disconnected()