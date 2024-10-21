from telethon.sync import TelegramClient, events
import asyncio
import os
from dotenv import load_dotenv

"""
Telegram Messaging Helper

This module provides functions to send messages, photos, documents, 
and videos to a Telegram channel using the Telethon library.

Background:
Telegram is a popular messaging app that is widely used for communication 
and sharing media. In this tutorial, we will learn how to use Python to send 
messages to a Telegram channel using the Telethon library.
"""

# Replace these with your actual values
api_id = ""  # Your API ID
api_hash = ""  # Your API Hash
bot_token = ""  # Your bot token
chat_id = ""  # The ID or username of the target chat

# Create a new client instance for the bot
bot = TelegramClient("bot", api_id, api_hash).start(bot_token=bot_token)


async def send_message(text, chat_id):
    async with bot:
        await bot.send_message(chat_id, text)


async def send_document(document, chat_id):
    async with bot:
        await bot.send_file(chat_id, document)


async def send_photo(photo, chat_id):
    async with bot:
        await bot.send_file(chat_id, photo)


async def send_video(video, chat_id):
    async with bot:
        await bot.send_file(chat_id, video)


async def main():
    # Sending a message
    await send_message(text="Hi Sujit!, How are you?", chat_id=chat_id)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
