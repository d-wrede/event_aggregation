# Import libraries
import asyncio, configparser
from telethon import TelegramClient
import spacy
from datetime import datetime
import json

# Load spacy model for English
nlp = spacy.load("de_core_news_sm")

# Reading Configs
config = configparser.ConfigParser()
config.read("config.ini")

# Setting configuration values
api_id = config['Telegram']['api_id']
api_hash = str(config['Telegram']['api_hash'])
phone = config['Telegram']['phone']
username = config['Telegram']['username']
channel = 'Ecstatic_Dance'
chat_id = int(config['Channels'][channel])

# Create a client object with your credentials
client = TelegramClient(username, api_id, api_hash)

# Define an async function to get messages from channel
async def get_messages():
    # Start the client
    await client.start()
    me = await client.get_me()
    print(f"{me.first_name} {me.last_name}")
    print("id ", me.id)
    print("phone +", me.phone)

    # print contact details
    dialogs = await client.get_dialogs()
    for dialog in dialogs:
        if 'dance'.lower() in dialog.name.lower():
            print(f"{dialog.name}, {dialog.id}") 
            # print phone number, if available    
            if hasattr(dialog.entity, 'phone'):
                print("\tphone: +", dialog.entity.phone)
    
    group_entity = await client.get_input_entity(chat_id)
    print("channel: ", group_entity.stringify())
    connection_status = client.is_connected()
    print("connection_status: ", connection_status)

    messages = await client.get_messages(group_entity, limit = 10)
    print("type(messages): ", type(messages))
    print("type(message): ", type(messages[0]))

    # print contact details
    dialogs = await client.get_dialogs()
    for dialog in dialogs:
        if dialog.id == chat_id:
            print(f"{dialog.name}, {dialog.id}")

    ## Save the messages to a json file
    messages_dict = [message.to_dict() for message in messages]
    with open(f'telegram_messages_{channel}_{datetime.date(datetime.now())}.json', 'w') as f:
        json.dump(messages_dict, f, indent=4, sort_keys=True, default=str)

                    
# Run the async function using asyncio.run()
asyncio.run(get_messages(),debug=True)