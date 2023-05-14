import json

json_filename = (
    "/Users/danielwrede/Documents/read_event_messages/telegram_messages.json"
)

with open(json_filename, "r", encoding="utf-8") as f:
    messages = json.load(f)

for message in messages:
    # check if message contains a timestamp and delete the timestamp
    if "timestamps" in message:
        del message["timestamps"]

with open('cleaned_tel_mes.json', "w", encoding="utf-8") as f:
    json.dump(messages, f, indent=4, sort_keys=True)