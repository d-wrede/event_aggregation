import json
from src.extract_timestamp import extract_timestamp

json_filename = 'telegram_messages.json'


def main():
    # Opening messages file
    with open(json_filename, 'r', encoding='utf-8') as f:
        messages = json.load(f)

    # timestamp extraction
    for message in messages:
        if 'message' in message:
            timestamps = extract_timestamp(message['message'])
            print(message['message'][0:200])
            print(timestamps, '\n')
            # add timestamps to message
            message.setdefault('timestamps', {})
            message['timestamps']['startstamp'] = timestamps[0]
            message['timestamps']['endstamp'] = timestamps[1]

    # save messages with timestamps
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(messages, f, indent=4, ensure_ascii=False)

    # further terms to be extracted:
    # - sender/author
    # - place
    # - category


if __name__ == "__main__":
    main()
