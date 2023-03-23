import json
from src.extract_timestamp import extract_timestamp, filter_string

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
            message['timestamps']['comment'] = timestamps[2]

    # save messages with timestamps
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(messages, f, indent=4, ensure_ascii=False)

    with open('file.txt', 'w', encoding='utf-8') as f:
        for message in messages:
            if 'message' in message:
                f.write(filter_string(message['message']) + '\n')
                f.write('-'*50 + '\n')
                f.write('startstamp: ' +
                        str(message['timestamps']['startstamp']) + '\n')
                f.write('endstamp: ' +
                        str(message['timestamps']['endstamp']) + '\n')
                f.write(str(message['timestamps']['comment']) + '\n\n')

    # further terms to be extracted:
    # - sender/author spaCy?
    # - place
    # - category
    # - URL



if __name__ == "__main__":
    main()
