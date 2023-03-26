import json
from src.extract_timestamp import extract_timestamp, filter_string

json_filename = 'telegram_messages.json'


def main():
    # Opening messages file
    with open(json_filename, 'r', encoding='utf-8') as f:
        messages = json.load(f)

    count_dates = 0
    # timestamp extraction
    for message in messages:
        if 'message' in message and message['message'] != '':
            timestamps = extract_timestamp(message['message'])
            # print(message['message'][0:200])
            print("timestamps: ", timestamps, '\n')
            # add timestamps to message dict
            message.setdefault('timestamps', {})
            message['timestamps']['startstamp'] = timestamps[0]
            message['timestamps']['endstamp'] = timestamps[1]
            message['timestamps']['comment'] = timestamps[2]
            if timestamps[0] is not None:
                count_dates += 1
    
    # sort messages by timestamp
    messages.sort(key=lambda x: str(x.get('timestamps', {}).get('startstamp', '')))


    print(f'found {count_dates} dates in {len(messages)} messages')
    # save messages with timestamps
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(messages, f, indent=4, ensure_ascii=False)

    with open('file.txt', 'w', encoding='utf-8') as f:
        for message in messages:
            if 'message' in message and message['message'] != '':
                f.write('startstamp: ' +
                        str(message['timestamps']['startstamp']) + '\n')
                f.write('endstamp: ' +
                        str(message['timestamps']['endstamp']) + '\n')
                if message['timestamps']['comment'] != '':
                    f.write(str(message['timestamps']['comment']) + '\n')
                f.write('-'*50 + '\n')
                f.write(filter_string(message['message']) + '\n\n')

    # further terms to be extracted:
    # - sender/author spaCy?
    # - place
    # - category
    # - URL



if __name__ == "__main__":
    main()
