import json
from src.extract_timestamp import extract_timestamp, filter_string
from src.organize_timestamps import dateparser_vs_ownparser



json_filename = '/Users/danielwrede/Documents/read_event_messages/telegram_messages.json'


def main():

    # read messages from json file
    with open(json_filename, 'r', encoding='utf-8') as f:
        messages = json.load(f)

    count_dates = 0
    count_messages = 0

    # timestamp extraction
    for message in messages[:]:
        
        # and 'approved' not in message['timestamps']['comment']:
        if 'message' not in message or message['message'] == '':
            continue

        count_messages += 1

        # extract timestamps as list of dicts with date and time using own parser
        time_matches = extract_timestamp(
            message['message'])

        # TODO: filter dates.

        

        # compare dateparser with own parser results
        dateparser_vs_ownparser(message, time_matches)

        # add timestamps to message dict
        #message.setdefault('timestamps', {})
        #message['timestamps'] = timestamps

        if time_matches is not None:
            count_dates += 1

        # print message to file, if it contains a timestamp
        if time_matches:
            with open('file.txt', 'a', encoding='utf-8') as f:
                for match in time_matches:
                    f.write(f'{match["matching_substring"]}: {match["timestamp"]}, priority: {match["priority"]}, pattern_type: {match["pattern_type"]}\n')
                f.write(str(filter_string(message['message'])) + '\n\n')
                

    print(
        f'found {count_dates} dates in {len(messages)} messages')

    exit()

    # sort messages by timestamp
    messages.sort(key=lambda x: str(
        x.get('timestamps', float('inf'))))

    # save messages with timestamps
    # with open('new_message_list.json', 'w', encoding='utf-8') as f:
    #     json.dump(messages, f, indent=4, ensure_ascii=False)

    with open('file.txt', 'w', encoding='utf-8') as f:
        for message in messages:
            if 'message' in message and message['message'] != '':
                if message['timestamps']:
                    timestamps_sorted = sorted(message['timestamps'], key=lambda x: str(
                        x if x is not None else float('inf')))
                    f.write(str(timestamps_sorted) + '\n')
                    print('timestamps: ', timestamps_sorted)
                if message['parsedstamps']:
                    parsedstamps_sorted = sorted(message['parsedstamps'], key=lambda x: x[1])
                    f.write(str(parsedstamps_sorted) + '\n\n')
                    print('parsedstamps: ', parsedstamps_sorted)


if __name__ == "__main__":
    main()
