import json
from src.extract_timestamp import extract_timestamp, filter_string
from src.organize_timestamps import dateparser_vs_ownparser



json_filename = '/Users/danielwrede/Documents/read_event_messages/telegram_messages.json'

def interpret_dates(time_matches):
    """Interpret dates by connecting date and time."""
    interpreted_dates = []
    for match in time_matches:
        if match['date1']:
            interpreted_dates.append(match)
    
    # if available, use double_clock match
    for match in time_matches:
        if 'double_clock' in match['pattern_type']:
            for date in interpreted_dates:
                if not date['clock1']:
                    date['clock1'] = match['clock1']
                    date['clock2'] = match['clock2']
    
    # else use min and max single clock matches
    valid_time_matches = [t for t in time_matches if t['clock1'] is not None or t['clock2'] is not None]
    minclock = min(valid_time_matches, key=lambda x: x['clock1'] or x['clock2'])
    maxclock = max(valid_time_matches, key=lambda x: x['clock1'] or x['clock2'])
    for date in interpreted_dates:
        if not date['clock1']:
            date['clock1'] = minclock['clock1'] or minclock['clock2']
            date['clock2'] = maxclock['clock1'] or maxclock['clock2']


    # for match in interpreted_dates:
    #     if not match['clock1']:
    #         match1, new_start_pos = next(((d['clock1'], d['start_pos']) for d in time_matches if d['start_pos'] > match['start_pos']), None)
    #         match2 = next((d['clock2'] for d in time_matches if d['start_pos'] > new_start_pos), None)
    print("time_matches: ", time_matches)
    print("interpreted_dates: ", interpreted_dates)
    return interpreted_dates


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

        # interpret dates by connecting date and time
        interpreted_dates = interpret_dates(time_matches)

        

        # compare dateparser with own parser results
        #dateparser_vs_ownparser(message, time_matches)

        # add timestamps to message dict
        #message.setdefault('timestamps', {})
        #message['timestamps'] = timestamps

        if time_matches is not None:
            count_dates += 1

        # print message to file, if it contains a timestamp
        if time_matches:
            with open('file.txt', 'a', encoding='utf-8') as f:
                for match in time_matches:
                    timestamp = ''
                    if match['date1']:
                        timestamp += match['date1'].strftime('%Y-%m-%d')
                    if match['clock1']:
                        timestamp += ' ' + match['clock1'].strftime('%H:%M')
                    if match['date2']:
                        timestamp += ' - ' + match['date2'].strftime('%Y-%m-%d')
                    if match['clock2']:
                        timestamp += ' ' + match['clock2'].strftime('%H:%M')

                    f.write(f'{match["matching_substring"]}: {timestamp}, priority: {match["priority"]}, pattern_type: {match["pattern_type"]}\n')
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
