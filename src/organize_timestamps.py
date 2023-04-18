import datetime
import re
import dateparser.search
from dateparser_data.settings import default_parsers


def get_min_date(message):
        if not message.get('timestamps') or not isinstance(message['timestamps'], list):
            return '9999-12-31'

        min_date = message['timestamps'][0]['date1']

        for entry in message['timestamps']:
            if entry['date1'] and entry['date1'] < min_date:
                min_date = entry['date1']

        return min_date


def interpret_dates(time_matches):
    """Interpret dates by connecting date and time."""
    interpreted_dates = []
    for match in time_matches:
        if match['date1']:
            interpreted_dates.append({'date1': match['date1'], 'clock1': match['clock1'],
                                      'date2': match['date2'], 'clock2' : match['clock2']})
    
    # if available, use double_clock match
    for match in time_matches:
        if 'double_clock' in match['pattern_type']:
            for date in interpreted_dates:
                if not date['clock1']:
                    date['clock1'] = match['clock1']
                    date['clock2'] = match['clock2']
    
    # else use min and max single clock matches
    valid_time_matches = [t for t in time_matches if t['clock1'] is not None or t['clock2'] is not None]
    if valid_time_matches:
        minclock = min(valid_time_matches, key=lambda x: x['clock1'] or x['clock2'])
        maxclock = max(valid_time_matches, key=lambda x: x['clock1'] or x['clock2'])
        for date in interpreted_dates:
            if not date['clock1']:
                date['clock1'] = minclock['clock1'] or minclock['clock2']
                date['clock2'] = maxclock['clock1'] or maxclock['clock2']
    else:
        print("no valid time matches found")

    # serialize timestamps
    for date in interpreted_dates:
        if date['clock1']:
            date['clock1'] = date['clock1'].strftime('%H:%M')
        if date['clock2']:
            date['clock2'] = date['clock2'].strftime('%H:%M')
        if date['date1']:
            date['date1'] = date['date1'].strftime('%Y-%m-%d')
        if date['date2']:
            date['date2'] = date['date2'].strftime('%Y-%m-%d')

    return interpreted_dates



def check_timestamps(timestamps, parsedstamps):
    """
    This function filters the parsedstamps. It checks if the parsedstamps are in the timestamps list or in the blacklist or blackregexlist.
    Check if parsedstamps are in timestamps, on the blacklist or blackregexlist. If not print them."""
    # blacklist for parsedstamps
    blacklist = ['mit', 'in die', 'in die', 'und die', 'so', 'die', '40',
        '2-3', '1,5', 'jahren', 'und mit', 'mit einem', '18', '14€/12€', '10€/8€', '15€ / 20€', '0163', '60€', '6 mit', '12-22']
    blackregexlist = [r'^\d{2}$', r'^\d{1,2}-\d{1,2}$',
        r'^\d{1,2}€[/-]\d{2}€', r'^\d{1-3}$', r'(€|Euro)']

    def match_patterns(blackregexlist, string):
        """Check if string matches any of the blackregexlist."""
        for blackregex in blackregexlist:
            if re.search(blackregex, string, flags=re.IGNORECASE):
                return True
        return False

    if len(parsedstamps) == 0:
        print("parsedstamps is empty")
        return
    if parsedstamps is None:
        print("parsedstamps is None")
        return
    switch_message = False
    for stamp in parsedstamps:
        # generate date and time tuples for comparison
        tups_ts = tuple(dt[1].strftime(
            '%Y-%m-%d') for dt in timestamps if dt is not None)  # if dt is not None else None
        tups_time = tuple(dt[1].strftime(
            '%H:%M') for dt in timestamps if dt is not None)

        if match_patterns(blackregexlist, stamp[0]):
            continue
        if stamp[1].strftime('%Y-%m-%d') in tups_ts:
            continue
        if stamp[1].year > 2025:
            continue

        if len(timestamps) > 0:
            min_timestamp = min(timestamps, key=lambda x: x[1])[1]
            max_timestamp = max(timestamps, key=lambda x: x[1])[1]
            if min_timestamp < stamp[1] < max_timestamp:
                continue
        if tups_ts and min(tups_ts) < stamp[1].strftime('%Y-%m-%d') < max(tups_ts):
            continue
        if tups_time and min(tups_time) < stamp[1].strftime('%H:%M') < max(tups_time):
            continue
        if stamp[0].lower() in blacklist:
            continue
        if stamp[1].strftime('%H:%M') in tups_time:
            continue
        if stamp[0].lower() in blacklist:
            continue

        print("parsedstamp: ", stamp)
        switch_message = True

    if switch_message:
        print("timestamps:  ", timestamps)
        print("")
        switch_message = False


def dateparser_vs_ownparser(message, time_matches):
    """Compare dateparser with own parser."""
    default_parsers.reverse()
    settings = {
        # Order to prioritize when parsing ambiguous dates (Day, Month, Year)
        'DATE_ORDER': 'DMY',
        'PREFER_DATES_FROM': 'future',  # Prefer dates from the future
        'STRICT_PARSING': False,  # Allow for approximate parsing
        'NORMALIZE': True,  # Normalize whitespace and remove extra spaces within the date string
        'RETURN_AS_TIMEZONE_AWARE': False,  # Return timezone-aware datetime objects
        'DEFAULT_LANGUAGES': ["de"],
        # 'LANGUAGE_DETECTION_CONFIDENCE_THRESHOLD': 0.5,
        'PARSERS': default_parsers
        # 'RETURN_TIME_AS_PERIOD': True,
        # 'PREFER_DAY_OF_MONTH': 'first',
    }

    # parse with dateparser
    date_message = dateparser.parse(message['date'])
    settings['RELATIVE_BASE'] = date_message
    parsedstamps = dateparser.search.search_dates(
        message['message'], languages=['de'], settings=settings)

    # Set seconds to 0 for each parsed datetime object
    parsedstamps_no_seconds = []
    if parsedstamps:
        # step through parsedstamps and remove seconds
        for date_str, date_obj in parsedstamps:
            date_obj = date_obj.replace(second=0)
            parsedstamps_no_seconds.append((date_str, date_obj))
        parsedstamps = parsedstamps_no_seconds

        #check if parsedstamps are in timestamps
        timestamps = [
            (stampdict['matching_substring'], stampdict['timestamp']) for stampdict in time_matches]

        check_timestamps(timestamps, parsedstamps)


