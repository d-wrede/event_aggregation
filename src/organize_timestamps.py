import datetime
import re
import dateparser.search
from dateparser_data.settings import default_parsers

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


