import datetime
import re


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

def dict_to_timestamp(datedict):
    """Convert dict to timestamp."""
    # If any of the date values is None, set them to 1 (January 1st).
    # If any of the time values is None, set them to 0 (midnight).
    if datedict['year'] is None:
        datedict['year'] = 0
    if datedict['month'] is None:
        datedict['month'] = 1
    if datedict['day'] is None:
        datedict['day'] = 1
    if datedict['hour'] is None or datedict['hour'] == '24':
        datedict['hour'] = 0
    if datedict['minute'] is None:
        datedict['minute'] = 0

    # all vars to int:
    datedict['year'] = int(datedict['year']) + 2000
    datedict['month'] = int(datedict['month'])
    datedict['day'] = int(datedict['day'])
    datedict['hour'] = int(datedict['hour'])
    datedict['minute'] = int(datedict['minute'])
    return datetime.datetime(year=datedict['year'], month=datedict['month'], day=datedict['day'], hour=datedict['hour'], minute=datedict['minute'])