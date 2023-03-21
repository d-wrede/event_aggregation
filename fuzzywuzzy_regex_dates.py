from fuzzywuzzy import process
import json
import re

# Example list of strings to search
dates = [
    "22. Mai 2023", "23-05-2023", "22 05 23", "23 Mai", "23. Mai", "Maerz 23",
    "22 Mai", "22. Mai", "22.05.2023", "22-05-2023", "22 05 2023", "22.Mai.2023",
    "22.05.23", "22.05", "22 Mai 23", "22. Mai 23", "22.5.23", "22/05/2023",
    "22 05.2023", "22 05-23", "22. Mai 2023 12:30", "22 Mai 23 12:30:45",
    "22.5.23 12:30:45", "22.05.2023 12:30:45", '1.1.22', '01-02-2022', '3. März 22',
    '4/4/2022', '05. Mai 2022', '6/6/22', '7. Juli 22', '08-08-22', '9. September 22',
    '10.10.2022', '11. Nov. 22', '12-Dezember-22', '13.13.22', '14.14.2022', 
    '15. Oktober 2022', '16. Januar 22', '17-Februar-22', '18. 18.22', '19/19/2022',
    '20. Dez. 2022', '21/21/22', '22-03-22', '23- April-2022', '24/24/22', '25.25.2022',
    '26. Juni 22', '27-07-22', '28. August 2022', '29/09/2022', '30. November 2022',
    '31/12/2022', '31. Januar 2022'
]
# with open('deutsche_Geschichte.txt', 'r', encoding='utf-8') as f:
#     text = f.read()
  
# Opening JSON file as dict
with open('telegram_messages.json', 'r', encoding='utf-8') as f:
    messages = json.load(f)

# Example patterns to match
patterns = [
    # "DD-MM-YYYY" or "DD.MM.YYYY" or "DD MM YYYY" or  -- "DD/MM/YYYY"
    #r"[1-9]|[12][0-9]|3[01][-.\s][1-9]|[12][0-9]|3[01][-.\s]([19]|[20])?(\d{2})?",
    # "DD. Monat YYYY" or "DD. Monat YY"
    #r"(?:J(anuar|u(n|li))|Februar|Mä(rz|i)|A(pril|ugust)|(((Sept|Nov|Dez)em)|Okto)ber)",
    # regex structured with groups
    r"\b(0?[1-9]|[1-2][0-9]|3[01])?[-.\s/]*(Jan|Feb|März|Maerz|April|Ma[iy]|Jun|Jul|Aug|Sep|Okt|Nov|Dez)[a-z]*[-.\s/]*(20)?(2[2-5])?\b",
    r"\b(0?[1-9]|[1-2][0-9]|3[01])[-.\s/]+0?([1-9]|1[0-2])[-.\s/]+(20)?(2[2-5])?\b"
    
    # "Monat, YYYY" or "Monat, YY"
    #"(Jan|Feb|März|Apr|Mai|Jun|Jul|Aug|Sept|Okt|Nov|Dez)[a-z]*[\s]?\d{2}(\d{2})?",
    # "Mon DD, YY(YY)"
    #r"(Jan|Feb|März|Apr|Mai|Jun|Jul|Aug|Sept|Okt|Nov|Dez)[a-z]*[-.\s]?((0?[1-9])|[12][0-9]|3[01])?[.,/\s][\s]?(\d{2})?(\d{2})?",
    # Finally he went crazy (and tried to combine all the patterns)
    #r"(?:[1-9]|[12][0-9]|3[01][-.\s])?(?:Jan(?:uar)?|Feb(?:ruar)?|März|Mar(?:ch)?|Apr(?:il)?|Mai|Jun(?:i)?|Jul(?:i)?|Aug(?:ust)?|Sep(?:tember)?|Okt(?:ober)?|Nov(?:ember)?|Dez(?:ember)?)[-.\s](?:[1-9]|[12][0-9]|3[01][-.\s])?(?:\d{2}\d{2})?(?:\d{2})?",
    # Maybe this grew even longer
    #r"\b(\d{1,2}[-.\s]\d{1,2}[-.\s]\d{2}(\d{2})?|\d{1,2}[-.\s]?[-.\s]?(jan|feb|märz|apr|mai|jun|jul|aug|sept|okt|nov|dez)[a-z]*[-.\s]?(\d{2})?(\d{2})?|(jan|feb|märz|apr|mai|jun|jul|aug|sept|okt|nov|dez)[a-z]*[-.\s]?((0?[1-9])|[12][0-9]|3[01])?[.,/\s][\s]?(\d{2})?(\d{2})?)\b",

    #r"(?<!\d)(?:[1-9]|[12]\d|3[01])[-.\s](?:Jan(?:uar)?|Feb(?:ruar)?|März|Ma[iy]|Jun|iul|Aug|Sep(?:tember)?|Okt(?:ober)?|Nov(?:ember)?|Dez(?:ember)?)[a-z]*[-.\s](?:\d{4}|\d{2})(?!\d)"
]
# Compile your patterns with the re.IGNORECASE flag
#compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
# join all patterns into one big pattern using | operator
# pattern = "|".join(patterns) [('18', '3', '', '', '')]

for message in messages:
    dateset = set()
    print("message: ", message['message'][0:100])

    for i, pattern in enumerate(patterns):
        dates = re.findall(pattern, message['message'], flags=re.IGNORECASE)
        # if the dates are not in the set, add them
        #print(f"dates related to pattern {i}: \n{dates}\n")
        for date in dates:
            day = date[0]
            month = date[1]
            year = date[3]
            print(f"Day: {day}, Month: {month}, Year: {'20' + year if len(year) else ''}")
        for date in dates:
            dateset.add(date)
    #print("len(dateset): ", len(dateset))
    #print("dateset: ", dateset)

    
    # Find best matches using fuzzywuzzy
    # matches = {}
    # for date in dateset:
    #     results = process.extract(pattern, date, limit=None)
    #     for result, score in results:
    #         if result in matches:
    #             if score > matches[result]:
    #                 matches[result] = score
    #         else:
    #             matches[result] = score
    
    # print("dateset based on findall: ", dateset)
    # print("matches: ", matches)
    # print("len(matches): ", len(matches))