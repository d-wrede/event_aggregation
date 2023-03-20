from fuzzywuzzy import process

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
with open('deutsche_Geschichte.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Example patterns to match
patterns = [
    # "DD-MM-YYYY" or "DD.MM.YYYY" or "DD MM YYYY" or  -- "DD/MM/YYYY"
    "[1-9]|[12][0-9]|3[01][.-/\s][1-9]|[12][0-9]|3[01][.-/\s]([19]|[20])?(\d{2})?",
    # "DD. Monat YYYY" or "DD. Monat YY"
    "[1-9]|[12][0-9]|3[01]?[.-/\s]?[.-/\s]?(Jan|Feb|März|Apr|Mai|Jun|Jul|Aug|Sept|Okt|Nov|Dez)[a-z]*[.-/\s]?(\d{2})?(\d{2})?",
    # "Monat, YYYY" or "Monat, YY"
    #"(Jan|Feb|März|Apr|Mai|Jun|Jul|Aug|Sept|Okt|Nov|Dez)[a-z]*[\s]?\d{2}(\d{2})?",
    # "Mon DD, YY(YY)"
    "(Jan|Feb|März|Apr|Mai|Jun|Jul|Aug|Sept|Okt|Nov|Dez)[a-z]*[.-/\s]?((0?[1-9])|[12][0-9]|3[01])?[.,/\s][\s]?(\d{2})?(\d{2})?",
    # Finally he went crazy (and tried to combine all the patterns)
    "(?:[1-9]|[12][0-9]|3[01][.-\s])?(?:Jan(?:uar)?|Feb(?:ruar)?|März|Mar(?:ch)?|Apr(?:il)?|Mai|Jun(?:i)?|Jul(?:i)?|Aug(?:ust)?|Sep(?:tember)?|Okt(?:ober)?|Nov(?:ember)?|Dez(?:ember)?)[.-\s](?:[1-9]|[12][0-9]|3[01][.-\s])?(?:\d{2}\d{2})?(?:\d{2})?",
    # Maybe this grew even longer
    "(?i)\b(\d{1,2}[.-/\s]\d{1,2}[.-/\s]\d{2}(\d{2})?|\d{1,2}[.-/\s]?[.-/\s]?(jan|feb|märz|apr|mai|jun|jul|aug|sept|okt|nov|dez)[a-z]*[.-/\s]?(\d{2})?(\d{2})?|(jan|feb|märz|apr|mai|jun|jul|aug|sept|okt|nov|dez)[a-z]*[.-/\s]?((0?[1-9])|[12][0-9]|3[01])?[.,/\s][\s]?(\d{2})?(\d{2})?)\b",

    "(?<!\d)(?:[1-9]|[12]\d|3[01])[.-/\s](?:Jan(?:uar)?|Feb(?:ruar)?|März|Ma[iy]|Jun|iul|Aug|Sep(?:tember)?|Okt(?:ober)?|Nov(?:ember)?|Dez(?:ember)?)[a-z]*[.-/\s](?:\d{4}|\d{2})(?!\d)"
]

# Search for similar strings to each pattern in the list of strings
matches = {}
for pattern in patterns:
    results = process.extract(pattern, text[0:5000], limit=None)
    for result, score in results:
        if result in matches:
            if score > matches[result]:
                matches[result] = score
        else:
            matches[result] = score
#print("dates: ", len(dates))

print(matches)
print("len(matches): ", len(matches))