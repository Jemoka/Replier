import os
import csv
import getch
from prettytable import PrettyTable

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

data = []
with open("./replier_talking_data.csv", "r") as df:
    reader = csv.reader(df)
    for row in reader:
        data.append([row[-1], row[0], row[1]])
       

with open("./human_talkback_data.csv", "r") as df:
    reader = csv.reader(df)
    for indx, row in enumerate(reader):
        data[indx].append(row[1])


REVIEWER = "Houjun Liu"

def review(db, index):
    data = db[index]
    os.system('clear')
    x = PrettyTable()
    x.field_names = [f'{color.GREEN}{REVIEWER}/{color.GREEN}{index}{color.END}{color.GREEN} {color.BOLD}0{color.END}', f'{color.PURPLE}{data[0]}{color.END}', f'{color.UNDERLINE}{color.CYAN}Question type categorization.{color.END}']
    print(x)

# print(color.CYAN + data[0][0] + color.END, "is cool.")
# print(getch.getch())
review(data, 0)
breakpoint()

## Question score just for me
# Non psycology | psycology

## Response scores a la 2002.01862

# Clarity         - Unclear or incoherent response   | Coherent illogical response          | Logical response    | Logical and clearly directed response
# Specificty      - Irrelevant to the question itself| Addressses the question              | Engages the question| Engages and opionates upon the quesitons
# P. Helpfulness  - Negatively Influences the Issue  | Non-psych/Does not address the Issue | Addresses the Issue | Positively Influences the Issue

## Spot the Bot Touring a la 2010.02140
# Response A      - Bot | Unsure | Human
# Response B      - Bot | Unsure | Human


