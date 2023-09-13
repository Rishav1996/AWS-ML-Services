from fuzzywuzzy import fuzz
import re


preds = []
actuals = []

acc = []

with open('./../data/long.txt') as f:
    actuals = f.readlines()

with open('./../data/transcript.txt') as f:
    preds = f.readlines()

for p, a in zip(preds, actuals):
    p = re.sub('[^A-Za-z0-9]+', '', p).lower()
    a = re.sub('[^A-Za-z0-9]+', '', a).lower()
    acc.append(fuzz.ratio(p, a))

print(sum(acc) / len(acc))