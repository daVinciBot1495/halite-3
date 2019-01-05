import fileinput, logging, statistics

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)

i = 0
lefts = []
rights = []

for line in fileinput.input():
    value = float(line.strip())
    
    if i % 2 == 0:
        lefts.append(value)
    else:
        rights.append(value)

    i = i + 1

pairs = zip(lefts, rights)
diffs = [pair[0] - pair[1] for pair in pairs]

logging.info("median left: %s", statistics.median(lefts))
logging.info("median right: %s", statistics.median(rights))
logging.info("median diff: %s", statistics.median(diffs))
