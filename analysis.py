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

logging.info("min left: %s", min(lefts))
logging.info("mean left: %s", statistics.mean(lefts))
logging.info("median left: %s", statistics.median(lefts))
logging.info("max left: %s\n", max(lefts))

logging.info("min right: %s", min(rights))
logging.info("mean right: %s", statistics.mean(rights))
logging.info("median right: %s", statistics.median(rights))
logging.info("max right: %s\n", max(rights))

trail = 300

leftTrail = lefts[-trail:]

logging.info("trailing %d min left: %s", trail, min(leftTrail))
logging.info("trailing %d mean left: %s", trail, statistics.mean(leftTrail))
logging.info("trailing %d median left: %s", trail, statistics.median(leftTrail))
logging.info("trailing %d max left: %s\n", trail, max(leftTrail))

rightTrail = rights[-trail:]

logging.info("trailing %d min right: %s", trail, min(rightTrail))
logging.info("trailing %d mean right: %s", trail, statistics.mean(rightTrail))
logging.info("trailing %d median right: %s", trail, statistics.median(rightTrail))
logging.info("trailing %d max right: %s", trail, max(rightTrail))
