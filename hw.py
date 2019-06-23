import pyprind
import time
import os
n = 100
timesleep = 0.001
flist = list(f for f in os.listdir("exports/") if f.startswith("u1"))
for i in pyprind.prog_bar(flist):
    time.sleep(timesleep) # your computation here
