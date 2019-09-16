import sys
from collections import Counter

with open(sys.argv[1], 'r') as flist:
  next(flist)
  info = [line.strip().split('\t') for line in flist]

fw = open("bacteria_genome.length.txt", "w")

for line in info:
  with open(line[1], "rt") as f:
    seqname = next(f).strip()
    c = Counter()
    for i in f:
      if i.startswith('>'):
        fw.write('\t'.join([line[0], seqname, str(sum(c.values()))])+'\n')
        seqname = i.strip()
        c = Counter()
      else:
        c.update(i.strip())
    fw.write('\t'.join([line[0], seqname, str(sum(c.values()))])+'\n')
fw.close()
