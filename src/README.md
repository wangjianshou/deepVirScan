# mock_from_genome.py
`mock_from_genome.py` is a script to genenrate mock NGS(next-generation sequencing) data. Running this command `python mock_from_genome.py -h` can get some help like below. And the file specified by parameter `--info` should be in the same format as `data/virus.mock.abundance`.
```
usage: mock_from_genome.py [-h] --info INFO [--outdir OUTDIR]
                           [--nsample NSAMPLE] [--fname FNAME]
                           [--readsNumber READSNUMBER]
                           [--log {WARNING,INFO,DEBUG}]

Generate Mock fastq from reference

optional arguments:
  -h, --help            show this help message and exit
  --info INFO, -i INFO  infomation for generate reads
  --outdir OUTDIR, -o OUTDIR
                        output direction
  --nsample NSAMPLE, -n NSAMPLE
                        the number samples to generate
  --fname FNAME, -f FNAME
                        fastq file name
  --readsNumber READSNUMBER, -r READSNUMBER
                        reads number to generate
  --log {WARNING,INFO,DEBUG}, -l {WARNING,INFO,DEBUG}
                        log level, WARNING, INFO, DEBUG
```

# data_processing.py

There are two functions in this script: `base2num` and `shuffleSample`.  
* `base2num(file, label)`:  This function can convert base sequences to numeric sequences. For example, ${\rm ATCGN} \rightarrow {\rm 01234}$. And at the end of each line, it can add `\t$label`,`$label=1` for virus reads and `$label=0` for bacteria reads.

* `shuffleSample(file)`: This function can arrange each line randomly.

# train.py

The model is trained by this script.


