import os,sys
from os import path
import gzip
import argparse
import logging
import pandas as pd

def parseArgs():
  parser = argparse.ArgumentParser(description='Generate Mock fastq from reference')
  parser.add_argument("--info", "-i", required=True, help="infomation for generate reads")
  parser.add_argument("--outdir", "-o", required=False, default='.', help="output direction")
  parser.add_argument("--nsample", "-n", required=False, default=1, type=int, help="the number samples to generate")
  parser.add_argument("--fname", "-f", required=False, default="Mock", help='fastq file name')
  parser.add_argument("--readsNumber", "-r", required=False, default=3333333, type=int, help="reads number to generate")
  parser.add_argument('--log', '-l', required=False, default='INFO', choices=['WARNING', 'INFO', 'DEBUG'], help='log level, WARNING, INFO, DEBUG')
  args = parser.parse_args()
  return args

class GenerateMockFromGenome:
  def __init__(self, info, outdir='./Mock',r1="Mock.R1.fastq.gz", r2="Mock.R2.fastq.gz", data=3333333):
    info['relAbundance'] = info.abundance/info.abundance.sum()
    info['readsNumber'] = data * info['relAbundance']
    info['readsNumber'] = info['readsNumber'].astype(int)
    self.info = info.loc[:,:]
    self.outdir = path.abspath(outdir)
    self.tmp = path.join(self.outdir, 'tmp')
    path.isdir(self.tmp) or os.mkdir(self.tmp)
    self.r1 = gzip.open(path.join(outdir, r1), "wb")
    self.r2 = gzip.open(path.join(outdir, r2), "wb")
    self.index = 0
  def close(self):
    self.r1.closed or self.r1.close()
    self.r2.closed or self.r2.close()
  def __iter__(self):
    self.index = 0
    return self
  def __next__(self):
    if self.index < self.sampleN:
      self.processOne()
      self.r1.flush()
      self.r2.flush()
      self.index += 1
    else:
      self.close()
      raise StopIteration
  @property
  def status(self):
    return self.info.iloc[self.index,:]
  @property
  def sampleN(self):
    return self.info.shape[0]
  def processOne(self):
    logger.info("start " + self.status.reference)
    logger.info(self.status.reference + "\tAbundance readsNumber: " + str(self.status.readsNumber))
    fna = path.join(self.tmp, self.status.reference.replace('.gz',''))
    suffix = ('>' + self.status.reference+'_').encode()
    logger.debug("zcat is starting")
    if self.status.path.endswith('.gz'):
      os.system("zcat " + self.status.path + ' ' + '>'+fna)
    else:
      os.system("ln -s "+self.status.path+' '+fna)
    logger.debug('zcat is completed')
    tmp_r1 = path.join(self.tmp,'tmp.mock.1')
    tmp_r2 = path.join(self.tmp,'tmp.mock.2')
    logger.debug('wgsim is running')
    os.system("wgsim -A 0.02 -N {reads} -1 150 -2 150 -e 0.005 {fna} {r1} {r2} >/dev/null".format(\
               reads=self.status.readsNumber, fna=fna, r1=tmp_r1, r2=tmp_r2))
    logger.debug('wgsim is completed and write is running')
    with open(tmp_r1, 'rb') as f:
      fq = bytearray(f.read()).strip().split(b'\n')
    with open(tmp_r2, 'rb') as f:
      fq2 = bytearray(f.read()).strip().split(b'\n')
    for i in range(0, len(fq), 4):
      fq[i][0:1] = suffix
      fq2[i][0:1] = suffix
      self.r1.write(fq[i]+b'\n')
      self.r1.write(fq[i+1]+b'\n')
      self.r2.write(fq2[i]+b'\n')
      self.r2.write(fq2[i+1]+b'\n')
    logger.debug('write fastq is completed')
    logger.info("complete: " + self.status.reference)
  def cleanUp(self):
    os.system('rm -r ' + self.tmp)

if __name__=='__main__':
  args = parseArgs()
  info = pd.read_csv(args.info, sep='\t', header=0)
  loggerLevel = 10 if args.log=='WARNING' else 20 if args.log=='INFO' else 30
  logger = logging.getLogger(__name__)
  logging.basicConfig(level=loggerLevel,format = '%(asctime)s - %(levelname)s - %(message)s',
                      filename=path.join(args.outdir, 'log.GenerateMockReads'), filemode='w')
  for i in range(args.nsample):
    mockName = args.fname + '_' + str(i)
    logger.info(mockName+'_readsNumber:'+str(args.readsNumber))
    outdir = path.join(path.abspath(args.outdir), mockName)
    os.path.isdir(outdir) or os.makedirs(outdir)
    mock = GenerateMockFromGenome(info=info, outdir=outdir, r1=mockName+".R1.fastq.gz", r2=mockName+".R2.fastq.gz", data=args.readsNumber)
    list(iter(mock))
    mock.cleanUp()

