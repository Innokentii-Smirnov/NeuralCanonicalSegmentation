import os
from itertools import starmap
from argparse import ArgumentParser
from segm import load_pairs
from alignment import align
from library.dm import DM
from library.write import write_list

parser = ArgumentParser(prog='print_alignments.py',
                        description='Print out aligned data for morphological string transduction in a readable way')
parser.add_argument('infile')
parser.add_argument('outdir')
args = parser.parse_args()

target_vocabulary = set[str]()
source_char_target_fragment_pairs = set[tuple[str, str]]()

data = load_pairs(args.infile, '~')
os.makedirs(args.outdir, exist_ok=True)
with DM(args.outdir):
  with open('Alignments.txt', 'w', encoding='utf-8') as fout:
    for word, segm, _ in data:
      for word_char, segm_fragm in zip(word, segm, strict=True):
        target_vocabulary.add(segm_fragm)
        source_char_target_fragment_pairs.add((word_char, segm_fragm))
      aligned = align((word, segm))
      print(aligned, file=fout, end='\n\n')
  write_list(sorted(target_vocabulary), 'Vocabulary.txt')
  write_list(starmap('{0}  {1}'.format, sorted(source_char_target_fragment_pairs)), 'Pairs.txt')
print('Target vocabulary size: ', len(target_vocabulary))
print('Source character - target fragment pair count: ', len(source_char_target_fragment_pairs))
