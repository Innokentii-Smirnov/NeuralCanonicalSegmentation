from argparse import ArgumentParser
from segm import load_pairs
from alignment import align

parser = ArgumentParser(prog='print_alignments.py',
                        description='Print out aligned data for morphological string transduction in a readable way')
parser.add_argument('infile')
parser.add_argument('outfile')
args = parser.parse_args()

data = load_pairs(args.infile, '~')
with open(args.outfile, 'w', encoding='utf-8') as fout:
  for word, segm, _ in data:
    aligned = align((word, segm))
    print(aligned, file=fout, end='\n\n')
