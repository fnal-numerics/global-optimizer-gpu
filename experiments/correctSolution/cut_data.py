import sys

#filename = "rastrigin_5psoit_10d_particledata.tsv"

#!/usr/bin/env python3
import argparse

def filter_top_k(input_path, output_path, k):
    with open(input_path, 'r') as fin, open(output_path, 'w') as fout:
        for line in fin:
            # split into up to k fields
            fields = line.rstrip('\n').split('\t')
            fout.write('\t'.join(fields[:k]) + '\n')

parser = argparse.ArgumentParser(
    description="Stream a TSV and keep only the first k columns."
)
parser.add_argument('-i', '--input',  required=True,
                    help="Path to input TSV file")
parser.add_argument('-o', '--output', required=True,
                    help="Path to output TSV file")
parser.add_argument('-k', '--columns', type=int, required=True,
                    help="Number of leading columns to keep")
args = parser.parse_args()

filter_top_k(args.input, args.output, args.columns)


