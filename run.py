#!env python3

import csv
import enum

SYMBOL_TO_NAME = {
    'b' : 'blank',
    't' : 'cauldron',
    'h' : 'hexes',
    'c' : 'clover',
    'p' : 'plus',
    's' : 'snake',
    'd' : 'diamond'
}
def s_to_n(x): return SYMBOL_TO_NAME.get(x)

NAME_TO_SYMBOL = dict(((b, a) for a, b in SYMBOL_TO_NAME.items()))
def n_to_s(x): return NAME_TO_SYMBOL.get(x)

SYMBOLS = frozenset((a for a, b in SYMBOL_TO_NAME.items()))
def valid_symbol(x): return x in SYMBOLS

NAMES= frozenset((b for a, b in SYMBOL_TO_NAME.items()))
def valid_name(x): return x in NAMES

def get_open_edges(x):
    return [int(i) for i in x.strip().split(',')]

def read_file(filename):
    ret = []
    with open(filename) as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter='\t')
        for line in tsvreader:
            line = [x.lower() for x in line]

            if len(line) != 8:
                continue

            center, open_sides = line[:2]
            edges = line[2:]

            if not valid_name(center):
                print("{} invalid", center)
                continue

            open_sides_decoded = None
            try:
                open_sides_decoded = get_open_edges(open_sides)
            except:
                print("{} invalid", open_sides)
                continue

            if not all((all(map(valid_symbol, x)) for x in edges)):
                print("{} invalid", edges)
                continue
            
            ret.append((center, open_sides_decoded, edges))
    return ret

[print(i) for i in read_file('initial_sheet.tsv')]
