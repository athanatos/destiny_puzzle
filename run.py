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

EMPTY = 'bbbbbbb'
def null_edge(x): return x == EMPTY

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
                continue

            open_sides_decoded = None
            try:
                open_sides_decoded = get_open_edges(open_sides)
            except:
                continue

            if not all((len(x) == 7 and all(map(valid_symbol, x)) for x in edges)):
                continue
            
            ret.append((center, open_sides_decoded, edges))
    return ret

data = read_file('initial_sheet.tsv')

def count_edges(x):
    edge_counts = {}
    for _, _, edges in x:
        for edge in edges:
            if null_edge(edge): continue
            if edge not in edge_counts:
                edge_counts[edge] = 0
            edge_counts[edge] += 1
    return list(sorted((count, edge) for edge, count in edge_counts.items()))

class Node(object):
    def __init__(self, symbol, open_edges, edges):
        self.__symbol = symbol
        self.__open_edges = open_edges
        self.__edges = edges
        self.__neighbors = [None for x in self.__edges]
        self.__tag = None

    def __repr__(self):
        return "Node(symbol={}, open_edges={}, edges={}".format(
            self.__symbol, self.__open_edges, self.__edges)

    def symbol(self):
        return self.__symbol

    def get_edges(self):
        return self.__edges

    def set_neighbors(self, edge_to_nodes):
        for idx, edge in zip(range(len(self.__edges)), self.__edges):
            if null_edge(edge):
                continue
            nodes = edge_to_nodes[edge]
            if len(nodes) > 2:
                print("{} has 3 nodes: {}", edge, nodes)
            for node in nodes:
                if node is not self:
                    self.__neighbors[idx] = node

    def get_tag(self):
        return self.__tag

    def tagged(self):
        return self.__tag is not None

    def visit(self, tag):
        self.__tag = tag
        nodes = [self]
        for neighbor in self.__neighbors:
            if neighbor is not None and not neighbor.tagged():
                nodes += neighbor.visit(tag)
        return nodes


nodes = [Node(*x) for x in data]

edge_to_nodes = {}
for node in nodes:
    for edge in node.get_edges():
        if null_edge(edge):
            continue
        if edge not in edge_to_nodes:
            edge_to_nodes[edge] = []
        edge_to_nodes[edge].append(node)

[x.set_neighbors(edge_to_nodes) for x in nodes]

tag = 0
tags = {}
for node in nodes:
    if node.tagged():
        continue
    nodes = node.visit(tag)
    tags[tag] = (len(nodes), nodes)
    tag += 1

[print(x) for x in sorted(((num, tag, [x.symbol() for x in nodes]) for tag, (num, nodes) in tags.items()))]

