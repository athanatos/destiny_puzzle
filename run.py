#!env python3

import csv
import enum

SYMBOL_TO_NAME = {
    'b' : 'blank',
    't' : 'cauldron',
    'h' : 'hexagon',
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

NAME_MAP = [
    (('none', 'black'), 'blank'),
    ((), 'cauldron'),
    (('hex', 'hexes'), 'hexagon'),
    (('c'), 'clover'),
    (('p', 'pluss'), 'plus'),
    (('sneak'), 'snake'),
    (('d'), 'diamond'),
]
def validate_center(x):
    x = x.strip()
    for aliases, canonical in NAME_MAP:
        if x == canonical or x in aliases:
            return canonical
    return None

def get_open_edges(x):
    if x == '' or x == "none":
        return []
    return [
        int(i) for i in
        x.strip()
        .strip(',')
        .replace('.', ',')
        .replace(', ', ',')
        .replace('side ', ',')
        .replace(' ', ',')
        .split(',')]

EMPTY = 'bbbbbbb'
def null_edge(x): return x == EMPTY

FILES = {
    'initial_sheet.tsv': lambda x: x,
    'dumbo.tsv': lambda x: x[1:9]
}

def validate_edge(edge):
    if len(edge) == 7 and all(map(valid_symbol, edge)):
        return edge
    elif edge == '':
        return EMPTY

def validate_edges(edges):
    if len(edges) != 6:
        return None

    ret = list(map(validate_edge, edges))
    if not all(ret):
        return None
    return ret 

def read_files():
    ret = []
    for f, func in FILES.items():
        with open(f) as tsvfile:
            tsvreader = csv.reader(tsvfile, delimiter='\t')
            for line in tsvreader:
                line = [x.lower() for x in func(line)]
                
                if len(line) != 8:
                    print("Rejected: {}, wrong length".format(line))
                    continue

                center, open_sides = line[:2]
                edges = line[2:]

                validated_center = validate_center(center)
                if not validated_center:
                    print("Rejected: {}, invalid center {}".format(line, center))
                    continue

                open_sides_decoded = None
                try:
                    open_sides_decoded = get_open_edges(open_sides)
                except:
                    print("Rejected: {}, invalid open sides {}".format(line, open_sides))
                    continue

                validated_edges = validate_edges(edges)
                if not validated_edges:
                    print("Rejected: {}, invalid edges {}".format(line, edges))
                    continue
            
                ret.append((validated_center, open_sides_decoded, validated_edges))
    return ret

data = read_files()

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
        self.__open_edges = tuple(open_edges)
        self.__edges = tuple(edges)
        self.__neighbors = [None for x in self.__edges]
        self.__tag = None

    def __eq__(self, other):
        return (self.__symbol == other.__symbol and
                self.__open_edges == other.__open_edges and
                self.__edges == other.__edges)

    def __lt__(self, other):
        return (
            (self.__symbol, self.__open_edges, self.__edges) < 
            (other.__symbol, other.__open_edges, other.__edges)
            )
            
    def __hash__(self):
        return (self.__symbol, self.__open_edges, self.__edges).__hash__()

    def __repr__(self):
        return "Node(symbol={}, open_edges={}, edges={}".format(
            self.__symbol, self.__open_edges, self.__edges)

    def symbol(self):
        return self.__symbol

    def get_edges(self):
        return self.__edges

    def set_neighbor(self, idx, node):
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

node_list = sorted(set([Node(*x) for x in data]))

edge_to_nodes = {}
for node in node_list:
    for idx, edge in zip(range(len(node.get_edges())), node.get_edges()):
        if null_edge(edge):
            continue
        if edge not in edge_to_nodes:
            edge_to_nodes[edge] = []
        edge_to_nodes[edge].append((idx, node))

for edge, nodes in edge_to_nodes.items():
    for idx, inode in nodes:
        for jdx, jnode in nodes:
            if inode is jnode:
                continue
            if idx != (jdx + 3) % 6:
                print("found misaligned pair for edge {}: {}, {}".format(edge, inode, jnode))
            else:
                inode.set_neighbor(idx, jnode)
                inode.set_neighbor(jdx, inode)

tag = 0
tags = {}
for node in node_list:
    if node.tagged():
        continue
    nodes = node.visit(tag)
    tags[tag] = (len(nodes), nodes)
    tag += 1

[print(x) for x in sorted(((num, tag, [x.symbol() for x in nodes]) for tag, (num, nodes) in tags.items()))]

