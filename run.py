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
#    'initial_sheet.tsv': lambda x: x,
#    'dumbo.tsv': lambda x: x[1:9]
    '2020-01-18.tsv': lambda x: x[1:9]
}

def validate_edge(edge):
    edge = edge.strip()
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

def rotate_seq(t, tup, rot, rev):
    val = -1 if rev else 1
    return t((tup[(val * (i + rot)) % len(tup)] for i in range(len(tup))))

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

    def get_neighbor(self, idx):
        return self.__neighbors[idx]

    def get_idx_neighbors(self):
        return zip(range(len(self.__neighbors)), self.__neighbors)

    def replace(self, other):
        for idx, neighbor in self.get_idx_neighbors():
            if neighbor is not None:
                assert neighbor.get_neighbor((idx + 3) % 6) == self
                neighbor.set_neighbor((idx + 3) % 6, other)
                if other.get_neighbor(idx) is None:
                    other.set_neighbor(idx, other)
                self.set_neighbor(idx, None)

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

    def rotate(self, tag, rotate, rev):
        mult = -1 if rev else 1
        ret = Node(
            self.__symbol,
            tuple(sorted((((i + rotate) * mult) % 6) for i
                         in self.__open_edges)),
            rotate_seq(tuple, self.__edges, rotate, rev))
        ret.__neighbors = rotate_seq(list, self.__neighbors, rotate, rev)
        ret.__tag = tag
        return ret

    def fixup_neighbors(self, neighbor_map):
        def replace(x):
            if x is None:
                return x
            else:
                return neighbor_map[id(x)]
        self.__neighbors = list(map(replace, self.__neighbors))

    def get_unpaired_edges(self):
        return list(map(lambda x: x[:2],
            filter(lambda idx, edge, neighbor: neighbor is None,
                   zip(range(len(self.__edges)), self.__edges, self.__neighbors)
            )))
                   


nodes = sorted(set([Node(*x) for x in data]))


class ConnectedComponent(object):
    def __init__(self, tag, nodes):
        self.__tag = tag
        self.__nodes = nodes;

    def __lt__(self, other):
        return id(self) < id(other)

    def size(self):
        return len(self.__nodes)

    def nodes(self):
        return self.__nodes

    def copy(self, tag):
        return self.rotate(tag, 0, False)

    def rotate(self, tag, rotate, reverse):
        nodemap = {}
        new_nodes = []
        for node in self.__nodes:
            n = node.rotate(tag, rotate, reverse)
            nodemap[id(node)] = n
            new_nodes.append(n)
        for node in new_nodes:
            node.fixup_neighbors(nodemap)

        #sanity
        for node in new_nodes:
            assert id(node) not in nodemap
        for node in new_nodes:
            for idx, neighbor in node.get_idx_neighbors():
                if neighbor is None:
                    continue
                assert neighbor.get_neighbor((idx + 3) % 6) is self
        return ConnectedComponent(tag, new_nodes)

    def get_exterior_edges(self):
        ret = {}
        for node in self.__nodes:
            for idx, edge in node.get_unpaired_edges():
                ret[edge] = (node, idx)

    def merge(self, other):
        merged = False
        extedges = self.get_exterior_edges()
        oextedges = other.get_exterior_edges()
        
        matches = []
        for node in self.__nodes:
            for onode in other.__nodes:
                if node == onode:
                    matches.append((node, onode))

        if len(matches) > 0:
            merged = True
        
        deduped = frozenset((id(x) for tup in matches for x in tup))
        for node, onode in matches:
            onode.replace(node)

        for edge, (node, idx) in extedges.items():
            if id(node) in removed:
                continue
            for oedge, (onode, oidx) in oextedges.items():
                if id(onode) in removed:
                    continue
            if edge == oedge:
                if idx != ((oidx + 3) % 6):
                    print("Found unmatched pair in merge: {} {} {} {}".format(
                        edge, node, oedge, onode))
                node.set_neighbor(idx, onode)
                onode.set_neighbor(oidx, node)
                merged = True

        if merge:
            other.__nodes = []
            return True
        else:
            return False

def generate_connected_components(node_list):
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
                    print("found misaligned pair for edge {}: {}, {}".format(
                        edge, inode, jnode))
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
    return [
        ConnectedComponent(tag, nodes) for tag, (_, nodes) in
        tags.items()
        ]

def try_rotate_merge_ccs(ccs):
    def try_merge(cc, ccs):
        for rotation in (cc.rotate(0, i, j) for i in range(1, 6) for j in [False, True]):
            for occ in ccs:
                if occ.merge(rotation):
                    return True
        return False
        
    unmatched = []
    while (len(ccs)):
        cc = ccs[0]
        ccs = ccs[1:]
        if not try_merge(cc, ccs):
            unmatched.append(cc)
    return unmatched

def by_size(components):
    return sorted(((x.size(), x) for x in
                   generate_connected_components(nodes)))

def print_top_n(bs, n):
    list(map(print, [(x[0], [i.symbol() for i in x[1].nodes()]) for x in bs][-1 * n:]))

ccs = generate_connected_components(nodes)
print_top_n(by_size(ccs), 5)

ccs = try_rotate_merge_ccs(ccs)
print_top_n(by_size(ccs), 5)
