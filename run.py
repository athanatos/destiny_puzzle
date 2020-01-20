#!env python3

import csv
import enum
import heapq

import sys
sys.setrecursionlimit(20000)

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

def get_open_edges(x, open_func):
    if x == '' or x == "none":
        return []
    ret = [
        open_func(int(i)) for i in
        x.strip()
        .strip(',')
        .replace('.', ',')
        .replace(', ', ',')
        .replace('side ', ',')
        .replace(' ', ',')
        .split(',')]
    if not all(map(lambda x: x >= 0 and x < 6, ret)):
        return None
    return ret
    

EMPTY = 'bbbbbbb'
def null_edge(x): return x == EMPTY

FILES = {
#    'initial_sheet.tsv': lambda x: x,
#    'dumbo.tsv': lambda x: x[1:9]
    #'2020-01-18.tsv': (lambda x: x[1:9], lambda x: x),
    #'2020-01-18-2.tsv': (lambda x: x[1:9], lambda x: x - 1),
    'raid_secrets.tsv': (lambda x: x[0:8], lambda x: x - 1),
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
    for f, (func, open_func) in FILES.items():
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
                    open_sides_decoded = get_open_edges(open_sides, open_func)
                except:
                    continue
                
                if open_sides_decoded is None:
                    print("Rejected: {}, invalid open sides {}".format(line, open_sides))
                    continue

                validated_edges = validate_edges(edges)
                if not validated_edges:
                    print("Rejected: {}, invalid edges {}".format(line, edges))
                    continue
            
                ret.append((validated_center, open_sides_decoded, validated_edges))
    return ret

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

    def get_checksum(self):
        return ''.join(self.__edges).upper()

    def __repr__(self):
        #return "Node(symbol={}, open_edges={}, edges={}".format(
        #    self.__symbol, self.__open_edges, self.__edges)
        return self.get_checksum()

    def symbol(self):
        return self.__symbol

    def get_edges(self):
        return self.__edges

    def get_idx_edges(self):
        return filter(lambda x: not null_edge(x[1]), zip(range(len(self.__edges)), self.__edges))

    def set_neighbor(self, idx, node):
        self.__neighbors[idx] = node

    def get_neighbor(self, idx):
        return self.__neighbors[idx]

    def get_idx_neighbors(self):
        return filter(lambda x: x[1] is not None, zip(range(len(self.__neighbors)), self.__neighbors))

    def replace(self, other):
        for idx, neighbor in self.get_idx_neighbors():
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

    def get_open_idx(self):
        return self.__open_edges

    def get_ext_edges(self):
        return list(filter(lambda idx: null_edge(self.__edges[idx]), self.get_open_idx()))

    def get_missing_edges(self):
        return list(filter(lambda x: self.__neighbors[x[0]] is None,
                           self.get_idx_edges()))

    def get_valid_neighbors(self):
        return list(filter(
            lambda x: x[0] in self.__open_edges,
            self.get_idx_neighbors()))

    def is_blank(self):
        return self.symbol() == 'blank'

    def has_symbol(self):
        return not self.is_blank() and not self.is_fake()

    def is_fake(self):
        return self.symbol() == 'fake'

    def cost(self):
        if self.is_fake():
            return 1000
        elif self.is_blank():
            return 0
        else:
            return 1

    def clear_open_edge(self, edge):
        assert(self.is_fake())
        self.__open_edges = tuple(filter(lambda x: x != edge, self.__open_edges))


class ConnectedComponent(object):
    def __init__(self, tag, nodes):
        self.__tag = tag
        self.__nodes = nodes
        self.__fake = []
        self.sanity()

    def generate_coords(self, exclude_list):
        def add(x, y):
            return tuple(map(sum, zip(x, y)))
        def sub(x, y):
            return tuple(map(lambda x: x[0] - x[1], zip(x, y)))
        cube = [(1, 1), (2, 0), (1, -1), (-1, -1), (-2, 0), (-1, 1)]
        root = self.__nodes[90]
        to_visit = [(0, (0, 0), (), None, 0, root)]
        coords_to_node = {}
        node_to_coords = {}
        node_to_path = {}
        missing_edges = []
        offsets = {}
        coord_collisions = 0
        while len(to_visit):
            _, coords, path, prev, idx, next = heapq.heappop(to_visit)
            if id(next) in node_to_coords:
                if node_to_coords[id(next)] != coords:
                    offset = sub(coords, node_to_coords[id(next)])
                    l = int(offset[0]/abs(offset[0]) if offset[0] != 0 else offset[1]/abs(offset[1]))
                    offset = tuple([x * l for x in offset])
                    if offset not in offsets:
                        offsets[offset] = ([], [])
                    offsets[offset][0 if l == 1 else 1].append((path, prev, idx, next))
                continue
                
            node_to_coords[id(next)] = coords
            node_to_path[id(next)] = path

            if coords in coords_to_node:
                coord_collisions += 1
            coords_to_node[coords] = next
            
            for idx, neighbor in next.get_idx_neighbors():
                ncoords = add(cube[idx], coords)
                npath = path + (idx,)

                heapq.heappush(
                    to_visit,
                    (exclude_list.get(tuple(sorted([id(next), id(neighbor)])), 0),
                     ncoords,
                     npath,
                     next,
                     idx,
                     neighbor))

        edge_remove_list = []
        excl_list = []
        inconsistencies = 0
        for offset, (l, r) in offsets.items():
            if len(l) == 1:
                edge_remove_list.append(l[0][1:])
                continue
            else:
                inconsistencies += len(l)
            def max_prefix(l):
                ret = ()
                for i in range(1, min(map(len, l))):
                    if not all(map(lambda x: x[:i] == l[0][:i], l)):
                        return ret
                    ret = l[0][:i]
                return ret
            cand = ()
            if len(l) == 1:
                cand = l[0][0][:-2]
            else:
                lpre = max_prefix([x[0] for x in l])
                rpre = max_prefix([x[0] for x in r])
                cand = lpre if len(lpre) > len(rpre) else rpre
            assert len(cand) > 1
            current = root
            while len(cand) > 1:
                current = current.get_neighbor(cand[0])
                cand = cand[1:]
            excl_list.append(tuple(sorted([id(current), id(current.get_neighbor(cand[0]))])))
            
        return inconsistencies, coord_collisions, edge_remove_list, frozenset(excl_list), coords_to_node, node_to_coords

    def edge_list(self):
        ret = []
        for node in self.__nodes:
            for _, neighbor in node.get_idx_neighbors():
                if id(node) < id(neighbor):
                    ret.append((id(node), id(neighbor)))
        return ret

    def fill(self):
        def add(x, y):
            return tuple(map(sum, zip(x, y)))
        cube = [(1, 1), (2, 0), (1, -1), (-1, -1), (-2, 0), (-1, 1)]

        excl_list = {}
        inconsistencies = 10
        collisions = 0
        edge_remove_list = []
        while inconsistencies:
            inconsistencies, collisions, edge_remove_list, nexcl_list, coords_to_node, node_to_coords = self.generate_coords(excl_list)
            for excl in nexcl_list:
                if not excl in excl_list:
                    excl_list[excl] = 1
                excl_list[excl] *= 2
            print(inconsistencies)
        assert inconsistencies == 0
        assert collisions == 0

        print("Eliminating ", edge_remove_list)
        for fro, idx, to in edge_remove_list:
            fro.set_neighbor(idx, None)
            to.set_neighbor((idx + 3) % 6, None)

        missing_edges = []
        for node in self.__nodes:
            for idx, edge in node.get_missing_edges():
                coords = add(cube[idx], node_to_coords[id(node)])
                if coords in coords_to_node:
                    neighbor = coords_to_node[coords]
                    assert neighbor.get_neighbor((idx + 3) % 6) is None
                    neighbor.set_neighbor((idx + 3) % 6, node)
                    node.set_neighbor(idx, neighbor)
                else:
                    missing_edges.append(coords)


        minx = min((x[0][0] for x in coords_to_node.items()))
        maxx = max((x[0][0] for x in coords_to_node.items()))
        miny = min((x[0][1] for x in coords_to_node.items()))
        maxy = max((x[0][1] for x in coords_to_node.items()))
        print(minx, maxx, miny, maxy)

        while len(missing_edges):
            ncoords = missing_edges.pop()
            if ncoords in coords_to_node:
                continue
            fake_node = Node('fake', range(6), ['ccccccc']*7)
            self.__fake.append(fake_node)
            coords_to_node[ncoords] = fake_node
            node_to_coords[id(fake_node)] = ncoords
            for idx, mcoord in ((i, add(ncoords, cube[i])) for i in range(6)):
                if not (mcoord[0] >= minx and mcoord[0] <= maxx
                        and mcoord[1] >= miny and mcoord[1] <= maxy):
                    continue
                node = coords_to_node.get(mcoord, None)
                if node is None:
                    missing_edges.append(mcoord)
                else:
                    nidx = (idx + 3) % 6
                    if node.get_neighbor(nidx) is not None:
                        x = node.get_neighbor(nidx)
                        assert False
                    node.set_neighbor(nidx, fake_node)
                    fake_node.set_neighbor(idx, node)
                    if nidx not in node.get_open_idx():
                        fake_node.clear_open_edge(nidx)

        print(len(coords_to_node), len(node_to_coords))
        self.sanity()

    def pp_path(self, i, path):
        current = i
        ret = ''
        fakes = 0
        blanks = 0
        symbols = 0
        while (True):
            if current.is_blank():
                blanks += 1
            else:
                if blanks > 0:
        #            ret += " blanks[{}]".format(blanks)
                    blanks = 0

            if current.is_fake():
                fakes += 1
            else:
                if fakes > 0:
        #            ret += " fakes[{}]".format(fakes)
                    fakes = 0

            if current.has_symbol():
                symbols += 1
                ret += ' ' + current.symbol().upper()

            if len(path) == 0:
                break

            current = current.get_neighbor(path[0])
            path = path[1:]

        assert(fakes == 0)
        if blanks > 0:
            ret += " blanks[{}]".format(blanks)
            blanks = 0
        print(symbols)
        return ret

    def shortest_path(self, i, j):
        heap = [(0, i, ())]
        costs = {}
        costs[id(i)] = 0
        checked = 0
        last_cost = 0
        while (len(heap)):
            checked += 1
            cost, next, path = heapq.heappop(heap)
            #print("pulling", next)
            for idx, node in next.get_valid_neighbors():
                npath = path + (idx,)
                if id(node) in costs:
                    continue
                if node is j:
                    return cost + node.cost(), npath
                costs[id(node)] = cost + node.cost()
                heapq.heappush(heap, (cost + node.cost(), node, path + (idx,)))
                last_cost = cost + node.cost()
        return -1, None

    def get_exterior_nodes(self):
        ret = []
        for node in self.__nodes:
            ext = node.get_ext_edges()
            if len(ext) > 0:
                ret.append(node)
        return ret

    def summarize(self):
        valid = [0] * 7
        connected = [0] * 7
        valid_open = [0] * 7
        connected_open = [0] * 7
        missing_connections = 0
        symbols = 0
        exterior_open_nodes = 0
        for node in self.__nodes:
            num_valid = len(list(node.get_idx_edges()))
            num_connected = len(list(node.get_idx_neighbors()))
            valid[num_valid] += 1
            connected[num_connected] += 1
            missing_connections += num_valid - num_connected
            if node.symbol() != 'blank':
                symbols += 1

        ext_nodes = self.get_exterior_nodes()
        print("exterior_open_nodes: ", len(self.get_exterior_nodes()))
        print("valid: ", valid)
        print("connected: ", connected)
        print("missing_connections: ", missing_connections)
        print("symbols: ", symbols)
        print("nodes: ", len(self.__nodes))
        print("fake: ", len(self.__fake))

        for i in range(len(ext_nodes)):
            for j in range(i + 1, len(ext_nodes)):
                cost, path = self.shortest_path(ext_nodes[j], ext_nodes[i])
                print("Shortest: ", cost) 
                print("symbols: ", self.pp_path(ext_nodes[j], path))

    def __lt__(self, other):
        return id(self) < id(other)

    def size(self):
        return len(self.__nodes)

    def nodes(self):
        return self.__nodes

    def copy(self, tag):
        return self.rotate(tag, 0, False)

    def sanity(self):
        for node in self.__nodes:
            for idx, neighbor in node.get_idx_neighbors():
                if neighbor.get_neighbor((idx + 3) % 6) is not node:
                    print(neighbor, node, neighbor.get_neighbor((idx + 3) % 6), id(node), id(neighbor), id(neighbor.get_neighbor((idx + 3) % 6)))
                    assert False

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
    for lnode in node_list:
        #skip = False
        #for idx, edge in lnode.get_idx_edges():
        #    cls = (idx % 3, edge)
        #    if cls not in edge_to_nodes:
        #        edge_to_nodes[cls] = []
        #    for jdx, onode in edge_to_nodes[cls]:
        #        if idx != (jdx + 3) % 6:
        #            print("Skipping node {}, non-matching edge {}".format(onode, edge))
        #            skip = True
        #    
        #if skip:
        #    continue
        for idx, edge in lnode.get_idx_edges():
            cls = (idx % 3, edge)
            if cls not in edge_to_nodes:
                edge_to_nodes[cls] = []
            edge_to_nodes[cls].append((idx, lnode))
        #    assert len(edge_to_nodes[cls]) < 3

    for (cls, edge), nodes in edge_to_nodes.items():
        if len(nodes) > 2:
            print("edge {} too many nodes {}".format(edge, nodes))
            continue
        for idx, inode in nodes:
            for jdx, jnode in nodes:
                if id(inode) <= id(jnode):
                    continue
                if idx != (jdx + 3) % 6:
                    print("found misaligned pair for edge {}: {}, {}".format(
                        edge, inode, jnode))
                else:
                    assert inode.get_neighbor(idx) is None
                    assert jnode.get_neighbor(jdx) is None
                    inode.set_neighbor(idx, jnode)
                    jnode.set_neighbor(jdx, inode)

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
    return sorted(((x.size(), x) for x in components))

def print_top_n(bs, n):
    list(map(print, bs[-1 * n:]))

data = read_files()

nodes = sorted(set([Node(*x) for x in data]))

ccs = generate_connected_components(nodes)
print_top_n(by_size(ccs), 5)

top = by_size(ccs)[-1]
print(top)

top[1].fill()
top[1].summarize()

#ccs = try_rotate_merge_ccs(ccs)
#print_top_n(by_size(ccs), 5)
