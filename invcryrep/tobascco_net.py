# -*- coding: utf-8 -*-
# created by Boyd, P. G.; K. Woo, T. A Generalized Method for Constructing Hypothetical Nanoporous Materials of Any Net Topology from Graph Theory. CrystEngComm 2016, 18 (21), 3777–3792. https://github.com/peteboyd/tobascco
# modified by Hang Xiao; fixed self-loop bug
# xiaohang07@live.cn
import distutils.util as du
import math
import sys
from logging import debug, error, info, warning
from os.path import dirname, join, realpath
from sys import version_info
from uuid import uuid4

import networkx as nx
import numpy as np
import sympy as sy






DEG2RAD = np.pi / 180.0

class SystreDB(dict):
    """A dictionary which reads a file of the same format read by Systre"""

    def __init__(self, filename=None):
        self.voltages = {}
        self.read_store_file(filename)
        # scale holds the index and value of the maximum length^2 for the
        # real vectors associated with edges of the net.  This is only
        # found after SBUs have been assigned to nodes and edges.
        self.scale = (None, None)

    def read_store_file(self, file=None):
        """Reads and stores the nets in the self.file file.
        Note, this is specific to a systre.arc file and may be subject to
        change in the future depending on the developments ODF makes on
        Systre.

        """
        # just start an empty list
        if file is None:
            return

        with open(file, "r") as handle:
            block = []
            while True:
                line = handle.readline()
                if not line:
                    break

                l = line.strip().split()
                if l and l[0].lower() != "end":
                    block.append(" ".join(l))
                elif l and l[0].lower() == "end":
                    name = self.get_name(block)
                    ndim, systre_key = self.get_key(block)
                    # g, v = self.gen_sage_graph_format(systre_key) # SAGE compliant
                    g, v = self.gen_networkx_graph_format(
                        systre_key, ndim
                    )  # networkx compliant
                    self[name] = g
                    self.voltages[name] = np.array(v)
                    block = []

    def get_key(self, block):
        for j in block:
            l = j.split()
            if l[0].lower() == "key":
                dim = int(l[1])
                return dim, list(self.Nd_chunks([int(i) for i in l[2:]], dim))
        return None

    def get_name(self, block):
        name = uuid4()
        for j in block:
            l = j.split()
            if l[0].lower() == "id":
                name = l[1]
        return name

    def Nd_chunks(self, list, dim):
        n = 2 + dim
        for i in range(0, len(list), n):
            yield tuple(list[i : i + n])

    def gen_networkx_graph_format(self, edges, dim=3):
        """Take the edges from a systre db file and convert
        to a networkx graph readable format.

        Assumes that the direction of the edge goes from
        [node1] ---> [node2]
        """
        x_dat = []
        voltages = []
        if dim == 2:
            for id, (v1, v2, e1, e2) in enumerate(edges):
                ename = "e%i" % (id + 1)
                voltages.append((e1, e2))
                x_dat.append(
                    (str(v1), str(v2), dict(label=ename))
                )  # networkx compliant

        elif dim == 3:
            for id, (v1, v2, e1, e2, e3) in enumerate(edges):
                ename = "e%i" % (id + 1)
                voltages.append((e1, e2, e3))
                x_dat.append(
                    (str(v1), str(v2), dict(label=ename))
                )  # networkx compliant
        else:
            error(
                "Embedding nets of dimension %i is not currently implemented." % dim
                + " Also, why?...."
            )


        return (x_dat, voltages)

    def gen_sage_graph_format(self, edges):
        """Take the edges from a systre db file and convert
        to sage graph readable format.

        Assumes that the direction of the edge goes from
        [node1] ---> [node2]
        """
        sage_dict = {}
        voltages = []
        for id, (v1, v2, e1, e2, e3) in enumerate(edges):
            ename = "e%i" % (id + 1)
            voltages.append((e1, e2, e3))
            try:
                # n1 = chr(v1-1 + ord("A"))
                n1 = str(v1)
            except ValueError:
                n1 = str(v1)
            try:
                # n2 = chr(v2-1 + ord("A"))
                n2 = str(v2)
            except ValueError:
                n2 = str(v2)

            sage_dict.setdefault(n1, {})
            sage_dict.setdefault(n2, {})
            sage_dict[n1].setdefault(n2, [])  # SAGE compliant
            sage_dict[n1][n2].append(ename)  # SAGE compliant
        return (sage_dict, voltages)


class Net:
    def __init__(self, graph=None, dim=3, options=None):
        self.name = None
        self.lattice_basis = None
        self.metric_tensor = None
        self.cycle = None
        self.cycle_rep = None
        self.cocycle = None
        self.cocycle_rep = None
        self.periodic_rep = None  # alpha(B)
        self.edge_labels = None
        self.node_labels = None
        self.colattice_dotmatrix = None
        self.colattice_inds = None  # keep track of all the valid colattice dot indices
        self.voltage = None
        self._graph = graph
        # n-dimensional representation, default is 3
        self.ndim = dim
        if graph is not None:
            '''
            self._graph = nx.MultiDiGraph(graph) # networkx compliant
            self.original_graph = nx.MultiDiGraph(graph) # networkx compliant
            '''
            self._graph = nx.MultiDiGraph()
            self.original_graph = nx.MultiDiGraph()
            for (e1, e2, d) in graph:
                self._graph.add_edge(e1, e2, **d, key=d["label"])
                self.original_graph.add_edge(e1, e2, **d, key=d["label"])

            # self._graph = DiGraph(graph, multiedges=True, loops=True) # SAGE compliant
            # self._graph = nx.MultiDiGraph(graph) # networkx compliant
            # print(graph, list(self._graph.edges))
            # Keep an original for reference purposes.
            # self.original_graph = DiGraph(graph, multiedges=True, loops=True) # SAGE compliant
            # self.original_graph = nx.MultiDiGraph(graph) # networkx compliant

        self.options = options

    def nodes_iter(self, data=True):
        """Oh man, fixing to networkx 2.0

        This probably breaks a lot of stuff in the code. THANKS NETWORKX!!!!!!!1

        """
        for node in self._graph.nodes():
            if data:
                d = self._graph.node[node]
                yield (node, d)
            else:
                yield node

    def edges_iter(self, data=True):
        for erp in self._graph.edges(data=data):
            # d=self.edges[(n1,n2)]
            if data:
                yield (erp[0], erp[1], erp[2])
            else:
                yield (erp[0], erp[1])


    def get_cocycle_basis(self):
        """The orientation is important here!"""
        size = self._graph.order() - 1
        length = self._graph.size()
        count = 0
        for vert in self.vertices():  # networkx compliant
            if count == size:
                break
            vect = np.zeros(length)
            inds_out = self.return_indices(self.out_edges(vert))
            inds_in = self.return_indices(self.in_edges(vert))
            # deal with in_out case by Hang 20221109
            inds_in_out = [value for value in inds_out if value in inds_in] 
            if inds_out :
                vect[inds_out] = 1.0
            if inds_in:
                vect[inds_in] = -1.0
            if inds_in_out:
                vect[inds_in_out] = 0 
            if self.cycle_cocycle_check(vect):  # or len(self.neighbours(vert)) == 2:
                count += 1
                self.cocycle = self.add_to_array(vect, self.cocycle)

        if count != size:
            print("ERROR - could not find a linearly independent cocycle basis!")
        # special case - pcu
        if size == 0:
            self.cocycle = None
            self.cocycle_rep = None
        else:
            self.cocycle = np.array(self.cocycle)
            self.cocycle_rep = np.zeros((size, self.ndim))

    def add_name(self):
        name = str(self.order + 1)
        # name = chr(order + ord("A"))
        return name

    def insert_and_join(self, vfrom, vto, edge_label=None):
        if edge_label is None:
            edge_label = "e%i" % (self.shape)
        self.add_vertex(vto)
        edge = (vfrom, vto, edge_label)
        self.add_edge(vfrom, vto, edge_label)
        return edge

    def add_edges_between(self, edge, N):
        newedges = []
        V1 = edge[0] if edge in self.out_edges(edge[0]) else edge[1]
        V2 = edge[1] if edge in self.in_edges(edge[1]) else edge[0]

        name = self.add_name()
        newedges.append(self.insert_and_join(V1, name, edge_label=edge[2]))
        vfrom = name
        d = self.ndim
        newnodes = []
        for i in range(N - 1):
            newnodes.append(vfrom)
            name = self.add_name()
            newedges.append(self.insert_and_join(vfrom, name))
            vfrom = name
            self.voltage = np.concatenate((self.voltage, np.zeros(d).reshape(1, d)))
        # final edge to V2
        newnodes.append(V2)
        lastedge = (vfrom, V2, "e%i" % (self.shape))
        newedges.append(lastedge)

        self.add_edge(vfrom, V2, "e%i" % (self.shape))
        self.delete_edge(edge)
        self.voltage = np.concatenate((self.voltage, np.zeros(d).reshape(1, d)))
        return newnodes, newedges

    def cycle_cocycle_check(self, vect):
        if self.cocycle is None and self.cycle is None:
            return True
        elif self.cocycle is None and self.cycle is not None:
            return self.check_linear_dependency(vect, self.cycle)
        else:
            return self.check_linear_dependency(
                vect, self.add_to_array(self.cocycle, self.cycle)
            )

    def get_cycle_basis(self):
        """Find the basis for the cycle vectors. The total number of cycle vectors
        in the basis is E - V + 1 (see n below). Once this number of cycle vectors is found,
        the program returns.

        NB: Currently the cycle vectors associated with the lattice basis are included
        in the cycle basis - this is so that the embedding of the barycentric placement
        of the net works out properly. Thus the function self.get_lattice_basis()
        should be called prior to this.

        """

        c = self.iter_cycles(
            node=self.vertices(0),
            edge=None,
            cycle=[],
            used=[],
            nodes_visited=[],
            cycle_baggage=[],
            counter=0,
        )
        n = self.shape - self.order + 1
        count = 0
        if self.lattice_basis is not None:
            self.cycle = self.add_to_array(self.lattice_basis, self.cycle)
            self.cycle_rep = self.add_to_array(np.identity(self.ndim), self.cycle_rep)
            count += self.ndim
        for id, cycle in enumerate(c):
            if count >= n:
                break
            vect = np.zeros(self.shape)
            vect[self.return_indices(cycle)] = self.return_coeff(cycle)
            volt = self.get_voltage(vect)
            # REPLACE WITH CHECK_LINEAR_DEPENDENCY()
            check = self.cycle_cocycle_check(vect)
            if np.all(np.abs(volt) < 1.001) and np.sum(np.abs(volt) > 0.0) and check:
                self.cycle = self.add_to_array(vect, self.cycle)
                self.cycle_rep = self.add_to_array(volt, self.cycle_rep)
                count += 1
        self.cycle = np.array(self.cycle)
        self.cycle_rep = np.array(self.cycle_rep)
        del c

    def add_to_array(self, vect, rep):
        """Works assuming the dimensions are the same"""
        if len(vect.shape) == 1:
            v = np.reshape(vect, (1, vect.shape[-1]))
        else:
            v = vect
        if rep is None:
            return v.copy()
        else:
            return np.concatenate((rep, v))

    def get_voltage(self, cycle):
        return np.dot(cycle, self.voltage)

    def debug_print(self, val, msg):
        print("%s[%d] %s" % ("  " * val, val, msg))

    def simple_cycle_basis(self):
        """Cycle basis is constructed using a minimum spanning tree.
        This tree is traversed, and all the remaining edges are added
        to obtain the basis.

        """
        edges = self.all_edges()
        st_vtx = np.random.choice(range(self.graph.order()))
        # mspt = self.graph.to_undirected().min_spanning_tree(starting_vertex=st_vtx) # SAGE compliant
        # tree = Graph(mspt, multiedges=False, loops=False) # SAGE compliant
        # cycle_completes = [i for i in edges if i not in mspt and (i[1], i[0], i[2]) not in mspt] # SAGE compliant
        tree = nx.minimum_spanning_tree(
            self.graph.to_undirected()
        )  # networkx compliant
        mspt_edges = [
            (i, j, d["label"]) for (i, j, d) in tree.edges(data=True)
        ]  # networkx compliant
        cycle_completes = [
            i
            for i in edges
            if i not in mspt_edges and (i[1], i[0], i[2]) not in mspt_edges
        ]  # networkx compliant
        # self.graph.show()
        self.cycle = []
        self.cycle_rep = []
        for (v1, v2, e) in cycle_completes:
            # path = tree.shortest_path(v1, v2) # SAGE compliant
            path = nx.shortest_path(tree, source=v1, target=v2)  # networkx compliant
            basis_vector = np.zeros(self.shape)
            cycle, coefficients = [], []
            for pv1, pv2 in zip(path[:-1], path[1:]):
                # edge = [i for i in tree.edges_incident([pv1, pv2]) if pv1 in i[:2] and pv2 in i[:2]][0] # SAGE compliant
                edge = [
                    (i, j, d["label"])
                    for (i, j, d) in tree.edges(nbunch=[pv1, pv2], data=True)
                    if pv1 in (i, j) and pv2 in (i, j)
                ][
                    0
                ]  # networkx compliant
                if edge not in edges:
                    edge = (edge[1], edge[0], edge[2])
                    if edge not in edges:
                        error(
                            "Encountered an edge (%s, %s, %s) not in " % (edge)
                            + " the graph while finding the basis of the cycle space!"
                        )
                coeff = 1.0 if edge in self.out_edges(pv1) else -1.0
                coefficients.append(coeff)
                cycle.append(edge)
            # opposite because we are closing the loop. i.e. going from v2 back to v1
            edge = (v1, v2, e) if (v1, v2, e) in edges else (v2, v1, e)
            coeff = 1.0 if edge in self.in_edges(v1) else -1.0
            coefficients.append(coeff)
            cycle.append(edge)
            basis_vector[self.return_indices(cycle)] = coefficients
            voltage = self.get_voltage(basis_vector)
            self.cycle.append(basis_vector)
            self.cycle_rep.append(voltage)
        self.cycle = np.array(self.cycle)
        self.cycle_rep = np.array(self.cycle_rep)

    def iter_cycles(
        self,
        node=None,
        edge=None,
        cycle=[],
        used=[],
        nodes_visited=[],
        cycle_baggage=[],
        counter=0,
    ):
        """Recursive method to iterate over all cycles of a graph.
        NB: Not tested to ensure completeness, however it does find cycles.
        NB: Likely produces duplicate cycles along different starting points
        **last point fixed but not tested**

        """
        if node is None:
            node = self.vertices(0)
        if node in nodes_visited:
            i = nodes_visited.index(node)
            nodes_visited.append(node)
            cycle.append(edge)
            used.append(edge[:3])
            c = cycle[i:]
            uc = sorted([j[:3] for j in c])
            # yield c
            if uc in cycle_baggage:
                pass
            else:
                cycle_baggage.append(uc)
                yield c
        else:
            nodes_visited.append(node)
            if edge:
                cycle.append(edge)
                used.append(edge[:3])
            e = [
                (x, y, z, 1)
                for x, y, z in self.out_edges(node)
                if (x, y, z) not in used
            ]
            e += [
                (x, y, z, -1)
                for x, y, z in self.in_edges(node)
                if (x, y, z) not in used
            ]
            for j in e:
                newnode = j[0] if j[0] != node else j[1]
                # msg = "test: (%s to %s) via %s"%(node, newnode, j[2])
                # self.debug_print(counter, msg)
                for val in self.iter_cycles(
                    node=newnode,
                    edge=j,
                    cycle=cycle,
                    used=used,
                    nodes_visited=nodes_visited,
                    cycle_baggage=cycle_baggage,
                    counter=counter + 1,
                ):
                    yield val
                nodes_visited.pop(-1)
                cycle.pop(-1)
                used.pop(-1)

    def linear_independent_vectors(self, R, dim):
        R = np.matrix(R)
        r = np.linalg.matrix_rank(R)
        index = np.zeros(
            r
        )  # this will save the positions of the li columns in the matrix
        counter = 0
        index[
            0
        ] = 0  # without loss of generality we pick the first column as linearly independent
        j = 0  # therefore the second index is simply 0
        for i in range(R.shape[1]):  # loop over the columns
            if i != j:  # if the two columns are not the same
                inner_product = np.dot(R[:, i].T, R[:, j])  # compute the scalar product
                norm_i = np.linalg.norm(R[:, i])  # compute norms
                norm_j = np.linalg.norm(R[:, j])

                # inner product and the product of the norms are equal only if the two vectors are parallel
                # therefore we are looking for the ones which exhibit a difference which is bigger than a threshold
                if np.abs(inner_product - norm_j * norm_i) > 1e-4:
                    counter += 1  # counter is incremented
                    index[counter] = i  # index is saved
                    j = i  # j is refreshed
                # do not forget to refresh j: otherwise you would compute only the vectors li with the first column!!

        R_independent = np.zeros((r, dim))

        i = 0
        # now save everything in a new matrix
        while i < r:
            R_independent[i, :] = R[index[i], :]
            i += 1
        return R_independent

    def get_lattice_basis(self):
        L = []
        inds = list(range(self.cycle_rep.shape[0]))
        np.random.shuffle(inds)
        cycle_rep = self.cycle_rep.copy()
        cycle = self.cycle.copy()
        # j = cycle_rep
        # determine the null space of the cycle_rep w.r.t. the lattice unit vectors.
        lattice = []
        for e in np.identity(self.ndim):
            kk = np.vstack((e, cycle_rep))
            j = sy.Matrix(kk.T)
            null = np.array(
                [np.array(k).flatten() for k in j.nullspace()], dtype=float
            )
            found_vector = False
            for nulv in null:
                if abs(nulv[0]) == 1.0:
                    v = -nulv[1:] * nulv[0]
                    nz = np.nonzero(v)
                    tv = np.sum(cycle[nz] * v[nz][:, None], axis=0)
                    if self.is_integral(tv):
                        found_vector = True
                        lattice.append(tv)
                        break
            if not found_vector:
                error("Could not obtain the lattice basis from the cycle vectors!")
                # Terminate(1)
                return -1
        self.lattice_basis = np.array(lattice)
        return 1

    def check_linear_dependency(self, vect, vset):
        if not np.any(vset):
            return True
        else:
            A = np.concatenate((vset, np.reshape(vect, (1, self.shape))))
        lrank = vset.shape[0] + 1
        # U, s, V = np.linalg.svd(A)
        # if np.all(s > 0.0001):
        if np.linalg.matrix_rank(A) == lrank:
            return True
        return False

    def get_index(self, edge):
        return int(edge[2][1:]) - 1

    def return_indices(self, edges):
        return [self.get_index(i) for i in edges]

    def return_coeff(self, edges):
        assert edges[0][3]
        return [i[3] for i in edges]

    def to_ind(self, str_obj):
        return tuple([int(i) for i in str_obj.split("_")[1:]])

    def assign_ip_matrix(self, mat, inds):
        """Get the colattice dot matrix from Builder.py. This is an inner
        product matrix of all the SBUs assigned to particular nodes.
        """
        max_ind, max_val = (
            mat[np.diag_indices_from(mat)].argmax(),
            mat[np.diag_indices_from(mat)].max(),
        )
        self.scale = (([max_ind], [max_ind]), max_val)
        # this sbu_tensor_matrix is probably not needed...
        self.sbu_tensor_matrix = mat
        self.colattice_inds = inds
        self.colattice_dotmatrix = np.zeros((mat.shape[0], mat.shape[1]))
        # self.colattice_dotmatrix = np.array(mat)
        # return
        for (i, j) in zip(*np.triu_indices_from(mat)):
            if i == j:
                self.colattice_dotmatrix[i, j] = mat[i, j] / max_val
            else:
                val = mat[i, j] / np.sqrt(mat[i, i]) / np.sqrt(mat[j, j])
                self.colattice_dotmatrix[i, j] = val
                self.colattice_dotmatrix[j, i] = val

    def convert_params(self, x, ndim, angle_inds, cocycle_size):
        cell_lengths = x[:ndim]
        angles = x[ndim : ndim + angle_inds]
        cocycle = x[ndim + angle_inds :]
        mt = np.empty((ndim, ndim))
        for i in range(ndim):
            mt[i, i] = x[i]
        count = i + 1
        # g = [i[::-1] for i in np.triu_indices(ndim, 1)]
        g = np.triu_indices(ndim, 1)
        for (i, j) in zip(*g):
            val = x[count] * np.sqrt(mt[i, i]) * np.sqrt(mt[j, j])
            mt[i, j] = val
            mt[j, i] = val
            count += 1
        # convention alpha --> b,c beta --> a,c gamma --> a,b
        # in the metric tensor, these are related to the
        # (1,2), (0,2), and (0,1) array elements, which
        # are in the reversed order of how they would
        # be iterated.
        # assuming the parameters are defined in 'x' as
        # x[3] --> a.b  \
        # x[4] --> a.c  |--> these are in reversed order.
        # x[5] --> b.c  /
        cocycle_rep = np.reshape(cocycle, (cocycle_size, ndim))
        return mt, cocycle_rep



    def report_errors_nlopt(self):
        la = np.dot(self.cycle_cocycle_I, self.periodic_rep)
        inner_p = np.dot(np.dot(la, self.metric_tensor), la.T)
        # sc_fact = np.diag(inner_p).max()
        # for (i, j) in zip(*np.triu_indices_from(inner_p)):
        #    val = inner_p[i,j]
        #    if i != j:
        #        v = val/np.sqrt(inner_p[i,i])/np.sqrt(inner_p[j,j])
        #        inner_p[i,j] = v
        #        inner_p[j,i] = v
        # inner_p[np.diag_indices_from(inner_p)]# /= sc_fact
        nz = self.colattice_inds
        cdmat = self.colattice_dotmatrix
        # fit = inner_p[nz] - cdmat[nz]
        edge_lengths = []
        angles = []
        count = 0

        for (i, j) in zip(*nz):
            if i != j:
                ang1 = np.arccos(
                    inner_p[i, j] / np.sqrt(inner_p[i, i]) / np.sqrt(inner_p[j, j])
                )
                ang2 = np.arccos(
                    cdmat[i, j]
                )  # /np.sqrt(cdmat[i,i])/np.sqrt(cdmat[j,j]))
                ang = (ang1 - ang2) ** 2
                angles.append(ang)
            else:
                len = (
                    np.sqrt(inner_p[i, j]) - np.sqrt(cdmat[i, j] * self.scale[1])
                ) ** 2
                edge_lengths.append(len)
            count += 1
        edge_average, edge_std = (
            np.sqrt(np.mean(edge_lengths)),
            np.sqrt(np.std(edge_lengths)),
        )
        debug(
            "Average error in edge length: %12.5f +/- %9.5f Angstroms"
            % (edge_average, edge_std)
        )
        # math.copysign(1, edge_average)*
        # np.sqrt(abs(edge_average)*self.scale[1]),
        # math.copysign(1, edge_std)*
        # np.sqrt(abs(edge_std))*self.scale[1])))
        angle_average, angle_std = np.sqrt(np.mean(angles)), np.sqrt(np.std(angles))
        debug(
            "Average error in edge angles: %12.5f +/- %9.5f degrees"
            % (angle_average / DEG2RAD, angle_std / DEG2RAD)
        )
        if self.options is not None:
            self.options.csv.add_data(**{"edge_length_err.1": edge_average})
            self.options.csv.add_data(**{"edge_length_std.1": edge_std})
            self.options.csv.add_data(**{"edge_angle_err.1": angle_average / DEG2RAD})
            self.options.csv.add_data(**{"edge_angle_std.1": angle_std / DEG2RAD})

    def report_errors(self, fit):
        edge_lengths = []
        angles = []
        nz = np.nonzero(np.triu(np.array(self.colattice_dotmatrix)))
        count = 0
        for (i, j) in zip(*nz):
            if i != j:
                angles.append(fit[count])
            else:
                edge_lengths.append(fit[count])
            count += 1
        edge_average, edge_std = np.mean(edge_lengths), np.std(edge_lengths)
        debug(
            "Average error in edge length: %12.5f +/- %9.5f Angstroms"
            % (
                math.copysign(1, edge_average)
                * np.sqrt(abs(edge_average) * self.scale[1]),
                math.copysign(1, edge_std) * np.sqrt(abs(edge_std) * self.scale[1]),
            )
        )
        angle_average, angle_std = np.mean(angles), np.std(angles)
        debug(
            "Average error in edge angles: %12.5f +/- %9.5f degrees"
            % (angle_average / DEG2RAD, angle_std / DEG2RAD)
        )

    def get_metric_tensor(self):
        # self.metric_tensor = self.lattice_basis*self.projection*self.lattice_basis.T
        self.metric_tensor = np.dot(
            np.dot(self.lattice_basis, self.eon_projection), self.lattice_basis.T
        )

    def print_edge_count(self):
        if self.cocycle is not None:
            self.cocycle_rep = np.zeros((self.order - 1, self.ndim))
            # self.cocycle_rep = (np.random.random((self.order-1, self.ndim)) - .5)/40. #used to make MOFs from unstable nets
            # self.cocycle_rep[0] = np.array([0.0, 0.0, -0.5]) # Used to show non-barycentric placement of pts net for Smit meeting.
            self.periodic_rep = np.concatenate(
                (self.cycle_rep, self.cocycle_rep), axis=0
            )
        else:
            self.periodic_rep = self.cycle_rep
        latt_counts = []
        for j in self.lattice_basis:
            latt_counts.append(np.sum(np.abs(j)))
        return ",".join(["%i" % i for i in latt_counts])

    def barycentric_embedding(self):
        if self.cocycle is not None:
            self.cocycle_rep = np.zeros((self.order - 1, self.ndim))
            # self.cocycle_rep = (np.random.random((self.order-1, self.ndim)) - .5)/40. #used to make MOFs from unstable nets
            # self.cocycle_rep[0] = np.array([0.0, 0.0, -0.5]) # Used to show non-barycentric placement of pts net for Smit meeting.
            self.periodic_rep = np.concatenate(
                (self.cycle_rep, self.cocycle_rep), axis=0
            )
        else:
            self.periodic_rep = self.cycle_rep
        self.get_metric_tensor()
        # for toz
        # m = self.metric_tensor[np.diag_indices_from(self.metric_tensor)].max()
        # self.metric_tensor = m*np.array([[1.,0.745, 0.745],[0.745, 1., 0.745],[0.745, 0.745, 1.]])
        # for toz

    def get_2d_params(self):
        self.metric_tensor = np.dot(
            np.dot(self.lattice_basis, self.projection), self.lattice_basis.T
        )
        lena = math.sqrt(self.metric_tensor[0, 0])
        lenb = math.sqrt(self.metric_tensor[1, 1])
        gamma = math.acos(self.metric_tensor[1, 0] / lena / lenb)
        return lena, lenb, gamma

    def get_3d_params(self):
        if self.ndim == 2:
            lena = math.sqrt(self.metric_tensor[0, 0])
            lenb = math.sqrt(self.metric_tensor[1, 1])
            lenc = self.options.third_dimension
            alpha = np.pi / 2.0
            beta = np.pi / 2.0
            gamma = math.acos(self.metric_tensor[0, 1] / lena / lenb)

        elif self.ndim == 3:
            lena = math.sqrt(self.metric_tensor[0, 0])
            lenb = math.sqrt(self.metric_tensor[1, 1])
            lenc = math.sqrt(self.metric_tensor[2, 2])
            alpha = math.acos(self.metric_tensor[1, 2] / lenb / lenc)
            beta = math.acos(self.metric_tensor[0, 2] / lena / lenc)
            gamma = math.acos(self.metric_tensor[0, 1] / lena / lenb)
        return lena, lenb, lenc, alpha, beta, gamma

    def vertex_positions(self, edges, used, pos={}, bad_ones={}):
        """Recursive function to find the nodes in the unit cell.
        How it should be done:

        Create a growing tree around the init placed vertex. Evaluate
        which vertices wind up in the unit cell and place them.  Continue
        growing from those vertices in the unit cell until all are found.
        """
        # NOTE: NOT WORKING - FIX!!!
        lattice_arcs = self.lattice_arcs
        if self.ndim == 2:
            lattice_arcs = np.hstack(
                (np.array(lattice_arcs), np.zeros((np.array(lattice_arcs).shape[0], 1)))
            )
        if len(pos.keys()) == self.graph.order():
            return pos
        else:
            # generate all positions from all edges growing outside of the current vertex
            # iterate through each until an edge is found which leads to a vertex in the
            # unit cell.
            e = edges[0]
            if e[0] not in pos.keys() and e[1] not in pos.keys():
                pass
            elif e[0] not in pos.keys() or e[1] not in pos.keys():
                from_v = e[0] if e[0] in pos.keys() else e[1]
                to_v = e[1] if e[1] not in pos.keys() else e[0]

                coeff = 1.0 if e in self.out_edges(from_v) else -1.0
                index = self.get_index(e)

                to_pos = coeff * np.array(lattice_arcs)[index] + pos[from_v]
                newedges = []
                # FROM HERE REMOVED IN-CELL CHECK
                to_pos = np.array([i % 1 for i in to_pos])
                pos.update({to_v: to_pos})
                used.append(e)
                ee = self.neighbours(to_v)
                newedges = [i for i in ee if i not in used and i not in edges]
                edges = newedges + edges[1:]
            else:
                used.append(e)
                edges = edges[1:]
            return self.vertex_positions(edges, used, pos, bad_ones)

    def indices_with_voltage(self, volt):
        return np.where([np.all(i == volt) for i in self.cycle_rep])

    def is_integral(self, vect):
        return np.all(np.equal(np.mod(vect, 1), 0)) and not np.all(np.equal(0, vect))
        # return np.all(np.logical_or(np.abs(vect) == 0., np.abs(vect) == 1.))

    @property
    def kernel(self):
        if hasattr(self, "_kernel"):
            return self._kernel
        kernel_vectors = []
        max_count = self.shape - self.ndim - (self.order - 1)
        # if no kernel vectors need to be found, just return
        # the cocycle vectors.
        if max_count == 0:
            self._kernel = self.cocycle.copy()
            return self._kernel
        self._kernel = None
        j = sy.Matrix(self.cycle_rep.T)
        null = np.array([np.array(k).flatten() for k in j.nullspace()], dtype=float)
        for null_vector in null:
            nz = np.nonzero(null_vector)
            cv_comb = np.sum(self.cycle[nz] * null_vector[nz][:, None], axis=0)
            if self.is_integral(cv_comb):
                kernel_vectors.append(cv_comb)
            if len(kernel_vectors) >= max_count:
                break
        # if not enough kernel vectors were found from the cycle basis,
        # then iterate over cycles which are linearly independent from
        # the vectors already in the kernel. NB: this is fucking slow.
        if len(kernel_vectors) != max_count:
            warning(
                "The number of vectors in the kernel does not match the size of the graph!"
            )
            # convert cycle to vect: e.g. [('1', '15', 'e1', 1), ('15', '16', 'e138', 1), ('1', '16', 'e12', -1)]
            edge_list=list(self.graph.edges)
            cycle_vect=[]
            for i in self.cycle:
                vect_temp=[]
                for j in range(len(i)):
                    if i[j] != 0:
                        x,y,z=edge_list[j]
                        vect_temp.append((x, y, z, int(i[j])))
                cycle_vect.append(vect_temp)

            c=iter(cycle_vect)  # modified by Hang @ 20221120
            while len(kernel_vectors) < max_count:
                try:
                    if version_info.major >= 3:
                        cycle = next(c)
                    else:
                        cycle = c.next()
                except StopIteration:
                    # give up, use the cocycle basis
                    self._kernel = self.cocycle.copy()
                    return self._kernel
                vect = np.zeros(self.shape)
                vect[self.return_indices(cycle)] = self.return_coeff(cycle)
                volt = self.get_voltage(vect)
                if np.allclose(
                    np.abs(volt), np.zeros(3)
                ) and self.check_linear_dependency(vect, np.array(kernel_vectors)):
                    kernel_vectors.append(vect)
        try:
            self._kernel = np.concatenate((np.array(kernel_vectors), self.cocycle))
        except ValueError:
            self._kernel = np.array(kernel_vectors)
        return self._kernel

    def vertices(self, vertex=None):
        if vertex is not None:
            # return self._graph.vertices()[vertex] # SAGE compliant
            return list(self._graph.nodes())[vertex]  # networkx compliant
        # return self._graph.vertices()  # SAGE compliant
        return list(self._graph.nodes())  # networkx compliant

    def out_edges(self, vertex):
        # out =  self.graph.outgoing_edges(vertex) # SAGE compliant
        out = [
            (i, j, d["label"]) for (i, j, d) in self.graph.out_edges(vertex, data=True)
        ]  # networkx compliant
        if out is None:
            return []
        return out

    def in_edges(self, vertex):
        # ine = self.graph.incoming_edges(vertex) # SAGE compliant
        ine = [
            (i, j, d["label"]) for (i, j, d) in self.graph.in_edges(vertex, data=True)
        ]  # networkx compliant
        if ine is None:
            return []
        return ine

    def all_edges(self):
        # return self.graph.edges() # SAGE compliant
        return [(i, j, d["label"]) for (i, j, d) in self.graph.edges(data=True)]

    def neighbours(self, vertex):
        return self.out_edges(vertex) + self.in_edges(vertex)

    def loop_edges(self):
        # return self.graph.loop_edges() # SAGE compliant
        return [
            (i, j, d["label"]) for (i, j, d) in self.graph.edges(data=True) if i == j
        ]

    def add_vertex(self, v):
        # self.graph.add_vertex(v) # SAGE compliant
        self.graph.add_node(v)  # networkx compliant

    def add_edge(self, v1, v2, name):
        # self.graph.add_edge(v1, v2, name) # SAGE compliant
        self.graph.add_edge(v1, v2, label=name, key=name)  # networkx compliant

    def delete_edge(self, e):
        # self.graph.delete_edge(e) # SAGE compliant
        for (v1, v2, d) in self._graph.edges(data=True):
            if (v1, v2, d["label"]) == e:
                self._graph.remove_edge(v1, v2, key=d["label"])
                return

        error("could not find the edge (%s, %s, %s) in the graph" % (tuple(e)))
        sys.exit()

    @property
    def minimal(self):
        if len(self.cycle) > self.ndim:
            return False
        elif len(self.cycle) == self.ndim:
            return True
        else:
            info(
                "Net is not periodic in the number of desired dimensions."
                + " This feature has not been implemented yet"
            )
            return False

    @property
    def eon_projection(self):
        if not self.minimal:
            d = np.dot(self.kernel, self.kernel.T)
            d_inv = np.array(np.matrix(d).I)
            sub_mat = np.dot(np.dot(self.kernel.T, d_inv), self.kernel)
            return np.identity(self.shape) - sub_mat
        # if the projection gets here this is a minimal embedding
        return np.identity(self.shape)

    @property
    def projection(self):
        la = self.lattice_arcs
        d = np.dot(la.T, la)
        d_inv = np.array(np.matrix(d).I)
        return np.dot(np.dot(la, d_inv), la.T)

    @property
    def lattice_arcs(self):
        return np.dot(self.cycle_cocycle_I, self.periodic_rep)

    @property
    def shape(self):
        return self._graph.size()

    @property
    def order(self):
        return self._graph.order()

    @property
    def graph(self):
        return self._graph

    @graph.setter
    def graph(self, g):
        # self._graph = DiGraph(g, multiedges=True, loops=True) # SAGE compliant
        self._graph = nx.MultiDiGraph(g)

    @property
    def cycle_cocycle_I(self):
        try:
            return self._cycle_cocycle_I
        except AttributeError:
            self._cycle_cocycle_I = np.array(np.matrix(self.cycle_cocycle).I)
            return self._cycle_cocycle_I

    @property
    def cycle_cocycle(self):
        try:
            return self._cycle_cocycle
        except AttributeError:
            if self.cocycle is None and self.cycle is None:
                raise AttributeError(
                    "Both the cycle and cocycle " + "basis have not been allocated"
                )
            elif self.cocycle is None:
                self._cycle_cocycle = self.cycle.copy()

            elif self.cycle is None:
                raise AttributeError("The cycle " + "basis has not been allocated")
            else:
                self._cycle_cocycle = np.concatenate((self.cycle, self.cocycle))
            return self._cycle_cocycle
