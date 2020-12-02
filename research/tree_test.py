import unittest
from graph import *


def setify(cycles):
    return frozenset(frozenset(cycle) for cycle in cycles)


def cycles_from_edges(edges):
    return setify(cycle_vertices(construct_graph_from_edges(edges)))


def count_overlap(s1, s2):
    return sum(s in s2 for s in s1)


class GraphCyclesAtLeastOne(unittest.TestCase):
    def test_no_cycles(self):
        edges = [[0, 1], [1, 2], [2, 3]]
        expected = {}
        count = count_overlap(expected, cycles_from_edges(edges))
        self.assertEqual(count, 0)

    def test_only_single(self):
        edges = [[0, 1], [1, 2], [2, 0]]
        expected = {frozenset([0, 1, 2])}
        count = count_overlap(expected, cycles_from_edges(edges))
        self.assertGreaterEqual(count, 1)

    def test_disconnected_double(self):
        edges = [[0, 1], [1, 2], [2, 0]] + [[3, 4], [4, 5], [5, 3]]
        expected = {frozenset([0, 1, 2]), frozenset([3, 4, 5])}
        count = count_overlap(expected, cycles_from_edges(edges))
        self.assertGreaterEqual(count, 1)

    def test_connected_double(self):
        edges = [[0, 1], [1, 2], [2, 0]] + [[1, 3], [3, 0]]
        expected = {frozenset([0, 1, 2]), frozenset([0, 1, 3])}
        count = count_overlap(expected, cycles_from_edges(edges))
        self.assertGreaterEqual(count, 1)

    def test_nested_cycles(self):
        edges = [[0, 1], [1, 2], [2, 3], [3, 0]] + [[1, 3]]
        expected = {frozenset([0, 1, 2, 3]), frozenset([1, 2, 3]), frozenset([0, 1, 3])}
        count = count_overlap(expected, cycles_from_edges(edges))
        self.assertGreaterEqual(count, 1)

    def test_k4(self):
        edges = [[0, 1], [1, 2], [2, 3], [3, 0]] + [[1, 3]] + [[0, 2]]
        expected = {frozenset([0, 1, 2, 3]), frozenset([0, 1, 2]), frozenset([1, 2, 3]), frozenset([0, 2, 3])}
        count = count_overlap(expected, cycles_from_edges(edges))
        self.assertGreaterEqual(count, 1)


if __name__ == '__main__':
    unittest.main()
