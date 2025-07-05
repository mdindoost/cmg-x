import numpy as np
import scipy.sparse as sp
from cmgx.core import cmgCluster

def test_tiny_graph():
    A = sp.csc_matrix([
        [1, -1,  0],
        [-1, 2, -1],
        [0, -1, 1]
    ])
    labels, nc = cmgCluster(A)
    assert nc >= 1
