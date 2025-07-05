import numpy as np
import scipy.sparse as sp
from numpy.ctypeslib import ndpointer
import ctypes
import os
from pathlib import Path

_MODULE_DIR = Path(__file__).parent.absolute()

class CSCMatrix(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_double)),
        ("indices", ctypes.POINTER(ctypes.c_size_t)),
        ("indptr", ctypes.POINTER(ctypes.c_size_t)),
        ("n_rows", ctypes.c_size_t),
        ("n_cols", ctypes.c_size_t),
        ("nnz", ctypes.c_size_t)
    ]

def _load_library():
    if os.name == 'nt':
        lib_name = 'cmgCluster.dll'
    elif os.name == 'posix':
        lib_name = 'libcmgCluster.dylib' if os.uname()[0] == 'Darwin' else 'libcmgCluster.so'
    lib_path = _MODULE_DIR / lib_name
    if not lib_path.exists():
        raise RuntimeError(f"Required library {lib_path} not found")
    return ctypes.CDLL(str(lib_path))

try:
    _lib = _load_library()
    _lib.steiner_group.argtypes = [
        ctypes.POINTER(CSCMatrix),
        ndpointer(dtype=np.uint32, flags='C_CONTIGUOUS')
    ]
    _lib.steiner_group.restype = ctypes.c_uint32
except Exception as e:
    raise ImportError(f"Failed to initialize cmgCluster library: {str(e)}")

def cmgCluster(A: sp.spmatrix) -> tuple:
    if not sp.issparse(A):
        raise TypeError("Input matrix must be a sparse matrix")
    if A.shape[0] != A.shape[1]:
        raise ValueError("Input matrix must be square")
    if not sp.isspmatrix_csc(A):
        A = A.tocsc()

    A_indices = np.ascontiguousarray(A.indices, dtype=np.intp)
    A_indptr = np.ascontiguousarray(A.indptr, dtype=np.intp)

    matrix = CSCMatrix(
        data=A.data.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        indices=A_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t)),
        indptr=A_indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t)),
        n_rows=A.shape[0],
        n_cols=A.shape[1],
        nnz=A.nnz
    )

    cI = np.zeros(A.shape[0], dtype=np.uint32)
    nc = _lib.steiner_group(ctypes.byref(matrix), cI)
    return cI, nc

__all__ = ['cmgCluster']
