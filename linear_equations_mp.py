"""
https://en.wikipedia.org/wiki/Gaussian_elimination

h := 1 / * Initialization of the pivot row * /
k := 1 / * Initialization of the pivot column * /

while h ≤ m and k ≤ n
 /* Find the k-th pivot: */
    i_max := argmax(i=h ... m, abs(A[i, k]))
    if A[i_max, k] = 0
        /* No pivot in this column, pass to next column * /
        k := k + 1
    else
        swap rows(h, i_max)
        /* Do for all rows below pivot: */
        for i = h + 1 ... m:
            f := A[i, k] / A[h, k]
             /* Fill with zeros the lower part of pivot column: */
            A[i, k] := 0
            /* Do for all remaining elements in current row: */
            for j = k + 1 ... n:
                A[i, j] := A[i, j] - A[h, j] * f
        /* Increase pivot row and column * /
        h := h + 1
        k := k + 1
"""
import numpy as np
from numba import njit
from numba.openmp import openmp_content as openmp
from numba.openmp import omp_get_num_threads


def swap_rows(A, h, i_max):
    if not h == i_max:
        print("swapping %s and %s" % (A[h], A[i_max]))
        t1 = A[i_max]
        t2 = A[h]
        A[i_max] = t2
        A[h] = t1
        # for i in range(len(A[h])):
            # A[i_max][i] += A[h][i]
            # A[h][i] = A[i_max][i] - A[h][i]
            # A[i_max][i] -= A[h][i]
    return A


def argmax(A, h, m, k):
    """Return row number of maximum abs value of elements along pivot column k."""
    i_max = h
    for i in range(h, m):
        if abs(A[i][k]) > abs(A[i_max][k]):
            i_max = i
    return i_max


def format_mat(m):
    return "[\n %s\n]" % "\n ".join([str(row) for row in m])

def format_mat_marked(m, i, j):
    return format_mat([["*%s*" % v if (ii == i and jj == j) else v 
                        for jj, v in enumerate(row)] for ii, row in enumerate(m)])


def main():
    _A = [ # 10x10
        [8,6,4,4,2,2,4,0,4,0,94],
        [10,2,2,2,2,2,2,2,2,2,80],
        [0,2,0,0,0,0,0,0,0,0,8],
        [0,0,2,1,0,0,0,0,0,0,9],
        [0,0,1,1,1,0,1,1,1,1,19],
        [1,1,1,1,1,1,1,1,1,1,28],
        [0,0,0,0,2,0,0,0,0,0,2],
        [0,0,0,0,0,0,1,0,0,0,1],
        [0,0,0,0,0,0,0,0,1,0,2],
        [0,0,0,0,0,0,0,0,0,3,9],
    ]

    # _A = [ # 3x3
    #     [1,1,1,12],
    #     [2,4,4,38],
    #     [0,1,2,10],
    # ]
    # result before mods
    # X: [38, -7.0, 3.0]

    A = _A.copy()
    h = 0 # pivot row
    k = 0 # pivot col
    m = len(A)
    print("m: %s" % m)
    n = len(A[0])
    print("n: %s" % n)

    while h < m and k < n:
        # find the k-th pivot
        i_max = argmax(A, h, m, k)
        if A[i_max][k] == 0:
            # no pivot
            k = k + 1
        else:
            A = swap_rows(A, h, i_max)
            w = A[h][k]
            print("w: %s" % w)
            if not w in [0, 1]: # change pivot to 1
                A[h] = [v/w for v in A[h]]
            
            # get multiplier a for equation: a * A_ij - b * A_hj
            a = A[h][k]
            
            # whole range of m except for where i is equal to h to 
            # reduce each row above and below the pivot row to
            # achieve reduced echelon form.
            # use range [h + 1: m] to achieve only echelon form.
            # for i in range(h + 1, m): # start at row after pivot row
            for i in range(m): # start at row after pivot row
                if i == h:
                    # skip pivot row
                    continue
                # do for all remaining elements in current row
                b = A[i][k]
                A[i][k] = 0.0
                for j in range(k + 1, n): # crawls through current row to update elements a * A_ij - b * A_hj
                    A[i][j] = a * A[i][j] - b * A[h][j]
            # increase pivot row and column
            h = h + 1
            k = k + 1
    
    print("result:\n%s\n" % format_mat(A))
    print("X: %s" % [x[-1] for x in A])
    V = [r.pop(-1) for r in _A]
    print("X (expected): %s" % np.linalg.solve(_A, V))


if __name__ == "__main__":
    main()