# """
# https://www.codesansar.com/numerical-methods/gauss-elimination-method-python-program.htm
# """
# # Importing NumPy Library
# import numpy as np
# import sys

# # Reading number of unknowns
# n = int(input('Enter number of unknowns: '))

# # Making numpy array of n x n+1 size and initializing
# # to zero for storing augmented matrix
# a = np.zeros((n, n+1))

# # Making numpy array of n size and initializing
# # to zero for storing solution vector
# x = np.zeros(n)

# # Reading augmented matrix coefficients
# print('Enter Augmented Matrix Coefficients:')
# for i in range(n):
#     for j in range(n+1):
#         a[i][j] = int(input('a['+str(i)+'][' + str(j)+']='))


# print("%s = a" % a)

# # a1 = np.array(a)
# # print(a1)
# # av = np.array([[v] for v in np.delete(a1, a1[-1])])
# # print(av)

# # Applying Gauss Elimination
# for i in range(n):
#     if a[i][i] == 0.0:
#         sys.exit('Divide by zero detected!')

#     for j in range(i+1, n):
#         ratio = a[j][i]/a[i][i]

#         for k in range(n+1):
#             a[j][k] = a[j][k] - ratio * a[i][k]

# # Back Substitution
# x[n-1] = a[n-1][n]/a[n-1][n-1]

# for i in range(n-2, -1, -1):
#     x[i] = a[i][n]

#     for j in range(i+1, n):
#         x[i] = x[i] - a[i][j]*x[j]

#     x[i] = x[i]/a[i][i]

# # Displaying solution
# print('\nRequired solution is: ')
# for i in range(n):
#     print('X%d = %0.2f' % (i, x[i]), end='\t')

# # print(a1.linalg.solve(a,b))




# import numpy as np
 
# a = np.array([[1, 4, 3],[1, 2, 9],[1, 6, 6]])
# b = np.array([[1],[1],[1]])
 
# x = np.linalg.solve(a, b)
 
# print(x)





from typing import List


def gaussian_elimination(mat):
    for i, row in enumerate(mat):
        for c in row[:i]:
            if c == 0:
                break


def add_row(r1: List[float], r2: List[float]):
    return[r1[i] + r2[i] for i in range(len(r1))]


def mul_row(row: List[float], target: float, index: int) -> List[float]:
    coef = target / row[index]
    return [row[i] * coef for i in range(len(row))]

def swp_row(mat, i, j):
    for idx in range(mat[i]):
        mat[i] += mat[j]
        mat[j] = mat[i] - mat[j]
        mat[i] -= mat[j]

"""

    Interchanging two rows

    Multiplying a row by a constant (any constant which is not zero)

    Adding a row to another row

"""