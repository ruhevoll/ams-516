# (c) Jacob White, 2025
# This MATLAB-style script solves the model problem
# min{c'x + 0.5*x'*C*x | Ax = b}
# by solving H*[x0, u] = [-c, b]
# for H = [C, A* // A, 0]
import numpy as np

# Dimensions
n = 2
m = 1

# Parameters
c = np.array([-4, -4])
C = np.array([[2, 0], [0, 2]])
A = np.array([[1, 1]])
b = np.array([2])

# Construct H and rhs
H = np.block([[C, A.T], [A, np.zeros((m, m))]])
rhs = np.concatenate([-c, b])

# Solve H @ y = rhs
y = np.linalg.solve(H, rhs)
x = y[0:n]
u = y[n:n+m]

print(f'Optimal solution x = {x}')
print(f'Multipliers for Ax = b: {u}')