import numpy as np
from scipy.linalg import solve_continuous_are

# Given LQR parameters
rho = 0.1
A = np.array([[0, 1], [-2, -3]])
B = np.array([[0], [1]])
G = np.array([[0.1, 0] , [0, 0.1]])
Q = np.array([[1, 0] , [0, 1]])
R = np.array([[1]])
beta = 0.1
A_tilde = A - 0.5*beta*np.eye(A.shape[0])

P = solve_continuous_are(A_tilde, B, Q, R)
print(P)