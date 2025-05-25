# -import numpy as np

def solve_system_cramer(A, B):
    """
    Решает систему линейных уравнений Ax = B методом Крамера.
    A — матрица коэффициентов (комплексные числа),
    B — вектор свободных членов (комплексные числа).
    Возвращает вектор решений или None, если система не имеет решения.
    """
    det_A = np.linalg.det(A)
    if abs(det_A) < 1e-12:
        print("Система не имеет единственного решения (детерминант равен нулю).")
        return None

    n = A.shape[0]
    solutions = []

    for i in range(n):
        Ai = np.copy(A)
        Ai[:, i] = B
        det_Ai = np.linalg.det(Ai)
        xi = det_Ai / det_A
        solutions.append(xi)

    return solutions

# Пример системы:
# 2 + 3i * x + (1 - i) * y = 4 + i
# (1 + 2i) * x + 3 * y = 5 - i

A = np.array([[2 + 3j, 1 - 1j],
              [1 + 2j, 3]])

B = np.array([4 + 1j, 5 - 1j])

solution = solve_system_cramer(A, B)

if solution is not None:
    for idx, val in enumerate(solution):
        print(f"x{idx+1} = {val}")
