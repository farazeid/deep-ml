import numpy as np

def matrixmul(
    a: list[list[int | float]],
    b: list[list[int | float]]
) -> list[list[int | float]] | int:
    a, b = np.array(a), np.array(b)

    if a.shape[1] != b.shape[0]:
        return -1
    return np.matmul(a, b)