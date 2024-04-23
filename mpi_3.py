from mpi4py import MPI
import numpy as np

def f(x, y):
    return np.sin(x) * np.cos(y)

def calculate_derivative(data, h=0.01):
    # Расчет производной по x используя центральные разности
    derivative = np.zeros_like(data)
    rows, cols = data.shape
    for i in range(1, rows-1):
        for j in range(cols):
            derivative[i][j] = (data[i+1][j] - data[i-1][j]) / (2 * h)
    return derivative

# Настройка MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
start_time = MPI.Wtime()
# Параметры сетки
N_arr = [10, 100, 1000, 10000]  # размер сетки, NxN
for N in N_arr:
    x = np.linspace(0, 2*np.pi, N)
    y = np.linspace(0, 2*np.pi, N)
    X, Y = np.meshgrid(x, y)

    # Разделение данных между процессами
    rows_per_proc = N // size
    extra = N % size

    start = rank * rows_per_proc + min(rank, extra)
    end = start + rows_per_proc + (1 if rank < extra else 0)

    # Локальная часть данных
    local_X = X[start:end, :]
    local_Y = Y[start:end, :]
    local_data = f(local_X, local_Y)

    # Расчет локальной производной
    local_derivative = calculate_derivative(local_data)

    # Сбор данных
    full_derivative = None
    if rank == 0:
        full_derivative = np.zeros_like(X)

    comm.Gather(local_derivative, full_derivative, root=0)

    end_time = MPI.Wtime()
    # Вывод результата на процессе 0
    if rank == 0:
        print(f"Size of mesh {N}")
        print(f"Execution time {end_time - start_time}\n")
