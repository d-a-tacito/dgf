import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # или 'Qt5Agg', если предпочтительнее
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List
from datetime import datetime

def powell_discrete(
        f: Callable[[np.ndarray], float],
        x0: np.ndarray,
        tol: float = 1e-3,
        max_iter: int = 1000,
        initial_step: float = 1.0,
        step_reduce_factor: float = 0.5,
        line_search_tol: float = 1e-7
) -> Tuple[np.ndarray, float, pd.DataFrame, List[np.ndarray]]:
    """
    Обобщённый метод (вариант метода Пауэлла) с дискретным шагом для поиска минимума функции f.
    Позволяет работать в произвольном числе измерений.

    Параметры
    ---------
    f : Callable[[np.ndarray], float]
        Целевая функция.
    x0 : np.ndarray
        Начальная точка (размерность определяет число переменных).
    tol : float
        Допуск (точность) остановки.
    max_iter : int
        Максимальное число итераций.
    initial_step : float
        Начальный шаг для линейного поиска.
    step_reduce_factor : float
        Множитель уменьшения шага.
    line_search_tol : float
        Порог прекращения дискретного поиска по направлению.

    Возвращает
    ----------
    (x_min, f_min, history, trajectory) :
        x_min : np.ndarray - найденная точка минимума;
        f_min : float - значение функции в найденной точке;
        history : pd.DataFrame - история итераций;
        trajectory : List[np.ndarray] - траектория точек.
    """
    records = []         # для истории итераций
    trajectory = [x0.copy()]  # сохраняем начальную точку
    x_current = x0.copy()
    f_current = f(x_current)

    n = len(x0)
    # Изначальный набор ортонормированных векторов (стандартный базис)
    directions = [np.eye(n)[i] for i in range(n)]

    for k in range(max_iter):
        x_old = x_current.copy()
        f_old = f_current

        # Проходим по всем направлениям базиса
        for i, d in enumerate(directions):
            print(f"Iteration {k+1}, Step {i+1}:")
            print(f"  x_before = {x_current}, f_before = {f_current}, direction = {d}")
            x_tmp = line_search_discrete(f, x_current, d,
                                         initial_step, step_reduce_factor, line_search_tol)
            f_tmp = f(x_tmp)
            print(f"  x_after  = {x_tmp}, f_after = {f_tmp}")
            records.append({
                'iteration': k+1,
                'step': f"step{i+1}",
                'x_before': tuple(float(val) for val in x_current),
                'f_before': f_current,
                'direction': tuple(float(val) for val in d),
                'x_after': tuple(float(val) for val in x_tmp),
                'f_after': f_tmp
            })
            x_current = x_tmp
            f_current = f_tmp

        # Вычисляем общее изменение за итерацию
        p = x_current - x_old
        norm_p = np.linalg.norm(p)
        if norm_p < 1e-14:
            print("Норма изменения точки слишком мала. Прерывание алгоритма.")
            break
        d_new = p / norm_p  # новое направление
        print(f"Iteration {k+1}, Extra Step:")
        print(f"  Используем x_old = {x_old}, f_before = {f_old}, direction = {d_new}")
        x_tmp = line_search_discrete(f, x_old, d_new,
                                     initial_step, step_reduce_factor, line_search_tol)
        f_tmp = f(x_tmp)
        print(f"  x_after  = {x_tmp}, f_after = {f_tmp}")
        records.append({
            'iteration': k+1,
            'step': "extra_step",
            'x_before': tuple(float(val) for val in x_old),
            'f_before': f_old,
            'direction': tuple(float(val) for val in d_new),
            'x_after': tuple(float(val) for val in x_tmp),
            'f_after': f_tmp
        })
        x_current = x_tmp
        f_current = f_tmp
        trajectory.append(x_current.copy())

        if np.linalg.norm(x_current - x_old) < tol or abs(f_current - f_old) < tol:
            print(f"Остановка: изменение точки или значения функции меньше tol = {tol}")
            break

        # Обновляем набор направлений: удаляем первый и добавляем новое
        directions = directions[1:] + [d_new]

    df_history = pd.DataFrame(records)
    return x_current, f_current, df_history, trajectory

def line_search_discrete(
        f: Callable[[np.ndarray], float],
        x: np.ndarray,
        d: np.ndarray,
        initial_step: float,
        step_reduce_factor: float,
        tol: float
) -> np.ndarray:
    """
    Дискретный поиск минимума вдоль направления d.
    """
    x_best = x.copy()
    step = initial_step

    improved = True
    while improved:
        improved = False

        x_forward = x_best + step * d
        f_forward = f(x_forward)

        x_backward = x_best - step * d
        f_backward = f(x_backward)

        f_best = f(x_best)

        if f_forward < f_best or f_backward < f_best:
            if f_forward < f_backward:
                x_best = x_forward
            else:
                x_best = x_backward
            improved = True
        else:
            step *= step_reduce_factor
            if step < tol:
                break

    return x_best

def f1(x: np.ndarray) -> float:
    """ F1(x1, x2) = 9*(x1)^2 + 16*(x2)^2 - 90*(x1) - 128*(x2) """
    x1, x2 = x
    return 9*(x1**2) + 16*(x2**2) - 90*x1 - 128*x2

def f2(x: np.ndarray) -> float:
    """ F2(x1, x2, x3) = (x1)^2 + 2*(x1)*(x2) + 2*(x2)^2 + (x3)^2 - (x2)*(x3) + (x1) + 3*(x2) - (x3) """
    x1, x2, x3 = x
    return (x1**2) + 2*x1*x2 + 2*(x2**2) + (x3**2) - x2*x3 + x1 + 3*x2 - x3

def plot_iterations(f: Callable[[np.ndarray], float],
                    trajectory: List[np.ndarray],
                    tol: float,
                    title: str):
    """
    Строит график с линиями уровня для 2D-функций или 3D-траекторию для 3D-функций.
    """
    dim = len(trajectory[0])
    traj_arr = np.array(trajectory)
    if dim == 2:
        margin = 1.0
        x1_min = min(np.min(traj_arr[:, 0]), -2) - margin
        x1_max = max(np.max(traj_arr[:, 0]), 10) + margin
        x2_min = min(np.min(traj_arr[:, 1]), -2) - margin
        x2_max = max(np.max(traj_arr[:, 1]), 10) + margin

        x1_vals = np.linspace(x1_min, x1_max, 400)
        x2_vals = np.linspace(x2_min, x2_max, 400)
        X1, X2 = np.meshgrid(x1_vals, x2_vals)
        # Векторизуем функцию для применения к массивам
        f_vec = np.vectorize(lambda a, b: f(np.array([a, b])))
        Z = f_vec(X1, X2)

        plt.figure(figsize=(8, 6))
        contour = plt.contour(X1, X2, Z, levels=20, cmap='viridis')
        plt.clabel(contour, inline=True, fontsize=8)
        plt.plot(traj_arr[:, 0], traj_arr[:, 1], 'ro-', label="Траектория")
        for i in range(len(traj_arr) - 1):
            start = traj_arr[i]
            end = traj_arr[i+1]
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            plt.arrow(start[0], start[1], dx, dy,
                      head_width=0.2, head_length=0.2, fc='k', ec='k')
        plt.title(f"{title} (tol={tol})")
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.legend()
        plt.grid(True)
        plt.show()
    elif dim == 3:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(traj_arr[:, 0], traj_arr[:, 1], traj_arr[:, 2], 'ro-', label="Траектория")
        ax.set_title(f"{title} (tol={tol})")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("x3")
        plt.legend()
        plt.show()
    else:
        print("Отображение траектории не поддерживается для размерности:", dim)

def main():
    # Обработка для первой функции (2D)
    try:
        tol_f1 = float(input("Введите epsilon: "))
    except ValueError:
        print("Неверный ввод. Используем значение по умолчанию tol=1e-3.")
        tol_f1 = 1e-3

    try:
        initial_point_str = input("Введите начальную точку для первой функции (через запятую, например, 0,0): ")
        initial_point_f1 = np.array([float(x.strip()) for x in initial_point_str.split(',')])
        if len(initial_point_f1) != 2:
            print("Неверное число координат для F1. Используем значение по умолчанию [0.0, 0.0].")
            initial_point_f1 = np.array([0.0, 0.0])
    except Exception:
        print("Ошибка при вводе начальной точки. Используем значение по умолчанию [0.0, 0.0].")
        initial_point_f1 = np.array([0.0, 0.0])

    print(f"\n=== Результаты для F1 при tol={tol_f1} ===")
    x_min_f1, val_f1, hist_f1, traj_f1 = powell_discrete(f1, initial_point_f1, tol=tol_f1)
    print(f"Найденная точка минимума F1: {x_min_f1}")
    print(f"Значение F1 в найденной точке: {val_f1}")
    print("Число записей в истории:", len(hist_f1))
    print("История итераций:")
    print(hist_f1.to_string(index=False))
    plot_iterations(f1, traj_f1, tol_f1, "Линии уровня F1 и траектория")

    # Обработка для второй функции (3D)
    try:
        tol_f2 = float(input("Введите epsilon: "))
    except ValueError:
        print("Неверный ввод. Используем значение по умолчанию tol=1e-3.")
        tol_f2 = 1e-3

    try:
        initial_point_str = input("Введите начальную точку для второй функции (через запятую, например, 3,5,1): ")
        initial_point_f2 = np.array([float(x.strip()) for x in initial_point_str.split(',')])
        if len(initial_point_f2) != 3:
            print("Неверное число координат для F2. Используем значение по умолчанию [3.0, 5.0, 1.0].")
            initial_point_f2 = np.array([3.0, 5.0, 1.0])
    except Exception:
        print("Ошибка при вводе начальной точки. Используем значение по умолчанию [3.0, 5.0, 1.0].")
        initial_point_f2 = np.array([3.0, 5.0, 1.0])

    print(f"\n=== Результаты для F2 при tol={tol_f2} ===")
    x_min_f2, val_f2, hist_f2, traj_f2 = powell_discrete(f2, initial_point_f2, tol=tol_f2)
    print(f"Найденная точка минимума F2: {x_min_f2}")
    print(f"Значение F2 в найденной точке: {val_f2}")
    print("Число записей в истории:", len(hist_f2))
    print("История итераций:")
    print(hist_f2.to_string(index=False))
    plot_iterations(f2, traj_f2, tol_f2, "Траектория оптимизации F2 (3D)")

    # Экспорт истории итераций в Excel-файл с двумя листами (F1 и F2)
    filename = datetime.now().strftime("%H-%M-%Y-%m-%d-rosenbrock.xlsx")
    with pd.ExcelWriter(filename) as writer:
        hist_f1.to_excel(writer, sheet_name="F1", index=False)
        hist_f2.to_excel(writer, sheet_name="F2", index=False)
    print("Истории итераций для F1 и F2 сохранены в файл:", filename)

if __name__ == "__main__":
    main()
