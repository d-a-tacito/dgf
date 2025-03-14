import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # или 'Qt5Agg', если предпочтительнее
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List

def rosenbrock_discrete(
        f: Callable[[np.ndarray], float],
        x0: np.ndarray,
        tol: float = 1e-3,
        max_iter: int = 1000,
        initial_step: float = 1.0,
        step_reduce_factor: float = 0.5,
        line_search_tol: float = 1e-7
) -> Tuple[np.ndarray, float, pd.DataFrame, List[np.ndarray]]:
    """
    Реализует метод Розенброка (с дискретным шагом) для нахождения минимума функции f.

    Параметры
    ---------
    f : Callable[[np.ndarray], float]
        Целевая функция, принимающая вектор x (np.ndarray) и возвращающая число.
    x0 : np.ndarray
        Начальная точка (например, np.array([0.0, 0.0])).
    tol : float
        Допуск (точность), при котором останавливается метод.
    max_iter : int
        Максимальное число итераций основного цикла.
    initial_step : float
        Начальный шаг для линейного поиска.
    step_reduce_factor : float
        Множитель уменьшения шага при отсутствии улучшения (0 < factor < 1).
    line_search_tol : float
        Порог, при котором прекращаем дискретный поиск по направлению.

    Возвращает
    ----------
    (x_min, f_min, history, trajectory) :
        x_min : np.ndarray
            Найденная точка минимума.
        f_min : float
            Значение функции в найденной точке.
        history : pd.DataFrame
            История итераций с подробной информацией.
        trajectory : List[np.ndarray]
            Список конечных точек каждой основной итерации (начальная точка включается).
    """
    # Изначальные ортонормированные направления в 2D: e1 = (1,0), e2 = (0,1)
    e1 = np.array([1.0, 0.0])
    e2 = np.array([0.0, 1.0])

    records = []  # для истории итераций
    trajectory = [x0.copy()]  # для хранения конечных точек каждой итерации

    x_current = x0.copy()
    f_current = f(x_current)

    for k in range(max_iter):
        # Сохраняем старую точку для сравнения
        x_old = x_current.copy()
        f_old = f_current

        # --- Шаг 1: Поиск вдоль направления e1 ---
        print(f"Iteration {k+1}, Step 1:")
        print(f"  x_before = {x_current}, f_before = {f_current}, direction = {e1}")
        x_tmp = line_search_discrete(f, x_current, e1,
                                     initial_step, step_reduce_factor, line_search_tol)
        f_tmp = f(x_tmp)
        print(f"  x_after  = {x_tmp}, f_after = {f_tmp}")
        records.append({
            'iteration': k + 1,
            'step': 'step1',
            'x_before': tuple(x_current),
            'f_before': f_current,
            'direction': tuple(e1),
            'x_after': tuple(x_tmp),
            'f_after': f_tmp
        })
        x_current = x_tmp
        f_current = f_tmp

        # --- Шаг 2: Поиск вдоль направления e2 ---
        print(f"Iteration {k+1}, Step 2:")
        print(f"  x_before = {x_current}, f_before = {f_current}, direction = {e2}")
        x_tmp = line_search_discrete(f, x_current, e2,
                                     initial_step, step_reduce_factor, line_search_tol)
        f_tmp = f(x_tmp)
        print(f"  x_after  = {x_tmp}, f_after = {f_tmp}")
        records.append({
            'iteration': k + 1,
            'step': 'step2',
            'x_before': tuple(x_current),
            'f_before': f_current,
            'direction': tuple(e2),
            'x_after': tuple(x_tmp),
            'f_after': f_tmp
        })
        x_current = x_tmp
        f_current = f_tmp

        # Сохраняем точку для формирования направления
        x_current_prev = x_current.copy()
        f_current_prev = f_current

        # --- Шаг 3: Поворот и поиск вдоль нового направления ---
        p = x_current - x_old
        norm_p = np.linalg.norm(p)
        if norm_p < 1e-14:
            print("  Норма изменения точки слишком мала. Прерывание алгоритма.")
            break

        e1_new = p / norm_p  # новое направление
        dot_e2_e1new = e2.dot(e1_new)
        e2_new = e2 - dot_e2_e1new * e1_new
        norm_e2_new = np.linalg.norm(e2_new)
        if norm_e2_new < 1e-14:
            e2_new = np.array([0.0, 1.0])
        else:
            e2_new /= norm_e2_new

        e1 = e1_new
        e2 = e2_new

        print(f"Iteration {k+1}, Step 3:")
        print(f"  Используем x_old = {x_old}, f_before = {f_old}")
        print(f"  Новое направление e1 = {e1}")
        x_tmp = line_search_discrete(f, x_old, e1,
                                     initial_step, step_reduce_factor, line_search_tol)
        f_tmp = f(x_tmp)
        print(f"  x_after  = {x_tmp}, f_after = {f_tmp}")
        records.append({
            'iteration': k + 1,
            'step': 'step3',
            'x_before': tuple(x_old),
            'f_before': f_old,
            'direction': tuple(e1),
            'x_after': tuple(x_tmp),
            'f_after': f_tmp
        })
        x_current = x_tmp
        f_current = f_tmp

        # Сохраняем конечную точку основной итерации
        trajectory.append(x_current.copy())

        # Проверка критерия остановки
        if np.linalg.norm(x_current - x_old) < tol:
            print(f"  Остановка: изменение точки меньше tol = {tol}")
            break
        if abs(f_current - f_old) < tol:
            print(f"  Остановка: изменение значения функции меньше tol = {tol}")
            break

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
    1) Начинаем с шага initial_step.
    2) Проверяем f(x + step*d) и f(x - step*d).
    3) Если улучшение есть – двигаемся в ту сторону, пока улучшается.
    4) Если улучшения нет – уменьшаем шаг.
    5) Останавливаемся, если шаг стал меньше tol.
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
    """ F1(x1, x2) = 9x1^2 + 16x2^2 - 90x1 - 128x2 """
    x1, x2 = x
    return 9*(x1**2) + 16*(x2**2) - 90*x1 - 128*x2


def f2(x: np.ndarray) -> float:
    """ F2(x1, x2) = 2x1^2 + 2x2^2 + 2x1x2 + 3x1 + x2 """
    x1, x2 = x
    return 2*(x1**2) + 2*(x2**2) + 2*x1*x2 + 3*x1 + x2


def plot_iterations_F1(trajectory: List[np.ndarray], tol: float):
    """
    Строит график с линиями уровня F1 и траекторией, где
    каждая точка отмечена, а стрелками показано направление движения.
    """
    traj_arr = np.array(trajectory)
    # Задаём диапазон для построения графика с небольшим запасом
    margin = 1.0
    x1_min = min(np.min(traj_arr[:, 0]), -2) - margin
    x1_max = max(np.max(traj_arr[:, 0]), 10) + margin
    x2_min = min(np.min(traj_arr[:, 1]), -2) - margin
    x2_max = max(np.max(traj_arr[:, 1]), 10) + margin

    x1_vals = np.linspace(x1_min, x1_max, 400)
    x2_vals = np.linspace(x2_min, x2_max, 400)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)
    # Вычисляем значения функции F1 на сетке
    Z = 9 * X1**2 + 16 * X2**2 - 90 * X1 - 128 * X2

    plt.figure(figsize=(8, 6))
    contour = plt.contour(X1, X2, Z, levels=20, cmap='viridis')
    plt.clabel(contour, inline=True, fontsize=8)
    # Отмечаем точки траектории и соединяем их линиями
    plt.plot(traj_arr[:, 0], traj_arr[:, 1], 'ro-', label="Траектория")
    # Добавляем стрелки для визуализации направлений
    for i in range(len(traj_arr) - 1):
        start = traj_arr[i]
        end = traj_arr[i+1]
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        plt.arrow(start[0], start[1], dx, dy,
                  head_width=0.2, head_length=0.2, fc='k', ec='k')
    plt.title(f"Линии уровня F1 и траектория (tol={tol})")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    tolerances = [1e-1, 1e-2, 1e-3]
    initial_point = np.array([0.0, 0.0])

    for eps in tolerances:
        print(f"\n=== Результаты для F1 при tol={eps} ===")
        x_min_f1, val_f1, hist_f1, traj_f1 = rosenbrock_discrete(f1, initial_point, tol=eps)
        print(f"Найденная точка минимума F1: {x_min_f1}")
        print(f"Значение F1 в найденной точке: {val_f1}")
        print("Число записей в истории:", len(hist_f1))
        print("История итераций:")
        print(hist_f1.to_string(index=False))
        # Строим график для F1 с линиями уровня и траекторией
        plot_iterations_F1(traj_f1, tol=eps)

        print(f"\n=== Результаты для F2 при tol={eps} ===")
        x_min_f2, val_f2, hist_f2, _ = rosenbrock_discrete(f2, initial_point, tol=eps)
        print(f"Найденная точка минимума F2: {x_min_f2}")
        print(f"Значение F2 в найденной точке: {val_f2}")
        print("Число записей в истории:", len(hist_f2))
        print("История итераций:")
        print(hist_f2.to_string(index=False))
        # Для F2 график не строится, т.к. требовались линии уровня F1.

if __name__ == "__main__":
    main()
