import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # или 'Qt5Agg', если предпочтительнее
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List
from datetime import datetime

def rosenbrock_discrete(
        f: Callable[[np.ndarray], float],
        x0: np.ndarray,
        tol: float = 1e-3,
        max_iter: int = 1000,
        initial_step: float = 1.0,
        step_reduce_factor: float = 0.5
) -> Tuple[np.ndarray, float, pd.DataFrame, List[np.ndarray]]:
    """
    Метод Розенброка с дискретным шагом для поиска минимума функции f.

    В этой версии используются только два направления – (0,1,0,...,0) и (1,0,0,...,0).
    Если ни одно из направлений не приводит к уменьшению значения функции, то
    производится обновление направлений с использованием процедуры Грама–Шмидта:
      - Вычисляется приближённый градиент по первым двум координатам (через центральные разности).
      - Новый первый вектор направления определяется как нормализованное отрицание градиента.
      - Второй вектор получается путём ортогонализации кандидата (по умолчанию (1,0,...,0))
        относительно нового первого вектора.

    Поиск продолжается до тех пор, пока шаг не станет меньше tol или не исчерпается max_iter.
    """
    records = []
    trajectory = [x0.copy()]
    x_current = x0.copy()
    f_current = f(x_current)
    n = len(x0)

    # Инициализируем только два направления.
    # Для n>=2: d1 = (0,1,0,...,0) и d2 = (1,0,0,...,0). Если n==1, то используем [1] и [-1].
    if n >= 2:
        d1 = np.zeros(n)
        d2 = np.zeros(n)
        d1[1] = 1
        d2[0] = 1
    else:
        d1 = np.array([1])
        d2 = np.array([-1])
    directions = [d1, d2]

    iter_num = 0
    step = initial_step
    while step > tol and iter_num < max_iter:
        improved = False
        # Перебор только двух направлений
        for d in directions:
            x_new = x_current + step * d
            f_new = f(x_new)
            records.append({
                'iteration': iter_num,
                'x_current': tuple(x_current),
                'f_current': f_current,
                'direction': tuple(d),
                'step': step,
                'x_new': tuple(x_new),
                'f_new': f_new
            })
            # Если улучшение найдено – переходим в новую точку
            if f_new < f_current:
                x_current = x_new
                f_current = f_new
                trajectory.append(x_current.copy())
                improved = True
                break  # переходим к следующей итерации
        # Если ни одно направление не улучшило значение функции:
        if not improved:
            # Обновляем направления с использованием процедуры Грама–Шмидта
            # (вычисляем приближённый градиент по первым двум координатам)
            subdim = min(n, 2)
            epsilon = 1e-6
            g = np.zeros(subdim)
            for i in range(subdim):
                unit_vec = np.zeros(n)
                unit_vec[i] = 1
                g[i] = (f(x_current + epsilon * unit_vec) - f(x_current - epsilon * unit_vec)) / (2 * epsilon)
            # Формируем новый первый вектор направления как -grad (в первых subdim координатах)
            d1_new = np.zeros(n)
            d1_new[:subdim] = -g
            norm_d1 = np.linalg.norm(d1_new)
            if norm_d1 > 1e-12:
                d1_new = d1_new / norm_d1
            else:
                # Если градиент слишком мал, оставляем старый вектор
                d1_new = directions[0]
            # Выбираем кандидата для второго направления
            candidate = np.zeros(n)
            candidate[0] = 1
            # Если кандидат почти параллелен d1_new, выбираем другой
            if abs(np.dot(candidate, d1_new)) > 0.99:
                candidate = np.zeros(n)
                if n >= 2:
                    candidate[1] = 1
                else:
                    candidate[0] = -1
            # Грамм-Шмидт: d2_new = candidate - (candidate·d1_new)*d1_new
            proj = np.dot(candidate, d1_new) * d1_new
            d2_new = candidate - proj
            norm_d2 = np.linalg.norm(d2_new)
            if norm_d2 > 1e-12:
                d2_new = d2_new / norm_d2
            else:
                d2_new = directions[1]
            directions = [d1_new, d2_new]
            # Уменьшаем шаг, как и раньше
            step *= step_reduce_factor
        iter_num += 1

    df_history = pd.DataFrame(records)
    return x_current, f_current, df_history, trajectory

def f1(x: np.ndarray) -> float:
    """ F1(x1, x2) = 9*x1^2 + 16*x2^2 - 90*x1 - 128*x2 """
    x1, x2 = x
    return 9*(x1**2) + 16*(x2**2) - 90*x1 - 128*x2

def f2(x: np.ndarray) -> float:
    """ F2(x1, x2, x3) = x1^2 + 2*x1*x2 + 2*x2^2 + x3^2 - x2*x3 + x1 + 3*x2 - x3 """
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
        # Векторизуем функцию для построения линий уровня
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

def format_value(val):
    """
    Функция для округления чисел до 4 знаков после запятой и явного преобразования numpy-типов в стандартные типы Python.
    Если значение является кортежем, то производится рекурсивное преобразование его элементов.
    """
    if isinstance(val, (np.floating, float)):
        return float(round(val, 4))
    elif isinstance(val, tuple):
        return tuple(format_value(v) for v in val)
    else:
        return val

def main():
    # Обработка для первой функции (2D)
    try:
        tol_f1 = float(input("Введите epsilon для F1: "))
    except ValueError:
        print("Неверный ввод. Используем значение по умолчанию tol=1e-3.")
        tol_f1 = 1e-3

    try:
        initial_point_str = input("Введите начальную точку для F1 (через запятую, например, 0,0): ")
        initial_point_f1 = np.array([float(x.strip()) for x in initial_point_str.split(',')])
        if len(initial_point_f1) != 2:
            print("Неверное число координат для F1. Используем значение по умолчанию [0.0, 0.0].")
            initial_point_f1 = np.array([0.0, 0.0])
    except Exception:
        print("Ошибка при вводе начальной точки. Используем значение по умолчанию [0.0, 0.0].")
        initial_point_f1 = np.array([0.0, 0.0])

    print(f"\n=== Результаты для F1 при tol={tol_f1} ===")
    x_min_f1, val_f1, hist_f1, traj_f1 = rosenbrock_discrete(f1, initial_point_f1, tol=tol_f1)
    print(f"Найденная точка минимума F1: {x_min_f1}")
    print(f"Значение F1 в найденной точке: {val_f1}")
    print("Число записей в истории:", len(hist_f1))
    print("История итераций:")
    print(hist_f1.to_string(index=False))
    plot_iterations(f1, traj_f1, tol_f1, "Линии уровня F1 и траектория")

    # Обработка для второй функции (3D)
    try:
        tol_f2 = float(input("Введите epsilon для F2: "))
    except ValueError:
        print("Неверный ввод. Используем значение по умолчанию tol=1e-3.")
        tol_f2 = 1e-3

    try:
        initial_point_str = input("Введите начальную точку для F2 (через запятую, например, 3,5,1): ")
        initial_point_f2 = np.array([float(x.strip()) for x in initial_point_str.split(',')])
        if len(initial_point_f2) != 3:
            print("Неверное число координат для F2. Используем значение по умолчанию [3.0, 5.0, 1.0].")
            initial_point_f2 = np.array([3.0, 5.0, 1.0])
    except Exception:
        print("Ошибка при вводе начальной точки. Используем значение по умолчанию [3.0, 5.0, 1.0].")
        initial_point_f2 = np.array([3.0, 5.0, 1.0])

    print(f"\n=== Результаты для F2 при tol={tol_f2} ===")
    x_min_f2, val_f2, hist_f2, traj_f2 = rosenbrock_discrete(f2, initial_point_f2, tol=tol_f2)
    print(f"Найденная точка минимума F2: {x_min_f2}")
    print(f"Значение F2 в найденной точке: {val_f2}")
    print("Число записей в истории:", len(hist_f2))
    print("История итераций:")
    print(hist_f2.to_string(index=False))
    plot_iterations(f2, traj_f2, tol_f2, "Траектория оптимизации F2 (3D)")

    # Экспорт истории итераций в Excel-файл с двумя листами (F1 и F2)
    # Применяем форматирование: округление всех числовых значений до 4 знаков после запятой
    hist_f1_formatted = hist_f1.map(format_value)
    hist_f2_formatted = hist_f2.map(format_value)

    filename = datetime.now().strftime("%H-%M-%Y-%m-%d-rosenbrock.xlsx")
    with pd.ExcelWriter(filename) as writer:
        hist_f1_formatted.to_excel(writer, sheet_name="F1", index=False)
        hist_f2_formatted.to_excel(writer, sheet_name="F2", index=False)
    print("Истории итераций для F1 и F2 сохранены в файл:", filename)

if __name__ == "__main__":
    main()
