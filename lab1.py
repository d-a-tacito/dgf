import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 1-я функция условия.
def F1(x):
    """
    Функция F1: f(x) = 3*x - x**3 - 1.
    """
    return 3 * x - x**3 - 1

# 2-я функция условия.
def F2(x):
    """
    Функция F2: f(x) = (4 - x**2) / (x * (x**2 + 3)).
    """
    return (4 - x**2) / (x * (x**2 + 3))

def dichotomy_method(f, a, b, epsilon, tol, max_iterations=1000):
    """
    Поиск минимума функции f на отрезке [a, b] методом дихотомии.

    Параметры:
        f : функция для минимизации.
        a, b : границы интервала.
        epsilon : небольшое число для разделения точек (используется в методе).
        tol : требуемая точность (условие остановки по длине интервала).
        max_iterations : максимальное число итераций.

    Возвращает:
        (x_opt, f(x_opt), iterations)
    """
    iterations = 0
    while (b - a) > tol and iterations < max_iterations:
        iterations += 1
        mid = (a + b) / 2
        x1 = mid - epsilon
        x2 = mid + epsilon
        if f(x1) < f(x2):
            b = x2
        else:
            a = x1
    x_opt = (a + b) / 2
    return x_opt, f(x_opt), iterations

def golden_section_method(f, a, b, epsilon, tol, max_iterations=1000):
    """
    Поиск минимума функции f на отрезке [a, b] методом золотого сечения.
    Параметр epsilon здесь не используется (оставлен для согласованности интерфейса).

    Параметры:
        f : функция для минимизации.
        a, b : границы интервала.
        epsilon : не используется.
        tol : требуемая точность (условие остановки по длине интервала).
        max_iterations : максимальное число итераций.

    Возвращает:
        (x_opt, f(x_opt), iterations)
    """
    iterations = 0
    phi = (1 + np.sqrt(5)) / 2  # золотое сечение
    x1 = b - (b - a) / phi
    x2 = a + (b - a) / phi
    while abs(b - a) > tol and iterations < max_iterations:
        if f(x1) <= f(x2):
            b = x2
            x2 = x1
            x1 = b - (b - a) / phi
        else:
            a = x1
            x1 = x2
            x2 = a + (b - a) / phi
        iterations += 1
    x_opt = (a + b) / 2
    return x_opt, f(x_opt), iterations

def fibonacci_method(f, a, b, epsilon, tol, max_iterations=1000):
    """
    Поиск минимума функции f на отрезке [a, b] методом Фибоначчи.
    Параметр epsilon здесь не используется (оставлен для согласованности интерфейса).

    Параметры:
        f : функция для минимизации.
        a, b : границы интервала.
        epsilon : не используется.
        tol : требуемая точность (определяет число итераций через числа Фибоначчи).
        max_iterations : максимальное число итераций (не используется, определяется tol).

    Возвращает:
        (x_opt, f(x_opt), iterations)
    """
    # Вычисляем числа Фибоначчи, пока последнее число не станет больше (b - a) / tol
    fib = [1, 1]
    while fib[-1] < (b - a) / tol:
        fib.append(fib[-1] + fib[-2])
    n = len(fib) - 1  # число итераций, необходимое для достижения точности

    # Начальное разбиение
    x1 = a + fib[n-2] / fib[n] * (b - a)
    x2 = a + fib[n-1] / fib[n] * (b - a)
    iterations = 0
    for k in range(1, n-1):
        iterations += 1
        if f(x1) > f(x2):
            a = x1
            x1 = x2
            x2 = a + fib[n - k - 1] / fib[n - k] * (b - a)
        else:
            b = x2
            x2 = x1
            x1 = a + fib[n - k - 2] / fib[n - k] * (b - a)
    x_opt = (a + b) / 2
    return x_opt, f(x_opt), iterations

def find_extremum(f, intervals, method, epsilon, tol):
    """
    Поиск минимума функции f на заданных интервалах с использованием указанного метода.

    Параметры:
        f : функция для минимизации.
        intervals : список кортежей (a, b) – интервалов поиска.
        method : функция-метод оптимизации.
        epsilon : параметр для метода (или игнорируется, если не используется).
        tol : требуемая точность.

    Возвращает:
        Список кортежей (x_opt, f(x_opt), iterations) для каждого интервала.
    """
    results = []
    for interval in intervals:
        a, b = interval
        x_opt, f_opt, iterations = method(f, a, b, epsilon, tol)
        results.append((x_opt, f_opt, iterations))
    return results

def plot_function(f, intervals, x_opt_points, f_opt_points):
    """
    Построение графика функции f, отображение интервалов и найденных оптимальных точек.

    Параметры:
        f : функция для построения.
        intervals : список интервалов в виде кортежей (a, b).
        x_opt_points : список найденных оптимальных x.
        f_opt_points : список соответствующих значений f(x).
    """
    # Определяем диапазон построения по заданным интервалам
    x_min = min(interval[0] for interval in intervals)
    x_max = max(interval[1] for interval in intervals)
    x = np.linspace(x_min, x_max, 1000)
    y = f(x)

    plt.figure(figsize=(8, 6))
    plt.plot(x, y, label="Функция")
    # Рисуем линии для границ интервалов, подписывая их только один раз
    for i, (a, b) in enumerate(intervals):
        if i == 0:
            plt.axvline(x=a, color='r', linestyle='--', label="Начало интервала")
            plt.axvline(x=b, color='g', linestyle='--', label="Конец интервала")
        else:
            plt.axvline(x=a, color='r', linestyle='--')
            plt.axvline(x=b, color='g', linestyle='--')
    # Отображаем найденные оптимальные точки
    plt.scatter(x_opt_points, f_opt_points, color='b', label="Оптимальная точка")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Оптимизация функции")
    plt.legend()
    plt.grid(True)
    plt.show()

# Заданные интервалы для F1 и F2.
intervals_F1 = [(-3, 0), (0.8, 5), (-10, 0.5)]
intervals_F2 = [(-5, 0), (0, 10), (-5, 5)]

# Параметры: epsilon используется в методе дихотомии, tol – требуемая точность.
epsilon_values = [0.1, 0.01, 0.001]
tol_values = [0.1, 0.01]

# Выполнение вычислений для F1 методом дихотомии.
print("Метод дихотомии для F1:")
for epsilon in epsilon_values:
    for tol in tol_values:
        print(f"F1, epsilon = {epsilon}, tol = {tol}")
        results = find_extremum(F1, intervals_F1, dichotomy_method, epsilon, tol)
        for i, (x_opt, f_opt, iterations) in enumerate(results):
            print(f"Интервал {intervals_F1[i]}: x_opt = {x_opt:.5f}, f(x_opt) = {f_opt:.5f}, итераций = {iterations}")
        print("-" * 40)

# Выполнение вычислений для F2 методом золотого сечения.
print("\nМетод золотого сечения для F2:")
for epsilon in epsilon_values:  # epsilon не используется, но передаётся для единообразия
    for tol in tol_values:
        print(f"F2, epsilon = {epsilon}, tol = {tol}")
        results = find_extremum(F2, intervals_F2, golden_section_method, epsilon, tol)
        for i, (x_opt, f_opt, iterations) in enumerate(results):
            print(f"Интервал {intervals_F2[i]}: x_opt = {x_opt:.5f}, f(x_opt) = {f_opt:.5f}, итераций = {iterations}")
        print("-" * 40)

# Демонстрация метода Фибоначчи для F1.
print("\nМетод Фибоначчи для F1:")
for epsilon in epsilon_values:  # epsilon не используется, но передаётся для единообразия
    for tol in tol_values:
        print(f"F1, epsilon = {epsilon}, tol = {tol}")
        results = find_extremum(F1, intervals_F1, fibonacci_method, epsilon, tol)
        for i, (x_opt, f_opt, iterations) in enumerate(results):
            print(f"Интервал {intervals_F1[i]}: x_opt = {x_opt:.5f}, f(x_opt) = {f_opt:.5f}, итераций = {iterations}")
        print("-" * 40)

# Пример построения графика для F1 с использованием метода дихотомии на первом интервале.
x_opt, f_opt, iters = dichotomy_method(F1, intervals_F1[0][0], intervals_F1[0][1], 0.1, 0.1)
print(f"\nПример для F1 (метод дихотомии) на интервале {intervals_F1[0]}:")
print(f"Оптимальное x: {x_opt:.5f}")
print(f"f(x_opt): {f_opt:.5f}")
print(f"Количество итераций: {iters}")

plot_function(F1, [intervals_F1[0]], [x_opt], [f_opt])
