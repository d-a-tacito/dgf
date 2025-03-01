import numpy as np
import matplotlib.pyplot as plt

# 1я функция условия.
def F1(x):
    return 3 * x - x**3 - 1

# 2я функция условия.
def F2(x):
    return (4 - x**2) / (x * (x**2 + 3))

# Метод дихотомии.
def dichotomy_method(f, a, b, epsilon, l, max_iterations=1000):
    k = 0
    while (b - a) > l and k < max_iterations: # Шаг 1.
        k += 1
        mid = (a + b) / 2
        x1 = mid - epsilon # x1 это та же лямбда.
        x2 = mid + epsilon # x2 это та же мю.
        if f(x1) < f(x2): # Шаг 2.
            b = x2
        else:
            a = x1
    return (a + b) / 2, f((a + b) / 2), k

# Метод золотого сечения.
def golden_section_method(f, a, b, l, max_iterations=1000):
    k = 0
    phi = (1 + np.sqrt(5)) / 2
    x1 = b - (b - a) / phi # x1 это та же лямбда.
    x2 = a + (b - a) / phi # x2 это та же мю.
    while abs(b - a) > l and k < max_iterations: # Шаг 1.
        if f(x1) <= f(x2): # Шаг 3.
            b = x2
            x2 = x1
            x1 = b - (b - a) / phi
        else: # Шаг 2.
            a = x1
            x1 = x2
            x2 = a + (b - a) / phi
        k += 1 # Шаг 4.
    return (a + b) / 2, f((a + b) / 2), k

# Метод Фибоначчи.
# TODO этот метод не совсем соответствует той логике, что у нас в лекциях (нет 5го шага).
def fibonacci_method(f, a, b, l, max_iterations=1000):
    k = 0
    fib = [1, 1]
    while fib[-1] < (b - a) / l:
        fib.append(fib[-1] + fib[-2])
    n = len(fib) - 1
    x1 = a + fib[n-2] / fib[n] * (b - a) # x1 это та же лямбда.
    x2 = a + fib[n-1] / fib[n] * (b - a) # x2 это та же мю.
    while n > 1 and k < max_iterations:
        if f(x1) <= f(x2): # Шаг 1.
            b = x2 # Шаг 3.
            x2 = x1
            n -= 1
            x1 = a + fib[n-2] / fib[n] * (b - a)
            # if k == n - 2:
            #     if f(x1) > f(x2):
            #         a = 
        else:
            a = x1 # Шаг 2.
            x1 = x2
            n -= 1
            x2 = a + fib[n-1] / fib[n] * (b - a)
        k += 1 # Шаг 4.
    return (a + b) / 2, f((a + b) / 2), k

# Поиск экстремума на интервалах.
def find_extremum(f, k, method, epsilon, l):
    results = []
    for interval in k:
        a, b = interval
        x_opt, f_opt, k = method(f, a, b, epsilon, l)
        results.append((x_opt, f_opt, k))
    return results

# Заданные интервалы.
intervals_F1 = [(-3, 0), (0.8, 5), (-10, 0.5)]
intervals_F2 = [(-5, 0), (0, 10), (-5, 5)]

# Параметры.
epsilon_values = [0.1, 0.01, 0.001]
l_values = [0.1, 0.01]

# Выполнение вычислений для F1.
for epsilon in epsilon_values:
    for l in l_values:
        print(f"F1, epsilon={epsilon}, l={l}")
        results = find_extremum(F1, intervals_F1, dichotomy_method, epsilon, l)
        for i, (x_opt, f_opt, k) in enumerate(results):
            print(f"Interval {intervals_F1[i]}: x_opt={x_opt}, f_opt={f_opt}, iterations={k}")

# Выполнение вычислений для F2.
for epsilon in epsilon_values:
    for l in l_values:
        print(f"F2, epsilon={epsilon}, l={l}")
        results = find_extremum(F2, intervals_F2, golden_section_method, epsilon, l)
        for i, (x_opt, f_opt, k) in enumerate(results):
            print(f"Interval {intervals_F2[i]}: x_opt={x_opt}, f_opt={f_opt}, iterations={k}")

# Построение графика.
def plot_function(f, intervals, x_opt, f_opt):
    x = np.linspace(min(intervals, key=lambda x: x[0])[0], max(intervals, key=lambda x: x[1])[1], 1000)
    y = f(x)
    plt.plot(x, y, label="Function")
    for interval in intervals:
        plt.axvline(x=interval[0], color='r', linestyle='--', label="Interval Start")
        plt.axvline(x=interval[1], color='g', linestyle='--', label="Interval End")
    plt.scatter(x_opt, f_opt, color='b', label="Optimal Point")
    plt.legend()
    plt.show()

# Пример построения графика для F1.
x_opt, f_opt, _ = dichotomy_method(F1, -3, 0, 0.1, 0.1)
plot_function(F1, [(-3, 0)], x_opt, f_opt)

# Пример вывода результатов для F1.
x_opt, f_opt, k = dichotomy_method(F1, -3, 0, 0.1, 0.1)
print(f"Optimal x: {x_opt}")
print(f"Optimal F1(x): {f_opt}")
print(f"Number of iterations: {k}")
