import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# -----------------------------
# 1. Определяем функции F1 и F2
# -----------------------------
def F1(x):
    """
    Функция F1(x) = 3*x - x^3 - 1.
    Задание требует найти минимум F1 на разных интервалах.
    """
    return 3*x - x**3 - 1

def F2(x):
    """
    Функция F2(x) = (4 - x^2) / (x*(x^2 + 3)).
    Задание требует найти максимум на ряде интервалов, а также минимум на одном интервале.
    """
    return (4 - x**2) / (x*(x**2 + 3))

# -----------------------------------------------------------------------------------
# 2. Реализуем методы оптимизации (дихотомии, золотого сечения, Фибоначчи) с выводом.
# -----------------------------------------------------------------------------------

def dichotomy_method(f, a, b, epsilon, tol, find_min=True, max_iterations=1000):
    """
    Метод дихотомии.
    Если find_min=True, ищем минимум f(x), иначе ищем максимум f(x).
    Для поиска максимума используем f_neg(x) = -f(x) и минимизируем её.

    Параметры:
    -----------
    f : функция
    a, b : границы интервала
    epsilon : константа различимости
    tol : допустимая длина интервала неопределённости (l)
    find_min : True для минимума, False для максимума
    max_iterations : ограничение на количество итераций

    Возвращает:
    -----------
    x_opt : найденная точка минимума/максимума
    f_opt : значение f(x_opt)
    iterations : число итераций
    f_calls : число вычислений функции
    """
    # Чтобы искать максимум, минимизируем -f(x)
    sign = 1 if find_min else -1
    def g(x):
        return sign * f(x)

    iteration_data = []
    iterations = 0
    f_calls = 0

    while (b - a) > tol and iterations < max_iterations:
        iterations += 1
        mid = (a + b) / 2
        x1 = mid - epsilon
        x2 = mid + epsilon

        f1 = g(x1); f_calls += 1
        f2 = g(x2); f_calls += 1

        # Сохраняем данные итерации (k, a_k, b_k, λ_k, μ_k, F(λ_k), F(μ_k))
        iteration_data.append((iterations, a, b, x1, x2, sign*f1, sign*f2))

        if f1 <= f2:
            b = x2
        else:
            a = x1

    x_opt = (a + b) / 2
    f_opt = f(x_opt)  # итоговое значение функции (уже без знака)
    f_calls += 1

    # Выводим таблицу итераций
    print("\nМетод дихотомии" + (" (минимум)" if find_min else " (максимум)") + f", epsilon={epsilon}, tol={tol}")
    print("k      a_k       b_k       λ_k       μ_k     F(λ_k)    F(μ_k)")
    for row in iteration_data:
        k, a_k, b_k, lam, mu, F_lam, F_mu = row
        print(f"{k:2d}  {a_k:8.5f}  {b_k:8.5f}  {lam:8.5f}  {mu:8.5f}  {F_lam:8.5f}  {F_mu:8.5f}")

    print(f"\nИтог: x_opt={x_opt:.5f}, f(x_opt)={f_opt:.5f}, итераций={iterations}, вычислений f={f_calls}\n")

    return x_opt, f_opt, iterations, f_calls


def golden_section_method(f, a, b, epsilon, tol, find_min=True, max_iterations=1000):
    """
    Метод золотого сечения.
    Параметр epsilon здесь не используется, но оставляем для единообразия.

    Параметры:
    -----------
    f : функция
    a, b : границы интервала
    epsilon : не используется
    tol : допустимая длина интервала неопределённости
    find_min : True для минимума, False для максимума
    max_iterations : ограничение на число итераций

    Возвращает:
    -----------
    x_opt, f_opt, iterations, f_calls
    """
    sign = 1 if find_min else -1
    def g(x):
        return sign * f(x)

    iteration_data = []
    iterations = 0
    f_calls = 0

    phi = (1 + np.sqrt(5)) / 2
    x1 = b - (b - a) / phi
    x2 = a + (b - a) / phi
    f1 = g(x1); f_calls += 1
    f2 = g(x2); f_calls += 1

    while (b - a) > tol and iterations < max_iterations:
        iterations += 1
        iteration_data.append((iterations, a, b, x1, x2, sign*f1, sign*f2))

        if f1 <= f2:
            b = x2
            x2 = x1
            f2 = f1
            x1 = b - (b - a) / phi
            f1 = g(x1); f_calls += 1
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = a + (b - a) / phi
            f2 = g(x2); f_calls += 1

    x_opt = (a + b) / 2
    f_opt = f(x_opt); f_calls += 1

    print("\nМетод золотого сечения" + (" (минимум)" if find_min else " (максимум)") + f", epsilon={epsilon}, tol={tol}")
    print("k      a_k       b_k       λ_k       μ_k     F(λ_k)    F(μ_k)")
    for row in iteration_data:
        k, a_k, b_k, lam, mu, F_lam, F_mu = row
        print(f"{k:2d}  {a_k:8.5f}  {b_k:8.5f}  {lam:8.5f}  {mu:8.5f}  {F_lam:8.5f}  {F_mu:8.5f}")

    print(f"\nИтог: x_opt={x_opt:.5f}, f(x_opt)={f_opt:.5f}, итераций={iterations}, вычислений f={f_calls}\n")

    return x_opt, f_opt, iterations, f_calls


def fibonacci_method(f, a, b, epsilon, tol, find_min=True, max_iterations=1000):
    """
    Метод Фибоначчи для поиска минимума/максимума на [a, b].
    Параметр epsilon не используется, но передаётся для единообразия.

    Параметры:
    -----------
    f : функция
    a, b : границы интервала
    epsilon : не используется
    tol : допустимая длина интервала неопределённости
    find_min : True для минимума, False для максимума
    max_iterations : ограничение на число итераций (для подстраховки)

    Возвращает:
    -----------
    x_opt, f_opt, iterations, f_calls
    """
    sign = 1 if find_min else -1
    def g(x):
        return sign * f(x)

    iteration_data = []
    iterations = 0
    f_calls = 0

    # Генерируем числа Фибоначчи, пока последнее не превысит (b-a)/tol
    fib = [1, 1]
    while fib[-1] < (b - a) / tol:
        fib.append(fib[-1] + fib[-2])
        if len(fib) > 10000:  # защита от бесконечного цикла
            break

    n = len(fib) - 1
    x1 = a + fib[n-2]/fib[n] * (b - a)
    x2 = a + fib[n-1]/fib[n] * (b - a)
    f1 = g(x1); f_calls += 1
    f2 = g(x2); f_calls += 1

    while iterations < (n - 2) and (b - a) > tol and iterations < max_iterations:
        iterations += 1
        iteration_data.append((iterations, a, b, x1, x2, sign*f1, sign*f2))

        if f1 > f2:
            a = x1
            x1 = x2
            f1 = f2
            x2 = a + fib[n - iterations - 1]/fib[n - iterations] * (b - a)
            f2 = g(x2); f_calls += 1
        else:
            b = x2
            x2 = x1
            f2 = f1
            x1 = a + fib[n - iterations - 2]/fib[n - iterations] * (b - a)
            f1 = g(x1); f_calls += 1

    x_opt = (a + b) / 2
    f_opt = f(x_opt); f_calls += 1

    print("\nМетод Фибоначчи" + (" (минимум)" if find_min else " (максимум)") + f", epsilon={epsilon}, tol={tol}")
    print("k      a_k       b_k       λ_k       μ_k     F(λ_k)    F(μ_k)")
    for row in iteration_data:
        k, a_k, b_k, lam, mu, F_lam, F_mu = row
        print(f"{k:2d}  {a_k:8.5f}  {b_k:8.5f}  {lam:8.5f}  {mu:8.5f}  {F_lam:8.5f}  {F_mu:8.5f}")

    print(f"\nИтог: x_opt={x_opt:.5f}, f(x_opt)={f_opt:.5f}, итераций={iterations}, вычислений f={f_calls}\n")

    return x_opt, f_opt, iterations, f_calls

# -----------------------------------------------------------------------------
# 3. Вспомогательная функция для построения графика с отмеченными результатами.
# -----------------------------------------------------------------------------
def plot_function(f, a, b, x_opt, f_opt, title=""):
    """
    Строит график функции на отрезке [a, b], отмечает начальные границы
    и точку найденного экстремума.
    """
    x_vals = np.linspace(a, b, 400)
    y_vals = [f(x) for x in x_vals]

    plt.figure(figsize=(6, 4))
    plt.plot(x_vals, y_vals, 'b-', label="Функция")
    plt.axvline(x=a, color='r', linestyle='--', label="Начало интервала")
    plt.axvline(x=b, color='g', linestyle='--', label="Конец интервала")
    plt.scatter(x_opt, f_opt, color='m', zorder=5, label="Оптимум")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True)
    plt.legend()
    plt.show()

# -----------------------------------------------------------------------------
# 4. Пример использования для F1 и F2
# -----------------------------------------------------------------------------

# Интервалы неопределённости из задания:
intervals_F1 = [(-3, 0), (0.8, 5), (-10, 0.5)]
# Для F2: max ищем на [-5,0], [0,10], а min — на [-5,5].
intervals_F2_max = [(-5, 0), (0, 10)]
intervals_F2_min = [(-5, 5)]

# Значения ε (epsilon) и l (tol):
epsilon_values = [0.1, 0.01, 0.001]
tol_values = [0.1, 0.01]

# ---------------------------
# Пример 1: F1 (минимизация)
# ---------------------------
print("=== Минимизация F1(x) = 3*x - x^3 - 1 ===")

# Для наглядности возьмём одно сочетание epsilon и tol, чтобы не перегружать вывод.
# При желании можно перебрать все combinations(epsilon_values, tol_values).
example_epsilon = 0.01
example_tol = 0.1

for (a, b) in intervals_F1:
    # 1) Метод дихотомии
    x_opt, f_opt, iters, f_calls = dichotomy_method(F1, a, b, example_epsilon, example_tol, find_min=True)
    # Строим график
    plot_function(F1, a, b, x_opt, f_opt, title=f"F1: дихотомия, интервал=({a},{b})")

    # 2) Метод золотого сечения
    x_opt, f_opt, iters, f_calls = golden_section_method(F1, a, b, example_epsilon, example_tol, find_min=True)
    plot_function(F1, a, b, x_opt, f_opt, title=f"F1: золотое сечение, интервал=({a},{b})")

    # 3) Метод Фибоначчи
    x_opt, f_opt, iters, f_calls = fibonacci_method(F1, a, b, example_epsilon, example_tol, find_min=True)
    plot_function(F1, a, b, x_opt, f_opt, title=f"F1: Фибоначчи, интервал=({a},{b})")


# --------------------------------
# Пример 2: F2 (максимизация, min)
# --------------------------------
print("=== Оптимизация F2(x) = (4 - x^2) / (x*(x^2 + 3)) ===")

# a) Ищем максимум на интервалах [-5,0] и [0,10]
print("--- Максимизация F2 ---")
for (a, b) in intervals_F2_max:
    x_opt, f_opt, iters, f_calls = dichotomy_method(F2, a, b, 0.01, 0.1, find_min=False)
    plot_function(F2, a, b, x_opt, f_opt, title=f"F2: дихотомия (max), интервал=({a},{b})")

    x_opt, f_opt, iters, f_calls = golden_section_method(F2, a, b, 0.01, 0.1, find_min=False)
    plot_function(F2, a, b, x_opt, f_opt, title=f"F2: золотое сечение (max), интервал=({a},{b})")

    x_opt, f_opt, iters, f_calls = fibonacci_method(F2, a, b, 0.01, 0.1, find_min=False)
    plot_function(F2, a, b, x_opt, f_opt, title=f"F2: Фибоначчи (max), интервал=({a},{b})")

# b) Ищем минимум F2 на интервале [-5,5]
print("\n--- Минимизация F2 на [-5,5] ---")
(a, b) = intervals_F2_min[0]
x_opt, f_opt, iters, f_calls = dichotomy_method(F2, a, b, 0.01, 0.1, find_min=True)
plot_function(F2, a, b, x_opt, f_opt, title=f"F2: дихотомия (min), интервал=({a},{b})")

x_opt, f_opt, iters, f_calls = golden_section_method(F2, a, b, 0.01, 0.1, find_min=True)
plot_function(F2, a, b, x_opt, f_opt, title=f"F2: золотое сечение (min), интервал=({a},{b})")

x_opt, f_opt, iters, f_calls = fibonacci_method(F2, a, b, 0.01, 0.1, find_min=True)
plot_function(F2, a, b, x_opt, f_opt, title=f"F2: Фибоначчи (min), интервал=({a},{b})")


print("\nВсе вычисления завершены.")
print("При необходимости можно перебрать все комбинации (epsilon_values, tol_values) для каждого интервала.")
print("Также в коде можно дополнительно сохранять результаты или строить все графики последовательно.")
