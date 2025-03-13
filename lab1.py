import os

import numpy as np
import pandas as pd

# Глобальный список для хранения результатов всех решений
solution_results = []  # Каждая запись – словарь для Excel

# -----------------------------------------------------------
# 1. Определяем функции F1 и F2
# -----------------------------------------------------------
def F1(x):
    """Функция F1(x) = 3*x - x^3 - 1."""
    return 3*x - x**3 - 1

def F2(x):
    """Функция F2(x) = (4 - x^2) / (x*(x^2 + 3))."""
    return (4 - x**2) / (x*(x**2 + 3))

# -----------------------------------------------------------
# 2. Методы оптимизации (дихотомии, золотого сечения, Фибоначчи)
#    Каждый метод печатает таблицу итераций в консоль.
# -----------------------------------------------------------
def dichotomy_method(f, a, b, epsilon, tol, find_min=True, max_iterations=1000):
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

        iteration_data.append((iterations, a, b, x1, x2, sign*f1, sign*f2))

        if f1 <= f2:
            b = x2
        else:
            a = x1

    x_opt = (a + b) / 2
    f_opt = f(x_opt)
    f_calls += 1

    print("\nМетод дихотомии" + (" (минимум)" if find_min else " (максимум)")
          + f", epsilon={epsilon}, tol={tol}")
    print("k      a_k       b_k       λ_k       μ_k     F(λ_k)    F(μ_k)")
    for row in iteration_data:
        k, a_k, b_k, lam, mu, F_lam, F_mu = row
        print(f"{k:2d}  {a_k:8.5f}  {b_k:8.5f}  {lam:8.5f}  {mu:8.5f}  {F_lam:8.5f}  {F_mu:8.5f}")

    print(f"\nИтог: x_opt={x_opt:.5f}, f(x_opt)={f_opt:.5f}, "
          f"итераций={iterations}, вычислений f={f_calls}\n")
    return x_opt, f_opt, iterations, f_calls

def golden_section_method(f, a, b, epsilon, tol, find_min=True, max_iterations=1000):
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

    print("\nМетод золотого сечения" + (" (минимум)" if find_min else " (максимум)")
          + f", epsilon={epsilon}, tol={tol}")
    print("k      a_k       b_k       λ_k       μ_k     F(λ_k)    F(μ_k)")
    for row in iteration_data:
        k, a_k, b_k, lam, mu, F_lam, F_mu = row
        print(f"{k:2d}  {a_k:8.5f}  {b_k:8.5f}  {lam:8.5f}  {mu:8.5f}  {F_lam:8.5f}  {F_mu:8.5f}")

    print(f"\nИтог: x_opt={x_opt:.5f}, f(x_opt)={f_opt:.5f}, "
          f"итераций={iterations}, вычислений f={f_calls}\n")
    return x_opt, f_opt, iterations, f_calls

def fibonacci_method(f, a, b, epsilon, tol, find_min=True, max_iterations=1000):
    sign = 1 if find_min else -1
    def g(x):
        return sign * f(x)

    iteration_data = []
    iterations = 0
    f_calls = 0

    fib = [1, 1]
    while fib[-1] < (b - a) / tol:
        fib.append(fib[-1] + fib[-2])
        if len(fib) > 10000:
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

    print("\nМетод Фибоначчи" + (" (минимум)" if find_min else " (максимум)")
          + f", epsilon={epsilon}, tol={tol}")
    print("k      a_k       b_k       λ_k       μ_k     F(λ_k)    F(μ_k)")
    for row in iteration_data:
        k, a_k, b_k, lam, mu, F_lam, F_mu = row
        print(f"{k:2d}  {a_k:8.5f}  {b_k:8.5f}  {lam:8.5f}  {mu:8.5f}  {F_lam:8.5f}  {F_mu:8.5f}")

    print(f"\nИтог: x_opt={x_opt:.5f}, f(x_opt)={f_opt:.5f}, "
          f"итераций={iterations}, вычислений f={f_calls}\n")
    return x_opt, f_opt, iterations, f_calls

# -----------------------------------------------------------
# 4. Функция для добавления результата решения в глобальный список
# -----------------------------------------------------------
def add_solution_result(solution_number, method_name, func_name, a, b, epsilon, tol,
                        find_min, x_opt, f_opt, iterations, f_calls):
    solution_results.append({
        "Номер решения": solution_number,
        "Метод решения": f"{method_name} ({'min' if find_min else 'max'})",
        "Функция": func_name,
        "a": a,
        "b": b,
        "epsilon": epsilon,
        "tol": tol,
        "x_opt": x_opt,
        "f_opt": f_opt,
        "Число итераций": iterations,
        "Число вычислений f": f_calls
    })

def save_plot(f, a, b, x_opt, f_opt, title, file_name, clamp=50):
    import numpy as np
    import plotly.graph_objs as go
    import plotly.io as pio

    # Определяем общий диапазон для графика (можно менять по необходимости)
    domain_min, domain_max = -10, 10
    x_vals = np.linspace(domain_min, domain_max, 800)

    # Вычисляем y-значения с учетом ограничений (clamp) и обработки деления на 0
    y_vals = []
    for x in x_vals:
        try:
            val = f(x)
            if abs(val) > clamp:
                y_vals.append(None)
            else:
                y_vals.append(val)
        except ZeroDivisionError:
            y_vals.append(None)

    # Создаем график функции
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_vals, y=y_vals, mode='lines', name='Функция', connectgaps=False
    ))

    # Рисуем вертикальные пунктирные линии, обозначающие границы искомого интервала [a, b]
    fig.add_shape(
        type="line",
        x0=a, y0=min([y for y in y_vals if y is not None]),
        x1=a, y1=max([y for y in y_vals if y is not None]),
        line=dict(color="black", width=2, dash="dashdot")
    )
    fig.add_shape(
        type="line",
        x0=b, y0=min([y for y in y_vals if y is not None]),
        x1=b, y1=max([y for y in y_vals if y is not None]),
        line=dict(color="black", width=2, dash="dashdot")
    )

    # Функция для безопасного вычисления f(x)
    def safe_val(x):
        try:
            val = f(x)
            return val if abs(val) <= clamp else None
        except ZeroDivisionError:
            return None

    # Отмечаем маркерами границы интервала
    fig.add_trace(go.Scatter(
        x=[a], y=[safe_val(a)], mode='markers', name='Левая граница',
        marker=dict(color='red', size=8)
    ))
    fig.add_trace(go.Scatter(
        x=[b], y=[safe_val(b)], mode='markers', name='Правая граница',
        marker=dict(color='green', size=8)
    ))

    # Если оптимальное значение определено, отмечаем его
    if x_opt is not None and safe_val(x_opt) is not None:
        fig.add_trace(go.Scatter(
            x=[x_opt], y=[f_opt], mode='markers', name='Оптимум',
            marker=dict(color='magenta', size=10)
        ))

    fig.update_layout(
        title=title,
        xaxis_title='x',
        yaxis_title='f(x)',
        hovermode='x unified'
    )

    # Сохраняем график в виде HTML (так его можно открыть в браузере)
    pio.write_html(fig, file_name.replace(".png", ".html"))
    print(f"График сохранён в {file_name.replace('.png', '.html')}")

# -----------------------------------------------------------
# 5. Главная функция: выбор ручного или автоматического режима
# -----------------------------------------------------------
def main(manual=False):
    # Создаем папку "plots" для сохранения графиков (если не существует)
    if not os.path.exists("plots"):
        os.mkdir("plots")

    solution_counter = 0

    if manual:
        # Ручной ввод параметров
        print("Ручной ввод параметров...")
        func_choice = input("Выберите функцию (1 или 2): ").strip()
        if func_choice == '1':
            f = F1
            func_name = "F1"
        else:
            f = F2
            func_name = "F2"

        method_choice = input("Выберите метод (dichotomy/golden/fibonacci): ").strip()
        extrema_choice = input("Ищем минимум (min) или максимум (max): ").strip()
        find_min = (extrema_choice == "min")

        a = float(input("Введите левую границу a: "))
        b = float(input("Введите правую границу b: "))
        epsilon = float(input("Введите epsilon: "))
        tol = float(input("Введите tol: "))

        solution_counter += 1
        if method_choice == "dichotomy":
            x_opt, f_opt, iters, f_calls = dichotomy_method(f, a, b, epsilon, tol, find_min)
            method_name = "Дихотомия"
        elif method_choice == "golden":
            x_opt, f_opt, iters, f_calls = golden_section_method(f, a, b, epsilon, tol, find_min)
            method_name = "Золотое сечение"
        else:
            x_opt, f_opt, iters, f_calls = fibonacci_method(f, a, b, epsilon, tol, find_min)
            method_name = "Фибоначчи"

        add_solution_result(solution_counter, method_name, func_name, a, b, epsilon, tol,
                            find_min, x_opt, f_opt, iters, f_calls)

        # Сохраняем интерактивный график в HTML-файл с помощью функции save_plot
        # (В ручном режиме график будет сохранён в корневой папке проекта)
        html_filename = f"manual_solution_{solution_counter}.png"  # расширение .png заменится на .html внутри save_plot
        save_plot(f, a, b, x_opt, f_opt, f"{method_name}, {func_name}", html_filename)

    else:
        # Автоматический режим: перебор заранее заданных вариантов
        print("Автоматический расчёт по заранее заданным параметрам...")
        intervals_F1 = [(-3, 0), (0.8, 5), (-10, 0.5)]
        intervals_F2_max = [(-5, 0), (0, 10)]
        intervals_F2_min = [(-5, 5)]
        epsilon_values = [0.01, 0.001]  # Примеры значений epsilon
        tol_values = [0.1, 0.01]         # Примеры значений tol

        # Пример: минимизация F1
        for (a, b) in intervals_F1:
            for eps in epsilon_values:
                for tol in tol_values:
                    # Дихотомия для F1 (минимизация)
                    solution_counter += 1
                    x_opt, f_opt, iters, f_calls = dichotomy_method(F1, a, b, eps, tol, True)
                    add_solution_result(solution_counter, "Дихотомия", "F1", a, b, eps, tol, True,
                                        x_opt, f_opt, iters, f_calls)
                    file_name = f"plots/sol_{solution_counter}_F1_dichotomy_min.png"
                    save_plot(F1, a, b, x_opt, f_opt, f"F1: Дихотомия, интервал=({a},{b})", file_name)

                    # Золотое сечение для F1 (минимизация)
                    solution_counter += 1
                    x_opt, f_opt, iters, f_calls = golden_section_method(F1, a, b, eps, tol, True)
                    add_solution_result(solution_counter, "Золотое сечение", "F1", a, b, eps, tol, True,
                                        x_opt, f_opt, iters, f_calls)
                    file_name = f"plots/sol_{solution_counter}_F1_golden_min.png"
                    save_plot(F1, a, b, x_opt, f_opt, f"F1: Золотое сечение, интервал=({a},{b})", file_name)

                    # Фибоначчи для F1 (минимизация)
                    solution_counter += 1
                    x_opt, f_opt, iters, f_calls = fibonacci_method(F1, a, b, eps, tol, True)
                    add_solution_result(solution_counter, "Фибоначчи", "F1", a, b, eps, tol, True,
                                        x_opt, f_opt, iters, f_calls)
                    file_name = f"plots/sol_{solution_counter}_F1_fibonacci_min.png"
                    save_plot(F1, a, b, x_opt, f_opt, f"F1: Фибоначчи, интервал=({a},{b})", file_name)

        # Обновлённые интервалы для F2 (максимизация), чтобы исключить 0:
        intervals_F2_max = [(-5, -1e-8), (1e-8, 10)]
        for (a, b) in intervals_F2_max:
            for eps in epsilon_values:
                for tol in tol_values:
                    # Дихотомия для F2 (максимизация)
                    solution_counter += 1
                    x_opt, f_opt, iters, f_calls = dichotomy_method(F2, a, b, eps, tol, False)
                    add_solution_result(solution_counter, "Дихотомия", "F2", a, b, eps, tol, False,
                                        x_opt, f_opt, iters, f_calls)
                    file_name = f"plots/sol_{solution_counter}_F2_dichotomy_max.png"
                    save_plot(F2, a, b, x_opt, f_opt, f"F2: Дихотомия (max), интервал=({a},{b})", file_name)

                    # Золотое сечение для F2 (максимизация)
                    solution_counter += 1
                    x_opt, f_opt, iters, f_calls = golden_section_method(F2, a, b, eps, tol, False)
                    add_solution_result(solution_counter, "Золотое сечение", "F2", a, b, eps, tol, False,
                                        x_opt, f_opt, iters, f_calls)
                    file_name = f"plots/sol_{solution_counter}_F2_golden_max.png"
                    save_plot(F2, a, b, x_opt, f_opt, f"F2: Золотое сечение (max), интервал=({a},{b})", file_name)

                    # Фибоначчи для F2 (максимизация)
                    solution_counter += 1
                    x_opt, f_opt, iters, f_calls = fibonacci_method(F2, a, b, eps, tol, False)
                    add_solution_result(solution_counter, "Фибоначчи", "F2", a, b, eps, tol, False,
                                        x_opt, f_opt, iters, f_calls)
                    file_name = f"plots/sol_{solution_counter}_F2_fibonacci_max.png"
                    save_plot(F2, a, b, x_opt, f_opt, f"F2: Фибоначчи (max), интервал=({a},{b})", file_name)

        # Пример: минимизация F2 на интервале [-5, 5]
        for (a, b) in intervals_F2_min:
            for eps in epsilon_values:
                for tol in tol_values:
                    # Дихотомия для F2 (минимизация)
                    solution_counter += 1
                    x_opt, f_opt, iters, f_calls = dichotomy_method(F2, a, b, eps, tol, True)
                    add_solution_result(solution_counter, "Дихотомия", "F2", a, b, eps, tol, True,
                                        x_opt, f_opt, iters, f_calls)
                    file_name = f"plots/sol_{solution_counter}_F2_dichotomy_min.png"
                    save_plot(F2, a, b, x_opt, f_opt, f"F2: Дихотомия (min), интервал=({a},{b})", file_name)

                    # Золотое сечение для F2 (минимизация)
                    solution_counter += 1
                    x_opt, f_opt, iters, f_calls = golden_section_method(F2, a, b, eps, tol, True)
                    add_solution_result(solution_counter, "Золотое сечение", "F2", a, b, eps, tol, True,
                                        x_opt, f_opt, iters, f_calls)
                    file_name = f"plots/sol_{solution_counter}_F2_golden_min.png"
                    save_plot(F2, a, b, x_opt, f_opt, f"F2: Золотое сечение (min), интервал=({a},{b})", file_name)

                    # Фибоначчи для F2 (минимизация)
                    solution_counter += 1
                    x_opt, f_opt, iters, f_calls = fibonacci_method(F2, a, b, eps, tol, True)
                    add_solution_result(solution_counter, "Фибоначчи", "F2", a, b, eps, tol, True,
                                        x_opt, f_opt, iters, f_calls)
                    file_name = f"plots/sol_{solution_counter}_F2_fibonacci_min.png"
                    save_plot(F2, a, b, x_opt, f_opt, f"F2: Фибоначчи (min), интервал=({a},{b})", file_name)

    # Сохраняем сводную таблицу результатов в Excel
    df = pd.DataFrame(solution_results)
    df.to_excel("results.xlsx", index=False)
    print("\nВсе результаты сохранены в 'results.xlsx'. Работа завершена.")

if __name__ == "__main__":
    # Запуск: для автоматического режима установите manual=False
    main(manual=False)
