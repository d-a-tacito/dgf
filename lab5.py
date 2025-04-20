# import numpy as np
# from scipy.optimize import minimize
# import matplotlib.pyplot as plt
# from matplotlib import cm
# import pandas as pd
# from openpyxl import load_workbook

# # 1. Определение функций задачи

# def objective(x):
#     """Целевая функция E(X₁, X₂) = 4X₁² - 5X₁X₂ + X₂²"""
#     return 4*x[0]**2 - 5*x[0]*x[1] + x[1]**2

# def constraint1(x):
#     """Первое ограничение: X₁² - X₂ + 2 <= 0"""
#     return x[0]**2 - x[1] + 2

# def constraint2(x):
#     """Второе ограничение: X₁ + X₂ - 6 <= 0"""
#     return x[0] + x[1] - 6

# # 2. Метод штрафных функций

# def penalty_function(x, mu):
#     """Штрафная функция"""
#     penalty = 0
#     # Квадратичные штрафы за нарушение ограничений
#     penalty += max(0, constraint1(x))**2
#     penalty += max(0, constraint2(x))**2
#     # Штрафы за отрицательные значения переменных
#     penalty += max(0, -x[0])**2
#     penalty += max(0, -x[1])**2
#     return objective(x) + mu * penalty

# def penalty_method(initial_point, mu_values, X_type='E2', max_iter=100, tol=1e-6):
#     """
#     Реализация метода штрафных функций
    
#     Параметры:
#     - initial_point: начальная точка
#     - mu_values: список значений параметра штрафа
#     - X_type: тип множества X ('E2', 'nonnegative', 'constraint3')
#     - max_iter: максимальное число итераций
#     - tol: критерий остановки
    
#     Возвращает:
#     - Результаты в виде таблицы
#     - Оптимальное решение
#     """
#     results = []
#     x_current = np.array(initial_point)
    
#     for k, mu in enumerate(mu_values, 1):
#         # Определяем ограничения в зависимости от типа множества X
#         constraints = []
#         bounds = None
        
#         if X_type == 'nonnegative':
#             bounds = [(0, None), (0, None)]
#         elif X_type == 'constraint3':
#             bounds = [(0, None), (0, None)]
#             constraints.append({'type': 'ineq', 'fun': lambda x: -constraint2(x)})
        
#         # Решаем задачу оптимизации с текущим параметром штрафа
#         res = minimize(lambda x: penalty_function(x, mu), 
#                        x_current, 
#                        method='SLSQP',
#                        bounds=bounds,
#                        constraints=constraints)
        
#         x_optimal = res.x
#         f_val = objective(x_optimal)
#         alpha = max(0, constraint1(x_optimal))**2 + max(0, constraint2(x_optimal))**2
#         theta = f_val + mu * alpha
#         mu_alpha = mu * alpha
        
#         results.append([k, mu, x_optimal[0], x_optimal[1], f_val, alpha, theta, mu_alpha])
        
#         # Проверка критерия остановки
#         if mu_alpha < tol:
#             break
            
#         x_current = x_optimal
    
#     return results, x_optimal

# def save_to_excel(results, filename, sheet_name):
#     """Сохраняет результаты в Excel файл с проверкой существования листа"""
#     df = pd.DataFrame(results, columns=['K', 'µk', 'X₁', 'X₂', 'F(Xk+1)', 'α(Xµk)', 'Θ(µk)', 'µkα(Xµk)'])
    
#     try:
#         # Пытаемся открыть существующий файл
#         with pd.ExcelWriter(filename, engine='openpyxl', mode='a') as writer:
#             # Проверяем, существует ли уже такой лист
#             if sheet_name in writer.book.sheetnames:
#                 # Удаляем существующий лист
#                 std = writer.book[sheet_name]
#                 writer.book.remove(std)
#             # Записываем данные
#             df.to_excel(writer, sheet_name=sheet_name, index=False)
#     except FileNotFoundError:
#         # Если файл не существует, создаем новый
#         with pd.ExcelWriter(filename, engine='openpyxl', mode='w') as writer:
#             df.to_excel(writer, sheet_name=sheet_name, index=False)

# # 3. Выполнение расчетов и сохранение в Excel

# # 3.1. Поиск оптимального решения для µ = 0.1, 1, 10, 100
# mu_values = [0.1, 1, 10, 100]
# initial_point = [0, 0]

# results_mu, opt_solution = penalty_method(initial_point, mu_values)
# save_to_excel(results_mu, 'optimization_results.xlsx', 'Разные µ')

# # 3.2. Поиск для нескольких начальных точек
# initial_points = [[0, 0], [1, 1], [2, 2], [3, 3]]
# for i, point in enumerate(initial_points):
#     results, _ = penalty_method(point, mu_values)
#     save_to_excel(results, 'optimization_results.xlsx', f'Начальная точка {i+1}')

# # 3.3. Решение задачи для разных типов множества X
# # a) X = E2
# results_e2, _ = penalty_method([0, 0], mu_values, 'E2')
# save_to_excel(results_e2, 'optimization_results.xlsx', 'X=E2')

# # b) X = {(X₁, X₂): X₁>= 0, X₂>=0}
# results_nonneg, _ = penalty_method([0, 0], mu_values, 'nonnegative')
# save_to_excel(results_nonneg, 'optimization_results.xlsx', 'X неотрицательные')

# # c) X = {(X₁, X₂): X₁ + X₂ <= 6, X₁>= 0, X₂>=0}
# results_constraint, _ = penalty_method([0, 0], mu_values, 'constraint3')
# save_to_excel(results_constraint, 'optimization_results.xlsx', 'X с ограничением')

# print("Результаты сохранены в файл 'optimization_results.xlsx'")

# # 4. Визуализация с указанием коэффициента штрафа

# def plot_with_penalty(results, initial_point):
#     """Визуализация с указанием коэффициента штрафа"""
#     plt.figure(figsize=(12, 8))
    
#     # Траектория оптимизации
#     x_path = [initial_point[0]] + [r[2] for r in results]
#     y_path = [initial_point[1]] + [r[3] for r in results]
    
#     # Рисуем траекторию и точки с подписями µ
#     for i, (x, y, mu) in enumerate(zip(x_path, y_path, [0] + [r[1] for r in results])):
#         if i == 0:
#             plt.plot(x, y, 'go', markersize=8, label='Начальная точка')
#         else:
#             plt.plot(x, y, 'ro', markersize=5)
#             plt.text(x, y, f'µ={mu}', fontsize=8, ha='right', va='bottom')
    
#     plt.plot(x_path, y_path, 'yo-', label='Траектория оптимизации')
#     plt.plot(results[-1][2], results[-1][3], 'b*', markersize=10, label='Оптимальное решение')
    
#     # Сетка и линии уровня
#     x1 = np.linspace(-1, 4, 400)
#     x2 = np.linspace(-1, 6, 400)
#     X1, X2 = np.meshgrid(x1, x2)
#     Z = 4*X1**2 - 5*X1*X2 + X2**2
    
#     levels = np.linspace(Z.min(), Z.max(), 50)
#     plt.contour(X1, X2, Z, levels=levels, cmap=cm.viridis, alpha=0.5)
    
#     # Границы ограничений
#     plt.plot(x1, x1**2 + 2, 'r-', label='X₁² - X₂ + 2 = 0')
#     plt.plot(x1, 6 - x1, 'b-', label='X₁ + X₂ - 6 = 0')
    
#     plt.xlim([-1, 4])
#     plt.ylim([-1, 6])
#     plt.xlabel('X₁')
#     plt.ylabel('X₂')
#     plt.title('Метод штрафных функций с указанием коэффициента штрафа')
#     plt.legend()
#     plt.grid(True)
#     plt.show()

# # Визуализация для начальной точки (0, 0)
# plot_with_penalty(results_mu, initial_point)


import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
from openpyxl import load_workbook

# 1. Определение функций задачи

def objective(x):
    """Целевая функция E(X₁, X₂) = 4X₁² - 5X₁X₂ + X₂²"""
    return 4*x[0]**2 - 5*x[0]*x[1] + x[1]**2

def constraint1(x):
    """Первое ограничение: X₁² - X₂ + 2 <= 0"""
    return x[0]**2 - x[1] + 2

def constraint2(x):
    """Второе ограничение: X₁ + X₂ - 6 <= 0"""
    return x[0] + x[1] - 6

# 2. Метод штрафных функций

def penalty_function(x, mu):
    """Штрафная функция"""
    penalty = 0
    # Квадратичные штрафы за нарушение ограничений
    penalty += max(0, constraint1(x))**2
    penalty += max(0, constraint2(x))**2
    # Штрафы за отрицательные значения переменных
    penalty += max(0, -x[0])**2
    penalty += max(0, -x[1])**2
    return objective(x) + mu * penalty

def penalty_method(initial_point, mu_values, X_type='E2', max_iter=100, tol=1e-6):
    """
    Реализация метода штрафных функций
    
    Параметры:
    - initial_point: начальная точка
    - mu_values: список значений параметра штрафа
    - X_type: тип множества X ('E2', 'nonnegative', 'constraint3')
    - max_iter: максимальное число итераций
    - tol: критерий остановки
    
    Возвращает:
    - Результаты в виде таблицы
    - Оптимальное решение
    """
    results = []
    x_current = np.array(initial_point)
    
    for k, mu in enumerate(mu_values, 1):
        # Определяем ограничения в зависимости от типа множества X
        constraints = []
        bounds = None
        
        if X_type == 'nonnegative':
            bounds = [(0, None), (0, None)]
        elif X_type == 'constraint3':
            bounds = [(0, None), (0, None)]
            constraints.append({'type': 'ineq', 'fun': lambda x: -constraint2(x)})
        
        # Решаем задачу оптимизации с текущим параметром штрафа
        res = minimize(lambda x: penalty_function(x, mu), 
                       x_current, 
                       method='SLSQP',
                       bounds=bounds,
                       constraints=constraints)
        
        x_optimal = res.x
        f_val = objective(x_optimal)
        alpha = max(0, constraint1(x_optimal))**2 + max(0, constraint2(x_optimal))**2
        theta = f_val + mu * alpha
        mu_alpha = mu * alpha
        
        results.append([k, mu, x_optimal[0], x_optimal[1], f_val, alpha, theta, mu_alpha])
        
        # Проверка критерия остановки
        if mu_alpha < tol:
            break
            
        x_current = x_optimal
    
    return results, x_optimal

def save_to_excel(results, filename, sheet_name):
    """Сохраняет результаты в Excel файл с проверкой существования листа"""
    df = pd.DataFrame(results, columns=['K', 'µk', 'X₁', 'X₂', 'F(Xk+1)', 'α(Xµk)', 'Θ(µk)', 'µkα(Xµk)'])
    
    try:
        # Пытаемся открыть существующий файл
        with pd.ExcelWriter(filename, engine='openpyxl', mode='a') as writer:
            # Проверяем, существует ли уже такой лист
            if sheet_name in writer.book.sheetnames:
                # Удаляем существующий лист
                std = writer.book[sheet_name]
                writer.book.remove(std)
            # Записываем данные
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    except FileNotFoundError:
        # Если файл не существует, создаем новый
        with pd.ExcelWriter(filename, engine='openpyxl', mode='w') as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)

def plot_with_directions(results, initial_point):
    """Визуализация с указанием направлений движения"""
    plt.figure(figsize=(12, 8))
    
    # Траектория оптимизации
    x_path = [initial_point[0]] + [r[2] for r in results]
    y_path = [initial_point[1]] + [r[3] for r in results]
    
    # Создаем стрелки для направлений движения (зеленые)
    for i in range(len(x_path)-1):
        dx = x_path[i+1] - x_path[i]
        dy = y_path[i+1] - y_path[i]
        plt.arrow(x_path[i], y_path[i], dx*0.9, dy*0.9, 
                 head_width=0.1, head_length=0.15, 
                 fc='green', ec='green', length_includes_head=True)
    
    # Отмечаем точки
    plt.plot(x_path, y_path, 'go-', markersize=5, label='Траектория оптимизации', color='green')
    plt.plot(initial_point[0], initial_point[1], 'ko', markersize=8, label='Начальная точка')  # Черная точка
    plt.plot(results[-1][2], results[-1][3], 'b*', markersize=10, label='Оптимальное решение')
    
    # Подписи точек с µ
    for i, (x, y, mu) in enumerate(zip(x_path, y_path, [0] + [r[1] for r in results])):
        if i > 0:  # Пропускаем начальную точку
            plt.text(x, y, f'µ={mu}', fontsize=8, ha='right', va='bottom')
    
    # Сетка и линии уровня
    x1 = np.linspace(-1, 4, 400)
    x2 = np.linspace(-1, 6, 400)
    X1, X2 = np.meshgrid(x1, x2)
    Z = 4*X1**2 - 5*X1*X2 + X2**2
    
    levels = np.linspace(Z.min(), Z.max(), 50)
    plt.contour(X1, X2, Z, levels=levels, cmap=cm.viridis, alpha=0.5)
    
    # Границы ограничений
    plt.plot(x1, x1**2 + 2, 'r-', label='X₁² - X₂ + 2 = 0')
    plt.plot(x1, 6 - x1, 'b-', label='X₁ + X₂ - 6 = 0')
    
    # Область допустимых решений
    feasible = (X1**2 - X2 + 2 <= 0) & (X1 + X2 - 6 <= 0) & (X1 >= 0) & (X2 >= 0)
    plt.contourf(X1, X2, feasible, levels=[0.5, 1.5], colors=['lightgreen'], alpha=0.3)
    
    plt.xlim([-1, 4])
    plt.ylim([-1, 6])
    plt.xlabel('X₁')
    plt.ylabel('X₂')
    plt.title('Метод штрафных функций с направлениями движения')
    plt.legend()
    plt.grid(True)
    plt.show()

# 3. Выполнение расчетов и сохранение в Excel

# 3.1. Поиск оптимального решения для µ = 0.1, 1, 10, 100
mu_values = [0.1, 1, 10, 100]
initial_point = [0, 0]

results_mu, opt_solution = penalty_method(initial_point, mu_values)
save_to_excel(results_mu, 'optimization_results.xlsx', 'Разные µ')

# 3.2. Поиск для нескольких начальных точек
initial_points = [[0, 0], [1, 1], [2, 2], [3, 3]]
for i, point in enumerate(initial_points):
    results, _ = penalty_method(point, mu_values)
    save_to_excel(results, 'optimization_results.xlsx', f'Начальная точка {i+1}')

# 3.3. Решение задачи для разных типов множества X
# a) X = E2
results_e2, _ = penalty_method([0, 0], mu_values, 'E2')
save_to_excel(results_e2, 'optimization_results.xlsx', 'X=E2')

# b) X = {(X₁, X₂): X₁>= 0, X₂>=0}
results_nonneg, _ = penalty_method([0, 0], mu_values, 'nonnegative')
save_to_excel(results_nonneg, 'optimization_results.xlsx', 'X неотрицательные')

# c) X = {(X₁, X₂): X₁ + X₂ <= 6, X₁>= 0, X₂>=0}
results_constraint, _ = penalty_method([0, 0], mu_values, 'constraint3')
save_to_excel(results_constraint, 'optimization_results.xlsx', 'X с ограничением')

print("Результаты сохранены в файл 'optimization_results.xlsx'")

# 4. Визуализация
plot_with_directions(results_mu, initial_point)