import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd

def objective(x):
    """Целевая функция E(X₁, X₂) = 4X₁² - 5X₁X₂ + X₂²"""
    return 4*x[0]**2 - 5*x[0]*x[1] + x[1]**2

def constraint1(x):
    """Первое ограничение: X₁² - X₂ + 2 <= 0"""
    return x[0]**2 - x[1] + 2

def constraint2(x):
    """Второе ограничение: X₁ + X₂ - 6 <= 0"""
    return x[0] + x[1] - 6

def penalty_function(x, mu, X_type):
    """Штрафная функция с учетом типа множества X"""
    penalty = 0
    
    # Общие штрафы за основные ограничения
    penalty += max(0, constraint1(x))**2
    penalty += max(0, constraint2(x))**2
    
    # Штрафы в зависимости от типа множества X
    if X_type == 'b':  # X₁, X₂ ≥ 0
        penalty += max(0, -x[0])**2 + max(0, -x[1])**2
    elif X_type == 'c':  # X₁ + X₂ ≤ 6 и X₁, X₂ ≥ 0
        penalty += max(0, -x[0])**2 + max(0, -x[1])**2
        penalty += max(0, constraint2(x))**2
    
    return objective(x) + mu * penalty

def penalty_method(initial_point, mu_init, beta, epsilon, X_type, max_iter=100):
    """
    Реализация метода штрафных функций с учетом типа множества X
    
    Параметры:
    - initial_point: начальная точка (X₁, X₂)
    - mu_init: начальное значение параметра штрафа
    - beta: коэффициент увеличения mu
    - epsilon: критерий остановки
    - X_type: тип множества X ('a', 'b' или 'c')
    - max_iter: максимальное число итераций
    
    Возвращает:
    - results: список с результатами каждой итерации
    - optimal_point: найденное оптимальное решение
    """
    results = []
    x_current = np.array(initial_point)
    mu = mu_init
    k = 1
    
    while k <= max_iter:
        # Определяем ограничения в зависимости от типа множества X
        constraints = []
        bounds = None
        
        if X_type == 'b':  # X₁, X₂ ≥ 0
            bounds = [(0, None), (0, None)]
        elif X_type == 'c':  # X₁ + X₂ ≤ 6 и X₁, X₂ ≥ 0
            bounds = [(0, None), (0, None)]
            constraints.append({'type': 'ineq', 'fun': lambda x: -constraint2(x)})
        
        # Шаг 1: Решаем задачу минимизации
        res = minimize(lambda x: penalty_function(x, mu, X_type), 
                      x_current, 
                      method='SLSQP',
                      bounds=bounds,
                      constraints=constraints)
        
        x_optimal = res.x
        f_val = objective(x_optimal)
        alpha = max(0, constraint1(x_optimal))**2 + max(0, constraint2(x_optimal))**2
        theta = f_val + mu * alpha
        mu_alpha = mu * alpha
        
        # Сохраняем результаты
        results.append({
            'K': k,
            'μk': mu,
            'X₁': x_optimal[0],
            'X₂': x_optimal[1],
            'F(X_{k+1})': f_val,
            'α(X_{μk})': alpha,
            'Θ(μk)': theta,
            'μkα(X_{μk})': mu_alpha
        })
        
        # Шаг 2: Проверка критерия остановки
        if mu_alpha < epsilon:
            break
            
        # Обновление параметров
        mu *= beta
        x_current = x_optimal
        k += 1
    
    return results, x_optimal

def save_to_excel(results, filename, sheet_name):
    """Сохраняет результаты в Excel файл"""
    df = pd.DataFrame(results)
    df = df[['K', 'μk', 'X₁', 'X₂', 'F(X_{k+1})', 'α(X_{μk})', 'Θ(μk)', 'μkα(X_{μk})']]
    with pd.ExcelWriter(filename, engine='openpyxl', mode='a') as writer:
        # Удаляем лист, если он уже существует
        if sheet_name in writer.book.sheetnames:
            del writer.book[sheet_name]
        df.to_excel(writer, sheet_name=sheet_name, index=False)
    print(f"Результаты сохранены в файл {filename} на листе '{sheet_name}'")

def plot_optimization(results, initial_point, X_type):
    """Визуализация процесса оптимизации"""
    plt.figure(figsize=(12, 8))
    
    # Траектория оптимизации
    x_path = [initial_point[0]] + [r['X₁'] for r in results]
    y_path = [initial_point[1]] + [r['X₂'] for r in results]
    
    # Создаем стрелки для направлений движения
    for i in range(len(x_path)-1):
        dx = x_path[i+1] - x_path[i]
        dy = y_path[i+1] - y_path[i]
        plt.arrow(x_path[i], y_path[i], dx*0.9, dy*0.9, 
                 head_width=0.1, head_length=0.15, 
                 fc='green', ec='green', length_includes_head=True)
    
    # Отмечаем точки
    plt.plot(x_path, y_path, 'go-', markersize=5, label='Траектория оптимизации')
    plt.plot(initial_point[0], initial_point[1], 'ko', markersize=8, label='Начальная точка')
    plt.plot(results[-1]['X₁'], results[-1]['X₂'], 'b*', markersize=10, label='Оптимальное решение')
    
    # Подписи точек с µ
    for i, (x, y, mu) in enumerate(zip(x_path, y_path, [0] + [r['μk'] for r in results])):
        if i > 0:
            plt.text(x, y, f'µ={mu:.1f}', fontsize=8, ha='right', va='bottom')
    
    # Линии уровня
    x1 = np.linspace(-1, 4, 400)
    x2 = np.linspace(-1, 6, 400)
    X1, X2 = np.meshgrid(x1, x2)
    Z = 4*X1**2 - 5*X1*X2 + X2**2
    
    levels = np.linspace(Z.min(), Z.max(), 50)
    plt.contour(X1, X2, Z, levels=levels, cmap=cm.viridis, alpha=0.5)
    
    # Границы ограничений
    plt.plot(x1, x1**2 + 2, 'r-', label='X₁² - X₂ + 2 = 0')
    plt.plot(x1, 6 - x1, 'b-', label='X₁ + X₂ - 6 = 0')
    
    # Область допустимых решений в зависимости от X_type
    if X_type == 'a':
        feasible = (X1**2 - X2 + 2 <= 0) & (X1 + X2 - 6 <= 0)
    elif X_type == 'b':
        feasible = (X1**2 - X2 + 2 <= 0) & (X1 + X2 - 6 <= 0) & (X1 >= 0) & (X2 >= 0)
    elif X_type == 'c':
        feasible = (X1**2 - X2 + 2 <= 0) & (X1 + X2 - 6 <= 0) & (X1 >= 0) & (X2 >= 0)
    
    plt.contourf(X1, X2, feasible, levels=[0.5, 1.5], colors=['lightgreen'], alpha=0.3)
    
    plt.xlim([-1, 4])
    plt.ylim([-1, 6])
    plt.xlabel('X₁')
    plt.ylabel('X₂')
    
    # Заголовок с указанием типа множества X
    x_type_names = {
        'a': 'X = E₂ (без дополнительных ограничений)',
        'b': 'X = {(X₁, X₂): X₁ ≥ 0, X₂ ≥ 0}',
        'c': 'X = {(X₁, X₂): X₁ + X₂ ≤ 6, X₁ ≥ 0, X₂ ≥ 0}'
    }
    plt.title(f'Метод штрафных функций\n{x_type_names[X_type]}')
    plt.legend()
    plt.grid(True)
    plt.show()

# Основная программа
def main():
    print("Метод штрафных функций для задачи оптимизации")
    print("Начальная точка фиксирована: (0, 0)")
    
    # Ввод параметров алгоритма
    mu_init = float(input("Введите начальное значение μ: "))
    beta = float(input("Введите коэффициент β (≥1): "))
    epsilon = float(input("Введите критерий остановки ε: "))
    
    # Выбор типа множества X
    print("\nВыберите тип множества X:")
    print("a) X = E₂ (без дополнительных ограничений)")
    print("b) X = {(X₁, X₂): X₁ ≥ 0, X₂ ≥ 0}")
    print("c) X = {(X₁, X₂): X₁ + X₂ ≤ 6, X₁ ≥ 0, X₂ ≥ 0}")
    X_type = input("Введите букву (a, b или c): ").lower()
    
    while X_type not in ['a', 'b', 'c']:
        print("Неверный ввод. Пожалуйста, введите a, b или c.")
        X_type = input("Введите букву (a, b или c): ").lower()
    
    initial_point = [0, 0]  # Фиксированная начальная точка
    
    # Запуск алгоритма
    results, optimal_point = penalty_method(initial_point, mu_init, beta, epsilon, X_type)
    
    # Сохранение в Excel
    sheet_names = {
        'a': 'X=E2',
        'b': 'X_nonnegative',
        'c': 'X_with_constraint'
    }
    save_to_excel(results, 'optimization_results.xlsx', sheet_names[X_type])
    
    # Вывод результатов
    print("\nРезультаты оптимизации:")
    print(f"Тип множества X: {X_type}")
    print(f"Начальная точка: ({initial_point[0]}, {initial_point[1]})")
    print(f"Оптимальное решение: ({optimal_point[0]:.4f}, {optimal_point[1]:.4f})")
    print(f"Значение целевой функции: {objective(optimal_point):.4f}")
    print(f"Количество итераций: {len(results)}")
    
    # Визуализация
    plot_optimization(results, initial_point, X_type)

if __name__ == "__main__":
    main()