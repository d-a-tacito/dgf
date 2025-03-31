Ниже приведён пример кода на Python, который решает описанную задачу. В коде:

- Генерируются данные согласно условию
- Определяется функция пороговой активации (шаговая функция)
- Реализуется обучение перцептрона с нулевыми начальными весами и смещением
- После обучения выводятся полученные коэффициенты и строится линия, разделяющая классы
- Также показан пример для уменьшённого числа точек (например, 10 штук) и эксперимент с различными начальными весами/смещениями

```python
import numpy as np
import matplotlib.pyplot as plt

# Функция пороговой активации: если net >= 0, то 1, иначе 0.
def step_activation(net):
    return np.where(net >= 0, 1, 0)

# Функция обучения перцептрона
def train_perceptron(X, y, weights, bias, lr=0.1, max_iter=100):
    n_samples = X.shape[0]
    for iteration in range(max_iter):
        error_count = 0
        for i in range(n_samples):
            net = np.dot(X[i], weights) + bias
            prediction = step_activation(net)
            error = y[i] - prediction
            # Если ошибка, корректируем веса и смещение
            if error != 0:
                weights += lr * error * X[i]
                bias += lr * error
                error_count += 1
        # Если ошибок нет – алгоритм сошелся
        if error_count == 0:
            print(f'Сошлось на итерации {iteration}')
            break
    return weights, bias

# Функция для визуализации результатов: точки и разделяющая линия
def plot_decision_boundary(X, y, weights, bias, title=''):
    plt.figure(figsize=(6,6))
    # Отображаем точки разных классов
    plt.scatter(X[y==0][:,0], X[y==0][:,1], color='blue', label='Класс 0')
    plt.scatter(X[y==1][:,0], X[y==1][:,1], color='red', label='Класс 1')
    
    # Вычисляем координаты для линии: w1*x + w2*y + bias = 0  =>  y = (-w1*x - bias)/w2
    x_vals = np.linspace(0, 1, 100)
    # Проверяем, чтобы w2 не был нулевым
    if weights[1] != 0:
        y_vals = -(weights[0]*x_vals + bias)/weights[1]
        plt.plot(x_vals, y_vals, 'k--', label='Разделяющая линия')
    else:
        # Если w2 == 0, линия вертикальная: x = -bias/weights[0]
        x_line = -bias/weights[0]
        plt.axvline(x=x_line, color='k', linestyle='--', label='Разделяющая линия')
    
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

# Генерация исходных данных
def generate_data(n_samples):
    X = np.random.rand(n_samples, 2)
    # Задаём класс: Y = True, если (2*X1 - 0.7) + (-X2 + 0.1) < 0.1, иначе False
    y = (((2*X[:, 0] - 0.7) + (-X[:, 1] + 0.1)) < 0.1).astype(int)
    return X, y

# Основной эксперимент
def experiment(n_samples=100, init_weights=None, init_bias=0.0, lr=0.1, max_iter=100, title_suffix=''):
    X, y = generate_data(n_samples)
    if init_weights is None:
        init_weights = np.zeros(2)  # нулевые начальные веса
    else:
        init_weights = np.array(init_weights, dtype=float)
    bias = init_bias
    print(f"Начальные веса: {init_weights}, смещение: {bias}")
    weights, bias = train_perceptron(X, y, init_weights, bias, lr=lr, max_iter=max_iter)
    print("Результирующие веса:", weights, "смещение:", bias)
    plot_decision_boundary(X, y, weights, bias, title=f"n_samples={n_samples} {title_suffix}")

# --- Эксперименты ---

# 1. Стандартный эксперимент: n_samples = 100, нулевые начальные веса и смещение
experiment(n_samples=100, init_weights=[0,0], init_bias=0.0, title_suffix="(нулевые начальные приближения)")

# 2. Эксперимент с уменьшенным числом точек (например, 10 точек)
experiment(n_samples=10, init_weights=[0,0], init_bias=0.0, title_suffix="(10 точек, нулевые начальные приближения)")

# 3. Эксперимент с другим начальным приближением (например, [0.5, -0.5] и смещение 0.2)
experiment(n_samples=100, init_weights=[0.5, -0.5], init_bias=0.2, title_suffix="(изменённые начальные приближения)")

# 4. Эксперимент с уменьшенным числом точек и другими начальными значениями
experiment(n_samples=15, init_weights=[0.5, -0.5], init_bias=0.2, title_suffix="(15 точек, изменённые начальные приближения)")
```

### Объяснение кода и результатов

1. **Генерация данных.**  
   Функция `generate_data` создаёт массив точек \(X\) размера `n_samples×2` с равномерным распределением на \([0,1]\) и вычисляет метки \(Y\) по условию  
   \[
   (2 \cdot x_1 - 0.7) + (-x_2 + 0.1) < 0.1.
   \]
   Полученные метки приводятся к типу `int` (0 или 1).

2. **Обучение перцептрона.**  
   Функция `train_perceptron` реализует классический алгоритм перцептрона. При каждом неверном предсказании веса и смещение корректируются по правилу:
   \[
   w \leftarrow w + \alpha \cdot (y_{\text{true}} - y_{\text{pred}}) \cdot x,
   \]
   \[
   b \leftarrow b + \alpha \cdot (y_{\text{true}} - y_{\text{pred}}),
   \]
   где \(\alpha\) – скорость обучения.

3. **Визуализация.**  
   Функция `plot_decision_boundary` строит график, где разные классы отображаются разными цветами, а разделяющая прямая рассчитывается по уравнению
   \[
   w_1 \cdot x + w_2 \cdot y + b = 0.
   \]

4. **Эксперименты.**  
   Функция `experiment` позволяет задать количество точек, начальные веса, смещение, скорость обучения и число итераций. Таким образом можно сравнить работу алгоритма:
    - При стандартном количестве точек (100) с нулевыми начальными весами.
    - При уменьшенном числе точек (10 или 15).
    - При различных начальных приближениях (например, [0.5, -0.5] и смещение 0.2).

При изменении начальных весов и числа точек результаты могут отличаться по числу итераций до сходимости и даже по положению разделяющей прямой (особенно на небольших выборках). Это позволяет проанализировать устойчивость и влияние начальных условий на результат обучения.

Запустив представленный код, вы получите как вывод весовых коэффициентов, так и визуализацию разделяющей линии для каждого эксперимента.