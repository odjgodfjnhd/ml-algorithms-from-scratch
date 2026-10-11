# ML Algorithms from Scratch

Реализация классических алгоритмов машинного обучения на Python с использованием NumPy и Pandas.

## Реализованные алгоритмы

### Linear Regression
- Batch и Stochastic Gradient Descent
- Регуляризация: L1 (Lasso), L2 (Ridge), Elastic Net
- Метрики: MSE, MAE, RMSE, MAPE, R²
- Динамическая скорость обучения

## Пример использования

```python
from linear_regression.MyLineReg import MyLineReg
import pandas as pd

data = pd.read_csv('data.csv')
X, y = data.drop('target', axis=1), data['target']

model = MyLineReg(n_iter=100, learning_rate=0.1, reg='l2', l2_coef=0.01, metric='mse')
model.fit(X, y, verbose=10)
predictions = model.predict(X)