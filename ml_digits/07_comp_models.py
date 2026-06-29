from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# 1. Подготовка данных
digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=42)

# 2. Логистическая регрессия (один слой)
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
print(f"Точность LogisticRegression: {accuracy_score(y_test, lr_pred):.4f}")

# 3. Переход к нейросети (MLPClassifier)
# hidden_layer_sizes=(100, 50) означает 2 скрытых слоя: 100 нейронов в первом, 50 во втором
mlp = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', max_iter=1000, random_state=42)
mlp.fit(X_train, y_train)
mlp_pred = mlp.predict(X_test)
print(f"Точность MLPClassifier: {accuracy_score(y_test, mlp_pred):.4f}")