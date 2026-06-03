# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# 1️⃣ ЗАГРУЗКА ДАННЫХ
df = pd.read_csv('synthetic_dog_breed_health_data.csv')

# 2️⃣ ПОДГОТОВКА ДАННЫХ
# Целевая переменная (то, что мы хотим предсказывать) - колонка 'Healthy'
df['Healthy'] = df['Healthy'].fillna('No')  # Заполняем пустые значения
y = df['Healthy']  # Это то, что будем предсказывать

# Выбираем признаки (то, на основе чего будем предсказывать)
feature_columns = ['Age', 'Weight (lbs)', 'Daily Walk Distance (miles)', 
                   'Hours of Sleep', 'Play Time (hrs)', 'Annual Vet Visits']
X = df[feature_columns]

# Заполняем пропуски в данных
X = X.fillna(X.mean())

# 3️⃣ КОДИРОВАНИЕ
# Превращаем 'Yes'/'No' в 1/0 для математических расчетов
le = LabelEncoder()
y_encoded = le.fit_transform(y)  # 'Yes' -> 1, 'No' -> 0

# 4️⃣ ОБУЧЕНИЕ МОДЕЛИ
# Разделяем данные на обучающую и тестовую части
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Создаем и обучаем модель (Random Forest - популярный алгоритм)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)  # Здесь происходит само обучение!

# Проверяем качество модели
print(f"Модель обучена. Точность на тесте: {model.score(X_test, y_test):.2f}")

# 5️⃣ СОХРАНЕНИЕ МОДЕЛИ
# Сохраняем модель и кодировщик в файл для последующего использования
with open('dog_health_model.pkl', 'wb') as f:
    pickle.dump((model, le, feature_columns), f)

print("Модель сохранена в файл 'dog_health_model.pkl'")
