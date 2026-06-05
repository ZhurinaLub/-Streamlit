import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 1️⃣ ЗАГРУЗКА ДАННЫХ
df = pd.read_csv("synthetic_dog_breed_health_data.csv")

# 2️⃣ ПРИВЕДЕНИЕ НАЗВАНИЙ КОЛОНОК К СТАНДАРТУ APP.PY
# Переименовываем старые колонки из CSV в новые названия для Streamlit
rename_dict = {
    "Hours of Sleep": "Sleep Hours",
    "Play Time (hrs)": "Play Hours",
    "Annual Vet Visits": "Vet Visits/Year",
}
df = df.rename(columns=rename_dict)

# 3️⃣ ПОДГОТОВКА ДАННЫХ
# Целевая переменная (то, что мы хотим предсказывать) - колонка 'Healthy'
df["Healthy"] = df["Healthy"].fillna("No")  # Заполняем пустые значения
y = df["Healthy"]  # Это то, что будем предсказывать

# Выбираем признаки (названия СТРОГО СОВПАДАЮТ с app.py)
feature_columns = [
    "Age",
    "Weight (lbs)",
    "Daily Walk Distance (miles)",
    "Sleep Hours",
    "Play Hours",
    "Vet Visits/Year",
]
X = df[feature_columns]

# Заполняем пропуски в данных средними значениями
X = X.fillna(X.mean())

# 4️⃣ КОДИРОВАНИЕ
# Превращаем 'Yes'/'No' в 1/0 для математических расчетов
le = LabelEncoder()
y_encoded = le.fit_transform(y)  # 'Yes' -> 1, 'No' -> 0

# 5️⃣ ОБУЧЕНИЕ МОДЕЛИ
# Разделяем данные на обучающую и тестовую части
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# Создаем и обучаем модель (Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)  # Здесь происходит само обучение!

# Проверяем качество модели
print(f"Модель обучена. Точность на тесте: {model.score(X_test, y_test):.2f}")

# 6️⃣ СОХРАНЕНИЕ МОДЕЛИ
# Сохраняем модель, кодировщик и список колонок в файл для последующего использования
with open("dog_health_model.pkl", "wb") as f:
    pickle.dump((model, le, feature_columns), f)

print("Модель сохранена в файл 'dog_health_model.pkl'")

