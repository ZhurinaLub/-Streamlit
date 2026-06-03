# app.py
import streamlit as st
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

# --- НАСТРОЙКА СТРАНИЦЫ ---
st.set_page_config(page_title="Dog Health Predictor", layout="wide")
st.title("🐕 Анализ и прогнозирование здоровья собак")
st.markdown("---")

# --- ЗАГРУЗКА ДАННЫХ (Задание 1) ---
# Кэшируем загрузку, чтобы данные не перечитывались при каждом взаимодействии
@st.cache_data
def load_data():
    df = pd.read_csv('synthetic_dog_breed_health_data.csv')
    # Небольшая предобработка
    df['Healthy'] = df['Healthy'].fillna('No')
    return df

df = load_data()

# --- БОКОВАЯ ПАНЕЛЬ С КОНТРОЛАМИ (Задание 1 - не менее 5 контролов) ---
st.sidebar.header("🔧 Панель управления")

# Контрол 1: Мультивыбор пород
breeds = st.sidebar.multiselect(
    "Выберите породы для фильтрации:",
    options=df['Breed'].dropna().unique(),
    default=[]
)

# Контрол 2: Слайдер для веса
weight_range = st.sidebar.slider(
    "Диапазон веса (lbs):",
    min_value=float(df['Weight (lbs)'].min()),
    max_value=float(df['Weight (lbs)'].max()),
    value=(float(df['Weight (lbs)'].min()), float(df['Weight (lbs)'].max()))
)

# Контрол 3: Выбор уровня активности
activity_level = st.sidebar.selectbox(
    "Уровень дневной активности:",
    options=["Все"] + sorted(df['Daily Activity Level'].dropna().unique())
)

# Контрол 4: Чекбокс "Только здоровые"
only_healthy = st.sidebar.checkbox("Показать только здоровых собак")

# Контрол 5: Радио-кнопка для выбора типа графика
plot_type = st.sidebar.radio(
    "Тип графика для визуализации:",
    options=["Гистограмма пород", "Средний вес по породам"]
)

# Применяем фильтры к данным
filtered_df = df.copy()
if breeds:
    filtered_df = filtered_df[filtered_df['Breed'].isin(breeds)]
filtered_df = filtered_df[(filtered_df['Weight (lbs)'] >= weight_range[0]) & 
                          (filtered_df['Weight (lbs)'] <= weight_range[1])]
if activity_level != "Все":
    filtered_df = filtered_df[filtered_df['Daily Activity Level'] == activity_level]
if only_healthy:
    filtered_df = filtered_df[filtered_df['Healthy'] == 'Yes']

# --- ОСНОВНАЯ ОБЛАСТЬ (Задание 1 - отображение данных) ---
st.header("📊 Исходные данные")
st.info(f"Показано записей: {len(filtered_df)} из {len(df)}")
st.dataframe(filtered_df, use_container_width=True)

# Визуализация
st.header("📈 Визуализация")
if plot_type == "Гистограмма пород":
    breed_counts = filtered_df['Breed'].value_counts().head(10)
    st.bar_chart(breed_counts)
else:
    avg_weight = filtered_df.groupby('Breed')['Weight (lbs)'].mean().sort_values(ascending=False).head(10)
    st.bar_chart(avg_weight)


# --- МОДЕЛЬ МАШИННОГО ОБУЧЕНИЯ (Задание 2) ---
st.header("🤖 Прогнозирование здоровья собаки")
st.markdown("Введите данные о собаке, чтобы предсказать, будет ли она здорова.")

# Загружаем предварительно обученную модель
@st.cache_resource
def load_model():
    try:
        with open('dog_health_model.pkl', 'rb') as f:
            model, label_encoder, feature_cols = pickle.load(f)
        return model, label_encoder, feature_cols
    except FileNotFoundError:
        st.error("Файл с моделью 'dog_health_model.pkl' не найден. Запустите сначала `train_model.py`.")
        return None, None, None

model, label_encoder, feature_cols = load_model()

# Интерфейс для ввода данных нового экземпляра
if model is not None:
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Возраст (лет)", min_value=0.0, max_value=25.0, value=5.0)
        weight = st.number_input("Вес (lbs)", min_value=0.0, max_value=150.0, value=40.0)
        walk_dist = st.number_input("Дистанция прогулки (миль)", min_value=0.0, max_value=20.0, value=2.0)
    with col2:
        sleep_hrs = st.number_input("Часы сна", min_value=0.0, max_value=24.0, value=10.0)
        play_hrs = st.number_input("Часы игр", min_value=0.0, max_value=10.0, value=1.0)
        vet_visits = st.number_input("Визитов к ветеринару в год", min_value=0, max_value=10, value=1)

    if st.button("🔮 Предсказать здоровье", type="primary"):
        # Создаем массив признаков в том же порядке, что и при обучении
        input_data = pd.DataFrame([[age, weight, walk_dist, sleep_hrs, play_hrs, vet_visits]], 
                                  columns=feature_cols)
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
        
        # Декодируем результат
        result = label_encoder.inverse_transform([prediction])[0]
        
        st.subheader("Результат прогноза:")
        if result == "Yes":
            st.success(f"✅ Собака, скорее всего, ЗДОРОВА (Вероятность: {probability[1]:.2f})")
        else:
            st.error(f"❌ Собака, возможно, НЕЗДОРОВА (Вероятность проблем: {probability[0]:.2f})")
            
        st.caption("Примечание: Прогноз основан на анализе 10 000 синтетических записей.")
