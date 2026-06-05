# app.py
import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder  # Обязательный импорт!

# --- НАСТРОЙКА СТРАНИЦЫ ---
st.set_page_config(page_title="Dog Health Predictor", layout="wide")
st.title("🐕 Анализ и прогнозирование здоровья собак")
st.markdown("---")

# --- ЗАГРУЗКА ДАННЫХ (Задание 1) ---
@st.cache_data
def load_data():
    df = pd.read_csv('synthetic_dog_breed_health_data.csv')
    df['Healthy'] = df['Healthy'].fillna('No')
    return df

df = load_data()

# --- БОКОВАЯ ПАНЕЛЬ С КОНТРОЛАМИ ---
st.sidebar.header("🔧 Панель управления")

breeds = st.sidebar.multiselect(
    "Выберите породы для фильтрации:",
    options=df['Breed'].dropna().unique(),
    default=[]
)

weight_range = st.sidebar.slider(
    "Диапазон веса (lbs):",
    min_value=float(df['Weight (lbs)'].min()),
    max_value=float(df['Weight (lbs)'].max()),
    value=(float(df['Weight (lbs)'].min()), float(df['Weight (lbs)'].max()))
)

activity_level = st.sidebar.selectbox(
    "Уровень дневной активности:",
    options=["Все"] + sorted(df['Daily Activity Level'].dropna().unique())
)

only_healthy = st.sidebar.checkbox("Показать только здоровых собак")

plot_type = st.sidebar.radio(
    "Тип графика для визуализации:",
    options=["Гистограмма пород", "Средний вес по породам"]
)

# Применяем фильтры
filtered_df = df.copy()
if breeds:
    filtered_df = filtered_df[filtered_df['Breed'].isin(breeds)]
filtered_df = filtered_df[(filtered_df['Weight (lbs)'] >= weight_range[0]) & 
                          (filtered_df['Weight (lbs)'] <= weight_range[1])]
if activity_level != "Все":
    filtered_df = filtered_df[filtered_df['Daily Activity Level'] == activity_level]
if only_healthy:
    filtered_df = filtered_df[filtered_df['Healthy'] == 'Yes']

# --- ОСНОВНАЯ ОБЛАСТЬ ---
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

# Модель обучается прямо здесь один раз и кэшируется
@st.cache_resource
def load_model():
    # Названия признаков, которые мы будем собирать с интерфейса
    feature_cols = ['Age', 'Weight (lbs)', 'Daily Walk Distance (miles)', 'Sleep Hours', 'Play Hours', 'Vet Visits/Year']
    
    # Словарь для переименования колонок, если в вашем CSV они называются иначе
    rename_dict = {
        'Hours of Sleep': 'Sleep Hours',
        'Play Time (hrs)': 'Play Hours',
        'Annual Vet Visits': 'Vet Visits/Year'
    }
    train_df = df.rename(columns=rename_dict)
    
    # Подготовка матриц
    X = train_df[feature_cols].fillna(train_df[feature_cols].mean())
    y = train_df['Healthy'].fillna('No')
    
    # Кодирование меток
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Обучение модели Random Forest
    model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    model.fit(X, y_encoded)
    
    return model, label_encoder, feature_cols

# Вызов функции обучения
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
        # Формируем DataFrame с правильными именами колонок
        input_data = pd.DataFrame([[age, weight, walk_dist, sleep_hrs, play_hrs, vet_visits]], 
                                  columns=feature_cols)
        
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
        
        result = label_encoder.inverse_transform([prediction])[0]
        
        st.subheader("Результат прогноза:")
        if result == "Yes":
            st.success(f"✅ Собака, скорее всего, ЗДОРОВА (Вероятность: {probability[1]:.2f})")
        else:
            st.error(f"❌ Собака, возможно, НЕЗДОРОВА (Вероятность проблем: {probability[0]:.2f})")
            
        st.caption("Примечание: Прогноз основан на автоматическом экспресс-обучении модели.")
