import streamlit as st
import requests
import os
import pandas as pd
import io
from PIL import Image
import time
import base64

# Настройки
API_URL = os.getenv("API_URL", "http://localhost:8000/analyze")
REQUEST_TIMEOUT = 90  

# Конфигурация страницы
st.set_page_config(
    page_title="Анализатор блок-схем",
    layout="wide"
)

# CSS для таблицы
st.markdown("""
<style>
.compact-table {
    font-size: 14px !important;
    border-collapse: collapse !important;
    width: 100% !important;
}
.compact-table th, .compact-table td {
    padding: 4px 8px !important;
    border: 1px solid #ddd !important;
    text-align: left !important;
    vertical-align: top !important;
    line-height: 1.3 !important;
}
.compact-table th {
    background-color: #333 !important;
    color: white !important;
    font-weight: bold !important;
}
.compact-table tr {
    background-color: #000 !important; 
    color: white !important;  
}
.compact-table tr:hover {
    background-color: #222 !important;
}

.result-container {
    border: 1px solid #ddd;
    border-radius: 5px;
    padding: 10px;
    margin: 10px 0;
    background-color: #fafafa;
    max-height: 400px;
    overflow-y: auto;
}

.download-btn {
    margin: 5px;
    padding: 8px 16px;
    background-color: #0066cc;
    color: white;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    text-decoration: none;
    display: inline-block;
    font-size: 14px;
    transition: background-color 0.3s;
}
.download-btn:hover {
    background-color: #0055aa;
    text-decoration: none;
    color: white;
}

.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
}

.time-info {
    font-size: 12px;
    color: #666;
    margin-top: 5px;
}

h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# Заголовок
st.title("Анализатор блок-схем")
st.markdown("Загрузите изображение блок-схемы для автоматического анализа")

# Информация о времени обработки
st.info("Время анализа зависит от сложности диаграммы и размера файла. Ожидаемое время обработки простых схем - 10-30 секунд, средних - 30-60 секунд, сложные - до 90 секунд.")

# Загрузка файла
uploaded_file = st.file_uploader(
    "Выберите файл блок-схемы",
    type=['png', 'jpeg']
)

if uploaded_file:
    # Предпросмотр
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Загруженный файл")
        st.write(f"**Имя:** {uploaded_file.name}")
        st.write(f"**Размер:** {uploaded_file.size / 1024:.2f} KB")
        
        image = Image.open(uploaded_file)
        st.write(f"**Разрешение:** {image.size[0]} × {image.size[1]} пикселей")
        
        # Оценка сложности
        if image.size[0] * image.size[1] > 1000000: 
            st.warning("Большое изображение, анализ может занять больше времени")
    
    with col2:
        st.image(image, caption="Предпросмотр", use_container_width=True)
    
    # Кнопка анализа
    if st.button("Анализировать блок-схему", type="primary", use_container_width=True):
        with st.spinner(f"Анализируем блок-схему (таймаут: {REQUEST_TIMEOUT} сек)..."):
            try:
                # Прогресс бар
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Таймер
                start_time = time.time()
                status_text.text("Подготовка запроса...")
                progress_bar.progress(10)
                time.sleep(0.5)
                
                # Отправка запроса с увеличенным таймаутом
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                
                # Настройка сессии с увеличенными таймаутами
                session = requests.Session()
                adapter = requests.adapters.HTTPAdapter(
                    pool_connections=10,
                    pool_maxsize=10,
                    max_retries=3,
                    pool_block=True
                )
                session.mount('http://', adapter)
                session.mount('https://', adapter)
                
                status_text.text("Отправка запроса на сервер...")
                progress_bar.progress(30)
                
                # Отправляем запрос с таймаутом
                response = session.post(
                    f"{API_URL}/analyze", 
                    files=files, 
                    timeout=(10, REQUEST_TIMEOUT)  # (connect timeout, read timeout)
                )
                
                elapsed_time = time.time() - start_time
                status_text.text(f"Анализ завершен за {elapsed_time:.2f} сек")
                progress_bar.progress(100)
                time.sleep(0.5)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Успешный анализ
                    if elapsed_time > 60:
                        st.success(f"Анализ завершен за {elapsed_time:.2f} секунд (долгая обработка)")
                    else:
                        st.success(f"Анализ завершен за {elapsed_time:.2f} секунд")
                    
                    # Отображение результата
                    result_text = result.get("description", "")
                    if result_text:
                        st.subheader("Результат анализа")
                        
                        # Обработка разных форматов результата
                        if "|" in result_text:
                            # Формат таблицы
                            lines = result_text.strip().split('\n')
                            if len(lines) > 0:
                                # Разделяем на заголовок и данные
                                headers = [h.strip() for h in lines[0].split('|')]
                                data = []
                                
                                for line in lines[1:]:
                                    if '|' in line:
                                        row = [cell.strip() for cell in line.split('|')]
                                        if len(row) >= len(headers):
                                            data.append(row[:len(headers)])
                                
                                if data:
                                    st.markdown(f'<div class="result-container">', unsafe_allow_html=True)
                                    
                                    table_html = '<table class="compact-table">'
                                    table_html += '<tr>' + ''.join(f'<th>{h}</th>' for h in headers) + '</tr>'
                                    for row in data:
                                        table_html += '<tr>' + ''.join(f'<td>{cell}</td>' for cell in row) + '</tr>'
                                    table_html += '</table>'
                                    
                                    st.markdown(table_html, unsafe_allow_html=True)
                                    st.markdown('</div>', unsafe_allow_html=True)
                                else:
                                    st.text_area("", value=result_text, height=200)
                        else:
                            st.text_area("", value=result_text, height=200)
                        
                        # Кнопка скачивания TXT
                        st.markdown("---")
                        st.subheader("Скачать результат")
                        
                        # Только TXT файл
                        txt_content = result_text
                        txt_b64 = base64.b64encode(txt_content.encode()).decode()
                        
                        st.markdown(
                            f'<a href="data:text/plain;base64,{txt_b64}" download="результат_анализа.txt" class="download-btn">Скачать результат (TXT)</a>',
                            unsafe_allow_html=True
                        )
                        
                        st.markdown("""
                        <div style="font-size: 12px; color: #666; margin-top: 5px;">
                            Результат будет сохранен в формате TXT для удобного просмотра и редактирования
                        </div>
                        """, unsafe_allow_html=True)
                        
                else:
                    error = response.json().get("detail", "Неизвестная ошибка")
                    st.error(f"Ошибка {response.status_code}: {error}")
                    
            except requests.exceptions.ReadTimeout:
                st.error(f"""
                Таймаут запроса ({REQUEST_TIMEOUT} секунд)
                
                Возможные причины:
                1. Слишком сложная блок-схема - требуется больше времени
                2. Большой размер изображения - попробуйте уменьшить разрешение
                3. Перегрузка сервера - попробуйте позже
                
                Рекомендации:
                - Уменьшите размер изображения до 1000×1000 пикселей
                - Упростите диаграмму
                - Попробуйте снова через несколько минут
                """)
                
            except requests.exceptions.ConnectionError:
                st.error("""
                Не удалось подключиться к серверу
                
                1. Убедитесь, что API сервер запущен:
                ```bash
                uvicorn app.main:app --reload --port 8000
                ```
                
                2. Проверьте, что сервер работает на http://localhost:8000
                
                3. Проверьте файрвол или антивирус
                """)
                
            except Exception as e:
                st.error(f"Неожиданная ошибка: {str(e)}")
            finally:
                # Очищаем прогресс бар
                if 'progress_bar' in locals():
                    progress_bar.empty()
                if 'status_text' in locals():
                    status_text.empty()

# Боковая панель с рекомендациями
st.sidebar.header("Рекомендации")
with st.sidebar.expander("Инструкция по использованию", expanded=True):
    st.markdown("""
    ### Как пользоваться:
    
    1. **Загрузите изображение** блок-схемы через кнопку загрузки
    2. **Проверьте предпросмотр** изображения
    3. **Нажмите кнопку "Анализировать блок-схему"**
    4. **Дождитесь результатов** анализа
    5. **Скачайте результат** в формате TXT
    
    ### Ограничения:
    - Поддерживаются **только блок-схемы** (flowcharts)
    - Форматы изображений: **PNG, JPEG**
    - Максимальный размер файла: **10 MB**
    - Максимальное время обработки: **90 секунд**
    """)

with st.sidebar.expander("Оптимизация для быстрого анализа", expanded=False):
    st.markdown("""
    **Для быстрого анализа:**
    
    - **Уменьшите разрешение:** идеально 800×600 пикселей
    - **Оптимизируйте схему:** четкий текст, минимум деталей
    - **Используйте PNG** для схем с текстом
    - **Используйте JPEG** для фотографий схем
    
    **Если анализ долгий:**
    - Закройте другие вкладки
    - Подождите до 90 секунд
    - При таймауте уменьшите размер файла
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 12px;">
    © 2026 SysCreators Team. Проект для МегаШколы ИТМО.
</div>
""", unsafe_allow_html=True)