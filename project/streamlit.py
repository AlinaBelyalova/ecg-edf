import streamlit as st
import pyedflib
import pandas as pd
import neurokit2 as nk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
import numpy as np
import pandas as pd
import neurokit2 as nk
import streamlit as st
import matplotlib.dates as mdates
from matplotlib.patches import Ellipse
import tempfile
from functions import (read_edf_to_dataframe,
                       get_sampling_rate,
                       clip_to_one_minute,
                       plot_cardiointervalogram,
                       plot_scattergram,
                       plot_double_ecg_cycle,
                       save_time_series_to_pdf,
                       split_time_series,
                       plot_rr_intervals
                       )

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Image
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import numpy as np
import pandas as pd
import neurokit2 as nk
import streamlit as st
import tempfile
import matplotlib.pyplot as plt
import os

pdfmetrics.registerFont(TTFont('DejaVuSans', 'dejavu-fonts-ttf-2.37/ttf/DejaVuSans.ttf'))

# Интерфейс Streamlit
st.title("Сервис анализа ЭКГ сигнала - Формирование PDF отчета")

st.header("Информация о пациенте")
first_name = st.text_input("Имя", "")
last_name = st.text_input("Фамилия", "")
age = st.number_input("Возраст", min_value=1, max_value=120, step=1)
height = st.number_input("Рост (см)", min_value=50, max_value=250, step=1)
weight = st.number_input("Вес (кг)", min_value=10, max_value=500, step=1)

if st.button("Подтвердить данные"):
    st.write(f"Пациент: {first_name} {last_name}, {age} лет, Рост: {height} см, Вес: {weight} кг")

uploaded_file = st.file_uploader("Загрузите EDF файл", type=["edf"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
        tmpfile.write(uploaded_file.read())
        tmpfile_path = tmpfile.name

    try:
        sampling_rate = get_sampling_rate(tmpfile_path)
        if not isinstance(sampling_rate, (int, float)):
            raise ValueError("Некорректная частота дискретизации")
        df = read_edf_to_dataframe(tmpfile_path)

        # Ограничение данных первой минутой
        first_minute_data = df[df['timestamp'] < (df['timestamp'][0] + pd.to_timedelta(1, unit='m'))]

        st.success("Файл успешно прочитан!")

        available_channels = first_minute_data.columns[:-1]
        st.subheader("Выбор каналов")
        selected_channels = st.multiselect("Выберите один или несколько каналов для создания отчета", available_channels)

        if len(selected_channels) > 0 and st.button("Сформировать PDF отчет"):
            st.write(f"Выбраны каналы: {', '.join(selected_channels)}")

            # Создаём временный файл для PDF
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf_file:
                pdf_path = tmp_pdf_file.name

            try:
                # Создаем PDF документ
                pdf = SimpleDocTemplate(pdf_path, pagesize=letter)
                styles = getSampleStyleSheet()
                story = []
                
                # Устанавливаем шрифт с поддержкой кириллицы для всех стилей
                styles = getSampleStyleSheet()
                styles["Normal"].fontName = "DejaVuSans"
                styles["Heading2"].fontName = "DejaVuSans"  # Для заголовков каналов
                styles["BodyText"].fontName = "DejaVuSans"  # Для текста

                # Добавляем информацию о пациенте
                patient_info_text = (
                    f"Информация о пациенте:<br/>"
                    f"Имя: {first_name}<br/>"
                    f"Фамилия: {last_name}<br/>"
                    f"Возраст: {age}<br/>"
                    f"Рост: {height} см<br/>"
                    f"Вес: {weight} кг<br/>"
                )
                story.append(Paragraph(patient_info_text, styles["Normal"]))
                story.append(Spacer(1, 12))

                all_channel_intervals = []

                # Обработка каждого канала
                for channel in selected_channels:
                    st.write(f"Обработка канала: {channel}")

                    # Добавляем заголовок для канала
                    story.append(Paragraph(f"Канал: {channel}", styles["Heading2"]))
                    story.append(Spacer(1, 12))  # Отступ после заголовка

                    # Обработка данных канала
                    ecg_signal = nk.ecg_clean(first_minute_data[channel].astype(float), sampling_rate=sampling_rate)
                    first_minute_data[f"{channel}_cleaned"] = ecg_signal

                    r_info, r_peaks = nk.ecg_peaks(ecg_signal, sampling_rate=sampling_rate)
                    r_peak_column = f"{channel}_R_Peak_Class"
                    first_minute_data[r_peak_column] = 0
                    if "ECG_R_Peaks" in r_peaks:
                        r_peak_indices = r_peaks["ECG_R_Peaks"]
                        first_minute_data.loc[r_peak_indices, r_peak_column] = 1

                    signals, waves_peak = nk.ecg_delineate(ecg_signal, r_peaks, sampling_rate=sampling_rate, method="dwt")
                    wave_peaks_df = pd.DataFrame(waves_peak)

                    # Вычисление интервалов
                    rr_intervals = np.diff(r_peaks['ECG_R_Peaks']) / sampling_rate * 1000  # В мс
                    pr_intervals = (wave_peaks_df['ECG_R_Onsets'] - wave_peaks_df['ECG_P_Onsets']).dropna().values / sampling_rate * 1000 if 'ECG_P_Onsets' in wave_peaks_df and 'ECG_R_Onsets' in wave_peaks_df else []
                    qrs_durations = (wave_peaks_df['ECG_R_Offsets'] - wave_peaks_df['ECG_R_Onsets']).dropna().values / sampling_rate * 1000 if 'ECG_R_Offsets' in wave_peaks_df and 'ECG_R_Onsets' in wave_peaks_df else []
                    st_durations = (wave_peaks_df['ECG_T_Onsets'] - wave_peaks_df['ECG_S_Offsets']).dropna().values / sampling_rate * 1000 if 'ECG_T_Onsets' in wave_peaks_df and 'ECG_S_Offsets' in wave_peaks_df else []
                    tp_intervals = (wave_peaks_df['ECG_P_Onsets'] - wave_peaks_df['ECG_T_Offsets'].shift(1)).dropna().values / sampling_rate * 1000 if 'ECG_T_Offsets' in wave_peaks_df and 'ECG_P_Onsets' in wave_peaks_df else []

                    interval_infos = [
                        ("RR Intervals", rr_intervals),
                        ("PR Intervals", pr_intervals),
                        ("QRS Durations", qrs_durations),
                        ("ST Durations", st_durations),
                        ("TP Intervals", tp_intervals),
                    ]

                    channel_intervals = []
                    for interval_name, intervals in interval_infos:
                        if intervals is not None and len(intervals) > 0:
                            mean_value = np.mean(intervals)
                            channel_intervals.append((interval_name, mean_value))
                            st.write(f"Среднее значение {interval_name} для канала {channel}: {mean_value:.2f} мс")

                    all_channel_intervals.append((channel, channel_intervals))

                    # Создаем графики и сохраняем их как изображения
                    time_series_parts = split_time_series(first_minute_data, interval=6)
                    image_paths = save_time_series_to_pdf(time_series_parts, channel)

                    # Вставляем изображения в PDF
                    for img_path in image_paths:
                        if os.path.exists(img_path):  # Проверяем, существует ли файл
                            story.append(Image(img_path, width=6 * inch, height=2 * inch))
                            story.append(Spacer(1, 12))  # Отступ между графиками
                        else:
                            st.warning(f"Файл изображения {img_path} не найден.")

                    r_peak_timestamps = first_minute_data["timestamp"].iloc[r_peaks['ECG_R_Peaks']]
                    rr_intervals_one_minute = np.diff(r_peaks['ECG_R_Peaks']) / sampling_rate * 1000

                    # Добавляем табличку с интервалами для текущего канала
                    interval_text = f"Средние значения интервалов для канала {channel}:<br/>"
                    for interval_name, mean_value in channel_intervals:
                        interval_text += f"{interval_name}: {mean_value:.2f} мс<br/>"
                    story.append(Paragraph(interval_text, styles["BodyText"]))
                    story.append(Spacer(1, 24))  # Отступ после таблички

                    fig_cardio = plot_cardiointervalogram(r_peak_timestamps, rr_intervals_one_minute)
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
                        cardio_img_path = tmpfile.name
                        fig_cardio.savefig(cardio_img_path, bbox_inches='tight', dpi=300)
                        plt.close(fig_cardio)

                    # Проверяем, существует ли файл, и вставляем его в PDF
                    if os.path.exists(cardio_img_path):
                        story.append(Image(cardio_img_path, width=6 * inch, height=3 * inch))
                        story.append(Spacer(1, 12))  # Отступ после кардиоинтервалограммы
                    else:
                        st.warning(f"Файл изображения {cardio_img_path} не найден.")

                    # Создаем и добавляем скаттерограмму
                    fig_scatter = plot_scattergram(rr_intervals_one_minute)
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
                        scatter_img_path = tmpfile.name
                        fig_scatter.savefig(scatter_img_path, bbox_inches='tight', dpi=300)
                        plt.close(fig_scatter)

                    # Проверяем, существует ли файл, и вставляем его в PDF
                    if os.path.exists(scatter_img_path):
                        story.append(Image(scatter_img_path, width=3 * inch, height=3 * inch))
                        story.append(Spacer(1, 24))  # Отступ после скаттерограммы
                    else:
                        st.warning(f"Файл изображения {scatter_img_path} не найден.")

                    # Удаляем временные файлы изображений после их использования
                    # if os.path.exists(cardio_img_path):
                    #     os.remove(cardio_img_path)
                    # if os.path.exists(scatter_img_path):
                    #     os.remove(scatter_img_path)

                # Сохраняем PDF
                pdf.build(story)

                with open(pdf_path, "rb") as report_file:
                    st.download_button(label="Скачать PDF отчет", data=report_file, file_name="ecg_report.pdf", mime="application/pdf")

            except Exception as e:
                st.error(f"Ошибка при создании PDF: {str(e)}")

    except Exception as e:
        st.error(f"Ошибка при обработке файла: {str(e)}")
