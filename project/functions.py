import pyedflib
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.dates as mdates
import tempfile
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
import tempfile
import os
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
from matplotlib.patches import Ellipse

pdfmetrics.registerFont(TTFont('DejaVuSans', 'dejavu-fonts-ttf-2.37/ttf/DejaVuSans.ttf'))

#Чтение edf файлов
def read_edf_to_dataframe(file_path):
    f = pyedflib.EdfReader(file_path)
    n_signals = f.signals_in_file
    df = pd.DataFrame()
    for i in range(n_signals):
        signal = f.readSignal(i)
        label = f.getLabel(i)
        df[label] = signal
    start_datetime = f.getStartdatetime()
    periods = df.shape[0]
    freq_sec = pd.to_timedelta(1000 / f.getSampleFrequencies()[0], unit="ms")
    idx = pd.date_range(start=start_datetime, periods=periods, freq=freq_sec)
    df["timestamp"] = idx
    f.close()
    return df
# def read_edf_to_dataframe(file_path):
#     f = pyedflib.EdfReader(file_path)
#     n_signals = f.signals_in_file
#     df = pd.DataFrame()
    
#     for i in range(n_signals):
#         signal = f.readSignal(i)
#         label = f.getLabel(i)
        
#         # Если хотите масштабировать только "ECG"-канал:
#         if label.upper().startswith("ECG"):
#             signal = signal / 1000.0
        
#         df[label] = signal

#     start_datetime = f.getStartdatetime()
#     periods = df.shape[0]
#     freq_sec = pd.to_timedelta(1000 / f.getSampleFrequencies()[0], unit="ms")
#     idx = pd.date_range(start=start_datetime, periods=periods, freq=freq_sec)
#     df["timestamp"] = idx
#     f.close()
#     return df

# Функция обрезки сигнала до одной минуты
def clip_to_one_minute(df, start_time):
    end_time = start_time + pd.Timedelta(minutes=1)
    df_one_minute = df[(df["timestamp"] >= start_time) & (df["timestamp"] < end_time)]
    return df_one_minute

def get_sampling_rate(file_path):
    f = pyedflib.EdfReader(file_path)
    sampling_rate = int(f.getSampleFrequencies()[0])
    f.close()
    return sampling_rate

#Отрисовка кардиоинтервалограммы
def plot_cardiointervalogram(r_peak_timestamps, rr_intervals):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(r_peak_timestamps[1:], rr_intervals, width=0.0000001, color='blue', edgecolor='black')
    ax.set_title("Кардиоинтервалограмма (первая минута)")
    ax.set_xlabel("Моменты появления R-зубцов")
    ax.set_ylabel("RR интервалы (мс)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig

#Функция, которая делит дф по 10 секунд
def split_time_series(df, interval=6):
    start_time = df['timestamp'].min()
    end_time = df['timestamp'].max()
    split_ranges = pd.date_range(start=start_time, end=end_time, freq=f'{interval}S')
    split_parts = [df[(df['timestamp'] >= start) & (df['timestamp'] < start + pd.Timedelta(seconds=interval))] 
                   for start in split_ranges]
    return split_parts


# Отрисовка скаттерограммы
def plot_scattergram(rr_intervals):
    fig, ax = plt.subplots(figsize=(4, 4))
    colors = []

    # Параметры эллипсов
    ellipse_good_center = np.array([850, 850])
    ellipse_good_width = 150
    ellipse_good_height = 300
    ellipse_norm_center = np.array([900, 900])
    ellipse_norm_width = 300
    ellipse_norm_height = 600
    angle = -45  # Угол поворота в градусах

    # Преобразование угла в радианы
    angle_rad = np.deg2rad(angle)
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)

    # Предварительные вычисления для матрицы поворота
    rotation_matrix = np.array([[cos_angle, sin_angle],
                                [-sin_angle, cos_angle]])

    # Полуоси эллипсов
    a_good = ellipse_good_width / 2
    b_good = ellipse_good_height / 2
    a_norm = ellipse_norm_width / 2
    b_norm = ellipse_norm_height / 2

    # Проверяем каждую точку на принадлежность эллипсам
    for i in range(len(rr_intervals) - 1):
        point = np.array([rr_intervals[i], rr_intervals[i + 1]])

        # Проверка для ellipse_good
        relative_point = point - ellipse_good_center
        rotated_point = rotation_matrix @ relative_point
        ellipse_eq = (rotated_point[0] / a_good)**2 + (rotated_point[1] / b_good)**2

        if ellipse_eq <= 1:
            colors.append('green')
            continue  # Если точка внутри ellipse_good, проверка завершена

        # Проверка для ellipse_norm
        relative_point = point - ellipse_norm_center
        rotated_point = rotation_matrix @ relative_point
        ellipse_eq = (rotated_point[0] / a_norm)**2 + (rotated_point[1] / b_norm)**2

        if ellipse_eq <= 1:
            colors.append('yellow')
        else:
            colors.append('red')

    # Отрисовка эллипсов
    ellipse_norm_patch = Ellipse(
        ellipse_norm_center,
        width=ellipse_norm_width,
        height=ellipse_norm_height,
        angle=angle,
        color='lightyellow',
        # alpha=0.5,
        zorder=1
    )
    ellipse_good_patch = Ellipse(
        ellipse_good_center,
        width=ellipse_good_width,
        height=ellipse_good_height,
        angle=angle,
        color='lightgreen',
        # alpha=0.5,
        zorder=2
    )
    ax.add_patch(ellipse_norm_patch)
    ax.add_patch(ellipse_good_patch)

    # Отрисовка точек поверх эллипсов
    ax.scatter(
        rr_intervals[:-1],
        rr_intervals[1:],
        color=colors,
        alpha=0.6,
        s=10,
        zorder=3  # Точки будут отображаться поверх эллипсов
    )

    # Отрисовка биссектрисы
    ax.plot(
        [0, 1500],
        [0, 1500],
        color='orange',
        linestyle='--',
        linewidth=0.5,
        zorder=4
    )

    # Настройка графика
    ax.set_xlim(300, 1500)
    ax.set_ylim(300, 1500)
    ax.set_title("Скаттерограмма RR-интервалов", fontsize=10)
    ax.set_xlabel("Предыдущий RR интервал (мс)", fontsize=8)
    ax.set_ylabel("Следующий RR интервал (мс)", fontsize=8)

    # Вычисление длины (L) и ширины (w) скаттерограммы
    L = np.max(rr_intervals) - np.min(rr_intervals)  # Длина «облака»
    distances = (rr_intervals[:-1] - rr_intervals[1:]) / np.sqrt(2)  # Проекция на биссектрису
    w = 2 * np.std(distances)  # Ширина скаттерограммы
    ellipse_area = np.pi * L  * w / 4  # Площадь эллипса

    # Отображение вычисленных параметров
    ax.text(
        1485, 1450,
        f'Area: {ellipse_area:.1f} ms²',
        fontsize=8,
        color='black',
        ha='right',
        zorder=4
    )
    ax.text(
        1485, 1400,
        f'Width (w): {w:.1f} ms',
        fontsize=8,
        color='black',
        ha='right',
        zorder=4
    )
    ax.text(
        1485, 1350,
        f'Length (L): {L:.1f} ms',
        fontsize=8,
        color='black',
        ha='right',
        zorder=4
    )

    # Настройка положения графика
    ax.set_position([0.15, 0.15, 0.8, 0.7])
    plt.tight_layout()
    return fig



def plot_double_ecg_cycle(ecg_signal, r_peaks, sampling_rate):
    try:
        # Проверяем наличие трех R-пиков
        if len(r_peaks["ECG_R_Peaks"]) < 3:
            print("Недостаточно R-пиков для выделения двух полных кардиоциклов.")
            return None

        # Извлечь участки, включающие два кардиоцикла
        start_index = r_peaks["ECG_R_Peaks"][0]
        end_index = r_peaks["ECG_R_Peaks"][2]
        double_cycle_segment = ecg_signal[start_index:end_index]
        time_values = np.arange(len(double_cycle_segment)) / sampling_rate

        # Нарисовать два кардиоцикла
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(time_values, double_cycle_segment, label='Два кардиоцикла')
        ax.set_title("Два ЭКГ кардиоцикла (на основе первой минуты)")
        ax.set_xlabel("Время (с)")
        ax.set_ylabel("Амплитуда (мВ)")
        ax.legend()
        plt.tight_layout()
        return fig
    except Exception as e:
        print(f"Ошибка при выделении кардиоциклов: {e}")
        return None
    
def plot_rr_intervals(ax, part, channel_name, sampling_rate, y_offset):
    """
    Рисует RR-интервалы для части временного ряда.
    """
    # Находим R-пики
    r_peak_indices = part.index[part[f"{channel_name}_R_Peak_Class"] == 1]
    r_peak_timestamps = part.loc[r_peak_indices, 'timestamp']
    
    rr_intervals = (np.diff(r_peak_indices) / sampling_rate) * 1000  # Перевод в мс
    for i in range(len(r_peak_indices) - 1):
        x_start = r_peak_timestamps.iloc[i]
        x_end = r_peak_timestamps.iloc[i + 1]
        
        # Проверка на нормальные RR интервалы
        if 1000 <= rr_intervals[i] <= 1200:
            color = 'darkorange'
        elif rr_intervals[i] < 600 or rr_intervals[i] > 1200:
            color = 'red'
        else:
            color = 'blue' 
        
        ax.annotate(
            '', 
            xy=(x_start, y_offset), 
            xytext=(x_end, y_offset), 
            arrowprops=dict(arrowstyle='<->', color=color, lw=1.5)  # Цвет стрелок
        )
        
        mid_x = x_start + (x_end - x_start) / 2  # Средняя точка по X
        ax.text(mid_x, y_offset + 0.05, f'{rr_intervals[i]:.0f} ms', 
                color=color, fontsize=10, ha='center', va='bottom') 
    
def save_time_series_to_pdf(time_series_parts, channel_name,sampling_rate, parts_per_page=10):
    """
    Сохраняет части временного ряда в изображения и возвращает список путей к ним.

    Args:
        time_series_parts: Список DataFrame с частями временного ряда.
        channel_name: Имя канала.
        parts_per_page: Количество частей на одной странице.

    Returns:
        Список путей к сохранённым изображениям.
    """
    image_paths = []
    cumulative_time_offset = 0  # Смещение времени между частями

    for i, part in enumerate(time_series_parts):
        # Проверяем и преобразуем тип столбца 'timestamp' в секунды
        if not np.issubdtype(part['timestamp'].dtype, np.datetime64):
            part['timestamp'] = pd.to_datetime(part['timestamp'], errors='coerce')

        # Переводим время в секунды с учётом сквозного смещения
        part['timestamp'] = (part['timestamp'] - part['timestamp'].min()).dt.total_seconds()
        part['timestamp'] += cumulative_time_offset
        cumulative_time_offset = part['timestamp'].max()  # Обновляем смещение для следующей части

        # Создаем временный файл для изображения
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
            image_path = tmpfile.name

        # Создаем график
        fig, ax = plt.subplots(figsize=(8.27, 3))  # Размер графика
        ax.plot(part['timestamp'], part[f"{channel_name}_cleaned"], color='black', lw=1)

        # Устанавливаем границы осей X и Y
        ax.set_xlim(part['timestamp'].min(), part['timestamp'].max())
        ax.set_ylim(-2, 2)

        # Настройка делений и сетки для миллиметровки
        major_ticks_x = 0.2  # 1 крупная клетка = 0.2 секунды (5 мм)
        minor_ticks_x = 0.04  # 1 мелкая клетка = 0.04 секунды (1 мм)
        ax.set_xticks(np.arange(part['timestamp'].min(), part['timestamp'].max() + major_ticks_x, major_ticks_x), minor=False)
        ax.set_xticks(np.arange(part['timestamp'].min(), part['timestamp'].max() + minor_ticks_x, minor_ticks_x), minor=True)

        major_ticks_y = 0.5  # 1 крупная клетка = 0.5 мВ (5 мм)
        minor_ticks_y = 0.1  # 1 мелкая клетка = 0.1 мВ (1 мм)
        ax.set_yticks(np.arange(-2, 2 + major_ticks_y, major_ticks_y), minor=False)
        ax.set_yticks(np.arange(-2, 2 + minor_ticks_y, minor_ticks_y), minor=True)

        # Настройка сетки
        ax.grid(which='major', color='red', linestyle='-', linewidth=0.5)
        ax.grid(which='minor', color='gray', linestyle=':', linewidth=0.3)

        # Убираем подписи, связанные с сеткой
        ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

        # Добавление подписей на ось X в формате MM:SS
        label_interval = 2
        xticks_labels = np.arange(part['timestamp'].min(), part['timestamp'].max() + label_interval, label_interval)
        for x in xticks_labels:
            minutes, seconds = divmod(int(x), 60)  # Преобразование секунд в минуты и секунды
            time_label = f"{minutes}:{seconds:02d}"  # Формат MM:SS
            ax.text(x, -2.2, time_label, fontsize=8, ha='center', va='top')  # Подпись ниже графика

        y_offset = part[f"{channel_name}_cleaned"].min() - 0.2  # Смещение для стрелок и подписей
        plot_rr_intervals(ax, part, channel_name,sampling_rate, y_offset)

        # Добавление подписей на ось Y
        ax.set_ylabel("Amplitude (mV)", fontsize=10)
        ax.set_yticklabels([f"{y:.1f}" for y in np.arange(-2, 2 + major_ticks_y, major_ticks_y)], fontsize=8)

        # Убираем сжатие графика
        fig.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.3)

        # Сохраняем график как изображение
        plt.savefig(image_path, dpi=300)
        plt.close(fig)

        image_paths.append(image_path)

    return image_paths
