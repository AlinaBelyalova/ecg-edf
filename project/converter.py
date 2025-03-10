import numpy as np
from scipy.io import wavfile
import pyedflib
import sys
import os
from datetime import datetime

def convert_wav_to_edf(wav_file_path, edf_file_path):
    sample_rate, data = wavfile.read(wav_file_path)
    
    if len(data.shape) > 1:
        data = data[:, 0]
    
    data = data.astype(np.float64) / 1000.0

    channel_info = {
        'label': 'ECG',
        'dimension': 'mV',            
        'sample_frequency': sample_rate,
        'physical_max': np.max(data), 
        'physical_min': np.min(data), 
        'digital_max': 32767,
        'digital_min': -32768,
        'transducer': 'ECG sensor',
        'prefilter': 'None'
    }
    

    f = pyedflib.EdfWriter(edf_file_path, 1)
    f.setSignalHeaders([channel_info])
    f.setStartdatetime(datetime.now())
    f.writeSamples([data])
    f.close()
    
    print(f"Конвертация завершена. EDF файл сохранен: {edf_file_path}")

def main():
    if len(sys.argv) != 2:
        print("python convert-wav-to-edf.py <путь_к_wav_файлу>")
        sys.exit(1)
        
    wav_file = sys.argv[1]
    if not os.path.exists(wav_file):
        print(f"Ошибка: файл {wav_file} не найден")
        sys.exit(1)
        
    edf_file = os.path.splitext(wav_file)[0] + '.edf'
    convert_wav_to_edf(wav_file, edf_file)

if __name__ == "__main__":
    main()