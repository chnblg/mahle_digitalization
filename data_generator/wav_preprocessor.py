from scipy.io import wavfile
from csv import writer
from pathlib import Path
from measurement_protocol_class import HardwareConfiguration


def channel_splitting(filepath: str, filename: str):
    fs, data = wavfile.read(f'{filepath}{filename}.wav')
    for i in range(data.shape[1]):
        wavfile.write(f'{filepath}{filename}_ch{i}.wav', fs, data[:, i])


def create_metadata(filepath: str, name_config: str):
    row = ['filename', 'hardware']
    with open(f'{filepath}{name_config}.csv', 'w') as csv_file:
        writer_csv = writer(csv_file, delimiter=';')
        writer_csv.writerow(row)
        csv_file.close()


def add_to_metadata(filepath: str, name_config: str, wav_filename: str,
                    hardware: HardwareConfiguration, audio_channels: int):
    csv_filepath = Path(f'{filepath}{name_config}.csv')
    if not csv_filepath.is_file():
        create_metadata(filepath, name_config)
    with open(csv_filepath, 'a') as csv_file:
        writer_csv = writer(csv_file, delimiter=';')
        for i in range(audio_channels):
            temp_filename = f'{wav_filename}_ch{i}.wav'
            hardware_str = f'{hardware.__dict__}'
            row = [temp_filename, hardware_str]
            writer_csv.writerow(row)
        csv_file.close()
