import threading
import pathlib
import os
import time
from wav_preprocessor import channel_splitting, add_to_metadata
from helper import ssh_scp_files, remove_files
from multiprocessing import Process
from audio_recorder import AudioRecorder, AudioRecorderConfig
from measurement_protocol_class import MeasurementStep, MeasurementProtocol
from datetime import datetime


class MeasurementManager:

    def __init__(self, protocol: MeasurementProtocol):
        self.audio_recorder = AudioRecorder()
        self.protocol = protocol
        self.thread = None
        self.current_step: int = 0
        self.test_running: bool = False
        self.stop_recording_flag = False

    def get_audio_recorder_config(self, step: MeasurementStep, filename: str) -> AudioRecorderConfig:
        return AudioRecorderConfig(self.protocol.audio_channels, filename, step.duration,
                                   f'samples/{self.protocol.config_name}/')

    def create_filename(self, repeat: int) -> str:
        date = datetime.now().strftime("%Y-%m-%d")
        return f'{date}_{self.protocol.config_name}_{self.current_step}_{repeat}'

    def create_filepath(self, config_name: str) -> str:
        dir_path = f'samples/{config_name}/'
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        return dir_path

    def _recorder_thread(self):
        for i in range(self.protocol.repeat):
            for step in self.protocol.steps:
                if self.stop_recording_flag:
                    self.test_running = False
                    return
                filename = self.create_filename(i)
                filepath = self.create_filepath(self.protocol.config_name)
                config = self.get_audio_recorder_config(step, filename)
                if self.audio_recorder.start_recording(config):
                    self.current_step = (self.current_step + 1) % len(self.protocol.steps)
                    time.sleep(1)
                    channel_splitting(filepath=filepath, filename=config.filename)
                    add_to_metadata(filepath=filepath,
                                    name_config=self.protocol.config_name,
                                    wav_filename=config.filename,
                                    hardware=self.protocol.steps[self.current_step].hardware,
                                    audio_channels=self.protocol.audio_channels)
                    ssh_scp_files(ssh_host='dest-bdagpul00',
                                    ssh_port='8122',
                                    ssh_user='iotmaster',
                                    ssh_password='cranberry314',
                                    source_volume=str(pathlib.Path(__file__).parent.resolve()) + '/samples',
                                    destination_volume='/home/iotmaster')
                    remove_files(filepath=filepath, filename='*.wav')
                    
        self.test_running = False

    def start_recording(self):
        if self.thread is None:
            self.test_running = True
            self.thread = threading.Thread(target=self._recorder_thread)
            self.thread.start()

    def stop_recording(self):
        self.stop_recording_flag = True

    def get_current_step(self) -> int:
        return self.current_step
