from dataclasses import dataclass
import subprocess
from typing import List


@dataclass
class AudioRecorderConfig:
    channels: int
    filename: str
    duration: int
    filepath: str


class AudioRecorder:

    def __init__(self):
        self.process = None

    def start_recording(self, config: AudioRecorderConfig) -> bool:
        self.process = subprocess.Popen(args=self.prepare_arguments(config))
        try:
            self.process.wait(timeout=config.duration)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            return True
        return False

    def stop_recording(self):
        self.process.terminate()

    def prepare_arguments(self, config: AudioRecorderConfig) -> List[str]:
        arguments = ['parecord']

        channels = config.channels
        arguments.append(f'--channels={channels}')

        channel_map_list = ['front-left', 'front-right', 'rear-left', 'rear-right', 'front-center',
                            'lfe', 'side-left', 'side-right', 'aux0', 'aux1']
        channel_map_string = ''
        channel_map_string += ",".join(channel for channel in channel_map_list if
                                       channel_map_list.index(channel) < config.channels)

        arguments.append(f'--channel-map={channel_map_string}')

        alsa_input = 'alsa_input.usb-BEHRINGER_UMC1820_1BB69429-00.multichannel-input'
        arguments.append('-d')
        arguments.append(alsa_input)

        arguments.append(f'{config.filepath}{config.filename}.wav')
        print(arguments)

        return arguments
