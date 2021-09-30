from dataclasses import dataclass
from datetime import date
from typing import List


@dataclass
class HardwareConfiguration:
    fan_1: float
    fan_2: float
    fan_3: float
    pump: float
    motor_1: float
    motor_2: float


@dataclass
class MeasurementStep:
    step_name: str
    duration: int
    hardware: HardwareConfiguration


@dataclass
class MeasurementProtocol:
    config_name: str
    date: date
    issuer: str
    repeat: int
    audio_channels: int
    steps: List[MeasurementStep]
