from measurement_manager import MeasurementManager
from measurement_protocol_class import HardwareConfiguration, MeasurementProtocol


class TestRigManager:

    def __init__(self):
        self.measurement_manager = None
        self.protocol = None

    def get_status(self) -> bool:
        return self.measurement_manager.test_running

    def load_new_protocol(self, protocol: MeasurementProtocol):
        self.protocol = protocol
        self.measurement_manager = MeasurementManager(self.protocol)

    def get_protocol(self) -> MeasurementProtocol:
        return self.protocol

    def start_measurement(self):
        self.measurement_manager.start_recording()

    def stop_measurement(self):
        self.measurement_manager.stop_recording()

    def get_current_hardware_config(self) -> HardwareConfiguration:
        return self.protocol.steps[self.measurement_manager.get_current_step()].hardware
