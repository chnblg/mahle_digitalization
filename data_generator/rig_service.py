#!/usr/bin/python3.7

from flask import Flask, jsonify
from flask_restx import Api, Resource
from test_rig_manager import TestRigManager
from measurement_protocol_class import MeasurementProtocol
import yaml
from werkzeug.datastructures import FileStorage
from dacite import from_dict

app = Flask(__name__)
api = Api(app)

upload_parser = api.parser()
upload_parser.add_argument('file', location='files', type=FileStorage, required=True)


@api.route('/status')
class TestRigStatusApi(Resource):
    def get(self):
        if test_rig_manager.measurement_manager is None:
            return False
        else:
            return test_rig_manager.get_status()


@api.route('/measurement_protocol')
class TestRigMeasurementProtocolApi(Resource):
    @api.expect(upload_parser)
    def post(self):
        args = upload_parser.parse_args()
        yaml_file = yaml.load(args['file'])
        protocol: MeasurementProtocol = from_dict(data_class=MeasurementProtocol, data=yaml_file)
        return test_rig_manager.load_new_protocol(protocol)

    def get(self):
        if test_rig_manager.measurement_manager is None:
            return False
        else:
            return jsonify(test_rig_manager.get_protocol())


@api.route('/measurement/start')
class TestRigMeasurementControlApiStart(Resource):
    def post(self):
        if test_rig_manager.measurement_manager is None:
            return False
        else:
            return test_rig_manager.start_measurement()


@api.route('/measurement/stop')
class TestRigMeasurementControlApiStop(Resource):
    def post(self):
        if test_rig_manager.measurement_manager is None:
            return False
        else:
            return test_rig_manager.stop_measurement()


@api.route('/hardware/current')
class TestRigHardwareConfigurationApi(Resource):
    def get(self):
        if test_rig_manager.measurement_manager is None or test_rig_manager.get_status() is False:
            hardware_config_idle = {'fan_1': 0.0, 'fan_2': 0.0, 'fan_3': 0.0, 
                                    'pump': 0.0, 'motor_1': 0.0, 'motor_2': 0.0}
            return jsonify(hardware_config_idle)
        else:
            return jsonify(test_rig_manager.get_current_hardware_config())


if __name__ == "__main__":
    test_rig_manager = TestRigManager()
    app.run(host='0.0.0.0')
