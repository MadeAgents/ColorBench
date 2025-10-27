from absl import logging
from dotenv import load_dotenv
from pathlib import Path

from server.utils import DeviceManager, get_devices


WORK_HOME = Path(__file__).parent.parent.parent
load_dotenv((WORK_HOME / ".env").as_posix())
logging.set_verbosity("info")


def test_get_devices():
    devices = get_devices()
    logging.info(devices)


def test_device_manager():
    device_manager = DeviceManager()
    logging.info(device_manager.get_available_devices())

    devices = []
    for _ in range(2):
        try:
            device = device_manager.request_device()
            devices.append(device)
            logging.info(device)
        except Exception as e:
            logging.info(e)
    logging.info(device_manager.get_available_devices())
    for device in devices:
        device_manager.release_device(device_name=device.device_name)
        logging.info(f"{device} has been released")
    logging.info(device_manager.get_available_devices())


if __name__ == "__main__":
    test_get_devices()
    test_device_manager()
