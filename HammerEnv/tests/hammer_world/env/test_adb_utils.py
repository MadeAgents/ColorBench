import os
from absl import logging
from dotenv import load_dotenv
from pathlib import Path

from hammer_world.env.adb_utils import (
    close_app,
    close_recents,
    double_tap,
    generate_swipe_command,
    get_all_package_names,
    install_apk,
    issue_generic_request,
    launch_app,
    long_press,
    press_back_button,
    press_enter_button,
    press_home_button,
    press_keyboard_generic,
    tap_screen,
    type_text,
    press_wakeup_button,
)
from hammer_world.env.device_env import DeviceEnv

WORK_HOME = Path(__file__).parent.parent.parent.parent
print(WORK_HOME)
load_dotenv((WORK_HOME / ".env").as_posix())
logging.set_verbosity("info")

device_env = DeviceEnv(
    device_name="6f24b6db", adb_path=os.environ.get("ADB_PATH") or "adb"
)


def test_tap_screen():
    resp = tap_screen(100, 200, env=device_env)
    logging.info(resp)


def test_double_tap():
    resp = double_tap(100, 200, env=device_env)
    logging.info(resp)


def test_long_press():
    resp = long_press(100, 200, env=device_env)
    logging.info(resp)


def test_press_home_button():
    resp = press_home_button(env=device_env)
    logging.info(resp)


def test_press_back_button():
    resp = press_back_button(env=device_env)
    logging.info(resp)


def test_press_enter_button():
    resp = press_enter_button(env=device_env)
    logging.info(resp)


def test_press_keyboard_generic():
    resp = press_keyboard_generic(keycode="187", env=device_env)
    logging.info(resp)


def test_type_text():
    resp = type_text(text="请输入", env=device_env)
    logging.info(resp)


def test_get_all_package_names():
    resp = get_all_package_names(env=device_env)
    logging.info(resp)


def test_launch_app():
    resp = launch_app(app_name="com.oplus.camera", env=device_env)
    logging.info(resp)


def test_close_recents():
    resp = close_recents(env=device_env)
    logging.info(resp)


def test_close_app():
    resp = close_app(app_name="com.oplus.camera", env=device_env)
    logging.info(resp)


def test_press_wakeup_button():
    resp = press_wakeup_button(env=device_env)
    logging.info(resp)


def test_install_apk():
    install_apk(apk_location="3rdparty/ADBKeyBoard/ADBKeyboard.apk", env=device_env)


def test_swipe():
    command = generate_swipe_command(540, 1188, 0, 1188, 50)
    resp = issue_generic_request(args=command, env=device_env)
    logging.info(resp)


if __name__ == "__main__":
    test_swipe()
