import os
import re
import unicodedata
from absl import logging
from subprocess import CompletedProcess
from typing import Collection, Iterable, Optional

import immutabledict

from hammer_world.env.device_env import DeviceEnv

_PATTERN_TO_ACTIVITY = immutabledict.immutabledict({})

_DEFAULT_URIS: dict[str, str] = {
    # "calendar": "content://com.android.calendar",
    # "browser": "http://",
    # "contacts": "content://contacts/people/",
    # "email": "mailto:",
    # "gallery": "content://media/external/images/media/",
}

"""
| 按键名称        | 键值名                   | KEYCODE |
| ----------- | --------------------- | ------- |
| Home        | KEYCODE_HOME         | 3       |
| Back        | KEYCODE_BACK         | 4       |
| Recent Apps | KEYCODE_APP_SWITCH  | 187     |
| Power       | KEYCODE_POWER        | 26      |
| Volume Up   | KEYCODE_VOLUME_UP   | 24      |
| Volume Down | KEYCODE_VOLUME_DOWN | 25      |
| Menu        | KEYCODE_MENU         | 82      |
| Enter       | KEYCODE_ENTER        | 66      |
| Escape      | KEYCODE_ESCAPE       | 111     |
| Tab         | KEYCODE_TAB          | 61      |
| Space       | KEYCODE_SPACE        | 62      |
| Up          | KEYCODE_DPAD_UP     | 19      |
| Down        | KEYCODE_DPAD_DOWN   | 20      |
| Left        | KEYCODE_DPAD_LEFT   | 21      |
| Right       | KEYCODE_DPAD_RIGHT  | 22      |
"""


def get_screen_size(env: DeviceEnv) -> tuple[int, int]:
    """Get the screen size in pixels of an Android device via ADB."""
    adb_command = ["shell", "wm size"]
    adb_response = issue_generic_request(adb_command, env)
    return _parse_screen_size_response(adb_response.stdout)


def get_logical_screen_size(env: DeviceEnv) -> tuple[int, int]:
    """Returns the logical screen size.

    The logical screen size is the screen size that applications use to render
    their interfaces which might be different than the physical screen size when
    orientation/resolution changes. The coordinates we get from A11y tree are
    based on the logical screen size.
    """
    response = issue_generic_request(args="shell dumpsys input | grep logicalFrame", env=env)
    if response.returncode == 0:
        raw_output = response.stdout
        pattern = r"logicalFrame=\[0, 0, (\d+), (\d+)\]"
        matches = re.findall(pattern, raw_output)
        for m in matches:
            if int(m[0]) == 0 and int(m[1]) == 0:
                continue
            width, height = (int(m[0]), int(m[1]))
            return (width, height)
    raise ValueError("Failed to get logical screen size.")


def _parse_screen_size_response(response: str) -> tuple[int, int]:
    """Parse the adb response to extract screen size."""
    match = re.search(r"Physical size: (\d+)x(\d+)", response)
    if match:
        width, height = map(int, match.groups())
        return width, height
    else:
        raise ValueError(f'Screen size information not found in adb response: "{response}"')


def tap_screen(x: int, y: int, env: DeviceEnv) -> CompletedProcess:
    """Issues an AdbRequest to tap the screen at the specified point."""
    logging.info("Attemting to tap the screen at (%d, %d)", x, y)
    adb_command = ["shell", f"input tap {x} {y}"]

    response = issue_generic_request(args=adb_command, env=env)

    if response.returncode != 0:
        logging.error("Failed to tap the screen")

    return response


def double_tap(x: int, y: int, env: DeviceEnv) -> CompletedProcess:
    """Issues two AdbRequests to double tap the screen at the specified point."""
    logging.info("Attempting to double tap the screen at (%d, %d)", x, y)
    first_tap = tap_screen(x, y, env)
    second_tap = tap_screen(x, y, env)
    logging.info(f"First tap: {first_tap}")
    logging.info(f"Second tap: {second_tap}")
    return second_tap


def long_press(x: int, y: int, env: DeviceEnv) -> CompletedProcess:
    """Issues an AdbRequest to long press the screen at the specified point."""
    logging.info("Attempting to long press the screen at (%d, %d)", x, y)
    return issue_generic_request(
        args=["shell", "input", "swipe", str(x), str(y), str(x), str(y), "1000"],
        env=env,
    )


def press_home_button(env: DeviceEnv) -> CompletedProcess:
    """Issues an AdbRequest to press the HOME button in the nav bar."""
    logging.info("Attempting to press the HOME button")

    adb_command = ["shell", "input keyevent KEYCODE_HOME"]
    response = issue_generic_request(args=adb_command, env=env)
    if response.returncode != 0:
        logging.error("Failed to press the HOME button")

    return response


def press_back_button(env: DeviceEnv) -> CompletedProcess:
    """Issues an AdbRequest to press the BACK button in the nav bar."""
    logging.info("Attemting to press the BACK button")
    adb_command = ["shell", "input keyevent KEYCODE_BACK"]
    response = issue_generic_request(args=adb_command, env=env)
    if response.returncode != 0:
        logging.error("Failed to press the BACK button")

    return response


def press_enter_button(env: DeviceEnv) -> CompletedProcess:
    """Issues an AdbRequest to press the ENTER button in the nav bar."""
    logging.info("Attemting to press the ENTER button123")
    adb_command = ["shell", "input keyevent KEYCODE_ENTER"]
    response = issue_generic_request(args=adb_command, env=env)
    if response.returncode != 0:
        logging.error("Failed to press the ENTER button")

    return response


def press_wakeup_button(env: DeviceEnv):
    adb_command = ["shell", "input keyevent KEYCODE_WAKEUP"]
    adb_response = issue_generic_request(adb_command, env)
    return adb_response.stdout


def press_keyboard_generic(keycode: str, env: DeviceEnv) -> CompletedProcess:
    """Issues an AdbRequest to press any button in the keyboard."""
    logging.info("Attemting to press the keyboard button: %s", keycode)
    response = issue_generic_request(
        args=["shell", "input", "keyevent", f"{keycode}"],
        env=env,
    )

    if response.returncode != 0:
        logging.error("Failed to press the keyboard button: %s", keycode)

    return response


def _adb_text_format(text: str) -> str:
    """Prepares text for use with adb."""
    to_escape = [
        "\\",
        ";",
        "|",
        "`",
        "\r",
        " ",
        "'",
        '"',
        "&",
        "<",
        ">",
        "(",
        ")",
        "#",
        "$",
    ]
    for char in to_escape:
        text = text.replace(char, "\\" + char)
    normalized_text = unicodedata.normalize("NFKD", text)
    return normalized_text.encode("ascii", "ignore").decode("ascii")


def _split_words_and_newlines(text: str) -> Iterable[str]:
    """Split lines of text into individual words and newline chars."""
    lines = text.split("\n")
    for i, line in enumerate(lines):
        words = line.split(" ")
        for j, word in enumerate(words):
            if word:
                yield word
            if j < len(words) - 1:
                yield "%s"
        if i < len(lines) - 1:
            yield "\n"


def type_text(text: str, env: DeviceEnv) -> None:
    """Issues an AdbRequest to type the specified text string word-by-word.

    It types word-by-word to fix issue where sometimes long text strings can be
    typed out of order at the character level. Additionally, long strings can time
    out and word-by-word fixes this, while allowing us to keep a lot timeout per
    word.
    """
    adb_command = ["shell", "ime enable com.android.adbkeyboard/.AdbIME"]
    response = issue_generic_request(args=adb_command, env=env)
    adb_command = ["shell", "ime set com.android.adbkeyboard/.AdbIME"]
    response = issue_generic_request(args=adb_command, env=env)

    words = _split_words_and_newlines(text)
    for word in words:
        if word == "\n":
            #logging.info("Found \\n, pressing enter button.")
            #press_enter_button(env)
            continue
        # formatted = _adb_text_format(word)
        # logging.info("Attempting to type word: %r", formatted)
        # adb_command = ["shell", f"input text {formatted}"]
        # response = issue_generic_request(args=adb_command, env=env)
        # if response.returncode != 0:
        #     logging.error("Failed to type word: %r", formatted)
        logging.info("Attempting to type word: %r", word)
        adb_command = ["shell", f"am broadcast -a ADB_INPUT_TEXT --es msg {word}"]
        response = issue_generic_request(args=adb_command, env=env)
        if response.returncode != 0:
            logging.error("Failed to type word: %r", word)
    adb_command = ["shell", "ime disable com.android.adbkeyboard/.AdbIME"]
    response = issue_generic_request(args=adb_command, env=env)


def get_all_package_names(env: DeviceEnv) -> list[str]:
    """Returns all packages installed on the device."""
    adb_command = ["shell", "pm", "list packages"]
    response = issue_generic_request(args=adb_command, env=env)
    if response.returncode != 0:
        logging.error("Failed to issue package manager request.")
        return []

    package_names = response.stdout.split()
    return package_names


def get_all_apps(env: DeviceEnv) -> list[str]:
    """Returns all apps installed on the device.

    Note: the output list will not be exhaustive as it is currently based on a
    mapping we define, so any apps not included in that mapping will not be
    output here.
    """
    packages = get_all_package_names(env)
    package_to_app = {v.split("/")[0]: k.split("|")[0] for k, v in _PATTERN_TO_ACTIVITY.items()}
    app_names = []
    for package in packages:
        if package in package_to_app:
            app_names.append(package_to_app[package])

    return app_names


def get_adb_activity(app_name: str) -> Optional[str]:
    """Get a mapping of regex patterns to ADB activities top Android apps."""
    for pattern, activity in _PATTERN_TO_ACTIVITY.items():
        if re.match(pattern.lower(), app_name.lower()):
            return activity


def start_activity(activity: str, extra_args: Optional[Collection[str]], env: DeviceEnv) -> CompletedProcess:
    """Issues an AdbRequest to launch the given activity."""
    logging.info("Attempting to launch %r", activity)
    adb_command = ["shell", "am", f"start -n {activity}"] + extra_args
    response = issue_generic_request(args=adb_command, env=env)
    if response.returncode != 0:
        logging.error("Failed to launch activity: %r", activity)

    logging.info("Launch package output %r", response.stdout)
    return response


def _launch_default_app(app_key: str, env: DeviceEnv) -> CompletedProcess:
    """Launches a default application with a predefined data URI."""
    if app_key not in _DEFAULT_URIS:
        raise ValueError(f"Unrecognized app key: {app_key}. Must be one of" f" {list(_DEFAULT_URIS.keys())}")
    data_uri = _DEFAULT_URIS[app_key]
    adb_command = [
        "shell",
        "am",
        "start",
        "-a",
        "android.intent.action.VIEW",
        "-d",
        data_uri,
    ]
    response = issue_generic_request(args=adb_command, env=env)
    return response


def launch_app(app_name: str, env: DeviceEnv) -> list[str]:
    """Uses regex and ADB activity to try to launch an app.

    Args:
      app_name: The name of the app, as represented as a key in
        _PATTERN_TO_ACTIVITY.
      env: The environment.

    Returns:
      The name of the app that is launched.
    """

    if app_name in _DEFAULT_URIS:
        _launch_default_app(app_name, env)
        return app_name

    activity = get_adb_activity(app_name)
    if activity is None:
        #  If the app name is not in the mapping, assume it is a package name.
        response = issue_generic_request(["shell", "monkey", "-p", app_name, "1"], env)
        logging.info("Launching app by package name, response: %r", response)
        return app_name
    start_activity(activity, extra_args=[], env=env)
    return app_name


def extract_package_name(activity: str) -> str:
    """Extract the package name from the activity string."""
    return activity.split("/")[0]


def close_recents(env: DeviceEnv):
    """Closes all recent apps."""
    response = issue_generic_request(args="shell dumpsys activity recents", env=env)
    if response.returncode != 0:
        return
    recents_ids = re.findall(r"id=(\d+)", response.stdout)
    for recents_id in recents_ids:
        issue_generic_request(args=["shell", "am", "stack", "remove", recents_id], env=env)


def close_app(app_name: str, env: DeviceEnv) -> Optional[str]:
    """Uses regex and ADB package name to try to directly close an app."""
    activity = get_adb_activity(app_name)
    if activity is None:
        logging.error("Failed to close app: %r", app_name)
        # return None
        activity = app_name
    package_name = extract_package_name(activity)
    issue_generic_request(args=["shell", "am", "force-stop", package_name], env=env)
    return app_name


def install_apk(apk_location: str, env: DeviceEnv) -> None:
    """Installs Android World APK."""
    if not os.path.exists(apk_location):
        raise ValueError("APK does not exist.")
    issue_generic_request(args=["install", apk_location], env=env)


def generate_swipe_command(
    start_x: int,
    start_y: int,
    end_x: int,
    end_y: int,
    duration_ms: Optional[int] = 50,
) -> list[str]:
    """Sends a swipe action to the simulator."""
    return [
        "shell",
        "input",
        f"swipe {start_x} {start_y} {end_x} {end_y} {duration_ms}",
    ]


def generate_drag_and_drop_command(
    start_x: int,
    start_y: int,
    end_x: int,
    end_y: int,
    duration_ms: Optional[int] = 4000,
) -> list[str]:
    """Sends a drag and drop action to the simulator.

    Args:
      start_x: The x-coordinate of the start of the drag and drop.
      start_y: The y-coordinate of the start of the drag and drop.
      end_x: The x-coordinate of the end of the drag and drop.
      end_y: The y-coordinate of the end of the drag and drop.
      duration_ms: If given, the duration of time in milliseconds to take to
        complete the drag and drop.

    Returns:
      List of adb arguments.
    """
    return [
        "shell",
        "input",
        f"draganddrop {start_x} {start_y} {end_x} {end_y} {duration_ms}",
    ]


def issue_generic_request(
    args: Collection[str] | str,
    env: DeviceEnv,
) -> CompletedProcess:
    """Issues a generic adb command.

    Example:
    ~~~~~~~

    issue_generic_request(['shell', 'ls'], env)
    # or
    issue_generic_request('shell ls', env)

    Args:
      args: Set of arguments to be issued with the ABD broadcast. Can also be a
        string.
      env: The environment.
      timeout_sec: A timeout to use for this operation.

    Returns:
      The adb response received after issuing the request.
    """
    if isinstance(args, str):
        args_str = args
        args = args.split(" ")
    else:
        args_str = " ".join(args)
    #print('args_str',args_str)
    response = env.execute_adb_call(args=args_str)
    if response.returncode != 0:
        logging.error(f"Failed to issue generic adb request: {args_str}")
    return response


def uiautomator_dump(env) -> str:
    """Issues a uiautomator dump request and returns the UI hierarchy."""
    dump_args = "shell uiautomator dump /sdcard/window_dump.xml"
    issue_generic_request(dump_args, env)

    read_args = "shell cat /sdcard/window_dump.xml"
    response = issue_generic_request(read_args, env)

    return response.stdout
