import time
from utils import read_from_queue
import os
import obsws_python as obs


cl = obs.ReqClient(host='192.168.1.24', port=4455, password='FGKezkUn96whKfub', timeout=3)


def update_obs_source(video_path: str):
    # pass conn info if not in config.toml
    while cl.get_current_program_scene().current_program_scene_name == "DemandeAJesus":
        print("Waiting for end of current question")
        time.sleep(0.1)
    settings_call = cl.get_input_settings("JesusAnswers")
    settings = settings_call.input_settings
    settings["local_file"] = video_path
    cl.set_input_settings("JesusAnswers", settings, True)
    cl.set_current_program_scene("DemandeAJesus")


def main_loop():
    for video_path in read_from_queue("video_response_queue", lambda line: line.decode("utf-8").strip()):
        if video_path is not None:
            print("Changing ", video_path)
            update_obs_source(video_path)
        else:
            time.sleep(1)


if __name__ == "__main__":
    main_loop()
