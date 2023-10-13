import random
import time
from utils import read_from_queue
import os
import obsws_python as obs
import subprocess


cl = obs.ReqClient(host='192.168.1.24', port=4455, password='FGKezkUn96whKfub', timeout=3)

MAIN_SCENE = "DemandeAJesus"
MAIN_MEDIA = "JesusAnswers"
BACKGROUND_SCENE = "DemandeAJesusBackground 3"
BACKGROUND_MEDIA = "JesusBackground"
QUEUE_NAME = "video_response_queue"
VIDEO_DIRECTORY_PATH = "/media/amor/Storage/Videos/JesusStreamFolder"


def update_obs_source(video_path: str):
    if cl.get_current_program_scene().current_program_scene_name != MAIN_SCENE:
        pass
    else:
        while cl.get_current_program_scene().current_program_scene_name == MAIN_SCENE:
            print("Waiting for end of current question")
            time.sleep(1)
    settings_call = cl.get_input_settings(MAIN_MEDIA)
    settings = settings_call.input_settings
    settings["local_file"] = video_path
    cl.set_input_settings(MAIN_MEDIA, settings, True)
    cl.set_current_program_scene(MAIN_SCENE)


def check_queue_size_is_zero():
    return len(open("./" + QUEUE_NAME + ".txt").read()) == 0


def random_vdieo_generator(video_directory_path):
    all_files = [os.path.join(video_directory_path, f) for f in os.listdir(video_directory_path)]
    random.shuffle(all_files)
    for f in all_files:
        if check_queue_size_is_zero():
            yield f
        else:
            return None


def update_random_obs_source(video_directory_path):
    # pass conn info if not in config.toml
    if cl.get_current_program_scene().current_program_scene_name == BACKGROUND_SCENE:
        for new_video in random_vdieo_generator(video_directory_path):
            if new_video is None:
                return
            else:
                while cl.get_media_input_status(BACKGROUND_MEDIA).media_state == "OBS_MEDIA_STATE_PLAYING":
                    time.sleep(1)
                settings_call = cl.get_input_settings(BACKGROUND_MEDIA)
                settings = settings_call.input_settings
                settings["local_file"] = new_video
                cl.set_input_settings(BACKGROUND_MEDIA, settings, True)
                print(f"Switch background to {new_video}")
                return new_video
    else:
        while cl.get_current_program_scene().current_program_scene_name == MAIN_SCENE:
            print("Waiting for end of current question")
            time.sleep(1)


def main_loop():
    background_path = None
    while True:
        for video_path in read_from_queue(QUEUE_NAME, lambda line: line.decode("utf-8").strip().split("|")[0]):
            if video_path is not None:
                print("Changing ", video_path)
                update_obs_source(video_path)
            else:
                background_path = update_random_obs_source(VIDEO_DIRECTORY_PATH)
                if background_path is not None:
                    time.sleep(3)


if __name__ == "__main__":
    main_loop()
