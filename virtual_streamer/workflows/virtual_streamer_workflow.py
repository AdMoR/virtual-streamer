import datetime
import time
from prompts import PROMPT, PROMPT_FR, PROMPT_FR_3, PROMPT_FR_2, SARCASTIC_PROMPT_FR, \
    STAND_UP_PROMPT, SARCASTIC_STANDUP, VERY_SARCASTIC_STANDUP_PROMPT, VERY_SARCASTIC_PROMPT
import random
import shutil
from llama_index.core.workflow import (
    Workflow,
    step,
    Event,
    Context,
    StartEvent,
    StopEvent
)
from llama_index.llms.openai import OpenAI
from virtual_streamer.wav2lip.main_logic import FaceDetectionGroup
from virtual_streamer.utils.utils import *


class ResponseEvent(Event):
    response: str


class VideoEvent(Event):
    video_path: str
    response: str


class VirtualStreamerWorkflow(Workflow):

    llm = OpenAI(model="gpt-4.1-mini", temperature=0.1)
    wav2lip_fn: Callable[[str, str, str, FaceDetectionGroup], str] = None
    txt_to_speech_call: Callable[[str, str, str], str] = None

    @step
    def generate_answer(self, ctx: Context, ev: StartEvent) -> ResponseEvent:
        """
        TODO : correct parameter unpacking
        to get : full_frames, face_det_results
        set in global context
        """
        question: Question = ev.question
        ctx.set("face_det_group", ev.face_det_group)
        ctx.set("question", question)
        # Step 1 - Get the response from GPT4o
        if question.prompt is None:
            question.prompt = random.choice([SARCASTIC_STANDUP, VERY_SARCASTIC_STANDUP_PROMPT, VERY_SARCASTIC_PROMPT])
        query = question.render()
        completion = self.llm.complete(query)
        text = completion.text
        json.dump({"query": query, "response": text, "question": question.question},
                  open(f"prompts/response_{hash(query) % 1000000}.json", "w"))
        return ResponseEvent(response=text)

    @step
    def answer_to_lip_sync(self, ctx: Context, ev: ResponseEvent) -> VideoEvent:
        """
        """
        # Step 1 : Resume
        # Param retrieval
        question = ctx["question"]
        face_det_group = ctx["face_det_group"]
        text = ev.response

        # Folder
        dirname = os.environ.get("OUT_VIDEO_FOLDER", "./out_video_folder")
        os.makedirs(dirname, exist_ok=True)
        TEMP_DIR = "./temp"
        os.makedirs(TEMP_DIR, exist_ok=True)

        # Step 2 - Get the audio for the response
        # prev p317
        audio_outpath = self.txt_to_speech_call(text, "male-pt-3%0A",
                                                f"{TEMP_DIR}/response_{hash(question.render()) % 100000}.wav")
        # character = CHARACTERS[question.character_name]
        # solero_language_switch(character.language, character.voice)
        # audio_outpath = txt_to_speech_call_solero(text, character.language,
        #                                          character.voice, f"./temp/response_{hash(query) % 100000}.wav")

        # Step 3 - Wav2lip video generation
        s = time.time()
        outfile_path = self.wave2lip_fn(TEMP_DIR, audio_outpath, question.question, face_det_group)
        print("wav2lip prediction time:", time.time() - s)

        # Step 4 - Recombination and add subtitles
        tag = str(datetime.datetime.now()).replace(" ", "-") + sanitize_str(question.question[:30])
        outfile_combined_path = f'{TEMP_DIR}/result_combined_{tag}.mp4'
        combine_video_and_audio(outfile_path, audio_outpath, outfile_combined_path)

        # Opt 4 : add subtitles
        if question.subtitle_mode == SubtitleMode.QUESTION:
            subtitle = f"Question de {question.name} : {question.question}"
        elif question.subtitle_mode == SubtitleMode.VOICE_SUBTITLE:
            subtitle = text
        if question.subtitle_mode in [SubtitleMode.QUESTION, SubtitleMode.VOICE_SUBTITLE]:
            outfile_titled_path = f'{TEMP_DIR}/result_titled_{tag}.mp4'
            add_subtitle(subtitle, outfile_combined_path, outfile_titled_path)
        else:
            outfile_titled_path = outfile_combined_path

        # Step 5 - Move the file
        final_outfile_path = os.path.join(dirname, f"result_{tag}.mp4")
        shutil.copyfile(outfile_titled_path, final_outfile_path)

        return VideoEvent(video_path=final_outfile_path, response=text)
