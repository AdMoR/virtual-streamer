import warnings
import itertools
import os, sys
from rich import print
import string
from spacy.cli import download
import spacy
import os
import json
import pysrt
from aeneas.executetask import ExecuteTask
from aeneas.task import Task



SPACY_MODEL_MAP = {
  "en": 'en_core_web_md',
  "ru": 'ru_core_news_md',
  "fr": 'fr_core_news_md',
  "ja": 'ja_core_news_md',
  "es": 'es_core_news_md',
  "de": 'de_core_news_md',
  "it": 'it_core_news_md',
  "zh": 'zh_core_web_md',
}


def get_spacy_model(language: str):
    model = SPACY_MODEL_MAP.get(language.lower(), "en_core_web_md")
    if language not in SPACY_MODEL_MAP:
        print(f"[yellow]Spacy model does not support '{language}', using en_core_web_md model as fallback...[/yellow]")
    return model

def init_nlp(language):
    try:
        model = get_spacy_model(language)
        print(f"[blue]⏳ Loading NLP Spacy model: <{model}> ...[/blue]")
        try:
            nlp = spacy.load(model)
        except:
            print(f"[yellow]Downloading {model} model...[/yellow]")
            print("[yellow]If download failed, please check your network and try again.[/yellow]")
            download(model)
            nlp = spacy.load(model)
    except:
        raise ValueError(f"❌ Failed to load NLP Spacy model: {model}")
    print(f"[green]✅ NLP Spacy model loaded successfully![/green]")
    return nlp

def analyze_connectors(doc, token):
    """
    Analyze whether a token is a connector that should trigger a sentence split.

    Processing logic and order:
     1. Check if the token is one of the target connectors based on the language.
     2. For 'that' (English), check if it's part of a contraction (e.g., that's, that'll).
     3. For all connectors, check if they function as a specific dependency of a verb or noun.
     4. Default to splitting for certain connectors if no other conditions are met.
     5. For coordinating conjunctions, check if they connect two independent clauses.
    """
    lang = doc.lang_
    if lang == "en":
        connectors = ["that", "which", "where", "when", "because", "but", "and", "or"]
        mark_dep = "mark"
        det_pron_deps = ["det", "pron"]
        verb_pos = "VERB"
        noun_pos = ["NOUN", "PROPN"]
    elif lang == "zh":
        connectors = ["因为", "所以", "但是", "而且", "虽然", "如果", "即使", "尽管"]
        mark_dep = "mark"
        det_pron_deps = ["det", "pron"]
        verb_pos = "VERB"
        noun_pos = ["NOUN", "PROPN"]
    elif lang == "ja":
        connectors = ["けれども", "しかし", "だから", "それで", "ので", "のに", "ため"]
        mark_dep = "mark"
        det_pron_deps = ["case"]
        verb_pos = "VERB"
        noun_pos = ["NOUN", "PROPN"]
    elif lang == "fr":
        connectors = ["que", "qui", "où", "quand", "parce que", "mais", "et", "ou"]
        mark_dep = "mark"
        det_pron_deps = ["det", "pron"]
        verb_pos = "VERB"
        noun_pos = ["NOUN", "PROPN"]
    elif lang == "ru":
        connectors = ["что", "который", "где", "когда", "потому что", "но", "и", "или"]
        mark_dep = "mark"
        det_pron_deps = ["det"]
        verb_pos = "VERB"
        noun_pos = ["NOUN", "PROPN"]
    elif lang == "es":
        connectors = ["que", "cual", "donde", "cuando", "porque", "pero", "y", "o"]
        mark_dep = "mark"
        det_pron_deps = ["det", "pron"]
        verb_pos = "VERB"
        noun_pos = ["NOUN", "PROPN"]
    elif lang == "de":
        connectors = ["dass", "welche", "wo", "wann", "weil", "aber", "und", "oder"]
        mark_dep = "mark"
        det_pron_deps = ["det", "pron"]
        verb_pos = "VERB"
        noun_pos = ["NOUN", "PROPN"]
    elif lang == "it":
        connectors = ["che", "quale", "dove", "quando", "perché", "ma", "e", "o"]
        mark_dep = "mark"
        det_pron_deps = ["det", "pron"]
        verb_pos = "VERB"
        noun_pos = ["NOUN", "PROPN"]
    else:
        return False, False

    if token.text.lower() not in connectors:
        return False, False

    if lang == "en" and token.text.lower() == "that":
        if token.dep_ == mark_dep and token.head.pos_ == verb_pos:
            return True, False
        else:
            return False, False
    elif token.dep_ in det_pron_deps and token.head.pos_ in noun_pos:
        return False, False
    else:
        return True, False


def split_by_connectors(text, context_words=5, nlp=None):
    doc = nlp(text)
    sentences = [doc.text]  # init

    while True:
        # Handle each task with a single cut
        # avoiding the fragmentation of a sentence into multiple parts at the same time.
        split_occurred = False
        new_sentences = []

        for sent in sentences:
            doc = nlp(sent)
            start = 0

            for i, token in enumerate(doc):
                split_before, _ = analyze_connectors(doc, token)

                if i + 1 < len(doc) and doc[i + 1].text in ["'s", "'re", "'ve", "'ll", "'d"]:
                    continue

                left_words = doc[max(0, token.i - context_words):token.i]
                right_words = doc[token.i + 1:min(len(doc), token.i + context_words + 1)]

                left_words = [word.text for word in left_words if not word.is_punct]
                right_words = [word.text for word in right_words if not word.is_punct]

                if len(left_words) >= context_words and len(right_words) >= context_words and split_before:
                    print(
                        f"[yellow]✂️  Split before '{token.text}': {' '.join(left_words)}| {token.text} {' '.join(right_words)}[/yellow]")
                    new_sentences.append(doc[start:token.i].text.strip())
                    start = token.i
                    split_occurred = True
                    break

            if start < len(doc):
                new_sentences.append(doc[start:].text.strip())

        if not split_occurred:
            break

        sentences = new_sentences

    return sentences


def split_sentences_main(sentences, nlp):
    all_split_sentences = []
    # Process each input sentence
    for sentence in sentences:
        split_sentences = split_by_connectors(sentence.strip(), nlp=nlp)
        all_split_sentences.extend(split_sentences)

    # output to sentence_splitbyconnector.txt
    rez = list()
    for sentence in all_split_sentences:
        print("sentence ", sentence)
        if len(sentence) == 0:
            continue
        if sentence[0] in string.punctuation:
            print("-------------")
            sentence = sentence[2:]
        rez.append(sentence)

    return rez


def is_valid_phrase(phrase):
    # 🔍 Check for subject and verb
    has_subject = any(token.dep_ in ["nsubj", "nsubjpass"] or token.pos_ == "PRON" for token in phrase)
    has_verb = any((token.pos_ == "VERB" or token.pos_ == 'AUX') for token in phrase)
    return (has_subject and has_verb)


def analyze_comma(start, doc, token):
    left_phrase = doc[max(start, token.i - 9):token.i]
    right_phrase = doc[token.i + 1:min(len(doc), token.i + 10)]

    suitable_for_splitting = is_valid_phrase(
        right_phrase)  # and is_valid_phrase(left_phrase) # ! no need to chekc left phrase

    # 🚫 Remove punctuation and check word count
    left_words = [t for t in left_phrase if not t.is_punct]
    right_words = list(
        itertools.takewhile(lambda t: not t.is_punct, right_phrase))  # ! only check the first part of the right phrase

    if len(left_words) <= 3 or len(right_words) <= 3:
        suitable_for_splitting = False

    return suitable_for_splitting


def split_by_comma(text, nlp):
    doc = nlp(text)
    sentences = []
    start = 0

    for i, token in enumerate(doc):
        if token.text.strip(" ") in [",", "，"]:
            suitable_for_splitting = analyze_comma(start, doc, token)

            if suitable_for_splitting:
                sentences.append(doc[start:token.i].text.strip())
                print(f"[yellow]✂️  Split at comma : {doc[start:token.i][-4:]},| {doc[token.i + 1:][:4]} [/yellow]")
                start = token.i + 1

    for i, token in enumerate(doc):
        if token.text == ":":  # Split at colon
            sentences.append(doc[start:token.i].text.strip())
            print(f"[yellow]✂️  Split at colon: {doc[start:token.i][-4:]}:| {doc[token.i + 1:][:4]}[/yellow]")

    sentences.append(doc[start:].text.strip())
    return sentences


def split_by_comma_main(sentences, nlp):
    all_split_sentences = []
    for sentence in sentences:
        split_sentences = split_by_comma(sentence.strip(), nlp)
        all_split_sentences.extend(split_sentences)

    return all_split_sentences


def split_by_mark(input_text, nlp, language):
    doc = nlp(input_text)
    assert doc.has_annotation("SENT_START")

    sentences_by_mark = [sent.text for sent in doc.sents]
    rez = list()
    buffer = ""
    for i, sentence in enumerate(sentences_by_mark):
        if sentence.strip() in [',', '.', '，', '。', '？', '！']:
            # ! If the current line contains only punctuation, merge it with the previous line, this happens in Chinese, Japanese, etc.
            # rez[i-1] += sentence # Add the punctuation
            continue
        else:
            rez.append(sentence)
    return rez


def split_long_sentence(doc, language):
    tokens = [token.text for token in doc]
    n = len(tokens)

    # dynamic programming array, dp[i] represents the optimal split scheme from the start to the ith token
    dp = [float('inf')] * (n + 1)
    dp[0] = 0

    # record optimal split points
    prev = [0] * (n + 1)

    for i in range(1, n + 1):
        for j in range(max(0, i - 100), i):  # limit search range to avoid overly long sentences
            if i - j >= 30:  # ensure sentence length is at least 30
                token = doc[i - 1]
                if j == 0 or (token.is_sent_end or token.pos_ in ['VERB', 'AUX'] or token.dep_ == 'ROOT'):
                    if dp[j] + 1 < dp[i]:
                        dp[i] = dp[j] + 1
                        prev[i] = j

    # rebuild sentences based on optimal split points
    sentences = []
    i = n
    joiner = get_joiner(language)
    while i > 0:
        j = prev[i]
        sentences.append(joiner.join(tokens[j:i]).strip())
        i = j

    return sentences[::-1]  # reverse list to keep original order


def split_extremely_long_sentence(doc, language):
    tokens = [token.text for token in doc]
    n = len(tokens)

    num_parts = (n + 59) // 60  # round up

    part_length = n // num_parts

    sentences = []
    joiner = get_joiner(language)
    for i in range(num_parts):
        start = i * part_length
        end = start + part_length if i < num_parts - 1 else n
        sentence = joiner.join(tokens[start:end])
        sentences.append(sentence)

    return sentences


def split_long_by_root_main(sentences, nlp):
    all_split_sentences = []
    for sentence in sentences:
        doc = nlp(sentence.strip())
        if len(doc) > 60:
            split_sentences = split_long_sentence(doc)
            if any(len(nlp(sent)) > 60 for sent in split_sentences):
                split_sentences = [subsent for sent in split_sentences for subsent in
                                   split_extremely_long_sentence(nlp(sent))]
            all_split_sentences.extend(split_sentences)
            print(f"[yellow]✂️  Splitting long sentences by root: {sentence[:30]}...[/yellow]")
        else:
            all_split_sentences.append(sentence.strip())

    punctuation = string.punctuation + "'" + '"'  # include all punctuation and apostrophe ' and "
    rez = list()
    for i, sentence in enumerate(all_split_sentences):
        stripped_sentence = sentence.strip()
        if not stripped_sentence or all(char in punctuation for char in stripped_sentence):
            print(f"[yellow]⚠️  Warning: Empty or punctuation-only line detected at index {i}[/yellow]")
            if i > 0:
                all_split_sentences[i - 1] += sentence
            continue
        rez.append(sentence)
    return rez


def decompose_text(input_text, nlp):
    sentences = [s for s in input_text.split(".") if len(s) > 0]
    sentences = split_sentences_main(sentences, nlp)
    sentences = split_by_comma_main(sentences, nlp)
    sentences = split_long_by_root_main(sentences, nlp)
    return sentences


def seconds_to_hmsm(seconds):
    seconds = float(seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    milliseconds = int(seconds * 1000) % 1000
    return f"{hours:02d}:{minutes:02d}:{int(seconds):02d},{milliseconds:03d}"


def align_text_audio(audio_path, text_path, output_json_path):
    config_string = "task_language=eng|is_text_type=plain|os_task_file_format=json"
    task = Task(config_string)
    task.audio_file_path_absolute = os.path.abspath(audio_path)
    task.text_file_path_absolute = os.path.abspath(text_path)
    task.sync_map_file_path_absolute = os.path.abspath(output_json_path)

    ExecuteTask(task).execute()
    task.output_sync_map_file()


def json_to_srt(sync_map_path, srt_path):
    with open(sync_map_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    subs = pysrt.SubRipFile()
    for idx, fragment in enumerate(data['fragments']):
        start = fragment['begin']
        end = fragment['end']
        text = fragment['lines'][0]
        start_time = seconds_to_hmsm(start)
        end_time = seconds_to_hmsm(end)
        subs.append(pysrt.SubRipItem(index=idx+1, start=start_time, end=end_time, text=text))
    subs.save(srt_path, encoding='utf-8')


nlp = None


def build_timed_srt(text, audio_file, temp_dir):
    out_text_file = f"{temp_dir}/transcript.txt"
    sync_map = f"{temp_dir}/sync_map.json"
    srt_path = f"{temp_dir}/sync.srt"

    language = "fr"

    global nlp
    if nlp is None:
        nlp = init_nlp(language)

    sentences = decompose_text(text, nlp)
    with open(out_text_file, "w") as f:
        f.write('\n'.join(sentences))
    align_text_audio(audio_file, out_text_file, sync_map)
    json_to_srt(sync_map, srt_path)
    return srt_path


"""
USAGE : 

# Example usage
text = "Eh dis donc, Jamy ! Figure-toi que j'étais à Rennes et j'ai eu une illumination après m'être pris un sacré coup de bambou avec trois canettes d'énergie. Alors Jamy, écoute-moi bien. On va lancer la 'Fred's kick', la boisson qui va déchirer tous ces petits machins marketing à deux balles genre Red Bull et Monster."
audio_file = "/home/amor/Downloads/audio.wav"

out_text_file = "/home/amor/Documents/code_dw/virtual-streamer/transcript.txt"
video_file = "/home/amor/Documents/code_dw/virtual-streamer/F&J_episode_1.mp4"

sync_map = "./sync_map.json"
srt_path = "./sync.srt"

language = "fr"
nlp = init_nlp(language)

sentences = decompose_text(text, nlp)
with open(out_text_file, "w") as f:
    f.write('\n'.join(sentences))
align_text_audio(audio_file, text_file, sync_map)
json_to_srt(sync_map, srt_path)
add_subtitle(video_file, srt_path, "./temp.mp4")
"""