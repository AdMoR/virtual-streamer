import json

import streamlit as st
from llama_index.retrievers.bm25 import BM25Retriever
import Stemmer
from virtual_streamer.utils.utils import (combine_video_and_short_audio, combine_part_in_concat_file,
                                          add_subtitle_from_srt)
from virtual_streamer.utils.subtitle_utils import build_timed_srt
from virtual_streamer.workflows.video_retriever import prepare_nodes, load_json_documents
from gradio_client import Client, handle_file


client = Client("http://localhost:7861/")


def txt_to_speech_call(text):
    reference_fred = '/home/amor/Downloads/FRED ET JAMY FONT TOUT POUR ÊTRE DANS LES TENDANCES YOUTUBE !! [DCf-EI5WgEw]-Scene-017.mp3'
    reference_text_fred = "Là tu vois Jamy, je suis dans le Data Center de Youtube où sont Entreposées des tonnes de vidéos de pranks D'unboxing et aussi les vidéos du Studio Bubble Tea, tu sais le mec qui s'est"
    result = client.predict(
        text=text,
        normalize=False,
        reference_id=None,
        reference_audio=handle_file(reference_fred),
        reference_text=reference_text_fred,
        max_new_tokens=0,
        chunk_length=200,
        top_p=0.7,
        repetition_penalty=1.2,
        temperature=0.7,
        seed=0,
        use_memory_cache="on",
        api_name="/partial"
    )
    audio_path = result[0]
    return audio_path


def process_():
    pass
    """
    preprocess(args, ref_video, ref_video, detector, face_detection_groups)
    print(len(face_detection_groups[ref_video].face_det_results))

    sample = face_detection_groups[ref_video]
    outfile_path = wav2lip_exec(out_folder, audio_path, "Fred se lance dans le twerk", sample)
    outfile_combined_path = f'{out_folder}/result_combined_{i}.mp4'
    combine_video_and_audio(outfile_path, audio_path, outfile_combined_path)
    video_chunks.append(outfile_combined_path)
    """


DEFAULT_SCRIPT = """
    Eh bien, dis donc Jamy, aujourd'hui, j'ai une idée qui va te retourner ton caleçon ! Tu sais quoi ? Je veux lancer ma propre chaîne de twerking sur TikTok ! Eh oui, twerking, le truc où on se frotte le cul sur quelqu'un d'autre, tu sais, comme à nos soirées bien arrosées avec Marcel.
    Hier soir, je me suis enfilé cinq heures de TikTok, et je te jure, Jamy, c'est un véritable festival  du boule ! Tu peux en croire mon instinct du bizness, Jamy, on va se faire du flouze comme à la grande  époque de c'est pas sorcier !
    Pour commencer, j'ai mis les petits plats dans les grands et j'ai  loué la salle des fêtes de Bourg-en-Gonesse. Tu sais Jamy, la grande salle où on a fait la fête pour le Nouvel An de 1998 avec la rédaction de France 3 Poitou charente. 
    Et pour filmer tout ça, j'ai emprunté le Nokia trente trois dix de ma cousine. C'est pas le dernier cri, mais ça fait le job !
    Et tu sais quoi, Jamy ? Avec notre école de twerk, tu vas enfin pouvoir pécho ! Oui, parce que le twerk, ça attire les gonzesses, c'est un fait. 
    Les filles adorent ça, et si tu leur montres tes maquettes, je suis sur qu'une ou deux se laisseront bien ramener au camion.
    Alors, qu'est-ce que t'en penses, Jamy ? 
    """

SEPARATOR = "\n"


@st.cache_resource
def load_retriever(directory_path = "/media/amor/data/Downloads/CPS/clip_infos", who="fred"):
    nodes = prepare_nodes(load_json_documents(directory_path))
    fred_nodes = [n for n in nodes if who == n.metadata["who"]]
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=fred_nodes,
        similarity_top_k=15,
        # Optional: We can pass in the stemmer and set the language for stopwords
        # This is important for removing stopwords and stemming the query + text
        # The default is english for both
        stemmer=Stemmer.Stemmer("french"),
        language="french",
    )
    return bm25_retriever


bm25_retriever = load_retriever()


def build_id(object_type, sentence, extra_index=None):
    str_ = f"{object_type}_{hash(sentence)}"
    if extra_index:
        str_ += f"_{extra_index}"
    return str_

def generate_text():
    st.session_state["llm_result"] = "Coucou. Coucou"


def search_videos(kw):
    retrieved_docs = bm25_retriever.retrieve(kw)
    videos = list()
    for x in retrieved_docs:
        videos.append(x.metadata["path"])
    return videos


def tab1_ui():
    st.title("Text Generation")
    st.text_area(label="Enter the LLM text here", key="llm_result", value=DEFAULT_SCRIPT)
    st.text_area(label="Enter the prompt text here")
    st.button("Generate Text", on_click=generate_text)


# Tab 2: Video Search
def make_search_fn(sentence_id, video_list_id):
    def search():
        kw = st.session_state[sentence_id]
        videos = search_videos(kw)
        st.session_state[video_list_id] = videos
    return search


def tab2_ui():
    st.title("Video Search")
    print(list(st.session_state.keys()))
    generated_sentences = [x for x in st.session_state["llm_result"].split(SEPARATOR) if len(x.replace(" ", "")) > 0]

    for i, sentence in enumerate(generated_sentences):
        with st.expander(sentence):
            keyword_id = build_id("keyword", sentence, i)
            video_list_id = build_id("videolist_", sentence, i)
            st.text_input("Enter a keyword to search a corresponding video", key=keyword_id)
            st.button("Search", key=build_id("button", sentence, i),
                      on_click=make_search_fn(keyword_id, video_list_id))

            if video_list_id in st.session_state:
                # Perform video search based on the keyword
                videos = st.session_state[video_list_id]
                st.selectbox(f"Video for {sentence[:15]}", options=videos,
                             key=build_id("selected_video", sentence, i),
                             format_func=lambda x: videos.index(x))
                for j, video in enumerate(videos):
                    st.subheader(j)
                    st.video(video)

# Tab 3: Audio Generation
def make_audio_gen(sentence_id, audio_id):
    def search():
        sentence = st.session_state[sentence_id]
        audio = txt_to_speech_call(sentence)
        st.session_state[audio_id] = audio
    return search


def generate_all():
    generated_sentences = [x for x in st.session_state["llm_result"].split(SEPARATOR) if len(x.replace(" ", "")) > 0]
    for i, sentence in enumerate(generated_sentences):
        selected_audio_id = build_id("selected_audio", sentence, i)
        audio = txt_to_speech_call(sentence)
        st.session_state[selected_audio_id] = audio


def tab3_ui():
    st.title("Audio Generation")
    generated_sentences = [x for x in st.session_state["llm_result"].split(SEPARATOR) if len(x.replace(" ", "")) > 0]

    st.button("Generate all", on_click=generate_all)

    for i, sentence in enumerate(generated_sentences):
        with st.expander(sentence):
            sentence_id = build_id("sentence", sentence, i)
            st.session_state[sentence_id] = sentence
            video_id = build_id("selected_video", sentence, i)
            if video_id in st.session_state:
                video = st.session_state[video_id]
                st.video(video)
            selected_audio_id = build_id("selected_audio", sentence, i)
            st.button("Generate Audio", key=build_id("audio", sentence, i),
                      on_click=make_audio_gen(sentence_id, selected_audio_id))
            if selected_audio_id in st.session_state:
                st.audio(st.session_state[selected_audio_id])


# Tab 4: Combined Results
def tab4_ui():
    st.title("Combined Results")
    generated_sentences = [x for x in st.session_state["llm_result"].split(SEPARATOR) if len(x.replace(" ", "")) > 0]
    video_chunks = list()

    ready = st.toggle("create final video", value=False)

    if ready:
        for i, sentence in enumerate(generated_sentences):
            selected_audio_id = build_id("selected_audio", sentence, i)
            selected_video_id = build_id("selected_video", sentence, i)
            if selected_audio_id in st.session_state and selected_video_id in st.session_state:
                video = st.session_state[selected_video_id]
                audio = st.session_state[selected_audio_id]
                outfile = f"./temp_{build_id('gen', sentence, i)}.mp4"
                combine_video_and_short_audio(video, audio, outfile)
                outfile_bis = f"./temp_{build_id('gen_sub', sentence, i)}.mp4"
                srt_path = build_timed_srt(sentence, audio, "./temp")
                add_subtitle_from_srt(outfile, srt_path, outfile_bis)
                video_chunks.append(outfile_bis)
            else:
                if selected_audio_id not in st.session_state:
                    st.text(f"No audio for sentence {i}: {sentence}")
                else:
                    st.text(f"No video for sentence {i}: {sentence}")

        if len(video_chunks) == len(generated_sentences):
            final_video = "./final.mp4"
            combine_part_in_concat_file(video_chunks, "./temp.txt", final_video)
            st.video(final_video)

    st.button("Dump state", on_click=lambda : json.dump({k: v for k, v in st.session_state.items() if "selected" in k},
                                                        open("session_dump.json", "w")))



# Main app
def main():
    st.sidebar.title("Navigation")
    tab1, tab2, tab3, tab4 = st.tabs(["Text Generation", "Video Search", "Audio Generation", "Combined Results"])

    with tab1:
        tab1_ui()
    with tab2:
        tab2_ui()
    with tab3:
        tab3_ui()
    with tab4:
        tab4_ui()

if __name__ == "__main__":
    main()

