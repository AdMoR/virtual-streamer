import json
import random
import streamlit as st
from llama_index.retrievers.bm25 import BM25Retriever
import Stemmer
from virtual_streamer.utils.utils import (combine_video_and_short_audio, combine_part_in_concat_file,
                                          add_subtitle_from_srt,)
from virtual_streamer.workflows.video_retriever import prepare_nodes, load_json_documents, prepare_nodes_v2
from gradio_client import Client, handle_file
import stable_whisper
from llama_index.llms.openai import OpenAI
from llama_index.llms.anthropic import Anthropic
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader


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

DEFAULT_PROMPT = """
    Tu es Fred de l'émission "C'est pas Sorcier". Cela fait 20 ans que l'émission s'est terminée. 
    Et avec Jamy, tu continues à expliquer des choses à Jamy et au téléspectateurs.

    Voici quelques examples de ton expression : 
     Alors, qu'est-ce qu'on peut acheter avec une crypto-monnaie, Jamy? Eh bien, je me suis procuré un petit bijou de technologie, un téléphone à clapet dont la batterie ne va pas me chier dans les bottes d'ici 5 mois. Nique l'obsolescence programmée et surtout nique Apple, Jamy. 
     Ça, c'est pas le genre de truc que tu peux trouver à l'Easy Cash d'Argenteuil, Jamy. C'est moi qui te le dis. 
     Eh dis donc, Jamy, je viens de vérifier nos comptes. On a tout ce qu'il faut pour s'installer à Dubaï et manger du steak de tigre. 
     Et dis donc Jamy, c'est super simple de créer de l'argent en France. Qui c'est qui a dit qu'on n'avait pas la possibilité de se faire de l'argent sur le dos des gogos? 
     Ah, bon Jamy, je perds pas espoir. Je vais aller checker les privilèges des autres cheminots. J'ai lu dans le Figaro qu'ils vivaient en meute le long des trains, un peu comme les musaraignes. Je vais bien finir par en trouver un qui va répondre à nos questions, Jamy. 
     Alors j'imagine, Jamy, que tu vas me demander mais comment as-tu pu trouver la thune pour t'acheter une centrale nucléaire? Eh bien, c'est très très simple, Jamy. C'est parce que c'est une centrale nucléaire en kit. Tu connais les éditions Altaïa? Eh bien, ils font pas que des maquettes de bateaux ou des figurines. Ils font aussi des centrales nucléaires. Et ça ne m'a coûté que 180 927 euros. 
     Je vais certainement être assassiné par le FBI ou la bande à Picsou parce que j'ai révélé toutes les vérités que les gens ne veulent pas entendre, Jamy.
     C'est vrai que vous rigolez au sketch de Tomer Sisley? Figure-toi, Jamy, qu'on n'est jamais allé sur la Lune. Eh oui, c'est Luc Besson qui a filmé le Luberon en pleine nuit, Jamy. 
     Ça coûte un RSA. C'est pas cher, finalement. Résultat, je n'ai jamais eu de cancer du col de l'utérus. 
     Sinon Jamy, il n'y a pas que le dropshipping, on peut aussi vendre des formations. Alors Jamy, je sais ce que tu vas me demander, qu'est-ce qu'une formation? Eh bien en fait, pour résumer, c'est une série de vidéos ou de PDF où on dit des choses qui font semblant d'être profondes tout en étant complètement débiles. 
     Eh dis donc, Jamy, aujourd'hui, on va arrêter d'être des bolosses de YouTube Game avec 3 abonnés et 12 vues. On va essayer de passer en tendance. 
     Tu vois, Jamy, il y a un proverbe du Wakanda qui dit que la lionne ne copule pas avec le lion, surtout s'il a une petite bite, c'est pour ça qu'on va s'acheter les abonnés Jamy, et c'est aussi ici qu'est entreposée ma petite vidéo là regarde, elle est juste là, fais coucou à la caméra, virtuelle, un peu comme les amis de Manuel Valls Jamy, quoi c'est tout? 23 vues? Ah putain Jamy, ça valait pas le coup de se casser le cul juste pour ce résultat-là. 
     Ah, alors moi, ce que je te propose, Jamy, c'est que pour totalement disrupter le réchauffement climatique, eh ben, c'est qu'on aille chercher la glace contenue dans les glaciers. Comme ça, on la met chez nous dans un grand frigo, et du coup, on résout le problème du réchauffement climatique. Mais ça m'a l'air un peu haut quand même, Jamy, tu trouves pas? 
     Eh dis donc, Jamy, j'étais tellement stressé cette nuit que j'ai même pas changé de vêtements. Du coup, je pue la transpiration. Mais c'est pas grave, on va voir ce qu'est devenu notre vidéo. 
     Et pour réussir une bonne vie, on s'y croirait. Avec moi Sabine et Marcel dans notre camion, je te propose, Jamy, qu'on devienne riche et pour ça, 
     Dis donc Jamy, 2018 c'est l'année de la bite non? Alors on va acheter des bitcoins. 

     Tu es emploie un langage fleuri et tu n'as pas peur d'utiliser des mots vulgaires pour te rapprocher de tes jeunes lecteurs.
     Tu fais souvent références à des localités pour rendre ton récits plus réel.
     Tu t'adresse à Jamy directement.

     Maintenant génère une présentation sur le thème suivant : 
     Thème : Fred veut trouver le trésor des templiers
     Les éléments de scénario suivant doivent apparaitre : 
     {story}
    """


SEPARATOR = "."


@st.cache_resource
def load_retriever_bm25(directory_path = "/media/amor/data/Downloads/CPS/clip_infos", who="fred"):
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

@st.cache_resource
def load_retriever(directory_path = "/media/amor/data/Downloads/CPS/clip_infos", who="fred"):
    embed_model = HuggingFaceEmbedding(model_name="lightonai/modernbert-embed-large")
    Settings.embed_model = embed_model

    # storage_context=storage_context
    nodes = prepare_nodes_v2(load_json_documents(directory_path))
    fred_nodes = [n for n in nodes if who == n.metadata["who"]]
    index = VectorStoreIndex(fred_nodes)
    index.storage_context.persist("/media/amor/data/Downloads/CPS/vector_store")
    retriever = index.as_retriever(verbose=True, similarity_top_k=5)
    return retriever


@st.cache_resource
def load_transcripter():
    return stable_whisper.load_hf_whisper('large-v3', batch_size=4)


@st.cache_resource
def load_llm():
    return Anthropic(model="claude-3-5-haiku-20241022")


bm25_retriever = load_retriever()
model = load_transcripter()
llm = load_llm()
DEFAULT_LENGTH = 35


def separation_fn(raw_text, max_length=DEFAULT_LENGTH):
    def split(txt, separator):
        return [x for x in txt.split(separator) if len(x.replace(" ", "")) > 0]
    parts = list()
    for p in split(raw_text, "\n"):
        if len(p) > max_length:
            broken_down = False
            for sep in [".", "!", "?"]:
                sub_parts = split(p, sep)
                if len(sub_parts) > 1:
                    broken_down = True
                    parts.extend(sub_parts)
                    break
            if not broken_down:
                parts.append(p)
        else:
            parts.append(p)
    return parts


def build_id(object_type, sentence, extra_index=None):
    str_ = f"{object_type}_{hash(sentence)}"
    if extra_index:
        str_ += f"_{extra_index}"
    return str_


def generate_text():
    PROMPT = st.session_state["prompt"]
    rez = llm.complete(PROMPT).text
    st.session_state["llm_result"] = rez


def search_videos(kw):
    retrieved_docs = bm25_retriever.retrieve(kw)
    videos = list()
    for x in retrieved_docs:
        videos.append(x.metadata["path"])
    return videos


def tab1_ui():
    st.title("Text Generation")
    st.text_area(label="LLM result here", key="llm_result", height=500, value=DEFAULT_SCRIPT)
    st.text_area(label="Prompt text here", key="prompt", height=800, value=DEFAULT_PROMPT)
    st.button("Generate Text", on_click=generate_text)


# Tab 2: Video Search
def make_search_fn(sentence_id, video_list_id):
    def search():
        kw = st.session_state[sentence_id]
        videos = search_videos(kw)
        st.session_state[video_list_id] = videos
    return search


def default_selection():
    generated_sentences = st.session_state["sentences"]

    for i, sentence in enumerate(generated_sentences):
        selected = build_id("selected_video", sentence, i)
        video_list_id = build_id("videolist_", sentence, i)
        keyword_id = build_id("keyword", sentence, i)
        videos = search_videos(sentence)
        st.session_state[keyword_id] = sentence
        st.session_state[video_list_id] = videos
        st.session_state[selected] = videos[random.choice(list(range(15)))]

def compute_generated_sentences():
    generated_sentences = separation_fn(st.session_state["llm_result"],
                                        st.session_state.get("max_length", DEFAULT_LENGTH))
    st.session_state["sentences"] = generated_sentences


def tab2_ui():
    st.title("Video Search")
    st.slider("max sentence length", 40, 80, DEFAULT_LENGTH, key="max_length", on_change=compute_generated_sentences)
    print(list(st.session_state.keys()))
    st.button("Default selection", key=build_id("button", "random"),
              on_click=default_selection)

    if "sentences" not in st.session_state:
        compute_generated_sentences()
    generated_sentences = st.session_state["sentences"]

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
    if "sentences" not in st.session_state:
        compute_generated_sentences()
    generated_sentences = st.session_state["sentences"]
    for i, sentence in enumerate(generated_sentences):
        selected_audio_id = build_id("selected_audio", sentence, i)
        audio = txt_to_speech_call(sentence)
        st.session_state[selected_audio_id] = audio


def tab3_ui():
    st.title("Audio Generation")
    if "sentences" not in st.session_state:
        compute_generated_sentences()
    generated_sentences = st.session_state["sentences"]

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
    if "sentences" not in st.session_state:
        compute_generated_sentences()
    generated_sentences = st.session_state["sentences"]
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
                srt_path = f"./{i}.srt"
                result = model.transcribe(audio)
                result.to_srt_vtt(srt_path)
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

