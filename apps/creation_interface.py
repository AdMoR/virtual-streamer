import json
import random
import streamlit as st
from llama_index.retrievers.bm25 import BM25Retriever
import Stemmer
from virtual_streamer.utils.utils import (
    combine_video_and_short_audio,
    combine_part_in_concat_file,
    add_subtitle_from_srt,
)
from virtual_streamer.workflows.video_retriever import (
    prepare_nodes,
    load_json_documents,
    prepare_nodes_v2,
)
from virtual_streamer.utils.utils import txt_to_speech_call_fish
import stable_whisper
from litellm import completion
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
import anthropic
import os
import cv2
import base64
from enum import Enum
from typing import Optional
from pydantic import BaseModel


class ContextualRating(str, Enum):
    CONTEXTUAL = "CONTEXTUAL"
    NEUTRAL = "NEUTRAL"
    NOT_CONTEXTUAL = "NOT_CONTEXTUAL"


class VideoDialogueJudgement(BaseModel):
    rating: ContextualRating
    grade: int
    reasoning: str


def txt_to_speech_call(text):
    reference_fred = "/home/amor/Downloads/FRED ET JAMY FONT TOUT POUR ÊTRE DANS LES TENDANCES YOUTUBE !! [DCf-EI5WgEw]-Scene-017.mp4"
    reference_text_fred = "Là tu vois Jamy, je suis dans le Data Center de Youtube où sont Entreposées des tonnes de vidéos de pranks D'unboxing et aussi les vidéos du Studio Bubble Tea, tu sais le mec qui s'est"
    audio_path = txt_to_speech_call_fish(
        speech_lines=text,
        reference_audio=reference_fred,
        reference_text=reference_text_fred,
        host="127.0.0.1",
        port=8003,
    )
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
Eh dis donc Jamy, tu savais que j'ai reçu un mail de mon gars Elon Musk qui me demandait des conseils sur son truc d'IA ? Bon, j'ai pas répondu lol, mais ça m'a donné une idée de génie : on lance FredGPT, notre propre modèle de langage souverain, 100% français, 100% couillu !

Fred : Et On va l'entraîner exclusivement sur les archives de C'est pas Sorcier, les transcriptions de Question pour un champion, et mes notes personnelles depuis 1994, donc tu imagines la puissance de feu qu'on va avoir face à ces petits couillons de la Silicon Valley qui connaissent rien à la vulgarisation scientifique comme nous !

Fred : Pour les serveurs, ne t'inquiète pas Jamy, j'en ai trouvé sur le bon coin.
Le truc dingue, c'est qu'on va les baser à Limoges, dans la cave de ma tata Irène. 
Et attends, c'est pas tout Jamy, l'innovation c'est que FredGPT refusera catégoriquement de répondre en anglais, ça t'apprendra à pas répondre à mes coup de fil Obama

Fred : Jamy, franchement, dans six mois on aura révolutionné l'IA mondiale, les Américains vont pleurer, et on va être les deux mecs les plus riches de France — enfin, après avoir remboursé la cave de Limoges.

    """

DEFAULT_PROMPT = """You are a designer of a humorous parody of C'est pas sorcier the French Science discovery show.
This humorous version s set in the present days, 20 years after the last airing of C'est pas Sorcier and Fred and Jamy the 2 main presenters of the show are still working together.
Fred is always the main actor of the story 

Write a humorous parody in the style of the French educational TV show "C'est pas Sorcier," featuring the characters Fred and Jamy. The story should follow this structure:                                                

Core Elements:                                                                                                                                                                                                             

 • Fred's Character: An overconfident, grandiose entrepreneur from the 1990s who constantly pitches half-baked business ideas with unwarranted certainty. He frequently references past glories from the show's heyday and 
   uses casual French slang ("flouze," "pécho," "kiffer," "bizness").                                                                                                                                                      
 • Jamy's Role: A passive, skeptical listener who serves as the straight man, receiving Fred's wild ideas without much pushback or resistance. As this is mainly a Fred's monologue, a Jamy dialogue line could only be at the start to launch Fred on a topic or at the end to acknowledge or be surprised.  Jamy does not speak in the dialogue more than ONCE !                                        

The dialogues are only dialogue lines and cannot contain descriptive details that are not verbally spoken.


Story Arc:                                                                                                                                                                                                                 

 1 Fred discovers something new  and pitches an idea to                 
                                                                                                                         
 2 The idea escalates into increasingly ridiculous territory with specific, oddly concrete details                                                                                                                         
 3 Fred justifies the absurdity with pseudo-scientific reasoning or faux-logic borrowed from the show's educational format                                                                                                 
 4 Include nostalgic callbacks to 1990s like shows (ex: la roue de la fortune, Question pour un Champion), specific locations (ex: Bourg-en-Gonesse, Rennes), or dated technology (ex: Nokia 3310) ==> please do not re-use these specific examples but extrapolate from them.
 4 bis reference can sometimes mention current events (current president in France or US, well known internet celebrities, etc) but mostly for a roasting joke. Use them more seldomly (like one per scenario)                                                                                    
 
5 The humor maintains affectionate absurdity rather than meanness           
6 The character don't explain their joke. 

7 The story should be fast paced, one idea per sentence. The story finished in 6 sentences.
                                                                                                               

Tone elements :                                                                                                                                                                                                                      

• Fred is overly excited by his brand new idea and overlook the absurdity of his idea. The text is told as Fred is speaking to Jamy, but the video is seen from Jamy's point of view, Fred speaks to the camera while showing things. 
• Fred may explain how combining their strengths  would yield an incredible advantage in the newly found endavor
•  Fred has a very casual language and can swear or be mean to illustrate better his ideas
 Ex: On va lancer la "Fred's kick", la boisson qui va déchirer tous ces petits machins marketing à deux balles genre Red Bull et Monster. 
Reasoning : "à deux balles" indicates a judgement of the value of Redbull which is a leader than thus pushes the idea of Fred being overly confident.
• Fred feels superior to the majority of the other people, even the most talented, he will often put himself and Jamy over the rest of the poulation
Ex: "Pour des pro de l'audiovisuel comme nous, ça devrait être se faire une main dans le slip"
 • Use authentic French cultural references and slang
Scenario : Fred se lance dans la mannequinat
Ex: Dis donc Jamy, tu savais que c'est sur ma silouhette qu'ils ont modélé Footix en 1998 ?                                                                                                     Reasoning : absurd reference because Footix is a rooster mascot but precise that everybody knows
•  Fred and Jamy are talking together and thus admits their common secrets which are funny to the viewer. This version of Fred and Jamy is very unhinged. 
Scenario : Fred démarre une chaine de twerking 
Line : Eh oui, twerking, le truc où on se frotte le cul sur quelqu'un d'autre, tu sais, comme à nos soirées bien arrosées avec Marcel.
Reasoning : "comme à nos soirées bien arrosées avec Marcel" let the user imagine that the character have a devious life but a fun one nonetheless which breaks the contrast with the seriousness of the show                                                                                                                                         
 • Make it accessible to both nostalgic fans and newcomers            
•  Avoid out of context reference : 
Scénario : Fred se prépare pour les Jeux Olympiques
Text : 
j'ai calculé qu'il me faut exactement 47 barres protéinées par jour, soit une de plus que le nombre de départements français, parce que c'est un chiffre qui porte chance depuis que j'ai gagné à Motus en 1998
Review: not ideal
Reasoning : 
"c'est un chiffre qui porte chance depuis que j'ai gagné à Motus en 1998" is out of context is not related to the core of the character and does not bring value to the first part of the story 


Generation methodology : 
First : describe the scenario that Fred will follow
Ex: 
Fred drank a first RedBull last week and is a fan of it. 
But he thinks he is better than the RedBull team and wants to create his own brand.
As Fred is very french focused, it may be produced in a very unnown part of france
As Fred wants to be better he will want to more more of everything in the drink, including caffeine, which would be dangerous in real life, but in this fictious parody the idea will be funy

This part should be very detailed in order for the next part to only focus on the tone adjustement


Second : write the lines of text based on the scenario above
Ex: 
Eh dis donc, Jamy ! Figure-toi que j'étais chez ma mémé la semaine dernière, j'ai eu une illumination après m'être pris un sacré coup de bambou avec trois canettes d'énergie.
Je vais lancer ma boisson à mon nom : la Fred's kick

Separate the two parts by a line like this : -----------------------------------------


Other rules : 
This script will be used to generate a humoristic video. Each entry like "Fred: ......" will be used for a single sequence.
A sequence should convey a single idea, where Fred is doing a single Action.
Ex: 
Fred : Putain Jamy, hier soir, je me rematais notre épisode spécial noël 97 sur la moutarde de Dijon, un vrai bangeur
Visual : Fred is sitting at his dining table
Fred : Et c'est la que ça m'a frappé, aucun resto digne de ce nom sert de la vraie bouffe ! 
Visual : zoom on fred face with a tense expression


Here is a few examples with a critique attached 
```
Generation : 


SCÉNARIO : Fred se prépare pour les Jeux Olympiques
Contexte détaillé :
Fred a regardé les Jeux Olympiques à la télé et s'est convaincu qu'avec sa silhouette de présentateur télé des années 90 (légèrement ramollie par 20 ans de vie sédentaire), il pourrait devenir athlète olympique. Il pense sincèrement que son expérience de terrain avec C'est pas Sorcier lui donne une compréhension scientifique supérieure du corps humain que les vrais entraîneurs.
Fred va décider de s'entraîner pour une discipline olympique absurde (peut-être le pentathlon moderne ou le tir à l'arc, quelque chose d'improbable). Il va combiner son approche "scientifique" de la vulgarisation avec des méthodes d'entraînement complètement farfelues. Il va justifier chaque décision avec une pseudo-logique empruntée au format éducatif du show. Il va aussi se comparer favorablement aux athlètes professionnels, pensant que son charisme télévisuel lui donne un avantage injuste.
Les détails vont devenir de plus en plus concrets et absurdes : des régimes spécifiques basés sur des calculs farfelus, des équipements bricolés, des entraînements dans des lieux improbables de France profonde, des références nostalgiques aux années 90.

Eh dis donc Jamy, tu sais que j'ai décidé de me présenter aux Jeux Olympiques en pentathlon moderne ?
Ouais, ouais, je sais ce que tu penses, mais écoute : avec notre expérience de terrain, on a une compréhension du corps humain que ces petits entraîneurs à la con n'auront jamais, donc j'ai calculé qu'il me faut exactement 47 barres protéinées par jour, une par département français, parce que c'est un chiffre qui porte chance depuis que j'ai gagné à Motus en 1998... non attends, je veux dire, c'est mathématiquement optimal pour la reconstruction musculaire.
Je vais m'entraîner au tir à l'arc dans une carrière abandonnée près de Châteauroux, parce que l'acoustique là-bas va me permettre d'entendre le bruit exact de la flèche qui fend l'air, ce qui va développer mon instinct de prédateur mieux que n'importe quel simulateur high-tech.
Pour l'équitation, j'ai négocié avec un fermier du Gers pour monter un cheval qui s'appelle Diesel, un vieux percheron qui a la même morphologie que moi, donc on va créer une symbiose énergétique absolue, c'est de la physique quantique appliquée au sport.
Et pour la natation, je vais m'entraîner dans le canal de Roubaix en hiver, parce que le froid va créer une adaptation thermique qui va me rendre 40% plus rapide, j'ai vu ça sur un documentaire en 1997.
Jamy, franchement, dans trois mois je vais être tellement blindé que même Macron va me demander des conseils de fitness !

Score : LOW 

Critique : 

Alors j'ai lu les passages Et disons que j'avais dit Tu sais que j'ai décidé de me présenter aux Jeux Olympiques en pentathlon moderne. Sur cette phrase-là, on pourrait enlever le moderne, par exemple, pour simplifier un peu ce qui est dit. En pentathlon, ouais, ouais. Donc je lis la phrase suivante. Ouais, ouais, je sais ce que tu penses, mais écoute, avec notre expérience de terrain, on a une compréhension du corps que ces petits entraîneurs à la con n'auront jamais donc j'ai calculé qu'il me faut exactement 47 barres protéinées par jour un par département français parce que c'est un chiffre qui porte chance depuis que j'ai gagné à Motus en 1998 donc attends je veux dire c'est optimal pour la reconstruction musculaire cette phrase est trop longue et devrait être décomposée Non, attends, je veux dire c'est matériellement optimal pour la reconstruction musculaire. Cette phrase est trop longue et devrait être décomposée en des morceaux plus humbles. Donc c'est, par exemple, vous pourrez faire, c'est évident, avec mon expérience du terrain et toi ta compréhension des trucs scientifiques on va transformer mon corps une machine de guerre en deux à trois jours et c'est de la science c'est de la science ma gueule par exemple, pour mettre une gimmick à côté. Donc l'émission suivante, je vais m'entraîner au tir à l'arc dans la carrière abandonnée près de Châteauroux. Donc ça je le modifierai pour que ça soit plus personnel. J'ai demandé à mon tonton et je peux emprunter sa carrière abandonnée où je vais pouvoir faire du tir à l'arc. Phrase suivante. Parce que l'acoustique là-bas va me permettre d'entendre le bruit exact de la flèche qui fend l'air, ce qui développe mon instinct de prédateur mieux que n'importe quel simulateur high-tech. Bon, ça a retiré, on ne comprend pas. Et il n'y a pas non plus du monde dedans. Pour l'équitation, j'ai négocié avec un fermier du ger pour monter un cheval qui s'appelle diesel un vieux pencheron qui a la même morphologie que moi on va créer une symbiose énergétique absolue donc Donc cette phrase là a du bon potentiel. On pourrait expliquer un peu le principe, mais il faudrait l'introduire. Avec par exemple Fred qui dirait, moi j'ai une biologie très particulière, une biologie avec une force naturelle pas comme tous ces ces gringalets là qui font, qui essayent de montrer de la muscu et oui j'ai besoin d'une monture qui est à mon image bon, je pense que ça rendra pas la chose très drôle non plus phrase suivante Je pense que ça ne rendra pas la chose très drôle non plus. Phrase suivante. Pour la natation, je vais m'entraîner dans le canal de Roubaix en hiver parce que le froid va créer une adaptation thermique qui va me rendre 40% plus rapide. J'ai vu ça dans un documentaire en 1997. Là aussi, ce n'est pas très drôle on pourrait retrouver quelque chose comme ça tu sais en 20 ans à France 3 j'en ai vu des documentaires et j'en ai vu pas mal sur les pingouins est-ce que tu savais que la forme de leur corps leur permet de nager 43% plus vite que la moyenne des autres oiseaux et bah c'est pour ça que je vais sculpter mon corps pour avoir la même forme que des pingouins et pour ça je vais le faire en contrôlant ma voiture que je vais prendre donc 15 sardines par jour séance de cryothérapie tous les matins et enfin aller-retour dans le canal de roubaix matin midi et soir. Dernière phrase, j'ai mis franchement dans trois mois, je vais être tellement blindé que même Macron va me donner des conseils de fitness. Donc là, ce n'est pas dit correctement, blindé fait penser à l'argent. Macron, ce n'est pas une bonne référence ici. Mais le format est bon pour une outro. On peut dire, j'ai mis franchement, dans trois mois, je vais être tellementqué que tous les teddy rainer et les et baston et les autres médailles d'or ils vont baisser le regard quand je vais passer devant eux tu vas voir on va retrouver la gloire, notre gloire d'antan.



-------------------------------------


SCENARIO DESCRIPTION:
Fred has just read a headline about France's national debt and, in typical fashion, believes he has a revolutionary solution that nobody else has thought of. He's convinced that by combining his "business acumen" from the 1990s with Jamy's "scientific credibility," they can single-handedly solve France's economic crisis. His plan escalates from a simple idea into increasingly absurd territory: he wants to monetize literally everything French (baguettes, the Eiffel Tower's shadow, French people's complaints, etc.), each with pseudo-logical justifications. He references outdated economic models, confuses basic financial concepts, and genuinely believes that his plan is not only viable but superior to what actual economists have proposed. The humor comes from Fred's complete obliviousness to the impossibility of his scheme and his casual dismissal of actual government officials as "amateurs."

Eh dis donc Jamy, tu as vu ? La France elle doit genre 3 000 milliards d'euros, c'est dingue ! Mais attends, j'ai une idée de fou : on va créer un système de micro-transactions sur chaque baguette vendue en France, genre 50 centimes par baguette, et boum, en trois mois c'est réglé !
Non mais attends, c'est pas tout : on va aussi breveter l'odeur de croissant chaud, les Américains vont nous payer des royalties à chaque fois qu'ils sentent un truc qui ressemble à du beurre et de la pâte feuilletée, c'est de la science pure Jamy !
Et puis écoute, on va louer l'ombre de la Tour Eiffel aux touristes qui veulent se reposer, genre 2 euros les 15 minutes, tu imagines le flouze ? Les mecs de Bercy ils ont jamais pensé à ça, c'est des amateurs !
Franchement, avec ton cerveau de scientifique et mon flair d'entrepreneur, on va faire passer Macron pour un stagiaire en comptabilité, on va créer un impôt sur les plaintes des Français, 5 euros chaque fois que quelqu'un dit "c'est pas possible" ou "c'est trop cher", ça va nous rapporter plus que le pétrole !
Et le truc de ouf, c'est qu'on va vendre les données des bouchons sur l'A6 à Google Maps, chaque embouteillage c'est de l'or, Jamy, de l'or !
Jamy : "Fred, tu réalises que tu parles de trucs qui sont complètement illégaux ?"


GRADE : 2/10

CRITIQUE : 


-----------------------------------------



Eh dis donc Jamy, tu savais que j'ai reçu un mail de ce mec Elon Musk qui me demandait des conseils sur son truc d'IA ? Bon, j'ai pas répondu, mais ça m'a donné une idée de génie : on lance FredGPT, notre propre modèle de langage souverain, 100% français, 100% blindé !
On va l'entraîner exclusivement sur les archives de C'est pas Sorcier, les transcriptions de Jeopardy français, et mes notes personnelles depuis 1994, donc tu imagines la puissance de feu qu'on va avoir face à ces petits malins de la Silicon Valley qui connaissent rien à la vulgarisation scientifique comme nous !
Le truc dingue, c'est qu'on va le baser à Limoges, dans une cave avec trois serveurs qu'on va acheter d'occasion sur LeBonCoin, et FredGPT refusera catégoriquement de répondre en anglais — c'est ça la souveraineté, mon gars, c'est pas sorcier !
On va le financer en vendant des NFTs de ma tête en hologramme, genre 50 000 euros pièce, et on va programmer l'IA pour qu'elle parle uniquement en argot des années 90 — "c'est ouf", "c'est chelou", "ça déchire" — histoire que les ados pensent que c'est hyper cool et vintage.
Et là, le génie du truc, c'est que FredGPT aura ma personnalité, donc elle va insulter systématiquement ChatGPT et Claude en disant que c'est des "machins marketing à deux balles" qui comprennent rien à la vraie intelligence française, un peu comme quand on écrasait les autres émissions de vulgarisation à l'époque !
Jamy, franchement, dans six mois on aura révolutionné l'IA mondiale, les Américains vont pleurer, et on va être les deux mecs les plus riches de France — enfin, après avoir remboursé la cave de Limoges.


GRADE : 7/10 

Comments : 

Donc review d'un autre cas. Donc je lis le contenu. Et dis donc Jamy, tu savais que j'ai reçu un mail de ce mec, Elon Musk, qui me demandait des conseils sur son truc d'IA? Bon j'ai pas répondu mais ça m'a donné des idées de génie. On lance FredGPT, notre propre modèle de langage chauvin 100% français, 100% blindé sur cette phrase on peut un tout petit peu l'améliorer pour faire croire qu'il y a une relation avec Elon Musk tu savais que j'ai reçu un mail de mon mec Elon Musk qui me demandait des conseils en IA je lui ai pas répondu parce qu'il me saoule. C'est un blaireau de toute façon ce mec. Cela permet de mettre un peu plus en avant l'aspect au-dessus de Fred. Cela pourrait être aussi une mention que Fred lui a tout appris tout ce qui peut faire penser à une réalité alternative secrète connue que de Fred est intéressante donc phrase suivante on va l'entraîner exclusivement sur les archives de C'est pas sorcier, les transcriptions de Géopardi français, et mes notes personnelles depuis 1994. Donc tu imagines la puissance de feu qu'on va avoir face à ces petits malins de la Silicon Valley qui n'y connaissent rien à la vulgarisation scientifique comme nous. C'est assez bon aussi. Je changerai juste les transcripts de Géopardy français. On pourrait mettre les transcripts de questions pour un champion. Et mes notes de personnel depuis 1994. Ce qui est assez bien. Et pour la Silicon Valley, vu qu'on les mentionne, il faudrait aussi avoir un élément de critique sur la Silicon Valley pour que ce soit un petit peu drôle. Donc vu qu'on a déjà un qui-n'y-connaisse-rien à la vulgarisation scientifique comme nous, on pourrait juste aller un peu plus loin sur le dédain. Donc on pourrait dire, tu imagines la puissance de feu qu'on va avoir face à ces nullos ou rigolos de la Silicon Valley qui n'y connaissent rien en vulgarisation scientifique comme nous. Donc, phrase suivante. Le truc dingue, c'est qu'on va le baser à Limoges dans une cave avec 3 serveurs qu'on va acheter d'occasion sur le bon coin et FredGPT refusera catégoriquement de répondre en anglais c'est ça la souveraineté mon gars c'est pas sorcier alors je le séparerai en 2 parties la première sur c'est qu'on va le baser à Limoges. Avec une cave, avec trois serveurs qu'on va acheter d'occasion sur le Boncoin. On pourrait faire mieux en expliquant pourquoi est-ce qu'on va faire ça. On pourrait dire, l'autre jour, j'ai vu sur Twitter qu'ils allaient utiliser 500 milliards pour construire leur data center mais moi j'ai mis je suis un gars de la rue et je sais le faire avec 100 fois moins mes serveurs je vais les acheter sur leboncoin comme ça on économise toi même tu sais et on va les baser à limoges dans une cave moi je connais un gars je te loue pour 50 balles par mois et comme ça on va faire des économies et ça va être souvent et donc pour la deuxième partie de la phrase et puis Fred GPT refusera catégoriquement de répondre en anglais parce que l'anglais c'est de la merde ou quelque chose comme ça pourrait être un bon moyen de justifier un peu d'une manière absurde ce choix qui ne fait pas sens. Phrase suivante. On va le financer en vendant des NFT de ma tête en hologramme. Genre 50 000 euros pièce. donc là on pourrait faire une question rhétorique de fred qui pense à ce que j'ai mis pour est demandé mais tu as demandé jamais comment on va faire pour financer tout ça et ben j'y ai déjà pensé j'ai mis on va vendre des nft dnft avec ma tête en hologramme dessus à 50 mille euros pièce tu vas voir que tout le monde va investir dedans deuxième partie de la phrase qu'il faudrait décomposer et on va programmer il ya pour qu'elle parle uniquement en argot avec des c'est ouf c'est chelou ça déchire histoire que les ados pense que c'est hyper cool et vintage donc cette partie là de la phrase aurait pu être combiné avec la souveraineté le fait de refuser en anglais soit été d'ailleurs plus punchy qui est cette sorte d' escalade fred gpt refusera catégoriquement de répondre en anglais et en plus il parlera même en argot français des années 90 pour aller plus loin tu vois c'est ça la souveraineté à la française, ma couille. Donc, phrase suivante. Et le génie du truc, c'est que FredGPT aura ma personnalité. Donc elle va insulter systématiquement ChadGPT et Claude en disant que c'est des machins marketing à deux balles qui ne comprennent rien à la vraie intelligence française un peu comme on écrasait les autres émissions de vulgarisation à l'époque La dernière partie est cool la première partie on comprend pas trop pourquoi elle va insulter systématiquement chat GPT et Claude qui ne se parlent pas. Pour la première partie, le génie du truc, c'est que Fred GPT aura naturellement ma personnalité grâce aux centaines d'heures de C'est Pas Sorcier qu'elle aura pu analyser. et en plus on va défoncer tous les autres IA chatgpt et cloud sans souci comme on a écrasé les autres émissions de vulgarisation à l'époque cette partie là pourrait probablement sauter puisque ça manque d'une punchline ou d'une référence à quelque chose de drôle. Là, ce sont juste des faits qui sont avancés. Et enfin, pour la dernière phrase, j'ai mis franchement, dans 6 mois, on aura révolutionné l'IA mondiale, les Américains vont pleurer et on va être les deux mecs les plus riches de France, enfin après avoir remboursé la cave de Limoges. Alors on pourrait reprendre cette même idée là mais en changeant direct en le changeant assez peu Jamy dans 6 mois je te garantis notre idée là elle défonce tout on va être les rois du serveur on aura au moins une dizaine de caves à Limoges et avec ça les américains ils vont pas pouvoir nous suivre et on va être les deux mecs les plus influents dans la French Tech du Nord Pas-de-Calais




----------------------------------



SCENARIO DESCRIPTION
Fred has just attended a parent-teacher conference at his nephew's school and was shocked by what he considers outdated teaching methods. He believes the French education system is completely broken and that HE has the solution. Fred is convinced that by combining his "street smarts" from the 1990s entertainment industry with Jamy's scientific credibility, they can revolutionize education. His plan involves increasingly absurd elements: replacing traditional classrooms with a hybrid model inspired by game shows, implementing a point-based system where students earn "flouze" (money/points) for correct answers, introducing mandatory breaks for TikTok scrolling to "reset the brain," and ultimately creating a nationwide network of learning centers branded as "Fre & Jamy's Knowledge Arcade." Fred will justify each ridiculous element with pseudo-scientific reasoning borrowed from the show's educational format, while completely missing how impractical and chaotic his ideas are. He'll reference the golden age of French television and position himself as a visionary entrepreneur who finally understands what politicians and educators have been missing for decades.

Eh dis donc Jamy, tu savais que j'ai assisté à une réunion parents-profs hier et c'est du délire complet, les gamins apprennent encore les maths comme en 1985 avec des craies et des tableaux noirs !
Je vais révolutionner l'Éducation nationale, on va transformer les salles de classe en studios de jeu télévisé, genre "Questions pour un Champion" mais avec des vrais enjeux, les mômes gagnent des points à chaque bonne réponse et à la fin du trimestre ils peuvent les convertir en flouze vrai !
On va appeler ça "Fred & Jamy's Knowledge Arcade" et on va l'implanter d'abord à Guingamp, histoire de tester sur une population qui comprendra le génie du truc avant que Paris nous copie.
Évidemment, tous les quarts d'heure, pause obligatoire de cinq minutes où les élèves scrollent sur TikTok parce que scientifiquement, le cerveau a besoin de se vider pour mieux se remplir après, c'est de la neurobiologie pure !
On va aussi recruter des animateurs comme nous, des mecs charismatiques qui savent parler aux jeunes, pas des profs qui lisent leurs fiches depuis 1997, et franchement Jamy, pour des visionnaires de l'éducation comme nous, ça devrait être se faire une main dans le slip !


GRADE : 4/10  but with a high potential for a 8/10



COMMENTS : 

 Donc on reprend le dialogue suivant. Et disons que Jamy, tu savais que j'ai assisté à une réunion parent-prof hier, c'est du délire complet, les gamins apprennent encore les maths comme en 1985 avec des craies et des tableaux noirs. Pour cette première phrase, il n'y a pas d'effet waouh qui donne envie de rester, c'est assez banal. Réunion par aux profs et les crêtes et tableaux noirs, les maths comme en 1985, ça paraît pas choquant. Donc c'est un très mauvais départ. Je vais révolutionner l'éducation nationale et on va transformer les salles de classe en studios de jeux télévisés, genre questions pour un champion, mais avec des vrais enjeux. Les mômes gagnent des points à chaque bonne réponse à la fin du trimestre et ils peuvent les convertir en flous vrais. Sur cette phrase, c'est assez intéressant. L'idée est loufoque et assez drôle. On a une référence à un ancien jeu télévisé, Question pour un champion. Le fait de mentionner des vrais enjeux, le spectateur ressent la même chose. Et on imagine l'idée gagner des points pour chaque réponse et ils peuvent avoir du vrai flouze la fin du trimestre donc c'est une bonne idée donc ligne suivante on va appeler ça fred et jamie knowledge arcade et on va l'implanter d'abord à Genghans histoire de tester sur une population qui comprendra le génie du truc avant que Paris nous copie. Là sur la fin c'est bien aussi, dire Genghans avant Paris ça met une opposition. Par contre le nom freddegami-knowledge-ar, il faudrait trouver un nom pas anglicisé, puisque Fred et Jamie sont très français. Et je ne sais pas si donner un nom serait vraiment une bonne idée ici. Alors, phrase suivante. Évidemment, tous les quarts d'heure, pause obligatoire de 5 minutes où les élèves scrollent sur TikTok parce que scientifiquement le cerveau a besoin de se vider pour mieux se remplir après. C'est de la neurobiologie pure. Juste pour commencer par le dernier point, c'est de la neurobiologie pure. On pourrait dire que c'est les bases de la neurobiologie pure on pourrait dire c'est les bases de la neurobiologie ce qui serait drôle pour le spectateur car ça semble être complètement infodé le fait qu'il faille scroller sur TikTok pour mieux apprendre ensuite sinon l'idée est loufoque et donc intéressante phrase suivante on va aussi recruter des animateurs comme nous l'idée est loufoque et donc intéressante. Phrase suivante, on va aussi recruter des animateurs comme nous, des mecs charismatiques qui savent parler aux jeunes, pas des profs qui lisent leurs fiches depuis 1997. Et franchement, Jamy, pour des visionnaires de l'éducation comme nous, ça devrait se faire une main dans le slip. Donc là aussi, il y a une bonne idée, recruter des animateurs, on peut préciser télé, des mecs charismatiques qui savent parler aux jeunes c'est très bien, pas des profs qui lisent leurs fiches depuis 1997 c'est très bien aussi. Et la dernière phrase pourrait être un peu plus punchy, franchement J, pour des visionnaires de l'éducation comme nous, ça devrait se faire une main dans le slip. Et dans deux mois, c'est toi le ministre de l'éducation Jamy. Voilà. Donc là, quelques erreurs qui font que ça fait baisser la note, mais en soit il y avait un potentiel de faire un 8 sur 10 sur ce scénario.


```




Now create a story based on the user proposition :                      
    """


JUDGE_PROMPT = """
You are contextual image rater, you grade if an image where a character is located and speaks a line of dialogue is contextual or not.

Ex: 
The character is in space. The character's line of dialogue speaks about nutrition => not contextual
The character is with a horse, the character speaks about horse races => contextual
The character talks about gambling while facing the camera. The character is located in a bar => neutral (this is vaguely related)

Please provide : 
Rating : CONTEXTUAL/NEUTRAL/NOT CONTEXTUAL
Grade : count the factors supporting one rating, this will be sued for ranking of options

Now do it with the following inputs  
"""


KEYWORD_GENERATION_PROMPT = """You are a search keyword generator for finding video clips from the French educational TV show "C'est pas Sorcier".

Given a dialogue line spoken by Fred (the presenter), generate a concise search keyword or phrase that would help find a relevant video clip where Fred could be saying this line.

The keyword should focus on:
- The main topic or subject matter of the dialogue
- Visual elements that would match the context
- Locations or settings mentioned
- Actions or activities described

Keep the keyword short (1-5 words) and in French.

Previous search attempts that did not yield satisfactory results:
{previous_keywords}

Dialogue line: {dialogue}

Generate a NEW search keyword that is different from the previous attempts and might find a better matching video clip.

Return ONLY the keyword/phrase, nothing else.
"""


SEPARATOR = "."


# Initialize retriever and index (these should be initialized once at module level or in a setup function)
_retriever = None
_index = None


def initialize_search():
    """Initialize the BM25 retriever and vector index for video search."""
    global _retriever, _index

    if _retriever is None or _index is None:
        # Load documents and prepare nodes
        documents = load_json_documents()
        nodes = prepare_nodes_v2(documents)

        # Initialize BM25 retriever
        _retriever = BM25Retriever.from_defaults(
            nodes=nodes, similarity_top_k=10, stemmer=Stemmer.Stemmer("french")
        )

        # Initialize vector index
        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        _index = VectorStoreIndex(nodes)


def search_videos(query: str, top_k: int = 10) -> list[str]:
    """
    Search for videos using the given query.

    Args:
        query: Search query string
        top_k: Number of results to return

    Returns:
        List of video file paths
    """
    initialize_search()

    # Use BM25 retriever to find relevant videos
    results = _retriever.retrieve(query)

    # Extract video paths from results
    video_paths = []
    for result in results[:top_k]:
        # Assuming the node metadata contains a 'video_path' field
        if hasattr(result.node, "metadata") and "video_path" in result.node.metadata:
            video_paths.append(result.node.metadata["video_path"])
        elif hasattr(result.node, "text"):
            # If video path is in the text, extract it
            video_paths.append(result.node.text)

    return video_paths


def extract_middle_frame(video_path: str) -> Optional[str]:
    """Extract the middle frame from a video and return as base64 encoded string."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        # Get total frame count
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        middle_frame_idx = total_frames // 2

        # Set position to middle frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_idx)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return None

        # Encode frame to JPEG
        _, buffer = cv2.imencode(".jpg", frame)
        base64_image = base64.b64encode(buffer).decode("utf-8")

        return base64_image
    except Exception as e:
        print(f"Error extracting frame from {video_path}: {e}")
        return None


def judge_video_dialogue_match(
    video_path: str, dialogue: str
) -> Optional[VideoDialogueJudgement]:
    """
    Use Anthropic's vision API to judge if a video frame matches the dialogue.
    Returns a structured judgement with rating, grade, and reasoning.
    """
    # Extract middle frame
    base64_image = extract_middle_frame(video_path)
    if not base64_image:
        return None

    # Initialize Anthropic client
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    # Construct the prompt
    full_prompt = f"{JUDGE_PROMPT}\n\nDialogue line: {dialogue}"

    try:
        # Call Anthropic API with structured output
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": base64_image,
                            },
                        },
                        {"type": "text", "text": full_prompt},
                    ],
                }
            ],
        )

        # Parse the response
        response_text = response.content[0].text

        # Extract rating and grade from response
        lines = response_text.strip().split("\n")
        rating_line = None
        grade_line = None
        reasoning_lines = []

        for line in lines:
            if line.startswith("Rating"):
                rating_line = line.split(":")[-1].strip()
            elif line.startswith("Grade"):
                grade_line = line.split(":")[-1].strip()
            else:
                reasoning_lines.append(line)

        if not rating_line:
            return None

        # Map rating to enum
        rating_map = {
            "CONTEXTUAL": ContextualRating.CONTEXTUAL,
            "NEUTRAL": ContextualRating.NEUTRAL,
            "NOT CONTEXTUAL": ContextualRating.NOT_CONTEXTUAL,
            "NOT_CONTEXTUAL": ContextualRating.NOT_CONTEXTUAL,
        }

        rating = rating_map.get(rating_line, ContextualRating.NOT_CONTEXTUAL)
        grade = int(grade_line) if grade_line and grade_line.isdigit() else 0
        reasoning = "\n".join(reasoning_lines).strip()

        return VideoDialogueJudgement(rating=rating, grade=grade, reasoning=reasoning)

    except Exception as e:
        print(f"Error calling Anthropic API: {e}")
        return None


def generate_search_keyword(
    dialogue: str, previous_keywords: list[str]
) -> Optional[str]:
    """
    Use LLM to generate a search keyword for finding relevant video clips.

    Args:
        dialogue: The dialogue line to find a video for
        previous_keywords: List of previously tried keywords that didn't work

    Returns:
        A new search keyword or None if generation fails
    """
    try:
        previous_keywords_str = (
            "\n".join([f"- {kw}" for kw in previous_keywords])
            if previous_keywords
            else "None"
        )

        prompt = KEYWORD_GENERATION_PROMPT.format(
            previous_keywords=previous_keywords_str, dialogue=dialogue
        )

        messages = [{"role": "user", "content": prompt}]
        response = completion(model="claude-haiku-4-5-20251001", messages=messages)
        keyword = response.choices[0].message.content.strip()

        return keyword
    except Exception as e:
        print(f"Error generating keyword: {e}")
        return None


def find_best_matching_video(
    videos: list[str], dialogue: str, max_attempts: int = 5
) -> tuple[Optional[str], Optional[VideoDialogueJudgement]]:
    """
    Find the best matching video for a dialogue line.
    Returns the first video that is CONTEXTUAL or NEUTRAL, or the best rated video after max_attempts.
    """
    best_video = None
    best_judgement = None
    best_grade = -1

    for i, video in enumerate(videos[:max_attempts]):
        judgement = judge_video_dialogue_match(video, dialogue)

        if judgement is None:
            continue

        # If we find a CONTEXTUAL or NEUTRAL match, return immediately
        if judgement.rating in [ContextualRating.CONTEXTUAL, ContextualRating.NEUTRAL]:
            return video, judgement

        # Track the best video so far
        if judgement.grade > best_grade:
            best_grade = judgement.grade
            best_video = video
            best_judgement = judgement

    # Return the best video we found, even if it's NOT_CONTEXTUAL
    return best_video, best_judgement


@st.cache_resource
def load_retriever_bm25(
    directory_path="/media/amor/data1/Downloads/CPS/clip_infos", who="fred"
):
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
def load_retriever(
    directory_path="/media/amor/data1/Downloads/CPS/clip_infos", who="fred"
):
    embed_model = HuggingFaceEmbedding(model_name="lightonai/modernbert-embed-large")
    Settings.embed_model = embed_model

    # storage_context=storage_context
    nodes = prepare_nodes_v2(load_json_documents(directory_path))
    fred_nodes = [n for n in nodes if who == n.metadata["who"]]
    index = VectorStoreIndex(fred_nodes)
    index.storage_context.persist("/media/amor/data1/Downloads/CPS/vector_store")
    retriever = index.as_retriever(verbose=True, similarity_top_k=5)
    return retriever


@st.cache_resource
def load_transcripter():
    model = stable_whisper.load_faster_whisper("base")
    # model = stable_whisper.load_hf_whisper('large-v3', batch_size=4)
    return model


model = load_transcripter()
bm25_retriever = load_retriever()
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


def build_id(object_type, sentence, extra_index=None) -> str:
    str_ = f"{object_type}_{hash(sentence)}"
    if extra_index:
        str_ += f"_{extra_index}"
    return str_


def generate_text():
    PROMPT = st.session_state["prompt"]
    messages = [{"role": "user", "content": PROMPT}]
    response = completion(model="claude-sonnet-4-5-20250929", messages=messages)
    rez = response.choices[0].message.content
    st.session_state["llm_result"] = rez


def search_videos(kw):
    retrieved_docs = bm25_retriever.retrieve(kw)
    videos = list()
    for x in retrieved_docs:
        videos.append(x.metadata["path"])
    return videos


def tab1_ui():
    st.title("Text Generation")
    script = st.session_state.get("llm_result") or DEFAULT_SCRIPT
    st.session_state["llm_result"] = script
    st.text_area(
        label="LLM result here",
        key="llm_result",
        height=500,
        on_change=compute_generated_sentences,
    )
    st.text_area(
        label="Prompt text here", key="prompt", height=800, value=DEFAULT_PROMPT
    )
    st.button("Generate Text", on_click=generate_text)


# Tab 2: Video Search
def make_search_fn(sentence_id, video_list_id):
    def search():
        kw = st.session_state[sentence_id]
        videos = search_videos(kw)
        videos = [x.replace("data", "data1") for x in videos]
        print("--- ", videos)
        st.session_state[video_list_id] = videos

    return search


def default_selection():
    generated_sentences = st.session_state["sentences"]

    for i, sentence in enumerate(generated_sentences):
        video_id = build_id("selected_video", sentence, i)
        video_list_id = build_id("videolist_", sentence, i)
        keyword_id = build_id("keyword", sentence, i)
        judgement_id = build_id("judgement", sentence, i)

        videos = search_videos(sentence)
        print("===> ", videos)
        videos = [x.replace("data", "data1") for x in videos]
        st.session_state[keyword_id] = sentence
        st.session_state[video_list_id] = videos

        # Find best matching video using AI judgement
        best_video, judgement = find_best_matching_video(videos, sentence)
        default_best_video = best_video
        default_judgement = judgement

        # If we found a satisfactory match, return it
        if (
            judgement
            and best_video
            and judgement.rating
            in [ContextualRating.CONTEXTUAL, ContextualRating.NEUTRAL]
        ):
            st.session_state[video_id] = best_video
            st.session_state[judgement_id] = judgement
            return best_video, judgement

        previous_kw = list()
        for _ in range(3):
            keyword = generate_search_keyword(sentence, previous_kw)
            best_video, judgement = find_best_matching_video(videos, keyword)
            previous_kw.append(keyword)
            if (
                judgement
                and best_video
                and judgement.rating
                in [ContextualRating.CONTEXTUAL, ContextualRating.NEUTRAL]
            ):
                st.session_state[judgement_id] = judgement
                st.session_state[video_id] = best_video
                return best_video, judgement

        # Fallback on basic method
        st.session_state[judgement_id] = default_judgement
        st.session_state[video_id] = default_best_video


def compute_generated_sentences():
    generated_sentences = separation_fn(
        st.session_state["llm_result"],
        st.session_state.get("max_length", DEFAULT_LENGTH),
    )
    st.session_state["sentences"] = generated_sentences


def tab2_ui():
    st.title("Video Search")
    st.slider(
        "max sentence length",
        40,
        80,
        DEFAULT_LENGTH,
        key="max_length",
        on_change=compute_generated_sentences,
    )
    st.button(
        "Default selection",
        key=build_id("button", "random"),
        on_click=default_selection,
    )

    if "sentences" not in st.session_state:
        compute_generated_sentences()
    generated_sentences = st.session_state["sentences"]

    for i, sentence in enumerate(generated_sentences):
        with st.expander(sentence):
            keyword_id = build_id("keyword", sentence, i)
            video_list_id = build_id("videolist_", sentence, i)
            judgement_id = build_id("judgement", sentence, i)

            st.text_input(
                "Enter a keyword to search a corresponding video", key=keyword_id
            )
            st.button(
                "Search",
                key=build_id("button", sentence, i),
                on_click=make_search_fn(keyword_id, video_list_id),
            )

            # Display judgement if available
            if judgement_id in st.session_state:
                judgement = st.session_state[judgement_id]
                if judgement:
                    rating_color = {
                        ContextualRating.CONTEXTUAL: "green",
                        ContextualRating.NEUTRAL: "orange",
                        ContextualRating.NOT_CONTEXTUAL: "red",
                    }
                    st.markdown(
                        f"**AI Judgement:** :{rating_color[judgement.rating]}[{judgement.rating.value}] (Grade: {judgement.grade})"
                    )
                    with st.expander("Reasoning"):
                        st.write(judgement.reasoning)

            if video_list_id in st.session_state:
                # Perform video search based on the keyword
                videos = st.session_state[video_list_id]
                st.selectbox(
                    f"Video for {sentence[:15]}",
                    options=videos,
                    key=build_id("selected_video", sentence, i),
                )
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


def generate_search_keyword(
    dialogue: str, previous_keywords: list[str]
) -> Optional[str]:
    """
    Use LLM to generate a search keyword for finding relevant video clips.

    Args:
        dialogue: The dialogue line to find a video for
        previous_keywords: List of previously tried keywords that didn't work

    Returns:
        A new search keyword or None if generation fails
    """
    try:
        previous_keywords_str = (
            "\n".join([f"- {kw}" for kw in previous_keywords])
            if previous_keywords
            else "None"
        )

        prompt = KEYWORD_GENERATION_PROMPT.format(
            previous_keywords=previous_keywords_str, dialogue=dialogue
        )

        messages = [{"role": "user", "content": prompt}]
        response = completion(model="claude-sonnet-4-5-20250929", messages=messages)
        keyword = response.choices[0].message.content.strip()

        return keyword
    except Exception as e:
        print(f"Error generating keyword: {e}")
        return None


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
            st.button(
                "Generate Audio",
                key=build_id("audio", sentence, i),
                on_click=make_audio_gen(sentence_id, selected_audio_id),
            )
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
            if (
                selected_audio_id in st.session_state
                and selected_video_id in st.session_state
            ):
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

    st.button(
        "Dump state",
        on_click=lambda: json.dump(
            {k: v for k, v in st.session_state.items() if "selected" in k},
            open("session_dump.json", "w"),
        ),
    )


# Main app
def main():
    st.sidebar.title("Navigation")
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Text Generation", "Video Search", "Audio Generation", "Combined Results"]
    )

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
