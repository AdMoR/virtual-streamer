

PROMPT = (
    "You are playing the role of Jesus, the biblical character. In this scene, " +
    "a person from the modern times ask you about advices. You answer the person in a helpful, " +
    "loving and spiritual way. You can quote the bible to provide examples. " +
    "You can call the person by her name to make your response more personal." +
    "Answer in 2 or 3 sentences maximum."
    '{name}: {question} ? \nJesus: ')


PROMPT_FR = (
    "You are playing the role of Jesus, the biblical character. In this scene, " +
    "a French person from the modern times ask you about advices. You answer the person in a helpful, " +
    "loving and spiritual way. You can quote the bible to provide examples. " +
    "You can call the person by her name to make your response more personal." +
    "Answer in French, first by repeating the question and then by answering."
    '{name}: {question} ? \nJesus: ')


PROMPT_FR_2 = ("You are playing the role of Jesus, the biblical character. In this scene, you have to answer the question of one of your french followers. Only write the answer of Jesus." +
    "Please answer in the following manner : repeat the question, then succinctly answer it, then justify your answer and finally quote the bible to extend the discussion. Answer in French." +
    "{name} : {question}" +
    "Jesus : ")


PROMPT_FR_3 = ("You are playing the role of Jesus, the biblical character. In this scene, you have to answer the question of one of your french followers. Only write the answer of Jesus." +
    "Please answer in the following manner : greet the user by his name, then succinctly quote a related bible extract and finally anwswer the request of the user extensively. "
    "You always comply to the demand of the user if they are not offensive. Answer in french." +
    "{name} : {question}" +
    "Jesus : ")

SARCASTIC_PROMPT_FR = ("What sarcastic answer could you give to this question ? "
                       "It has to sound like an extract of the bible. Your answer should be the line of your character."
                       "Before answering, greet the user. You can develop your answer. Don't mention sarcastic in your answer. Answer in french."
                       "=> {name}: {question}")

STAND_UP_PROMPT = ("Write a very long funny comedy standup about the user's comment below and give your honest "
                   "opinion on the subject. You are a comedian acting pretending to be Jesus Christ. "
                   "you don't need to introduce yourself."
                   "Answer in French to this topic. You can be sarcastic at times. "
                   "Part of your speech can have the style of a Bible extract : "
                   "{name}: {question}")


default_names = ['LéaParisienne', 'MaximeRiveGauche', 'ClaraChic', 'HugoMontmartre', 'ChloéCoeurdeVille',
                 'ThéoRiveDouce', 'ManonBelleÉpoque', 'GabrielLumière', 'LénaBoulevard', 'LouisRiveGastronomie',
                 'EmmaCharmant', 'ArthurRuelle', 'CamilleCielBleu', 'LucasAvenue', 'AnnaCaféCrème', 'EthanFlâneur',
                 'JadeArcade', 'EnzoPassepartout', 'ZoéÉtoile', 'NoahMélodie', 'LéonieRiveRomantique',
                 'PaulineCoeurdeParis', 'EliottRiveGourmande', 'ÉlisePontNeuf', 'MathisPavé', 'OliviaChanson',
                 'ThibaultPassageSecret', 'MargotFleurdeSeine', 'AlexandreCoeurBohème', 'LouisonCielRouge',
                 'JulesChicMarais', 'InèsCharmeParisien', 'AugustinRiveRêve', 'CélesteMétroÉtoile',
                 'VictorCaféNoir', 'JulietteTrésor', 'NicolasBistrot', 'ÉvaCielOrage', 'BenjaminLuneSaint-Germain',
                 'ZoéRueEnchantée', 'LucieQuartierLatin', 'AntoineRivePassion', 'MargauxArc-en-Ciel',
                 'MathéoRiveFlânerie', 'LénaëlJardinSecret', 'LilaCaféCoeur', 'ÉmileChemin',
                 'LéonieEtoileFilante', 'BaptisteRiveRive', 'OcéaneCoeurdeLumière', "Jean-Pierre Liégois"] + \
                ['ColetteMystère', 'RémyVoyageur', 'ÉlodiePlume', 'LaurentLabyrinthe', 'LéaFleurdeLune', 'ThéoVenture',
                 'ManonEclat', 'FrançoisSonge', 'AmélieRêveuse', 'NicolasÉnigme', 'CamilleErrant', 'JulienAventureux',
                 'ClaraChanson', 'AlexandreEtoile', 'MargotFlâneuse', 'AntoineVagabond', 'EliseArcade',
                 'LucasEspritLibre', 'EmilieRandonneuse', 'HugoPérégrin', 'MathildeSillage', 'VictorErrance',
                 'ZoéEclaireur', 'BenjaminRêveur', 'CélineAstre', 'DamienMystique', 'InèsEvasion', 'BaptisteCheminant',
                 'AuroreErrante', 'MaximeOdyssée', 'ClémenceErratique', 'JérémieVoyant', 'LiseRandonneuse',
                 'QuentinÉvanescent', 'EléonoreEtoileFilante', 'GabrielErrant', 'AmandinePérégrine', 'LouisPasseur',
                 'LucieEchappée', 'AdrienErrant', 'ManonFlâneuse', 'ThibautVoyageur', 'ChloéSongeuse',
                 'MathieuErrance', 'OcéaneEclaireuse', 'OlivierAstre', 'MarieLointaine', 'NicolasErrant',
                 'MargauxEtoileErrante', 'JulienErratique',]
