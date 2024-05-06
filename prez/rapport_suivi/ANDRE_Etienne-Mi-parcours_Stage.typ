#import "neurips2023.typ": *
#set par(justify: true)
#set page(
  paper:"a4",
  numbering: "1"
)
#set text(
  lang: "fr",
  size: 11pt,
  // font: "New Computer Modern"
  // font: "Noto Serif"
)
#show link: underline
#set heading(numbering: "1.")
#set text(size: 10pt)

// look like latex
#set par(leading: 0.55em, first-line-indent: 1.8em, justify: true)
#set text(font: "New Computer Modern")
#show par: set block(spacing: 0.55em)
#show heading: set block(above: 1.4em, below: 1em)

#let author(name, mail, affl) = box(align(center)[
  #strong(name) \ #link("mailto:"+mail, raw(mail)) \ #affl
])



#align(center)[
  #text(19pt)[
    #smallcaps([*Ingénierie inverse de mix DJ*])
  ]

  #box(height: 5em, image("ATIAM_logo_LONG_RVB.jpg"))
  
  #text(12pt)[
    05/03/2024 --- 30/08/2024
  ]

  #text(11pt)[
    IRCAM -- 1 Place Igor Stravinsky, 75004 Paris
  ]
  #v(2em)

  #author([Stagiaire: \ Étienne ANDRÉ], "andre@ircam.fr", "IRCAM")
  #h(5em)
  #author([Encadrant: \ Diemo SCHWARZ], "schwarz@ircam.fr", "IRCAM")
  #h(5em)
  #author([Encadrant: \ Dominique FOURER], "dominique.fourer@univ-evry.fr", "IBISC")

]
#place(center+bottom, float: true)[#box(height: 3em, image("IRCAM.CP.jpg"))
  #h(5em)
  #box(height: 3em, image("LOGO_SU_HORIZ_SEUL.jpg"))
  #h(5em)
  #box(height: 3em, image("Logo_Télécom_Paris.jpg"))
]
= Sujet de stage

#text(size: 9pt)[
Les pratiques des _Disc-Jockeys_ (DJ) constituent une part importante de la culture musicale populaire, et leur compréhension reste encore un défi. Celle-ci pourrait notamment être bénéfique pour la recherche musicologique, les études culturelles sur les pratiques des DJ et leur réception critique, le développement de la technologie musicale liée au _DJing_, l’automatisation de la performance DJ à des fins de divertissement ou commerciales, et d’autres encore.

Toutefois, malgré un intérêt grandissant pour la recherche d’informations musicales liées à la pratique DJ (DJ-MIR) @cliffHangDJAutomatic2000 @fujioSystemMixingSongs2003 @ishizakiFullAutomaticDJMixing2009 @kellEmpiricalAnalysisTrack2013 @yadatiDetectingDropsElectronic2014 @sonnleitnerLandmarkBasedAudioFingerprinting2016 @kimAutomaticDjMix2017 @bittnerAutomaticPlaylistSequencing2017 @schwarzExtractionGroundTruth @schwarzUnmixDBDatasetDJMix2018, les techniques DJ ne sont pas encore suffisamment étudiées par les chercheurs, en partie à cause du manque de jeux de données annotées de mix DJ.

Ce stage vise à combler ces lacunes en améliorant les méthodes de déconstruction et d’annotation automatiques des mixes enregistrés dont les morceaux constitutifs sont connus. Dans des recherches antérieures @schwarzMethodsDatasetsDJMix2021, un alignement approximatif permet d’abord d’estimer à quel endroit du mix chaque morceau commence et quel facteur d’étirement temporel a été appliqué. Ensuite, un alignement précis est appliqué pour déterminer le décalage exact de chaque morceau dans le mix. Enfin, les points de repère (_cue points_) ansi que les courbes de fondu sont estimés dans le domaine temps-fréquence.

Ces méthodes ont été évaluées sur le jeu de données public UnmixDB @schwarzMethodsDatasetsDJMix2021 @schwarzUnmixDBDatasetDJMix2018. Il contient des mixes générés procéduralement avec des morceaux de musique sous licence libre, avec la vérité terrain sur l’alignement, les transformations et les effets appliqués.

Le stage peut prendre plusieurs directions:

+ Revisiter et étendre le jeu de données UnmixDB :

  - valider et étendre les effets appliqués dans UnmixDB par des mixes DJ plus "écologiquement valides";

  - étendre le jeu de donnéees aux courbes de tempo non constantes;

+ Utiliser ou créer des jeux de données de mixes DJ annotés @werthen-brabantsGroundTruthExtraction2018 @kimComputationalAnalysisRealWorld2020;

+ Améliorer et rechercher de nouvelles méthodes d'ingénierie inverse de mix DJ. Quelques possibilités:

  - affiner la phase d’alignement pour gérer des mixes avec des courbes de tempo non constantes;

  - gérer la présence partielle et répétée de morceaux ou de boucles dans le mix @smithNonnegativeTensorFactorization2018;

  - tester l’influence de différentes représentations de signal (e.g. _MFCC_, spectre, chroma, _scattering transform_, _PANNs_) sur les résultats @demanAnalysisEvaluationAudio2014 @yangAligningUnsynchronizedPart2021 @cramerLookListenLearn2019 @zehrenAutomaticDetectionCue2020.

Avec ces améliorations, la méthode pourrait devenir sufisamment robuste et précise pour permettre l’inversion des fondus, de l’égalisation, et d’autres traitements appliqués lors de la constitution du mix @barchiesiReverseEngineeringMix2010 @ramonaSimpleEfficientFader2011.
]

#show: rest => columns(2, rest)

= Contexte
Nous nous focalisons sur la constitution de jeux de données, pour laquelle nous présentons trois pistes; ainsi que sur l'amélioration des méthodes d'ingénierie inverse de mix DJ, pour lesquelles nous proposons une méthode de décomposition et d'estimation de paramètres de mix par NMF#footnote([Factorisation en matrices non-négatives]).

Nous utilisons dans la suite de ce document les termes suivants:
/ Track: Un enregistrement d'un morceau de musique.
/ Mix: Un enregistrement d'une performance DJ.


= Constitution de jeux de données
Les jeux de données utiles au DJ-MIR se constituent de mixes auquels sont associés des tracks, généralement accompagnées de leur position dans le temps. En fonction des annotations disponibles, on pourra trouver des informations sur les manipulations effectuées par le DJ lors de la création du mix, comme le volume de chaque track, les facteurs d'étirement temporel et de transposition, la nature et les paramètres des effets appliqués, etc. Nous supposons que la connaissance exacte de tous les traitements appliqués aux tracks d'origine nous permet de construire exactement le signal de mix.

De tels jeux de données peuvent se distinguer en deux catégories:
- *synthétiques*, c'est-à-dire générés selon un certain nombre de règles ;
- *réels*, issus de mixes et annotés manuellement ou automatiquement.

Les jeux de données synthétiques ont l'avantage d'être précis et complets, les paramètres de création du mixes étant connus par construction. En revanche, étant basés sur des règles, ils peuvent pâtir en diversité et donc en validité écologique. UnmixDB @schwarzUnmixDBDatasetDJMix2018 (basé sur le dataset `mixotic` de #cite(<sonnleitnerLandmarkBasedAudioFingerprinting2016>, form:"prose")) et le dataset de #cite(<werthen-brabantsGroundTruthExtraction2018>, form:"prose") sont les seuls représentants de cette catégorie. On notera que ces jeux de données sont tous basés sur les mixes du netlabel `mixotic.net` #footnote(link("https://www.mixotic.net")), et partagent donc le même style musical.

Les jeux de données réels n'ont pas ce problème, étant directement représentatifs des usages du monde du DJing. Ils requièrent cependant une annotation manuelle (potentiellement chronophage), ou automatique (potentiellement erronée). La nature commerciale des mixes et des tracks les constituant présente également des problèmes d'ordre légal pour une diffusion dans un contexte académique. Il n'existe en effet pas (à notre connaissance) de tel jeu de données librement disponible: le jeu de données de #cite(<kimComputationalAnalysisRealWorld2020>, form:"prose") et le jeu de données `disco` de #cite(<sonnleitnerLandmarkBasedAudioFingerprinting2016>, form:"prose") ne sont pas accessibles, respectivement pour des raisons commerciales et de droit d'auteur.

== Utilisation de bases de données communautaires existantes
L'intérêt des spectateurs de connaître les tracks entendues lors d'une preformance DJ a amené à la création de nombreuses bases de données en ligne de _track ID_, telles que `1001tracklists`#footnote(link("https://www.1001tracklists.com"))<1001tracklists>,  `MixesDB`#footnote(link("https://www.mixesdb.com"))<mixesdb>, `trackid.net`#footnote(link("https://trackid.net"))<trackid.net>, `CueNation`#footnote(link("https://cuenation.com"))<cuenation>, `LiveTracklist`#footnote(link("https://www.livetracklist.com"))<livetracklist> et `setlist.fm`#footnote(link("https://www.setlist.fm"))<setlist.fm>. De manière moins structurée, les commentaires sur les sites de streaming regorgent d'identification de tracks plus ou moins complètes.

Ces bases de données sont pour la plupart communautaires (alimentées et vérifiées par les utilisateurs), mais peuvent inclure des éléments d'identification et de segmentation automatique. Le niveau de détail est variable, et est généralement constitué au moins de l'identification et de l'ordre des tracks dans le mix, et peut inclure les positions de début et de fin approximatives. À l'exception de `mixesdb`@mixesdb dont le contenu est sous licence libre, tous ces sites sont à vocation commerciale. Aucun n'offrent d'accès structuré aux données.

À ce jour, nous avons seulement pu récupérer les données annotées de `trackid.net`@trackid.net. Il s'agit d'un service d'identification automatique de tracks au sein de mix, basé sur un algorithme propriétaire de _fingerprinting_ et de segmentation. Les utilisateurs du site peuvent demander l'annotation du mix de leur choix en fournissant un lien vers un site de streaming musical.

Le jeu de données fourni par `trackid.net`@trackid.net est constitué de métadonnées pour 136231 mix (d'une durée cumulée de 189512h) et de 666625 tracks uniques. Les métadonnées incluent:
 - L'URL, la durée et les styles estimés des mixes;
 - Pour chaque mix, la liste des tracks détectées avec:
  - Titre, artistes(s) et label
  - Position de début et de fin dans le mix

Le jeu de données ne contient pas d'audio: ceux-ci doivent être obtenus par nos soins.

L'étude de quelques indicateurs statistiques montrent que ces métadonnées sont assez peu précises, empêchant leur utilisation directe. Cependant, devant la grande diversité et taille du jeu de données, nous espérons filtrer celles-ci pour obtenir un jeu de données plus petit mais de meilleure qualité.

== Extraction de vérité terrain sur DAW#footnote([Digital Audio Workstation])
Il est possible de produire des mix en studio, par exemple par l'utilisation d'un DAW, en mimant les usages de la pratique DJ. Dans ce cas, et dans une démarche similaire à celle de #cite(<werthen-brabantsGroundTruthExtraction2018>, form:"prose"), on peut associer le rendu audio du DAW avec une vérité terrain très précise, munis de la description symbolique du mix et de la connaissance du fonctionnement du logiciel.

Nous avons développé un ensemble d'outils permettant de transformer un projet Ableton Live#footnote(link("https://ableton.com")) associé à ses fichiers sources (les _clips_ audio) en une représentation symbolique adaptée au DJ-MIR.

Ces outils pourraient être utilisés pour constituer un jeu de données à partir de mix élaborés par nos soins. Cependant, il est possible que de tels mixs ne soient pas entièrement représentatifs de la culture DJ, entachés par cette technique de production particulière et par nos propres biais.

== Mesure en situation réelle par instrumentation de logiciel DJ
Dans le but de constituer un jeu de données à la fois précis et écologiquement valide, nous proposons de mesurer la vérité terrain en situation réelle, par l'usage de matériel et/ou de logiciel instrumenté par nos soins, dans une démarche similaire à celle de #cite(<hansenAcousticsPerformanceDJ2010>, form:"prose").

Nous espérons:
- Modifier le logiciel DJ libre `mixxx`#footnote(link("https://mixxx.org")) en y ajoutant l'enregistrement de divers paramètres d'intérêt au cours du mix;
- Organiser des sessions d'enregistrement sur ce logiciel instrumenté avec des DJs d'horizons et de styles différents.

Des entretiens sont en cours auprès de DJs afin de déterminer la pertinence et la faisabilité de cette démarche.

= Estimation des paramètres de mix via NMF

Les platines (_decks_), tables de mixage, contrôleurs et logiciels conditionnent le champ d'expression des DJ. Les marques et modèles différentes présentent des fonctionnalités variées, mais le flux de travail reste toujours similaire. On peut ainsi supposer que le trajet du signal depuis les tracks jusqu'au mix et les transformations pouvant y être appliquées suivent toujours le même schéma général, qu'on représente @dj-deck.

#figure(
  image("../2024-04-26/dj-deck.png"),
  caption: [Schéma-bloc généralisé de matériels/logiciels DJ typiques],
  placement: none
) <dj-deck>

Ensuite, en supposant que l'on puisse changer l'ordre des opérations, on peut séparer les transformations temporelles (étirement temporel, _delay_, boucles, sauts...) des transformations de timbre (distortion, compression, transposition, filtres, égalisation...). En considérant les spectrogrammes des tracks, on peut formuler le processus de création de mix DJ sous une forme matricielle qui se prête bien à l'application de la NMF (@nmf-djmix), qui connaît de nombreuses applications en audio @fevotteSingleChannelAudioSource2018. En particulier, nous exploitons l'angle de la transcription automatique et de la séparation de sources.

#figure(
  image("../2024-04-26/nmf-djmix.png"),
  caption: [Formulation du processus de mix DJ sous forme NMF],
  placement: none
) <nmf-djmix>donc

Étant donnée une matrice non-négative $bold(V)$, l'algorithme NMF permet de la décomposer en deux matrices non-négatives $bold(W)$ et $bold(H)$ telles que $ bold(V) approx bold(W H) $

On note $forall i in {1 ... M}$, où $M$ est le nombre de tracks dans le mix: 
- $bold(W)_i$ le spectrogramme de la $i$ème track de référence;
- $bold(H)_i$ la $i$ème matrice d'activation;
- $bold(V)_i$ le spectrogramme de la $i$ème track après les transformations temporelles;
- $f_i$ une fonction quelconque représentant les transformations de timbre de la $i$ème track;
- $bold(V)$ le spectrogramme du mix.

Alors:
$ bold(V) = sum_(i = 1)^M f_i lr((bold(W)_i bold(H)_i)) $

Et si de plus $forall i, f_i = bb(1)$:
$ bold(V) = sum_(i = 1)^M bold(W)_i bold(H)_i = underbrace(lr([bold(W)_1 bold(W)_2  ...  bold(W)_M]), bold(W)) underbrace(mat(delim: "[", bold(H)_1; bold(H)_2; dots.v; bold(H)_M), bold(H)) $

On peut alors fixer $bold(W)$ et estimer $bold(H)$ #footnote([Si le mix comporte des éléments inconnus (une boîte à rythmes, de la voix, la foule...), ajouter quelques colonnes à $bold(W)$ et quelques lignes à $bold(H)$ qu'on estime conjointement permet d'améliorer l'estimation.]).

Nous espérons alors retrouver dans les $bold(H)_i$ des structures similaires à celles @courbes-temps-temps.
#figure([#image("../2024-03-26/temps-temps.drawio.png")],
  caption: [
    Exemples de structures de matrice d'activation
  ]
) <courbes-temps-temps>

Une fois l'estimation effectuée, nous pouvons extraire des $bold(H_i)$ des paramètres tels que les points de coupe, le volume relatif de chaque track, ou encore les facteurs d'étirement temporel.

Nous utilisons l'algorithme NMF avec $beta$-divergence @fevotteAlgorithmsNonnegativeMatrix2011, et obtenons des résultats très  encourageants dans le cas particulier de la divergence d'Itakura-Saito associée aux spectres de puissance.

Toutefois, cette méthode présente actuellement plusieurs problèmes, qui nous offrent des pistes de travail:
- La longueur typique d'un mix DJ (d'une à plusieurs heures) rend aberrantes la durée de calcul et la consommation mémoire de l'algorithme. L'utilisation de variantes _online_ @cichockiFastLocalAlgorithms2009 et multi-résolution @leplatMultiResolutionBetaDivergenceNMF2022 de celui-ci permettrait de les rendre plus raisonnables. La connaissance a priori de la position approximative des tracks dans le mix permettrait également d'éviter un grand nombre d'opérations.
- La représentation spectrale utilisée (transformée de Fourier à court terme) requiert un recouvrement assez grand pour obtenir de bons résultats. Nous souhaitons essayer d'autres représentations spectrales, notamment en utilisant des fenêtres beat-synchrones.
- Toutes les solutions de l'algorithme ne sont pas forcément valides dans notre cas: on souhaite favoriser l'apparition dans $bold(H)$ les structures décrites @courbes-temps-temps. Nous comptons ajouter des termes de régularisation @fevotteSingleChannelAudioSource2018 à la fonction de coût pour notamment favoriser les activations lisses  @fevotteMajorizationminimizationAlgorithmSmooth2011. Une autre piste @driedgerLetItBee2015 est d'appliquer des transformations à $bold(H)$ entre chaque itération.
- L'hypothèse qu'il n'y a aucune transformation  de timbre ($forall i, f_i=bb(1)$) est très forte et est rarement vérifiée en réalité. Nous voulons explorer les possibilités d'estimation de ces effets.
- La transposition d'une track est également très courante dans les mix DJ. Nous comptons tirer parti de l'algorithme NMF2D @aarabiMusicRetilerUsing2018 afin d'y être invariants.


#bibliography("../../zotero.bib")
