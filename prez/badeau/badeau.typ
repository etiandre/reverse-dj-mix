#import "@preview/polylux:0.3.1": *
#import themes.metropolis: *
#set math.equation(numbering: "1.")
#show: metropolis-theme.with()

#set text(font: "Fira Sans", weight: "light", size: 20pt)
// #show math.equation: set text(font: "Fira Math")
#set strong(delta: 200)
// #set par(justify: true)

#title-slide(
  title: "Ingénierie inverse de mix DJ",
  author: "Étienne ANDRÉ",
  date: datetime.today().display(),
)

#slide(title:"Plan")[
#metropolis-outline
]

#new-section-slide("Contexte")

#slide(title: "Pratiques DJ")[
  #side-by-side[
    - Part importante de la musique populaire
    - Peu exploitées dans la recherche
    - Usages codifiés par le matériel
  ][
  #figure(image("../2024-03-26/ddj-sb3.png"), caption:[Pioneer DDJ-SB3])
  ]
]

#slide(title: "Définitions")[
  - *_track_*: un morceau de musique se suffisant à lui-même
  - *_mix_*: enchaînement de _tracks_ potentiellement transformées et partiellement superposées
]

#slide(title: "Objectifs")[
  #strong[Sachant]

  - Un enregistrement d’un _mix_
  - Les enregistrements des _tracks_ le composant

  #strong[Estimer]

  - Les transformations temporelles (vitesse, boucles, sauts, ...)
  - Les transformations de timbre (transposition, filtres, …)
  - Les éléments supplémentaires (voix, foule, …)
  - Leur évolution au cours du temps

]

#new-section-slide("Formulation matricielle d'un mix DJ")

#slide(title: "Chemin du signal usuel")[
  #figure([#image("../2024-04-26/dj-deck.png")],
  caption: [
    Chaîne de traitement de matériel DJ usuel
  ]
)
  #figure([#image("../2024-04-26/separate-boxes.png")],
  caption: [
    Séparation des éléments de la chaîne de traitement
  ]
)
]

#slide(title:"Mise sous forme matricielle")[
  #side-by-side[
    #figure([#image("../2024-04-26/nmf-djmix.png", height: 70%)],
      caption: [
        Chaîne de traitement sous forme matricielle
      ]
    )
  ][
    - $bold(W)_i$: spectrogramme des morceaux de
      référence
    - $bold(H)_i$: transformations temporelles
      (timestretch, boucles, delays…) + gain
    - $bold(V)_i$: spectrogramme transformés temporellement
    - $f_i$: le reste des transformations (filtres, pitch,
      distortion…)
  ]


$ bold(V) = sum_(i = 1)^M f_i lr((bold(W)_i bold(H)_i)) $

Si de plus $forall i,  f_i = bold(1)$ alors:

$ bold(V) &= sum_(i = 1)^M bold(W)_i bold(H)_i = underbrace([bold(W)_1 bold(W)_2 ... bold(W)_M], bold(W)) underbrace(mat(delim: "[", bold(H)_1; bold(H)_2; dots.v; bold(H)_M), bold(H)) $

]

#new-section-slide("Transcription de mix DJ par NMF")

#slide(title: "Non-negative Matrix Factorization")[
  #strong[Principe]: sachant une matrice $bold(V)$, estimer $bold(W)$ et
  $bold(H)$ tels que:

  $ bold(V) approx bold(W H) $


  #strong[Application à l’audio]:

  - Séparation de sources
  - Transcription
  - Restauration
  ]
  #slide(title: [$beta$-NMF])[

  #strong[Objectif]: minimiser la distance $d$ entre $bold(V)$ et
  $bold(W) bold(H)$:

  $ min_(bold(W) comma bold(H)) D lr((bold(V) bar.v bold(W) bold(H))) upright(" avec ") bold(W) gt.eq 0 comma bold(H) gt.eq 0 $

]

#slide(title: [$beta$-NMF])[
  - *Distance*: $beta$-divergence avec $beta = 0$: divergence d'Itakura-Saito  @basuRobustEfficientEstimation1998
  - *Réprésentation spectrale:* Spectrogrammes de puissance @fevotteSingleChannelAudioSource2018
  - *Algorithme*: _Multiplicative Updates_ @fevotteAlgorithmsNonnegativeMatrix2011
]

#new-section-slide[Stratégies]

#slide(title: [Structures espérées dans $bold(H)$])[
  #figure(image("../2024-05-17/creuse.png"), caption:[Résultat idéal])
]

#slide(title: [Estimation du gain et du _warp_])[
  #strong[Gain de la _track_ $i$ au temps mix $tau$]

  $=$ ampltiude de la colonne $tau$ de $bold(H)_i$
$ tilde(g) [tau] = sqrt(sum_(t=0)^(T-1) (bold(H)_i)_(t tau) ) $

  #strong[_Warp_ de la _track_ $i$ au temps mix $tau$]

  $=$ centre de masse de la colonne $tau$ de $bold(H)_i$
$ tilde(f) [tau] = (sum_(t=0)^(T-1) t (bold(H)_i)_(t tau)) / (sum_(t=0)^(T-1) (bold(H)_i)_(t tau)) $

]

#let torange(x) = text(fill: orange, $#x$)
#let tblue(x) = text(fill: blue, $#x$)

#slide(title: [_Tracks_ connues])[
  Si toutes les _tracks_ sont connues: on fixe les $bold(W)_i$

  $
  bold(V) &= underbrace([tblue(bold(W)_1 bold(W)_2 ... bold(W)_M)], bold(W)) underbrace(mat(delim: "[", torange(bold(H)_1); torange(bold(H)_2); torange(dots.v); torange(bold(H)_M)), bold(H))
  $
]
#slide(title: [Éléments inconnus])[
  - Foule, bruit, chant, boite à rythme ...
  - Représentés par un couple supplémentaire $(bold(W)_a, bold(H)_a)$ non fixé
  $
     bold(V) &= underbrace([tblue(bold(W)_1 bold(W)_2 ... bold(W)_M) torange(bold(W)_a)], bold(W)) underbrace(mat(delim: "[", torange(bold(H)_1); torange(bold(H)_2); torange(dots.v); torange(bold(H)_M); torange(bold(H)_a)), bold(H))
  $
]

#slide(title: [Compression de l'information])[
  - Utilisation de pas larges dans la STFT ($~$ 0.1 à 1 s.)
  - Utilisation du mel-spectrogramme ($~$ 128 à 512 _bins_)
]

#slide(title: [Utilisation du recouvrement])[
  
  - Exploite la continuité des signaux
  - Mais moyenne les résultats sur la longueur de la fenêtre
]


#slide(title: [NMF multi-passes])[
  + *Première passe*: très grand pas ($~$ 5 à 20 s.)
    + Initialiser $bold(H)$ aléatoirement
    + NMF
  + "Creuser" $bold(H)$
  + *Deuxième passe*: petit pas ($~$ 0.1 à 1 s.)
    + Initialiser $bold(H)$ avec celui de la première passe (redimensionné)
    + NMF
]

#new-section-slide[Expériences]

#slide(title:[Estimation du _warp_])[
  #figure([#image("../2024-04-26/results/boucles/out.png")],
  caption: [
    Morceau découpé et bouclé
  ]
)
]

#slide(title:[Estimation du _warp_])[
  #figure([#image("../2024-04-26/results/timestretch/out.png")],
  caption: [
    Morceau _timestretché_
  ]
)
]

#slide(title:[Estimation du gain])[
  #figure([#image("../2024-04-26/results/fondu/out.png")],
  caption: [
    Mélange beat-synchrone de 2 morceaux de _tech house_
  ]
)
]

#new-section-slide[Extensions et pistes]

#slide(title: [Structure de $bold(H)$])[
    #figure(image("../2024-05-17/image.png", height: 77%), caption: [Essai sur UnmixDB @schwarzUnmixDBDatasetDJMix2018])
]
#slide(title: [Structure de $bold(H)$])[
    #figure(image("../2024-05-17/image-2.png"), caption: [Indéterminations liées à des boucles dans le morceau original])
]

#slide(title: [Structure de $bold(H)$])[
    #figure(image("../2024-05-30/resultat.png"), caption: [indéterminations liés au décalage des trames des spectrogrammes])
]
#slide(title: [Structure de $bold(H)$])[
    #figure(image("../2024-05-30/longue-note.png", width: 50%), caption: [Indéterminations dues à de longues notes])
]

#slide(title: [Structure de $bold(H)$])[
  Deux pistes:
  - Ajout de termes de régularisation ?
  - Post-traitement @mullerEnhancingSimilarityMatrices2006 de $bold(H)$?
]

#slide(title: [Robustesse aux transformations de timbre])[
  - EQ 3 bandes:
    - découpage des spectres en bandes

  - Transposition:
    - déjà robuste aux petites transpositions avec le mel-spectrogramme
    - NMF2D @aarabiMusicRetilerUsing2018 ? 
    - Mesure de divergence invariante à la transposition @hennequinSpectralSimilarityMeasure ? 
]
#new-section-slide[Questions]

#slide(title: [Questions])[
  #line-by-line[
     - Divergence d'Itakura-Saito + spectrogramme de puissance: adapté ?
     - Séparation de sources + transcription: objectifs compatibles ?
     - Propriétés à rechercher dans une fonction de régularisation NMF ? Garanties de convergence, monotonicité ?
     - Habituellement, dimension de $bold(W)$ < dimension de $bold(V)$. Ici ce n'est pas le cas. Problématique ?
]
]

#slide(title: "Bibliographie")[
  #bibliography("../../zotero.bib", title: none, style: "chicago-notes")
]