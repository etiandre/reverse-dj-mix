#import "@preview/polylux:0.3.1": *
#import themes.simple: *
#set math.equation(numbering: "1.")
#show: simple-theme.with()


#title-slide[
  = Avancement DJ-unmixing
  18/07/2024
]

#slide[
  == Creusage entre passes
  - *Filtrage morphologique* @mullerEnhancingSimilarityMatrices2006 \ Pour _pente_ dans _min_pente_, _max_pente_:
    + Générer une image de d'une ligne longueur _n_morpho_ et de pente _pente_
    + Accumuler l'_opening_ de $bold(H)$ par cette image
  - Pour chaque $bold(H)_i$ (sous-partie de $bold(H)$):
    + *Flou gaussien*: Flouter bold(H_i) par un noyau de taille _n_flou_
    + *Seuillage*: Mettre à 0 les cellules inférieures à _threshold_
  - *Redimenssionnement*: redimensionner $bold(H)$ à la taille de la passe suivante
]

#slide[
  == KL-divergence vs IS-divergence

  - Besoin d'associer @fevotteNonnegativeMatrixFactorization2009
    - KL avec spectre d'amplitude
    - IS avec spectre de puissance
  - KL Semble diminiuer l'influence de l'initialisation, mais pas mentionné dans la littérature ?
]

#slide[
  == EQ sur les mixeurs DJ
  
  - EQ 3 bandes ou 4 bandes (rare)
  - $f_c$ typiques: 250 Hz, 2500 Hz
  - crossover à somme unité: respecte $sum bold(W)_i = bold(V)$
  
  *Estimation*
  - Filtrer chaque _track_ d'entrée en bandes de fréquences délimitées par les $f_c$
  - Traiter chacune de ces bandes comme une _track_ indépendante
]

#bibliography("../../zotero.bib", style: "chicago-notes")
