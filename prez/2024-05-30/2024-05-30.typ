#import "@preview/polylux:0.3.1": *
#import themes.simple: *
#set math.equation(numbering: "1.")
#show: simple-theme.with()

#title-slide[
  = Avancement DJ-unmixing
  30/05/2024
]
#centered-slide[
  = Contexte
]
#slide[
  == Objectif
  *Sachant*

  - Un enregistrement d’un mix DJ (#emph[mix])
  - Les enregistrements des morceaux composant le mix (#emph[reference
    tracks])

  *Estimer*

  - Les transformations temporelles (alignement, boucles, sauts)
  - Les transformations de timbre (pitch shift, filtres, …)
  - Les éléments supplémentaires (voix, foule, …)
  - Leur évolution au cours du temps
]

#slide[
  == Matériels et logiciels DJ usuels
  #figure([#image("../2024-04-26/dj-deck.png")],
    caption: [
      Chaîne de traitement de matériel DJ usuel
    ]
  )
]

// #slide[
//   #figure([#image("../2024-04-26/separate-boxes.png")],
//     caption: [
//       Séparation des éléments de la chaîne de traitement
//     ]
//   )
// ]

#slide[
  == Formulation matricielle

  #side-by-side[
    #figure([#image("../2024-04-26/nmf-djmix.png")],
      caption: [
        Chaîne de traitement sous forme matricielle
      ]
    )
    ][#text(size: 17pt)[
  $ bold(V) eq sum_(i eq 1)^M f_i lr((bold(W)_i bold(H)_i)) $

  Si de plus $forall i med f_i eq bold(1)$ alors:

  $ bold(V) eq sum_(i eq 1)^M bold(W)_i bold(H)_i eq underbrace(lr([bold(W)_1 bold(W)_2 ... bold(W)_M]), bold(W)) underbrace(mat(delim: "[", bold(H)_1; bold(H)_2; dots.v; bold(H)_M), bold(H)) $
    ]]
]

#slide[
  == Non-negative Matrix Factorization (NMF)
  <non-negative-matrix-factorization-nmf>
  #strong[Principe]: sachant une matrice $bold(V)>=0$, estimer $bold(W)>=0$ et $bold(H)>=0$ tels que: $ bold(V) approx bold(W H) $
    #figure(image("../2024-04-26/nmf-audio.png", width: 70%), caption: [Transcription avec la NMF (R. Badeau)])
]

#slide[
  == Résultats
  #figure(image("resultat.png", width: 60%), caption: [Matrice d'activation estimée pour un mix de 2 pistes])
]

#slide[

  == Amélioration: NMF multi-passes avec creusage
#figure([#image("../2024-05-17/creuse.png", height:70%)])

]

#slide[


  - Un 0 dans H restera toujours 0 au fil des itérations
  - Utilisation d'une structure de données de matrice creuse par blocs

  == Algorithme
  <algorithme>
  + Initialiser $bold(H)$ aléatoirement
  + Pour `hop_len` dans {_grand_, _petit_}:
    + Effectuer la STFT avec le pas `hop_len`
    + Effectuer la NMF
    + "Creuser" la matrice $bold(H)$
    + Redimensionner $bold(H)$ pour initialiser le tour d’après
]

// #slide[
//   #strong[Objectif]: minimiser la distance $d$ entre $bold(V)$ et
//   $bold(W) bold(H)$:

//   $ min_(bold(W) comma bold(H)) D lr((bold(V) bar.v bold(W) bold(H))) upright(" avec ") bold(W) gt.eq 0 comma bold(H) gt.eq 0 $

//   #strong[Distance]: Divergence d'Itakura-Saito

//   $ D_(I S) lr((bold(V) bar.v bold(W) bold(H))) eq sum_(f eq 1)^F sum_(n eq 1)^N d lr((bold(V)_(f n) bar.v lr((bold(W H)))_(f n))) $

//   $ d_(I S) lr((x bar.v y)) = x / y - log(x y) - 1 $
// ]

#centered-slide[
  = Caractérisation des transformations temporelles
]

#slide[
#figure(image("../2024-03-26/temps-temps.drawio.png"),
  caption: [
    Exemples de $bold(H_i)$ espérés
  ]
)
]

#slide[
  == Définitions
  #figure(image("remap-gain.png"))
  Soient:
  - $x[t]$ un signal ;
  - $f: tau |-> t$ une fonction injective de remappage temporel;
  - $g[tau]$ un signal de gain à valeurs dans [0,1];
  - $w$ une fenêtre de taille $M$ ;
  - $y[tau]$ le signal transformé.
]

#slide[
  == STFT de $x$ et $y$
  $ bold(X)_(m t) = abs(sum_(n=0)^(M-1) x[n+t] w[n] e^(-j 2 pi n m / M))^2 $
  $ bold(Y)_(m tau) &= abs(g[tau] sum_(n=0)^(M-1) x[n+f[tau]] w[n] e^(-j 2 pi n  m/ M))^2 $
]

#slide[ 
  == Matrice d'activation
  Trouver une transformation temporelle de $bold(X)$ vers $bold(Y)$ \
  $<=>$ trouver une matrice d'activation $H$ qui satisfait $forall omega, tau$:
  $ bold(Y) &= bold(X) bold(H) <=> bold(Y)_(m tau) = sum_(t=0)^(T-1) bold(X)_(m t) bold(H)_(t tau) $

  Une solution: la matrice d'activation _idéale_ $ bold(H)_(t tau) eq.def g[tau]² delta_(t,f[tau]) $
]

#slide[
  == Estimateurs de $f$ et $g$

  - Magnitude : $tilde(g)[tau]^2 = sum_(t=0)^(T-1) bold(H)_(t tau) $

  - Centre de masse : $tilde(f)[tau]=(sum_(t=0)^(T-1) t bold(H)_(t tau)) / (sum_(t=0)^(T-1) bold(H)_(t tau)) $

  Dans le cas de la matrice d'activation idéale:
  $ tilde(g)[tau] = g[tau] "et" tilde(f)[tau] = f[tau] $

]

#focus-slide[
  Pas tout à fait vrai en pratique :(
]

#slide[
  == En pratique
  === NMF
  
  $=>$ bruit dans $bold(H)$

]

#slide[
  == En pratique
  === STFT avec pas temporel
  
  $=>$ désalignement des trames

]

#slide[
  === similarité des spectres de 2 instants voisins
  #figure([#image("etalement.png", height: 69%)])

  $=>$ *étalement des activations*

]

#slide[
  === Boucles dans $x$
  #figure([#image("../2024-05-17/image-2.png", height: 69%)])

  $=>$ *indétermination des $f[tau]$*
]

#slide[
  === Sons longs et stationnaires
  #figure([#image("longue-note.png", height: 69%)])

  $=>$ *indétermination des $f[tau]$*
]

#centered-slide[
  = Suite
]

#slide[
  == Evaluation des performances

  === Avec ground truth
  Erreur d'estimation des $f$ et $g$
  
  === Sans ground truth
  Erreur de reconstruction du mix
]

#slide[
  == Ajout de pénalités
  
  - $f$ devrait être affine par morceaux $=>$ récompenser la _lignitude_ de $bold(H)$
  - $g$ ne devrait pas varier trop vite $=>$ récompenser la _smoothness_ de $bold(H)$
]

#slide[
  == Robustesse des estimateurs

  $f$ et $g$ sont-ils robustes en présence:

  - de bruit ?
  - d'étalement des activations ?
]

#slide[
  == Ajout d'éléments au modèle

  - repitch
  - filtres + EQ
]