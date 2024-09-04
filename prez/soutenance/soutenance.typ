#import "@preview/polylux:0.3.1": *
#import "@preview/lovelace:0.3.0": *
#import themes.metropolis: *
#show: metropolis-theme.with()
#set text(lang: "fr", size: 20pt)
#show figure.caption: set text(size: 15pt, style: "italic")
#set text(font: "Inria Sans", weight: "light", size: 17pt)
// #show math.equation: set text(font: "TeX Gyre Termes Math")
// #set math.equation(numbering: "(1)")
#show math.equation.where(block: true): it => rect(width: 100%, fill: rgb("#def4ff"))[
  #v(0.5em)
  #it
  #v(0.5em)
]
#pdfpc.config(duration-minutes: 15)
#let my-lovelace-defaults = (
  booktabs: true,
)
#let pseudocode = pseudocode.with(..my-lovelace-defaults)
#let pseudocode-list = pseudocode-list.with(..my-lovelace-defaults)

#let HH = math.bold(math.upright([H]))
#let XX = math.bold(math.upright([X]))
#let YY = math.bold(math.upright([Y]))
#let NN = math.bold(math.upright([N]))
#let argmax = math.op("argmax", limits: true)

#title-slide(
  title: "Transcription de mix DJ par factorisation de matrices non-négatives à passes multiples",
  author: [Étienne André \ _Encadrants:_ Dominique Fourer - Diemo Schwarz],
  date: "06/09/2024",
)

#slide(title: "Plan")[
  #metropolis-outline
]

#new-section-slide("Introduction")

#slide(title: [DJs et DJing])[
  #figure(
    image("DaveMcNaughton50s.jpg"),
    caption: [Studio de la radio canadienne CFRC, fin des années 50. (Archives de l'université Queen's, Kingston, Canada. Ralfe Clench fonds 5064.7-1-7)],
  )

  #pdfpc.speaker-note(```md
  DJ = Disc-Jockey, littéralement pilote de disques

  Origine: radio, besoin de passer de la musique de manière continue.

  DJ = sorte de juke-box humain.

  Matériel rudimentaire: besoin de "meubler" entre les morceaux.
  ```)
]

#slide(title: [DJs et DJing])[
  #figure(
    image("grandmaster-flash.jpg", width: 60%),
    caption: [Le DJ _Grandmaster Flash_ sur des platines à tempo ajustable. Vers 1980. (Cornell University digital library)],
  )
  #pdfpc.speaker-note(```md
  Spécialisation du matériel années '70, platines à tempo ajustable, tables de mixages + sophistiquées, donne lieu notamment dans le hip hop au scratching.

  Passe d'opérateur à performance artistique
  ```)
]

#slide(title: [DJs et DJing])[
  #side-by-side[
    #figure(
      image("serato.png"),
      caption: [Logiciel de DJing],
    )

  ][
    #figure(
      image("modern-dj.jpg"),
      caption: [Matériel DJ moderne avec platines et lecteurs numériques, mixeur, sampleur et console d'effets],
    )
  ]
  #pdfpc.speaker-note(```md
  Aujourd'hui, DJ et DJing omniprésents dans la culture populaire.

  Matériel peut être totalement numérique, ou hybride.
  ```)
]

#slide(title: [Mix DJ])[
  *Mix:* performance artistique, consiste à sélectionner et transformer de manière créative des morceaux de musique existants.
]

#slide(title: [Transcription de mix DJ])[
  #line-by-line[
    *Sachant*
    - Un enregistrement d’un mix;
    - Les enregistrements des morceaux composant le mix;
    *Estimer*
    - Les transformations temporelles utilisées (étirement, boucles, sauts...) ;
    - L'évolution des gains de mix ;
    - Tout effet supplémentaire appliqué aux morceaux et/ou mix.
  ]
  #pdfpc.speaker-note(```md
    mix DJ: performance artistique, consiste à sélectionner et transformer de manière créative des morceaux de musique existants.

    effets supplémentaires: (distortion, compression, transposition, filtrage, égalisation...)
  ```)
]

#new-section-slide("Formulation matricielle d'un mix DJ")

#slide(title: [Généralisation du chemin du signal])[
  #figure(
    image("dj-process.drawio.svg", width: 80%),
    caption: [Vue généralisée d'une performance DJ.],
  ) <dj-signal-path>

  #pdfpc.speaker-note(```md
  lecteur=source sonore

  mixeur
  ```)
]

#slide(title: [Simplification du modèle])[
  #side-by-side[
    #figure(
      image("separate-boxes.drawio.svg"),
      caption: [Modèle simplifié.],
    ) <separate-boxes>
  ][
    / $x^((i))[t]$ : signal du morceau $i$
    / $y^((i))[tau]$ : signal transformé
    / $f^((i)): tau |-> t$ : déformation temporelle
    / $g^((i))[t]$ : gain
  ]
]

#slide(title: [Mise sous forme matricielle])[
  #one-by-one[
    Spectrogrammes : $XX^((i))$ et $YY^((i))$][

    Lien entre $YY^((i))$ et $XX^((i))$:
    $ YY^((i))_(m tau) &= g[tau]^2 XX^((i))_(m,f^((i))[tau]) $
  ][
    Peut s'exprimer sous la forme:
    $ YY^((i)) &= XX^((i)) HH^((i)) $
  ][
    $HH^((i))$: *matrice d'activation* du morceau $i$
    }]
]


#slide(title: [Matrice d'activation])[
  #side-by-side[
    Solution particulière: la matrice d'activation "idéale"
    $ tilde(HH)^((i))_(t tau) eq.def g^((i))[tau]^2 delta_(t,f^((i))[tau]) $
    #pause
    Estimateurs de $f$ et $g$:
    $ tilde(f)^((i)) [tau] = argmax_(t in [1...T]) tilde(HH)_(t tau) $ <time_estimator_argmax>
    $ tilde(g)^((i)) [tau] = sqrt(sum_(t=1)^(T) tilde(HH)_(t tau) ) $ <gain_estimator_sum>
    #pause
  ][
    #figure(
      image("H_exemple.svg"),
      caption: [Exemple d'une matrice d'activation idéale pour une transformation avec saut, accélération, et fondus],
    )
  ]
]

#slide(title: [Équation matricielle d'un mix DJ])[
  #side-by-side[
    #figure(
      image("nmf-djmix.drawio.svg"),
      caption: [Formulation matricielle.],
    ) <nmf-djmix>
  ][
    $ YY &= NN + sum_(i = 1)^N XX^((i)) HH^((i)) $
    #pause
    Factorisation de la matrice de bruit: $NN = overline(XX) #h(0.2em) overline(HH)$

  ]
]

#slide(title: [Mise sous forme de produit matriciel])[
  $
    YY = underbrace(mat(overline(XX) #h(0.2em) XX^((1)) XX^((2)) ... XX^((M))), XX) underbrace(mat( overline(HH); HH^((1)); HH^((2)); dots.v; HH^((M))), HH)
  $
]
#let tcolor(x) = text(fill: blue, $#x$)
#slide(title: [Mise sous forme de produit matriciel])[

  *Transcrire un mix $<==>$ déterminer $HH^((1)), HH^((2)) ...$*

  $
    YY = underbrace(mat(tcolor(overline(XX)) #h(0.2em) XX^((1)) XX^((2)) ... XX^((M))), XX) underbrace(mat( tcolor(overline(HH)); tcolor(HH^((1))); tcolor(HH^((2))); tcolor(dots.v); tcolor(HH^((M)))), HH)
  $
]
#new-section-slide("Factorisation en matrices non-négatives (NMF) à passes multiples")

#slide(title: "Factorisation en matrices non-négatives (NMF)")[
  #strong[Principe]: sachant une matrice _cible_ $bold(Y)>=0$, estimer le _dictionnaire_ $bold(X)>=0$ et l'_activation_ $HH>=0$ tels que:

  $ bold(Y) approx bold(X H) $

  #strong[Application à l’audio]:

  - Séparation de sources
  - Transcription
  - Restauration
]


#slide(title: [Problèmes])[
  #figure(
    image("similarity.png"),
    caption: [Matrice d'activation estimée par la NMF pour un mix de 3 morceaux],
  )
]

#slide(title: [NMF à passes multiples])[
  *Principe* :

  - Effectuer plusieurs passes de la NMF sur des spectrogrammes de résolution croissantes
  - Utiliser le résultat d'une passe précédente pour initialiser la suivante
  - Entre chaque passe, filtrer et "creuser" les matrices d'activation
  - Utiliser une structure de données adaptée aux matrice creuses
]

#slide(title: [Filtrage inter-passes])[
  #figure(
    image("interpass.svg", width: 120%),
    caption: [Étapes du filtrage inter-passes de la matrice d'activation],
  )
]

#slide(title: [NMF à passes multiples, exemple])[
  #figure(
    image("multipass1.svg"),
    caption: [Passe 1: résolution = 15 s.],
  )
]

#slide(title: [NMF à passes multiples, exemple])[
  #figure(
    image("multipass2.svg"),
    caption: [Passe 2: résolution = 5 s.],
  )
]

#slide(title: [NMF à passes multiples, exemple])[
  #figure(
    image("multipass3.svg"),
    caption: [Passe 3: résolution = 1 s.],
  )
]

#slide(title: [NMF à passes multiples, exemple])[
  #figure(
    image("multipass4.svg"),
    caption: [Passe 4: résolution = 0.5 s.],
  )
]

#new-section-slide([Résultats])

#slide[

]

#new-section-slide([Conclusion])

#slide(title: "Bibliographie")[
  #bibliography("../../zotero.bib", title: none, style: "chicago-notes")
]