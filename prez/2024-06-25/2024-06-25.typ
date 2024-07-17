#import "@preview/polylux:0.3.1": *
#import themes.simple: *
#set math.equation(numbering: "1.")
#show: simple-theme.with()


#title-slide[
  = Avancement DJ-unmixing
  25/06/2024
]

#centered-slide[
  = NMF pénalisée
]

#slide[
  == On cherche à corriger:
  #side-by-side[
  #figure([#image("../2024-05-17/image-2.png", height: 69%)], caption: "boucles dans les ref.")
  ][

  #figure([#image("../2024-05-30/longue-note.png", height: 69%)], caption: "sons longs/stationnaires")
  ]
]

#slide[
  == NMF Pénalisée
  Soient $bold(X)_(F times K)$, $bold(H)_(K times N)$ et
$bold(Y)_(F times N)$.

*Objectif:* minimiser une fonction de coût $cal(C)$

$ min_(bold(X) , bold(H)) cal(C)(bold(V), bold(W), bold(H)) "avec" bold(W) >= 0 , bold(H) >= 0 $

$ cal(C)(bold(V), bold(W), bold(H)) = D(bold(V) | bold(W) bold(H)) + lambda_1 P_1(bold(H)) + lambda_2 P_2(bold(H)) + ...  "où" lambda_i > 0$
]

#slide[
  == Algorithme "Multiplicative Updates"

 - calculer $gradient_bold(H) cal(C)$
 - séparer en termes positifs et négatifs: $ gradient_bold(H) cal(C) = gradient_bold(H)^+ cal(C) - gradient_bold(H)^- cal(C) $
 - obtenir la règle de mise à jour par:

$
bold(H) <- bold(H) (gradient_bold(H)^- cal(C)) / (gradient_bold(H)^+ cal(C))
$
 - idem pour $gradient_bold(W) cal(C)$
]


#slide[
  == Pénalisations de la littérature
  
  - *Régularisation L1 (Lasso)*: $P(bold(H)) = sum_(t,tau) abs(bold(H)_(t,tau))$
  - *Régularisation L2 (Ridge)*: $P(bold(H)) = sum_(t,tau) bold(H)_(t,tau)^2$
  - *Continuité sur lignes*: $P(bold(H)) = sum_(t,tau) (bold(H)_(t,tau) - bold(H)_(t, tau-1))^2$
  - *Continuité sur colonnes*: $P(bold(H)) = sum_(t,tau) (bold(H)_(t,tau) - bold(H)_(t-1, tau))^2$
  - *Continuité temporelle*@virtanenMonauralSoundSource2007: $P(bold(H)) = sum_t 1/sigma_t^2 sum_tau (bold(H)_(t,tau) - bold(H)_(t, tau-1))^2$
]

#slide[
  == Pénalisations pour transcription DJ
  === Continuité des diagonales
  
  $ P(bold(H)) = sum_(t=1)^(T-1) sum_(tau=1)^(K-1) (bold(H)_(t,tau) - bold(H)_(t-1, tau-1))^2 $
  
]

#slide[
  == Pénalisations pour transcription DJ
  === "Lignitude"

$
P(bold(H)) = sum_(t=0)^(T-2) sum_(tau=0)^(K-2) bold(H)_(t,tau) (&bold(H)_(t,tau+1) bold(H)_(t+1,tau+1) \ &+ bold(H)_(t+1,tau) bold(H)_(t+1,tau+1) \ &+ bold(H)_(t+1,tau) bold(H)_(t,tau+1) )
$
]

#centered-slide[
  = Cadre de travail
]

#slide[
  == Librairie NMF modulaire
  
```python
    divergence = BetaDivergence(beta=0)
    penalties = [
        (L1(), 1e2),
        (L2(), 1e-1),
        (VirtanenTemporalContinuity(), 1e3),
        ...
    ]
    nmf = NMF(V, W_init, H_init, divergence, penalties)
    nmf.fit(max_iter=500)
```
]

#slide[
  == Visualisation des pénalités

  cf. notebook
]

#slide[
  == Porté en pytorch

  - Basé sur pytorch-nmf#footnote("https://github.com/yoyololicon/pytorch-NMF")
  - Tourne sur CPU, GPU

  $=>$ "Petites" matrices: mieux vaut lancer $n_"cores"$ jobs sur CPU

  $=>$ "Grandes" matrices: mieux vaut lancer 1 job sur GPU
]

#slide[
  == Optimisation des hyperparamètres

  $ cal(C)(bold(V), bold(W), bold(H)) = D(bold(V) | bold(W) bold(H)) + lambda_1 P_1(bold(H)) + lambda_2 P_2(bold(H)) + ...$

  - quelles $P_i$ sont pertinentes ?
  - avec quels $lambda_i$ ?
  - saut de la STFT ?
  - taille de la fenêtre STFT ($<=>$ quantité d'overlap) ?
  - nombres de bins STFT ?

]

#slide[
  == Optimisation des hyperparamètres

  Mesures objectives de performance:

  - $D(bold(V) | bold(W H))$
  - erreur d'estimation des gains
  - erreur d'estimation du time-remapping

  $==>$ Problème d'optimisation multi-objectif
]

#slide[
  == Optimisation des hyperparamètres
  
  Utilisation d'un algorithme génétique implémenté dans optuna #footnote[https://optuna.org/]

  Dans un premier temps, tentative d'"overfit" sur un exemple

  cf. dashboard
]

#slide[
  == Pistes NMF
  
  - Essayer d'autres pénalités
  - "$lambda$-warmup" @sonderbyLadderVariationalAutoencoders2016 @driedgerLetItBee2015
]

#slide[
  == Pistes hors NMF
  #side-by-side[
  #figure([#image("../2024-05-17/image-2.png", height: 70%)])
  ][
    - NMF non régularisée
    - $bold(H)$ est une sorte de matrice de similarité (?)
    - Postprocessing
    - Exploiter la littérature sur la matrice d'autosimilarité (estimation de structure)
  ]
  
]


#bibliography("../../zotero.bib", style: "chicago-notes")