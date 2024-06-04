= Transcription de mix par NMF
<transcription-de-mix-par-nmf>
== Non-negative Matrix Factorization (NMF)
<non-negative-matrix-factorization-nmf>
#strong[Principe]: sachant une matrice $bold(V)$, estimer $bold(W)$ et
$bold(H)$ tels que:

$ bold(V) approx bold(W H) $

#figure([#image("../2024-04-26/nmf-audio.png")],
  caption: [
    Transcription avec la NMF (R.Badeau)
  ]
)

== Beta-NMF
<beta-nmf-fevottealgorithmsnonnegativematrix2011>
Soient $bold(W)_(F times K)$, $bold(H)_(K times N)$ et
$bold(V)_(F times N)$.

#strong[Objectif]: minimiser la distance $d$ entre $bold(V)$ et
$bold(W) bold(H)$:

$ min_(bold(W) comma bold(H)) D lr((bold(V) bar.v bold(W) bold(H))) upright(" avec ") bold(W) gt.eq 0 comma bold(H) gt.eq 0 $

#strong[Distance]: $beta$-divergence

$ D_beta lr((bold(V) bar.v bold(W) bold(H))) eq sum_(f eq 1)^F sum_(n eq 1)^N d lr((bold(V)_(f n) bar.v lr((bold(W H)))_(f n))) $
$ d lr((x bar.v y)) eq cases(delim: "{", frac(1, beta lr((beta minus 1))) lr((x^beta plus lr((beta minus 1)) y^beta minus beta x y^(beta minus 1))) & beta eq.not brace.l 0 comma 1 brace.r, x log x / y minus x plus y & beta eq 1, x / y minus log x y minus 1 & beta eq 0) $

== Algorithme d’inférence (MU)
<algorithme-dinférence-mu>
notation: $bold(hat(V)) eq bold(W H)$

+ Initialiser $bold(W) gt.eq 0$ et $bold(H) gt.eq 0$
+ Mettre à jour successivement $bold(W)$ et
  $bold(H)$#footnote[$dot.op times dot.op$ et
  $frac(med dot.op med, med dot.op med)$ sont élément par élément]:
  $ bold(H) arrow.l bold(H) times frac(bold(W)^T lr((bold(hat(V))^(beta minus 2) times bold(V))), bold(W)^T bold(hat(V))^(beta minus 1)) $
  $ bold(W) arrow.l bold(W) times frac(lr((bold(hat(V))^(beta minus 2) times bold(V))) bold(H)^T, bold(hat(V))^(beta minus 1) bold(H)^T) $
+ Répéter l’étape 2 jusqu’à convergence ou nombre d’itérations maximum

== Mix DJ sous forme matricielle
<mix-dj-sous-forme-matricielle>
#block[
#block[
#figure([#image("../2024-04-26/nmf-djmix.png", height: 70%)],
  caption: [
    Chaîne de traitement sous forme matricielle
  ]
)

]
#block[
- matrices de bases $bold(W)_i$: spectre de puissance des morceaux de
  référence
- matrices d’activations $bold(H)_i$: transformations temporelles
  (timestretch, boucles, delays…) + gain
- $bold(V)_i$: spectres transformés temporellement
- fonctions $f_i$: le reste des transformations (filtres, pitch,
  distortion…)

]
]


#block[
#block[
#figure([#image("../2024-04-26/nmf-djmix.png", height: 70%)],
  caption: [
    Chaîne de traitement sous forme matricielle
  ]
)

]
#block[
$ bold(V) eq sum_(i eq 1)^M f_i lr((bold(W)_i bold(H)_i)) $

Si de plus $forall i med f_i eq bold(1)$ alors:

$ bold(V) eq sum_(i eq 1)^M bold(W)_i bold(H)_i eq underbrace(lr([bold(W)_1 lr(|bold(W)_2|) dot.basic dot.basic dot.basic bar.v bold(W)_M]), bold(W)) underbrace(mat(delim: "[", bold(H)_1; bold(H)_2; dots.v; bold(H)_M), bold(H)) $

]
]


Si tous les morceaux sont connus, on fixe les $bold(W)_i$

\$\$ \\mathbf{V} \= {\\color{orange}\\underbrace{\[\\mathbf{W}\_1|\\mathbf{W}\_2|...|\\mathbf{W}\_M\]}\_\\text{fixé}}
  {\\color{blue}\\underbrace{
    \\begin{bmatrix}
    \\mathbf{H}\_{1} \\\\
    \\mathbf{H}\_{2} \\\\
    \\vdots \\\\
    \\mathbf{H}\_{M}
  \\end{bmatrix}
  }\_\\text{estimé}}
\$\$

= Améliorations, essais
<améliorations-essais>
== Choix hyperparamètres
<choix-hyperparamètres>
Meilleurs résultats avec

- grandes fenêtres d’analyse ($gt.eq$ 0.5s) et overlap $gt.eq$ 75%:
  favorise la continuité temporelle
- `n_mels` \= 256: réduction de dimensionalité et bons résultats


- Critère d’arrêt: stabilisation de la divergence ($Delta d lt$ seuil)
- Parallélisation du chargement des audios + génération des spectres

== NMF multi-résolution et matrices creuses
<nmf-multi-résolution-et-matrices-creuses>
#figure([#image("creuse.png", height: 70%)],
  caption: [
    H idéalisé
  ]
)


- Propriété NMF avec MU: Un 0 dans H (ou W) restera toujours 0
- Si connaissance a priori de position des tracks: remplir de 0 à
  l’initialisation

== Algorithme
<algorithme>
+ Initialiser $bold(H)$ aléatoirement
+ Pour `wlen` dans {32,16,8,4,2,1}:
  + Transformer les entrées avec la taille de fenêtre `wlen`
  + Effectuer la NMF
  + Creuser la matrice $bold(H)$
  + Redimensionner $bold(H)$ pour initialiser tour d’après

== Essais non concluants
<essais-non-concluants>
- MFCC: fonctionne moins bien à `n_features` égal
- fenêtres d’analyse beat-synchrones
- smooth NMF

= Problèmes restants et pistes
<problèmes-restants-et-pistes>
== Invariance / détection du repitch
<invariance-détection-du-repitch>
- NMF2D
- Divergence invariante au pitch ?
  

== Robustesse au filtrage
<robustesse-au-filtrage>
- EQ 3 bandes typique: toujours les mêmes $f_c$
- #strong[Découper les spectres en bandes]

== Incertitude dans les reference tracks
<incertitude-dans-les-reference-tracks>
#figure([#image("image.png", height: 70%)],
  caption: [
    Incertitudes dans H (`wlen`\=2s)
  ]
)


#figure([#image("image-2.png", height: 80%)],
  caption: [
    Incertitudes dans H (`wlen`\=1s, zoomé)
  ]
)
