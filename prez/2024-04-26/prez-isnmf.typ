#bibliography("../../zotero.bib")
= Factorisation de matrices non-négatives (NMF)
<factorisation-de-matrices-non-négatives-nmf>
== Non-negative Matrix Factorization (NMF)
<non-negative-matrix-factorization-nmf>
#strong[Principe]: sachant une matrice $bold(V)$, estimer $bold(W)$ et
$bold(H)$ tels que:

$ bold(V) approx bold(W H) $


#strong[Application à l’audio]:

- Séparation de sources
- Transcription
- Restauration

#figure([#image("nmf-audio.png")],
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

== Représentation spectrale et choix de la distance
<représentation-spectrale-et-choix-de-la-distance>
Pour de l’audio :

- divergence d’Itakura-Saito ($beta eq 0$)
- spectres de puissance

= Formulation du problème
<formulation-du-problème>
== Objectif
<objectif>
#strong[Sachant]

- Un enregistrement d’un mix DJ (#emph[mix])
- Les enregistrements des morceaux composant le mix (#emph[reference
  tracks])

#strong[Estimer]

- Les transformations temporelles (alignement, boucles, sauts)
- Les transformations de timbre (pitch shift, filtres, …)
- Les éléments supplémentaires (voix, foule, …)
- Leur évolution au cours du temps

== Matériels et logiciels DJ typiques
<matériels-et-logiciels-dj-typiques>
#figure([#image("dj-deck.png", height: 50%)],
  caption: [
    Chaîne de traitement de matériel DJ usuel
  ]
)



#figure([#image("separate-boxes.png")],
  caption: [
    Séparation des éléments de la chaîne de traitement
  ]
)

== Formulation NMF: transcription et séparation de sources
<formulation-nmf-transcription-et-séparation-de-sources>
#block[
#block[
#figure([#image("nmf-djmix.png", height: 70%)],
  caption: [
    Chaîne de traitement sous forme NMF
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
#figure([#image("nmf-djmix.png", height: 70%)],
  caption: [
    Chaîne de traitement sous forme NMF
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

== Éléments inconnus
<éléments-inconnus>
S’il y a des éléments inconnus, on les représente par un couple
supplémentaire $lr((bold(W)_a comma bold(H)_a))$

\$\$ \\mathbf{V} \= {\\color{orange}\\underbrace{\[\\mathbf{W}\_1|\\mathbf{W}\_2|...|\\mathbf{W}\_M}\_\\text{fixé}}
  {\\color{blue}\\underbrace{|\\mathbf{W}\_a\]
    \\begin{bmatrix}
    \\mathbf{H}\_{1} \\\\
    \\mathbf{H}\_{2} \\\\
    \\vdots \\\\
    \\mathbf{H}\_{M} \\\\
    \\mathbf{H}\_a
  \\end{bmatrix}
  }\_\\text{estimé}}
\$\$

== Reconstruction des sources
<reconstruction-des-sources>
"source" $i$ \= le morceau $i$ transformé temporellement

$ bold(V)_i eq bold(V) times frac(bold(W)_i bold(H)_i, bold(W H)) $

\=\> évaluation de la qualité de la séparation

== Résultat espéré
<résultat-espéré>
#figure([#image("../2024-03-26/temps-temps.drawio.png", height: 80%)],
  caption: [
    Exemples de courbes temps-temps
  ]
)

== Estimation des paramètres
<estimation-des-paramètres>
#strong[Volume relatif du morceau $i$ à la frame $n$]

$approx$ ampltiude relative de la colonne $n$ de $bold(H)_i$

$ g_i lr((n)) eq frac(sum_(k eq 1)^K lr((bold(H)_i))_(k n), sum_(k eq 1)^K bold(H)_(k n)) $

#strong[Position dans le morceau $i$ à la frame $n$]

$approx$ centre de masse de la colonne $n$ de $bold(H)_i$

$ tau_i lr((n)) eq frac(sum_(k eq 1)^K k lr((bold(H)_i))_(k n), sum_(k eq 1)^K lr((bold(H)_i))_(k n)) $

= Essais & Résultats
<essais-résultats>



#figure([#image("results/cas-simple.png", height: 70%)],
  caption: [
    cas simple: ref\=mix (FS\=22050; NFFT\=1024; HOP\=256)
  ]
)

On retrouve bien $bold(H) eq bold(H)_1 eq bold(I)$



#figure([#image("results/boucles/out.png", height: 70%)],
  caption: [
    morceau découpé et bouclé
  ]
)

#link("results/boucles/original.wav")[ref.wav]
#link("results/boucles/boucled.wav")[mix.wav]
#link("results/boucles/estimated-0.wav")[est.wav]



#figure([#image("results/timestretch/out.png", height: 70%)],
  caption: [
    morceau timestretché
  ]
)

#link("results/timestretch/nuttah.wav")[ref.wav]
#link("results/timestretch/nuttah-timestretch.wav")[mix.wav]
#link("results/timestretch/estimated-0.wav")[est.wav]



#figure([#image("results/fondu/out.png", height: 70%)],
  caption: [
    mélange beat-synchrone de 2 morceaux de tech house
  ]
)

#link("results/fondu/linear-mix-1.wav")[ref0.wav]
#link("results/fondu/linear-mix-2.wav")[ref1.wav]
#link("results/fondu/linear-mix.wav")[mix.wav]
#link("results/fondu/estimated-0.wav")[est0.wav]
#link("results/fondu/estimated-1.wav")[est1.wav]



#figure([#image("results/bruit-no-wa/out.png", height: 70%)],
  caption: [
    même mélage + bruit
  ]
)

#link("results/bruit-no-wa/maya.wav")[bruit.wav]
#link("results/bruit-no-wa/estimated-0.wav")[est0.wav]
#link("results/bruit-no-wa/estimated-1.wav")[est1.wav]



#figure([#image("results/bruit-wa/out.png", height: 70%)],
  caption: [
    même mélage + bruit avec apprentissage de
    $lr((bold(W)_a comma bold(H)_a))$, $F eq 100$
  ]
)

#link("results/bruit-wa/estimated-0.wav")[est0.wav]
#link("results/bruit-wa/estimated-1.wav")[est1.wav]
#link("results/bruit-wa/estimated-2.wav")[est2.wav]

= Pistes d’amélioration
<pistes-damélioration>
== Complexité
<complexité>
- #strong[complexité mémoire]
  - NMF en mini-batchs 
- #strong[complexité de calcul]
  - NMF multi-résolution
    
  - Essayer d’autres représentations spectrales
  - Utiliser la connaissance a priori de position approximative des
    morceaux dans le mix

== Robustesse
<robustesse>
- Convergence de $bold(H)$ vers la forme voulue
  - influencer par l’initialisation (pour l’instant aléatoire)
  - régularisation
    
  - traiter $bold(H)$ entre chaque itération
    

== Estimation ou robustesse aux effets supplémentaires
<estimation-ou-robustesse-aux-effets-supplémentaires>
- invariance au pitch: NMF2D 
- estimation aveugle des $f_i$ ?
