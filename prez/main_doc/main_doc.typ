#import "@preview/ilm:1.1.1": *

#show: ilm.with(
  title: [bla bli blo],
  author: "Étienne ANDRÉ",
  date: datetime.today(),
  abstract: [],
  preface: [],
  // bibliography: bibliography("refs.bib"),
  figure-index: (enabled: true),
  table-index: (enabled: true),
  listing-index: (enabled: true)
)


// #set math.equation(numbering: "(1)")
// #set heading(numbering: "1.")

#let oslash = symbol("⊘")

= Contexte
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

- matrices de bases $bold(X)_((i))$: spectre de puissance des morceaux de
  référence
- matrices d’activations $bold(H)_((i))$: transformations temporelles
  (timestretch, boucles, delays…) + gain
- $bold(H)_((i))$: spectres transformés temporellement
- fonctions $f_i$: le reste des transformations (filtres, pitch,
  distortion…)

= Non-negative matrix factorization
== Beta-NMF 
<beta-nmf-fevottealgorithmsnonnegativematrix2011>
Soient $bold(X)_(F times K)$, $bold(H)_(K times N)$ et
$bold(Y)_(F times N)$.

#strong[Objectif]: minimiser la distance $d$ entre $bold(Y)$ et
$bold(X) bold(H)$:

$ min_(bold(X) , bold(H)) D (bold(Y) | bold(X) bold(H)) "avec" bold(X) >= 0 , bold(H) >= 0 $

#strong[Distance]: $beta$-divergence

$ D_beta lr((bold(Y) | bold(X) bold(H))) = sum_(f = 1)^F sum_(n = 1)^N d lr((bold(Y)_(f n) | lr((bold(W H)))_(f n))) $
$ d lr((x | y)) = cases(
  frac(1, beta lr((beta - 1))) lr((x^beta + lr((beta - 1)) y^beta - beta x y^(beta - 1))) & "si" beta != {0, 1},
  x log x / y - x + y & "si" beta = 1,
  x / y - log x y - 1 & "si" beta = 0
) $

== Algorithme d’inférence (MU)
$dot.circle$ et #oslash désignent respectivement le produit et la division d'Hadamard (terme-à-terme).
+ Initialiser $bold(X) >= 0$ et $bold(H) >= 0$
+ Mettre à jour successivement $bold(X)$ et
  $bold(H)$
  $ bold(H) arrow.l bold(H) dot.circle [bold(X)^T ((bold(W H))^(beta - 2) dot.circle bold(Y))] #oslash [bold(X)^T (bold(W H))^(beta - 1)] $
  $ bold(X) arrow.l bold(X) dot.circle [((bold(W H))^(beta - 2) dot.circle bold(Y)) bold(H)^T] #oslash [(bold(W H))^(beta - 1) bold(H)^T] $
+ Répéter l’étape 2 jusqu’à convergence ou nombre d’itérations maximum

== Mix DJ sous forme de combinaison de spectogrammes

$ bold(Y) = sum_(i = 1)^M f_((i)) lr((bold(X)_((i)) bold(H)_((i)))) $

Si de plus on néglige tout effet de timbre ($forall i, f_i = bold(1)$), alors:

$ bold(Y) = sum_(i = 1)^M bold(X)_((i)) bold(H)_((i)) = underbrace(mat(bold(X)_((1)) bold(X)_((2)) ... bold(X)_((M))), bold(X)) underbrace(mat( bold(H)_((1)); bold(H)_((2)); dots.v; bold(H)_((M))), bold(H)) $


S’il y a des éléments inconnus, on les représente par un couple
supplémentaire $lr((bold(X)_((a)) , bold(H)_((a))))$:

$ bold(Y) = sum_(i = 1)^M bold(X)_((i)) bold(H)_((i)) = underbrace(mat(bold(X)_((1)) bold(X)_((2)) ... bold(X)_((M))bold(X)_((a))), bold(X)) underbrace(mat( bold(H)_((1)); bold(H)_((2)); dots.v; bold(H)_((M)); bold(H)_((a))), bold(H)) $

== Reconstruction des morceaux
Il est possible de reconstruire les spectrogrammes transformés $bold(Y)_((i))$:

$ bold(Y)_((i)) = bold(Y) dot.circle [bold(X)_((i)) bold(H)_((i))] #oslash [bold(W H)] $

Cela permet d'évaluer objectivement la qualité de l'estimation. En récupérant l'information de phase des morceaux originaux, les audios peuvent être reconstruits et soumis à une évaluation objective et perceptive.

= Characterization of time-remapping and gain

In this section, we will be characterizing what happens to the $bold(H)_((i))$ in the case of a time-remapping and gain transformations. To simplify the notations, we will drop the $(i)$ subscripts in this whole section.

== Continuous formulation

Let:
- $x$ be a real-valued signal;
- $f$ a time-remapping injective function (with values in $RR$), that maps a time of $x(t)$ to another;
- $g$ be a gain signal with values in [0,1];
- $w$ be a symmetric window function of length $F$;
- $y$ to the time-remapped and gain-affected transformation of $x(t)$;
- $t$ denotes time in the $x$ signal, $tau$ denotes time in the $y$ signal.

We define $X$, the power-STFT of $x$, to be:
$ X (omega, t) eq.def abs(integral_RR x(u+t) w(u) e^(-j omega u) d u)^2 $

Similarly, we define $Y$, the power-STFT of $y$, and show that it can be expressed in terms of $X$:

$ Y (omega, tau) &eq.def abs(g(tau) integral_RR x(u+f(tau)) w(u) e^(-j omega u) d u)^2 \
&= g(tau)² X(omega, f(tau)) $

Now, our goal is to decompose $Y$ into a continuous linear combination of $X$, with fixed $omega$. In other words, find an integral transform of $X$ that yields $Y$. This involves determining a kernel $H$ that satisfies:
$ Y(omega, tau) = integral_RR X(omega, t) H (t, tau) d t $ <integral_transform>

The kernel $H(t, tau) eq.def g(tau)^2 delta(t-f(tau))$ (where $delta(0) = 1$ and $0$ elsewhere) is a particular solution to @integral_transform:
$ Y(omega, tau) &= integral_RR X(omega, t) g(tau)² delta(t-f(tau)) d t \
&= g(tau)² X(omega, f(tau))
$
== Discrete formulation

Let:
- $x[n]$ be a real-valued signal;
- $f[n]$ a time-remapping injective sequence with values in $NN$, that maps a frame of $x[n]$ to another;
- $g[n]$ be a gain signal with values in [0,1];
- $w[n]$ be a symmetric window function of length $M$.

We define $bold(X) = (bold(X)_(m t))$ to be the matrix containing the power spectrogram of $x$ ($M$ frequency bins $times T$ time steps):
$ bold(X)_(m t) = abs(sum_(n=0)^(M-1) x[n+t] w[n] e^(-j 2 pi n m / M))^2 $

We then define the signal $y$ to be the time-remapped and gain-affected transformation of $x$.

Similarly, we define $bold(Y) = (bold(Y)_(m tau))$ to be the matrix containing the power spectrogram of $y$ ($M$ frequency bins $times K$ time steps):

$ bold(Y)_(m tau) &= abs(g[tau] sum_(n=0)^(M-1) x[n+f[tau]] w[n] e^(-j 2 pi n  m/ M))^2 \
&= g[tau]² bold(X)_(m,f[tau])
$

Then we can find a matrix $bold(H) = (bold(H)_(t tau))$ (of dimensions $T$ time steps $times K$ time steps) that satisfies:
$ bold(Y) &= bold(X) bold(H) <=> bold(Y)_(m tau) = sum_(t=0)^(T-1) bold(X)_(m t) bold(H)_(t tau) $ <matmul>

The _ideal kernel_ $bold(H)_(t tau) eq.def g[tau]² delta_(t,f[tau])$ <ideal-kernel> is a particular solution to @matmul and can be seen as the discretized version of the ideal kernel from the previous section.

== Case of the mel-spectrogram
Let $bold(M)$ be a matrix of mel filter bank coefficients : $bold(X)^"mel" = bold(M)bold(X)$ and $bold(Y)^"mel" = bold(M)bold(Y)$. Then:
$
bold(Y)^"mel"_(m tau) &= sum_i bold(M)_(m i) bold(Y)_(i tau) \
&= g[tau]^2 sum_i bold(M)_(m i)  bold(X)_(i,f[tau]) \
&= g[tau]^2 bold(X)^"mel"_(m, f[tau])
$

So the ideal kernel $bold(H)$ above is still clearly a solution of the transform between $bold(X)^"mel"$ and $bold(Y)^"mel"$.

We can exploit this representation to reduce the number of frequency bins, thus reducing the algorithmic complexity.

= Estimator robustness
We define the following estimators:

$ tilde(g)_"sum" [tau] = sqrt(sum_(t=0)^(T-1) bold(H)_(t tau) ) $ <gain_estimator_sum>
$ tilde(g)_"max" [tau] = sqrt(max_(t in [0...T-1]) bold(H)_(t tau) ) $ <gain_estimator_max>
$ tilde(f)_"com" [tau] = (sum_(t=0)^(T-1) t bold(H)_(t tau)) / (sum_(t=0)^(T-1) bold(H)_(t tau)) $ <time_estimator_com>
$ tilde(f)_"argmax" [tau] = "argmax"_(t in [0...T-1]) bold(H)_(t tau) $ <time_estimator_argmax>
These can be understood as estimators operating on the columns of $bold(H)$.

In the case of the ideal kernel (@ideal-kernel), $ tilde(g)_"sum" [tau] = tilde(g)_"max" [tau] = g[tau] "and" tilde(f)_"sum" [tau] = tilde(f)_"argmax" [tau] = f[tau] $

In practice, the $bold(H)$ matrix is estimated using NMF, which offers no guarantee that the algorithm converges towards the particular kernel described above. This highlights the need of imposing penalties in the NMF algorithm to steer it towards our idealized kernel.

We will now study the robustness of our estimators to other sources of indetermination.

== Case of column-normalized spectrograms

To improve the numeric stability of NMF, the columns of the spectrograms are usually normalized to sum to 1:
$bold(X)^"norm"_(m t) = bold(X)_(m t) / (sum_i bold(X)_(i t))$ and $bold(Y)^"norm"_(m tau) = bold(Y)_(m tau) / (sum_i bold(Y)_(i tau))$.

$
bold(Y)^"norm"_(m t) &= (g[tau]² bold(X)_(m,f[tau])) / (sum_i g[tau]² bold(X)_(i,f[tau])) \
&= bold(X)^"norm"_(m, f[tau])
$

We clearly see that this would remove any effect of $g$ in $bold(H)$ and would hinder us from estimating it.

== Similar sounds in the source signal
Given the nature of musical signals, two columns of $bold(X)$ could be almost identical, for example in the presence of a loop or of a particularly long note.

Let $t_1$ and $t_2$ be the time steps at which this is true, and $tau_1=f^(-1)(t_1)$ and $tau_2=f^(-1)(t_2)$ their antecedents. We then have $forall m$:
$ bold(Y)_(m tau_1) = bold(Y)_(m tau_2) = g[tau_1]^2 bold(X)_(m t_1) = g[tau_2]^2 bold(X)_(m t_2) $

This would introduce indeterminations in $bold(H)$, highlighting the need to steer the NMF algorithm towards the ideal kernel.

== Hop in the STFT

Usually, the STFT is not calculated for every sample of a signal, but at regular intervals of an _hop size_ $h$. This means that the time steps are replaced with $overline(t) = h t$ and $overline(tau) = h tau$
$ bold(X)_(m overline(t)) = abs(sum_(n=0)^(M-1) x[n+h t] w[n] e^(-j 2 pi n m / M))^2 $
$ bold(Y)_(m overline(tau)) &= abs(g[tau] sum_(n=0)^(M-1) x[n+f[h tau]] w[n] e^(-j 2 pi n  m/ M))^2 $

Thus, there may not be an exact match between $overline(t)$ and $overline(tau)$: the activations in $bold(H)$ may be "smeared" between two "pixels". 

= Penalty functions for NMF

Given the nature of musical signals, we expect $f$ and $g$ to have certain properties:
+ $g(tau)$ should be relatively smooth <g-smooth>
  - $g'(tau)$ should be small
+ $f(tau)$ should be piecewise linear (alternatively, piecewise continuous in the case of varying speeds)
+ $f(tau)$ should be injective
  - there should be only one (or one cluster of, in the case of smearing) nonzero element(s) in a given column of $bold(H)$

== Gain smoothness

we want to minimize $forall tau in [1...K]$:
$
g[tau]^2 - g[tau-1]^2 &= sum_(t=0)^(T-1) bold(H)_(t tau) - sum_(t=0)^(T-1) bold(H)_(t, tau-1) \
&= sum_(t=0)^(T-1) (bold(H)_(t tau) - bold(H)_(t, tau-1)) 
$
So we define the penality function:

$
cal(C)_g^"sum" (bold(H)) &= sum_(tau=1)^(K-1) sum_(t=0)^(T-1) (bold(H)_(t tau) - bold(H)_(t, tau-1)) \
$

Alternatively, using the max gain estimator ?
$
cal(C)_g^"max" (bold(H)) = sum_(tau=1)^(K-1) (max_(t in [0...T-1]) bold(H)_(t tau) - max_(t in [0...T-1]) bold(H)_(t, tau-1))
$