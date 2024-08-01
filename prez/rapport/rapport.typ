#import "@preview/ilm:1.1.1": *
#set text(lang: "en")
#let oslash = symbol("⊘")


#show: ilm.with(
  title: [Rapport de stage],
  author: "Étienne ANDRÉ",
  date: datetime.today(),
  abstract: [],
  preface: [],
  bibliography: bibliography("../../zotero.bib"),
  figure-index: (enabled: true),
  table-index: (enabled: true),
  listing-index: (enabled: true),
  date-format: auto,
  table-of-contents: none
)

= Sujet de stage

Understanding disc jockey (DJ) practices remains a challenging important part of popular music culture. The outcomes from such an understanding are numerous for musicological research in popular music, cultural studies on DJ practices and critical reception, music technology for computer support of DJing, automation of DJ mixing for entertainment or commercial purposes, and others.

However, despite a growing interest in DJ-related music information retrieval (DJ-MIR) @cliffHangDJAutomatic2000 @fujioSystemMixingSongs2003 @ishizakiFullAutomaticDJMixing2009 @kellEmpiricalAnalysisTrack2013 @yadatiDetectingDropsElectronic2014 @sonnleitnerLandmarkBasedAudioFingerprinting2016 @kimAutomaticDjMix2017 @bittnerAutomaticPlaylistSequencing2017 @schwarzExtractionGroundTruth @schwarzUnmixDBDatasetDJMix2018, DJ techniques are not yet sufficiently investigated by researchers partly due to the lack of annotated datasets of DJ mixes.

This project aims at filling this gap by improving methods to automatically deconstruct and annotate recorded mixes for which the constituent tracks are known. In previous research @schwarzMethodsDatasetsDJMix2021, a rough alignment first estimates where in the mix each track starts, and which time-stretching factor was applied. Second, a sample-precise alignment is applied to determine the exact offset of each track in the mix. Third, we estimate the cue points and the fade curves in the time-frequency domain.

These methods were evaluated on our publicly available DJ-mix dataset UnmixDB @schwarzMethodsDatasetsDJMix2021 @schwarzUnmixDBDatasetDJMix2018. This dataset contains automatically generated beat-synchronous mixes based on freely available music tracks, and the ground truth about the placement, transformations and effects of tracks in a mix.

There are several possible directions and objectives of this internship :

+ Revisit and extend the UnmixDB dataset :
  - validate and extend the effects applied in UnmixDB by more "ecologically valid" DJ-mix
  - extend to non-constant tempo curves
+ Use or create a real-world annotated DJ mix data @werthen-brabantsGroundTruthExtraction2018 @kimComputationalAnalysisRealWorld2020.
+ Improve and research novel methods of DJ-mix reengineering : Some possibilities are :
  - refine the alignment phase to handle mixes with non-constant tempo curves, via a joint iterative estimation algorithm of alignment and time-scaling
  - handle the partial and repeated presence of tracks or loops in the mix @smithNonnegativeTensorFactorization2018
  - test the influence of other signal representations (MFCC, spectrum, chroma, scattering transform, PANNs) on the results @demanAnalysisEvaluationAudio2014 @yangAligningUnsynchronizedPart2021 @cramerLookListenLearn2019 @zehrenAutomaticDetectionCue2020

With these refinements, the method could become robust and precise enough to allow the inversion of fading, EQ and other processing @barchiesiReverseEngineeringMix2010 @ramonaSimpleEfficientFader2011.


= Introduction

In this section, we introduce DJ culture and techniques, and DJ-MIR. We discuss prior art and available datasets.

== DJs and DJing
DJs (_Disc-Jockeys_) and DJing have been a staple of our musical landscape for decades.

While the act of DJing can be understood in multiple ways, our understanding consists of playing recorded media (_tracks_) in a continuous manner, to a live audience, on the radio (or any other broadcast media) or to a recorded medium (such as _mixtapes_, or for streaming services).

These _mixes_ consist of music tracks, carefully selected by the DJ, played sequentially. But the act of DJing is a transformative one: the DJ may overlap one or more tracks in order to create new musical pieces (_mashups_), or more frequently to prevent silence between sequential tracks. The overlapping regions between songs are called _transitions_, the simplest of which is a simple cross-fade.

In dance music, these transitions are usually made to be as seamless as possible, in order to keep the energy up and the people dancing. As such, the DJ can use various techniques, such as syncrhonizing the tempo and downbeats of the overlapping tracks, using EQs to add or remove parts of the tracks, and use various effects such as reverbs and delays.

While the vinyl DJ may only choose _cue points_ (a point in track time at which it starts playing) change the speed at which the record is playing (sometimes in an extreme manner, as in _scratching_), modern digital DJ equipment allows for a larger variety of transformations: transposition, time-warping, loops, jumps...

Additionally, there can be additional elements in the mix, such as speech or singing from a _MC_, rhythm machines, sound effects such as sirens or jingles, etc.

== DJ-MIR and prior art
#text(fill: blue)[Redite de @schwarzMethodsDatasetsDJMix2021 : estce que c'est ok ?]

Despite the prevalence of DJing in pop culture, it remains a understudied field in academia. The outcomes from understanding them are numerous for musicological research in popular music, cultural studies on DJ practices and critical reception, music technology for computer support of DJing, automation of DJ mixing for entertainment or commercial purposes, and others.

This highlights the need, and explains the rise of the field of DJ-MIR (DJ Music Information Retrieval), which aims to infer metadata from recorded DJ mixes.

#text(fill: red)[Présenter les sous-tâches du DJ-MIR: identification, segmentation, alignement, estimation cue points, cf @kimComputationalAnalysisRealWorld2020 + refs]

#text(fill: red)[Ici, on s'intéresse à l'alignement + segmentation]

== DJ-MIR Datasets
#text(fill: red)[Présenter jeux de données]

Despite the enormous amount of available mixes, thanks to decades of recorded mixes and specialized streaming services, they are rarely sufficiently and correctly annotated. 

Useful datasets for DJ-MIR consist of mixes, accompanied by ..........

Les jeux de données utiles au DJ-MIR se constituent de mixes auquels sont associés des tracks, généralement accompagnées de leur position dans le temps. En fonction des annotations disponibles, on pourra trouver des informations sur les manipulations effectuées par le DJ lors de la création du mix, comme le volume de chaque track, les facteurs d'étirement temporel et de transposition, la nature et les paramètres des effets appliqués, etc. Nous supposons que la connaissance exacte de tous les traitements appliqués aux tracks d'origine nous permet de construire exactement le signal de mix.

De tels jeux de données peuvent se distinguer en deux catégories:
- *synthétiques*, c'est-à-dire générés selon un certain nombre de règles ;
- *réels*, issus de mixes et annotés manuellement ou automatiquement.

=== Synthetic datasets
#text(fill: red)[État de l'art]

Synthetic datasets have the advantage of being precise and complete, as the parameters of the mixes' creation are known by construction. However, being based on rules, they may suffer in diversity and therefore in ecological validity. UnmixDB @schwarzUnmixDBDatasetDJMix2018 (based on the `mixotic` dataset from #cite(<sonnleitnerLandmarkBasedAudioFingerprinting2016>, form:"prose")) and the dataset from #cite(<werthen-brabantsGroundTruthExtraction2018>, form:"prose") are the only representatives of this category. It should be noted that these datasets are all based on mixes from the netlabel `mixotic.net` #footnote(link("https://www.mixotic.net")), and therefore share the same musical style.

=== Real datasets
#text(fill: red)[État de l'art + sources de données encore non exploitées, culture du track ID]

Datasets made of real DJ sets are directly representative of past and current DJing practices and do not suffer from the problems of synthetic datasets. However, they require time-consuming manual annotation or error-prone automatic annotation. The commercial nature of the mixes and the tracks constituting them also poses legal issues for use in an academic context. Indeed, to our knowledge, there is no such freely available dataset: the dataset from #cite(<kimComputationalAnalysisRealWorld2020>, form:"prose") and the `disco` dataset from #cite(<sonnleitnerLandmarkBasedAudioFingerprinting2016>, form:"prose") are not freely accessible, respectively for commercial and copyright reasons.
= Création de jeux de données

== Méthodes de mesure de vérité terrain
#text(fill: red)[Mesurer les paramètres de mix sur un mix réel: que des avantages]

=== Suivi optique de disques vinyles
#text(fill: red)[Présentation, résultats, github]

=== Extraction des métadonnées de fichiers de projet DAW
#text(fill: red)[Présentation, résultats, github]

=== Instrumentation de logiciel de mix
#text(fill: red)[Présentation]

== `trackid.net`
#text(fill: red)[Présentation, statistiques, données, problèmes de droits.]

= DJ mix transcription using Non-Negative Matrix Factorization (NMF)

In this section, we present the use of the well-known NMF algorithm to perform DJ mix transcription.
We first study DJ hardware and software to justify the transcription task as a matrix factorization problem, and introduce the penalized Beta-NMF algorithm. We then show that the matrix factorization can yield an intuitive representation of DJ mix parameters. We propose a multi-pass extension of the NMF algorithm. We then present experiments, reproducible results and discuss possible improvements.

== Objectives

*Given*:
- a recording of a DJ mix;
- the recordings of all the mix's constituting tracks (henceforth called _reference tracks_);

*Estimate*:
- any temporal transformations applied to the tracks (play time, duration, speed, loops, jumps);
- the mixing gain;
- any effects applied to the tracks/and or mix (distortion, compression, pitch-shifting, filtering, EQing...)

== DJ mixing hardware
#text(fill: red)[Justification de l'utilisation de la NMF]

The DJs' field of expression is defined by its hardware and/or software: decks, mixing tables, and controllers. Even if different brands and models present a varying feature set, the general workflow always remains similar: we assume that the signal path from recorded tracks to recorded mix can be summarized by @dj-signal-path.

#figure(
  image("../2024-04-26/dj-deck.png"),
  caption: [Signal path of a typical DJ mix],
  placement: none
) <dj-signal-path>

This signal path is derived directly from usual DJ setups#footnote[It can be noted that DJ software is usually made to emulate DJ hardware, so the signal path remains valid in this case.]. It can be understood as follows:
- Two or more DJ decks (in blue) are used as signal sources and play pre-recorded tracks and apply time-warping;
- The DJ mixer is used to sum the signal from the decks, and may apply various additional effects, the most common of which is a 3- or 4-band EQ. Any additional elements are also summed at this stage;
- Outside of the DJ setup, some additional processing may be applied on the mix, to suit distribution or venue needs, such as compression and egalization. Because this processing is usually quite light, we will consider it to be inexistant for the remainder of this report.

== Matrix representation

We will now use this knowledge to introduce a matrix formulation of DJ-mixing, by considering a spectrogram reprensentation of the signals. We achive this by grouping all non-time-based transformation as illustrated in @separate-boxes.

#figure(
  image("../2024-04-26/separate-boxes.png"),
  caption: [Separated signal path],
  placement: none
) <separate-boxes>

Assuming the $M$ constituting tracks of the mix are known, let $forall i in [1...M]$:
- $bold(W)_((i))$ the spectrogram of track $i$;
- $bold(H)_((i))$ the _activation matrix_ of track $i$, representing both the time-warping operations and the gain applied at the mixing stage;
- $bold(V)_((i)) = bold(W)_((i)) bold(H)_((i))$ the time-remapped spectrogram of track $i$ with gain appplied;
- $bold(N)$ a noise matrix representing the timbral applied to the mix and/or tracks and any additional elements;
- $bold(V)$ the spectrogram of the mix.

Using these notations, we then have:

$ bold(V) &= bold(N) + sum_(i = 1)^M bold(W)_((i)) bold(H)_((i)) $

An illustration of this equation is given on @nmf-djmix.
#figure(
  image("../2024-04-26/nmf-djmix.png"),
  caption: [Formulation du processus de mix DJ sous forme NMF],
  placement: none
) <nmf-djmix>


Then, by defining two additional matrices $bold(W)_((a))$ and $bold(H)_((a))$ of compatible dimensions so that $bold(N) = bold(W)_((a)) bold(H)_((a))$, we can rewrite the equation as a simple matrix multiplication of two large matrices by concatenation:

$ bold(V) &= bold(W)_((a)) bold(H)_((a)) + sum_(i = 1)^M bold(W)_((i)) bold(H)_((i)) \
&= underbrace(mat(bold(W)_((1)) bold(W)_((2)) ... bold(W)_((M))bold(W)_((a))), bold(W)) underbrace(mat( bold(H)_((1)); bold(H)_((2)); dots.v; bold(H)_((M)); bold(H)_((a))), bold(H)) $

Thus, estimating the gain and time-warping amounts to determining the coefficients of the $bold(H)$ matrix. Additionally, by determining $bold(W)_((a))$, any additional elements and timbral effects can be estimated. Such an estimation task is well-suited for the NMF family of algorithms, which has proven effective in a variety of audio-related problems.

== NMF

Let $bold(W)_(F times K)$, $bold(H)_(K times N)$ and $bold(V)_(F times N)$ non-negative matrices.

== Beta-NMF
#text(fill: red)[Présentation générale de la beta-nmf, avec les propriétés intéressantes pour la suite. Lien entre beta et puissance du spectrogramme]


#strong[Objectif]: minimise a divergence $D$ between $bold(V)$ and
$bold(W) bold(H)$:

$ min_(bold(W) , bold(H)) D (bold(V) | bold(W) bold(H)) "avec" bold(W) >= 0 , bold(H) >= 0 $

#strong[Distance]: $beta$-divergence

$ D_beta lr((bold(V) | bold(W) bold(H))) = sum_(f = 1)^F sum_(n = 1)^N d lr((bold(V)_(f n) | lr((bold(W H)))_(f n))) $
$ d lr((x | y)) = cases(
  frac(1, beta lr((beta - 1))) lr((x^beta + lr((beta - 1)) y^beta - beta x y^(beta - 1))) & "if" beta != {0, 1},
  x log x / y - x + y & "if" beta = 1,
  x / y - log x y - 1 & "if" beta = 0
) $

$dot.circle$ et #oslash désignent respectivement le produit et la division d'Hadamard (terme-à-terme).
+ Initialiser $bold(X) >= 0$ et $bold(H) >= 0$
+ Mettre à jour successivement $bold(X)$ et
  $bold(H)$
  $ bold(H) arrow.l bold(H) dot.circle [bold(X)^T ((bold(W H))^(beta - 2) dot.circle bold(Y))] #oslash [bold(X)^T (bold(W H))^(beta - 1)] $
  $ bold(X) arrow.l bold(X) dot.circle [((bold(W H))^(beta - 2) dot.circle bold(Y)) bold(H)^T] #oslash [(bold(W H))^(beta - 1) bold(H)^T] $
+ Répéter l’étape 2 jusqu’à convergence ou nombre d’itérations maximum

== Caractérisation des transformations temporelles
#text(fill: red)[Justification de la forme attendue des résultats. Ne garder que la version discrète et virer la version continue ?]

In this section, we will be characterizing what happens to the $bold(H)_((i))$ in the case of a time-remapping and gain transformations. To simplify the notations, we will drop the $(i)$ subscripts in this whole section.

=== Continuous formulation

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
&= g(tau)^2 X(omega, f(tau)) $

Now, our goal is to decompose $Y$ into a continuous linear combination of $X$, with fixed $omega$. In other words, find an integral transform of $X$ that yields $Y$. This involves determining a kernel $H$ that satisfies:
$ Y(omega, tau) = integral_RR X(omega, t) H (t, tau) d t $ <integral_transform>

The kernel $H(t, tau) eq.def g(tau)^2 delta(t-f(tau))$ (where $delta(0) = 1$ and $0$ elsewhere) is a particular solution to @integral_transform:
$ Y(omega, tau) &= integral_RR X(omega, t) g(tau)^2 delta(t-f(tau)) d t \
&= g(tau)^2 X(omega, f(tau))
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
&= g[tau]^2 bold(X)_(m,f[tau])
$ <xy-relation>

Then we can find a matrix $bold(H) = (bold(H)_(t tau))$ (of dimensions $T$ time steps $times K$ time steps) that satisfies:
$ bold(Y) &= bold(X) bold(H) <=> bold(Y)_(m tau) = sum_(t=0)^(T-1) bold(X)_(m t) bold(H)_(t tau) $ <matmul>

The _ideal kernel_ $bold(H)_(t tau) eq.def g[tau]^2 delta_(t,f[tau])$ <ideal-kernel> is a particular solution to @matmul and can be seen as the discretized version of the ideal kernel from the previous section.

== Estimation du gain et du _warp_
#text(fill: red)[Justification de l'exactitude des estimateurs dans le cas idéal]

== Améliorations algorithmiques

=== Compression de l'information
#text(fill: red)[Réduction des dimensions des matrices $=>$ problème devient tractable. Parler de mel spec + gros hop avec justif que c'est "musical"]

Let $bold(M)$ be a matrix of mel filter bank coefficients : $bold(X)^"mel" = bold(M)bold(X)$ and $bold(Y)^"mel" = bold(M)bold(Y)$. Then:
$
bold(Y)^"mel"_(m tau) &= sum_i bold(M)_(m i) bold(Y)_(i tau) \
&= g[tau]^2 sum_i bold(M)_(m i)  bold(X)_(i,f[tau]) \
&= g[tau]^2 bold(X)^"mel"_(m, f[tau])
$

So the ideal kernel $bold(H)$ above is still clearly a solution of the transform between $bold(X)^"mel"$ and $bold(Y)^"mel"$.

We can exploit this representation to reduce the number of frequency bins, thus reducing the algorithmic complexity.

=== Recouvrement
#text(fill: red)[Exploitation de la continuité temporelle des signaux]

=== Normalisation
#text(fill: red)[Normalisation des matrices et impact sur les estimateurs]

=== Régularisation
#text(fill: red)[Définition des fonctions de régularisation utilisées et poids $lambda$ recommandés]

=== _Warm-up_
#text(fill: red)[Warm-up de $beta$, $lambda$, puissance du spectrogramme $=>$ pas de compromis entre robustesse et performance]

=== NMF multi-passes
#text(fill: red)[Explication de l'algorithme, + creusage des matrices]

=== Estimation des EQ
#text(fill: red)[Découpage des matrices en bandes + justification matérielle]

=== Invariance à la transposition
#text(fill: red)[cf. divergence invariante à la transposition, ou NMFD, ou PLCA... ou etude que le melspec suffit]

== Implémentation
#text(fill: red)[pytorch, discussion de cpu vs. gpu, github]

== Résultats
=== Sur mix synthétiques
=== Sur unmixdb
=== Sur mix réels

= Conclusion
