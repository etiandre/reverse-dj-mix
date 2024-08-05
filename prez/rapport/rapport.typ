#import "@preview/ilm:1.1.1": *
#import "@preview/lovelace:0.3.0": *
#import "@preview/ctheorems:1.1.2": *
#set text(lang: "en")
#let proof = thmproof("proof", "Proof")
#show: thmrules.with(qed-symbol: $square$)

#show: ilm.with(
  title: [Internship dissertation],
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

#set figure(placement: auto)
#let my-lovelace-defaults = (
  booktabs: true
)

#let pseudocode = pseudocode.with(..my-lovelace-defaults)
#let pseudocode-list = pseudocode-list.with(..my-lovelace-defaults)

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

These _mixes_ consist of music tracks, carefully selected by the DJ, played sequentially. But the act of DJing is a transformative endeavour: the DJ may overlap one or more tracks in order to create new musical pieces (_mashups_), or more frequently to prevent silence between sequential tracks. The overlapping regions between songs are called _transitions_, the simplest of which is a simple cross-fade.

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
- $bold(H)_((i))$ the so-called _activation matrix_ of track $i$, representing both the time-warping operations and the gain applied at the mixing stage;
- $bold(V)_((i)) = bold(W)_((i)) bold(H)_((i))$ the time-remapped spectrogram of track $i$ with gain appplied;
- $bold(N)$ a noise matrix representing the timbral applied to the mix and/or tracks and any additional elements;
- $bold(V)$ the spectrogram of the mix.

Using these notations, we can write:

$ bold(V) &= bold(N) + sum_(i = 1)^M bold(W)_((i)) bold(H)_((i)) $ <eq:mix>

An illustration of @eq:mix is given @nmf-djmix.

#figure(
  image("../2024-04-26/nmf-djmix.png"),
  caption: [Formulation du processus de mix DJ sous forme NMF],
  placement: none
) <nmf-djmix>


Then, by defining two additional matrices $bold(W)_((a))$ and $bold(H)_((a))$ of compatible dimensions so that $bold(N) = bold(W)_((a)) bold(H)_((a))$, we can rewrite the equation as a simple matrix multiplication of two large matrices by concatenation:

$ bold(V) &= bold(W)_((a)) bold(H)_((a)) + sum_(i = 1)^M bold(W)_((i)) bold(H)_((i)) \
&= underbrace(mat(bold(W)_((1)) bold(W)_((2)) ... bold(W)_((M))bold(W)_((a))), bold(W)) underbrace(mat( bold(H)_((1)); bold(H)_((2)); dots.v; bold(H)_((M)); bold(H)_((a))), bold(H)) $

Thus, estimating the gain and time-warping amounts to determining the coefficients of the $bold(H)$ matrix. Additionally, by determining $bold(W)_((a))$, any additional elements and timbral effects can be estimated. Such an estimation task is well-suited for the NMF family of algorithms, which has proven effective especially in audio source separation tasks.

== NMF Algorithm
#text(fill: red)[Présentation générale de la beta-nmf, avec les propriétés intéressantes pour la suite. Lien entre beta et puissance du spectrogramme]

=== Beta-NMF and Multiplicative Update rules
Let $bold(W)_(F times K)$, $bold(H)_(K times N)$ and $bold(V)_(F times N)$ non-negative matrices. The NMF algorithm in its most basic form aims to minimise a similarity measure $cal(D)$ between the _target matrix_ $bold(V)$ and the _estimated matrix_ $bold(W) bold(H)$, and amounts to solving the following optimization problem:

$ min_(bold(W) , bold(H)) cal(D) (bold(V) | bold(W) bold(H)) "with" bold(W) >= 0 , bold(H) >= 0 $ <eq:optimization-problem>

The similarity measure we use is the beta-divergence, which is defined $forall beta in RR$ as follows:

$ cal(D)_beta lr((bold(V) | bold(W) bold(H))) = sum_(f = 1)^F sum_(n = 1)^N d lr((bold(V)_(f n) | lr((bold(W H)))_(f n))) $
$ d_beta lr((x | y)) = cases(
  frac(1, beta lr((beta - 1))) lr((x^beta + lr((beta - 1)) y^beta - beta x y^(beta - 1))) & "if" beta != {0, 1},
  x log x / y - x + y & "if" beta = 1,
  x / y - log x y - 1 & "if" beta = 0
) $ <eq:beta-divergence>

It can be noted that the beta-divergence is equivalent to:
- the Euclidian distance if $beta = 2$;
- the Kullblack-Liebler divergence if $beta = 1$;
- the Itakura-Saito divergence if $beta = 0$.

As shown in @fevotteNonnegativeMatrixFactorization2009 and later extended in @fevotteAlgorithmsNonnegativeMatrix2011, an efficient and simple gradient descent algorithm for $bold(W)$ and $bold(H)$ can be derived from the beta-divergence by separating its gradient w.r.t. a parameter $bold(theta)$ into its positive and negative parts:

$
gradient_bold(bold(theta)) cal(D)_beta (bold(V) | bold(W) bold(H)) = gradient_bold(bold(theta))^+ cal(D)_beta (bold(V) | bold(W) bold(H)) - gradient_bold(bold(theta))^- cal(D)_beta (bold(V) | bold(W) bold(H))
$ <eq:gradient-separation>

Using the notation trick described in @fevotteAlgorithmsNonnegativeMatrix2011, multiplicative update rules can be obtained#footnote[$dot.circle$ and $.../...$ stand respectively for Hadamard's (element-wise) product and division.]:
$
bold(theta) <- bold(theta) dot.circle (gradient_bold(bold(theta))^- cal(D)_beta (bold(V) | bold(W) bold(H))) / (gradient_bold(bold(theta))^+ cal(D)_beta (bold(V) | bold(W) bold(H)))
$ <eq:mu-gradient>

With the beta-divergence, this yields @algo:mu-betadiv which can be very efficiently implemented, with strong monotonicity guarantees when $beta in [0,2]$:

#figure(
  kind: "algorithm",
  supplement: [Algorithm],

  pseudocode-list(numbered-title:[NMF Algorithm with Multiplicative Updates], line-gap:1.4em)[
  + *Initialize* $bold(W) >= 0$ and $bold(H) >= 0$
  + *Until* convergence criterion is reached:
    + $bold(H) arrow.l bold(H) dot.circle (bold(W)^T ((bold(W H))^(beta - 2) dot.circle bold(V))) / (bold(W)^T (bold(W H))^(beta - 1))$
    + $bold(W) arrow.l bold(W) dot.circle (((bold(W H))^(beta - 2) dot.circle bold(V)) bold(H)^T) / ((bold(W H))^(beta - 1) bold(H)^T)$
  ]
) <algo:mu-betadiv>

An interesting property of this algorithm is that any zeroes in $bold(H)$ or $bold(W)$, by property of multiplication, remains zero troughout the optimization process. We will exploit this property in @sec:multi-pass.

=== Penalized Beta-NMF

More complex objective functions can be crafted by adding supplementary functions to the similarity measure, for penalization or regularization of the solutions. To illustrate, we will consider a new objective function $cal(C)$ comprised of the beta-divergence and an additional penalty function $cal(P)$ on $bold(H)$ weighted by $lambda in RR^+$:

$
cal(C) = cal(D)_beta (bold(V) | bold(W H)) + lambda cal(P)(bold(H))
$

Supposing the gradients of the penalty functions are separable into their positive and negative parts, new multiplicative update rules can be easily derived by following the procedure from the previous section. However, the monotonicity mentioned above is no longer guanranteed, and is dependent on the form of $cal(P)$ and the choice of $lambda$; which both can be validated through experimentation.

=== Choosing the divergence and the type of spectrograms

In @virtanenMonauralSoundSource2007 it has been shown that the Kullblack-Liebler divergence gives good results on source separation tasks when paired with magnitude spectrogram. On the other hand, @fevotteMajorizationminimizationAlgorithmSmooth2011 uses a strong statistical model to pair the Itakura-Saito divergence with power spectrograms.

Furthermore, @fitzgeraldUseBetaDivergence2009 discusses fractional values for $beta$, and suggests that the spectrograms could also be taken to a fractional power; along with the so-called tempered beta-divergence which uses $beta$ as a temperature parameter that varies during the course of optimization. However, we did not manage to find prior art with conclusive results on the best possible approach for our specific task. In the end, we used the Itakura-Saito divergence with power spectrogram, which in our trials gave the best results.


== Characterization of warp and gain
#text(fill: red)[Justification de la forme attendue des résultats]

In this section, we introduce a model for time-warping and applying gain on a signal in the spectral domain, and show that there exists a particular solution for the activation matrix with exhibits intuitive structures. We then define gain and warp estimators, and study their robustness against noise and other indetermination sources.

From now on, and for clarity, $t$ denotes the discrete time in the track, and $tau$ the discrete time in the mix.

=== Ideal kernel

Let:
- $x[t]$ a real-valued signal;
- $f[tau]$ a time-warping injective sequence with values in $NN$ that maps a mix time step $tau$ to a track time step $t$;
- $g[tau]$ a gain signal with values in [0,1];
- $y[tau]$ the time-warped and gain-affected transformation of $x$;
- $w[n]$ an arbitrary window function of length $M$.

We define $bold(X) = (bold(X)_(m t))$ to be the the power spectrogram matrix of $x$ ($M$ frequency bins $times T$ time steps):
$ bold(X)_(m t) = abs(sum_(n=0)^(M-1) x[n+t] w[n] e^(-j 2 pi n m / M))^2 $ <eq:stft>

Similarly, we define $bold(Y) = (bold(Y)_(m tau))$ to be the matrix containing the power spectrogram of $y$ ($M$ frequency bins $times K$ time steps), and show that in can be expressed in terms of $bold(X)$:

$ bold(Y)_(m tau) &= abs(g[tau] sum_(n=0)^(M-1) x[n+f[tau]] w[n] e^(-j 2 pi n  m/ M))^2 \
&= g[tau]^2 bold(X)_(m,f[tau])
$ <xy-relation>

We can then find a matrix $bold(H) = (bold(H)_(t tau))$ (of dimensions $T$ time steps $times K$ time steps) that satisfies:
$ bold(Y) &= bold(X) bold(H) <=> bold(Y)_(m tau) = sum_(t=0)^(T-1) bold(X)_(m t) bold(H)_(t tau) $ <matmul>

The _ideal kernel_ $bold(H)^"ideal"$ (@ideal-kernel), a solution to @matmul, is of particular interest. Indeed, when the matrix is viewed as an image, it exhibits a very intuitive understanding of the concept of the transformations applided to $x[t]$, as is illustrated in @fig:time-time.

$ bold(H)^"ideal"_(t tau) eq.def g[tau]^2 delta_(t,f[tau]) $ <ideal-kernel>
where $delta_(a,b) = cases(1 "if" a = b, 0 "if not")$

#figure(
  image("../2024-03-26/temps-temps.drawio.png"),
  caption: [Some examples of the structures emerging in $bold(H)^"ideal"$, with the associated DJ nomenclature],
) <fig:time-time>

=== Estimation of the warp and gain values
#text(fill: red)[Justification de l'exactitude des estimateurs dans le cas idéal]

We define the following estimators:

$ tilde(g) [tau] = sqrt(sum_(t=1)^(T) bold(H)_(t tau) ) $ <gain_estimator_sum>
$ tilde(f) [tau] = "argmax"_(t in [1...T]) bold(H)_(t tau) $ <time_estimator_argmax>

Intuitively, $tilde(g)[tau]$ and $tilde(f)[tau]$ can respectively be understood as the energy of a column of $bold(H)$, and as the position of its peak.

In the case of the ideal kernel (@ideal-kernel), it can easily be shown that these are exact estimators. In practice, the bare optimization algorithm offers no guarantee that it converges towards this kernel. We discuss such sources of indetermination in the following sections. This highlights the need to introduce additional techniques to steer convergence towards the ideal kernel, and ensuring that the estimators are robust to noise.

=== Sources of indetermination

==== Similar sounds in the source signal

#figure([#image("../2024-05-30/longue-note.png")], caption: [Indeterminations in $bold(H)$ caused by spectrally similar frames.]) <fig:indeterminations>

#figure([#image("../2024-05-17/image-2.png")], caption: [Parallel line structures in $bold(H)$ caused by loops in the reference tracks.]) <fig:parallel-lines>

Given the nature of musical signals, two columns of $bold(X)$ could be almost identical (@fig:parallel-lines), for example in the presence of a loop in electronic music (@fig:indeterminations)

Let $t_1$ and $t_2$ be the time steps at which this is true, and $tau_1=f^(-1)(t_1)$ and $tau_2=f^(-1)(t_2)$ their antecedents. We then have $forall m$:
$ bold(Y)_(m tau_1) = bold(Y)_(m tau_2) = g[tau_1]^2 bold(X)_(m t_1) = g[tau_2]^2 bold(X)_(m t_2) $

Visually, this corresponds to having multiple activations per column of $bold(H)$, with the energy of the activations being distributed arbitrarily between $t_1$ and $t_2$ in both the $tau_1$ and $tau_2$ columns. Fortunately, such indeterminations do not invalidate $tilde(g)$, but the same can not be said of $tilde(h)$.

==== *Hop of the spectrogram*

Usually, the spectrogram is not calculated for every sample of a signal as in our definition (@eq:stft), but at regular intervals of a so-called hop size $h$. This means that the time steps are replaced with $overline(t) = h t$ and $overline(tau) = h tau$
$ bold(X)_(m overline(t)) = abs(sum_(n=0)^(M-1) x[n+h t] w[n] e^(-j 2 pi n m / M))^2 $
$ bold(Y)_(m overline(tau)) &= abs(g[tau] sum_(n=0)^(M-1) x[n+f[h tau]] w[n] e^(-j 2 pi n  m/ M))^2 $

Because of this discretization, there may not be an exact match between $overline(t)$ and $overline(tau)$: visually, the activations in $bold(H)$ may be distributed across two neighboring cells.

#text(fill: red)[ajouter une illustration des pics étalés dans $bold(H)$]
== Improvements to the algorithm <sec:improvements>

=== Information compression
#text(fill: red)[Réduction des dimensions des matrices $=>$ problème devient tractable. Parler de mel spec + gros hop avec justif que c'est "musical"]

DJ mixes are usually long, spanning from 30 minutes to multiple hours; and because they are musical signals, their frequency bandwidth is quite extensive. When using the STFT with typical hop durations and bin count for musical signals, the associated feature matrix can become quite large, causing high memory usage and resource consumption.

In order to mitigate these issues, we first choose to use relatively large hop durations, in the order of seconds. As a side benefit, these larger hop durations are better suited to capturing musical structures. The hop duration however is not fixed, as is explained in section @sec:multi-pass.

Additionally, we compress the frequency information using the mel-transform @stevensScaleMeasurementPsychological1937, which bins groups of close frequencies together according to a perceptual model of human hearing, which is well-suited for musical signals. This transform has no effect on the ideal kernel and our estimators.

#proof[
Let $bold(M)$ be a matrix of mel filterbank coefficients. The mel-spectrograms are calculated from the regular spectrograms: $bold(X)^"mel" = bold(M)bold(X)$ and $bold(Y)^"mel" = bold(M)bold(Y)$. Then we have:
$
bold(Y)^"mel"_(m tau) &= sum_i bold(M)_(m i) bold(Y)_(i tau) \
&= g[tau]^2 sum_i bold(M)_(m i)  bold(X)_(i,f[tau]) \
&= g[tau]^2 bold(X)^"mel"_(m, f[tau])
$

So the ideal kernel $bold(H)^"ideal"$ is still clearly a solution of @matmul.
]

=== Analysis window overlap
#text(fill: red)[Exploitation de la continuité temporelle des signaux]

A key parameter when working with spectrograms is the overlap factor of the analysis windows. In order to emphasize the temporal continuity of the musical signals, we use high overlap factors: our experiments have shown that a window size of 6 to 8 times the hop size give the best results. It has proven very effective for curbing indeterminations.

Such large window sizes would usually entail a very high number of frequency bins, but by using the mel-transform we avoid this problem.

=== Normalization
To improve the numeric stability of the NMF, the columns of $bold(X)$ are usually normalized to sum to 1. We also normalize $bold(Y)$ by a single factor#footnote[Normalizing by column as for $bold(X)$ would cancel out the gain information in $bold(H)$.]. This results in a simple scaling factor for $bold(H)$ and therefore the estimators.

#proof[
Let the scaling factors $bold(k)_t eq.def sum_i bold(X)_(i t)$ and $kappa eq.def sum_i sum_t bold(Y)_(i t)$.

The normalized spectrograms are:
$ bold(X)^"norm"_(m t) eq.def bold(X)_(m t) / bold(k)_t $

$ bold(Y)^"norm" eq.def bold(Y) / kappa  $


Using @xy-relation:
$
bold(Y)^"norm"_(m tau) =  bold(k)_t / kappa g[tau]^2 bold(X)^"norm"_(m, f[tau]) 
$

We can then deduce the ideal normalized kernel $bold(H)^"norm"$ as a solution to @matmul:

$
bold(H)^"norm"_(t tau) &eq.def bold(k)_t / kappa g[tau]^2 delta_(t,f[tau]) \
&= bold(k)_t / kappa bold(H)^"ideal"_(t tau)
$
]

=== Thresholding of low power frames
#text(fill: red)[Suppression des frames avec trop peu d'énergie, qui ont une forte probabilité d'introduire des indéterminations, justif spectrale (queue de reverbs) + justif musicale (peu probable que des moments sans énergie soient dans le mix)]

=== Penalty functions
#text(fill: red)[Définition des fonctions de régularisation utilisées et poids $lambda$ recommandés + warm-up/temperature/tempered. A garder même si non utilisé à la fin ?]

Given the nature of DJ mixes, we expect $f$ and $g$ to have certain properties, from which we define additional penalty functions:
+ $g(tau)$ should be relatively smooth
  - $g'(tau)$ should be small
+ $f(tau)$ should be piecewise linear (alternatively, piecewise continuous in the case of varying speeds)
+ $f(tau)$ should be injective
  - there should be only one (or one cluster of, in the case of smearing) nonzero element(s) in a given column of $bold(H)$

#text(fill: red)[ajouter L1 (et L2 si on s'en sert ?) voire L1+L2 (group lasso)]

==== Gain smoothness

We expect $g(tau)$ to be varying smoothly over time. Thus we introduce the *gain smoothness* penalty, that minimizes the difference between two consecutive gain values.

Given that the gain of the column is given by 
$ g[tau] = sqrt(sum_(t=0)^(T-1) bold(H)_(t tau) ) $

$
g[tau]^2 - g[tau-1]^2 &= sum_(t=0)^(T-1) bold(H)_(t tau) - sum_(t=0)^(T-1) bold(H)_(t, tau-1) \
&= sum_(t=0)^(T-1) (bold(H)_(t tau) - bold(H)_(t, tau-1)) 
$
So we define the penality function:

$
cal(P)_g (bold(H)) &= sum_(tau=1)^(K-1) sum_(t=0)^(T-1) (bold(H)_(t tau) - bold(H)_(t, tau-1))^2 \
$

*gradient calculation*
$
partial / (partial bold(H)_(i j)) (bold(H)_(t tau) - bold(H)_(t, tau-1))^2 = cases(
  2(bold(H)_(i j) - bold(H)_(i, j-1)) &"if" i=t "and" j=tau,
  -2(bold(H)_(i,j+1) - bold(H)_(i j)) &"if" i=t "and" j+1=tau,
  0 &"otherwise"
)
$

So:
$
(partial cal(P)_g) / (partial bold(H)_(i j)) &= 2(bold(H)_(i j) - bold(H)_(i, j-1)) -2(bold(H)_(i,j+1) - bold(H)_(i j)) \
 &= 4 bold(H)_(i j) - 2 (bold(H)_(i,j-1) + bold(H)_(i,j+1))
$

*gradient term separation*:
$
gradient_bold(H)^+ cal(P)_g = 4 bold(H) \
(gradient_bold(H)^- cal(P)_g)_(i j) = 2 (bold(H)_(i,j-1) + bold(H)_(i,j+1))
$

==== Diagonal smoothness

We hypothesize the tracks to be played near their original speed, and that there will be significant time intervals without any loops or jumps. This appears in $bold(H)$ as diagonal line structures. We define a *diagonal smoothness* penalty that minimises the difference between diagonal cells of $bold(H)$:

$
cal(P)_d (bold(H)) = sum_(t=1)^(T-1) sum_(tau=1)^(K-1) (bold(H)_(t,tau) - bold(H)_(t-1, tau-1))^2
$
*gradient calculation*
$
partial / (partial bold(H)_(i j)) (bold(H)_(t tau) - bold(H)_(t-1, tau-1))^2 = cases(
  2(bold(H)_(i j) - bold(H)_(i-1, j-1)) &"if" i=t "and" j=tau,
  -2(bold(H)_(i+1,j+1) - bold(H)_(i j)) &"if" i+1=t "and" j+1=tau,
  0 &"otherwise"
)
$

So:
$
(partial cal(P)_d) / (partial bold(H)_(i j)) &= 2(bold(H)_(i j) - bold(H)_(i-1, j-1)) -2(bold(H)_(i+1,j+1) - bold(H)_(i j)) \
 &= 4 bold(H)_(i j) - 2 (bold(H)_(i-1,j-1) + bold(H)_(i+1,j+1))
$

*gradient term separation*:
$
gradient_bold(H)^+ cal(P)_d = 4 bold(H) \
(gradient_bold(H)^- cal(P)_d)_(i j) = 2 (bold(H)_(i-1,j-1) + bold(H)_(i+1,j+1))
$

==== Lineness

The time-remapping function is expected to be piecewise continuous. In $bold(H)$, this means we can characterize the neighboring cells of a given activation. Given an activated cell $(i,j)$, only the up direction $(i+1,j)$, right direction $(i,j+1)$, or upper-right diagonal direction $(i+1, j+1)$ should be activated, but not any combination of the three.

Thus we define the *lineness* penalty below, that gets larger when more than one of these direction are activated near an activated cell:

$
cal(P)_l (bold(H)) &= sum_(t=0)^(T-2) sum_(tau=0)^(K-2) bold(H)_(t,tau) (bold(H)_(t,tau+1) bold(H)_(t+1,tau+1) + bold(H)_(t+1,tau) bold(H)_(t+1,tau+1) + bold(H)_(t+1,tau) bold(H)_(t,tau+1) )
$

*gradient calculation and separation*
$
(partial cal(P)_l) / (partial bold(H)_(i j))  =gradient_bold(H)^+ cal(P)_l &= bold(H)_(i,j+1) bold(H)_(i+1,j+1) + bold(H)_(i+1,j) bold(H)_(i+1,j+1) + bold(H)_(i+1,j) bold(H)_(i,j+1) \
&+ bold(H)_(i-1,j) bold(H)_(i,j+1) + bold(H)_(i-1,j) bold(H)_(i-1,j+1) \
&+ bold(H)_(i,j-1) bold(H)_(i+1,j) + bold(H)_(i,j-1) bold(H)_(i+1,j-1) \
&+ bold(H)_(i-1,j-1) bold(H)_(i-1,j) + bold(H)_(i-1,j-1) bold(H)_(i,j-1) \
gradient_bold(H)^- cal(P)_l &= 0
$

=== Multi-pass NMF <sec:multi-pass>
#text(fill: red)[Explication de l'algorithme, + creusage des matrices]

#figure(
  kind: "algorithm",
  supplement: [Algorithm],

  pseudocode-list(numbered-title:[Multi-pass NMF])[
    - *Inputs*: _hop_sizes_[], _overlap_
    - *Output*: estimated activation matrix
    + *Let* _Hs_[]
    + *For* $i$ in _hop_sizes_.length()
      + _hlen_ $<-$ _hop_sizes_[$i$]
      + _wlen_ $<-$ _hop_ \* _overlap_
      + $bold(W) <-$ spectrogram(concatenate(reference tracks), _hlen_, _wlen_))
      + $bold(V) <-$ spectrogram(mix, _hlen_, _wlen_))
      + *If* $i = 0$
        + $bold(H) <-$ noise
      + *Else*
        + $bold(H) <-$ filter_and_resize(_Hs_[$i-1$], _hlen_) (@algo:filter-and-resize)
      + *End If*
    + _Hs_.append($bold(H)$)
    + *End for*
    + *Return* _Hs_[$i$]
  ]
) <algo:multipass>

#figure(
  kind: "algorithm",
  supplement: [Algorithm],

  pseudocode-list(numbered-title:[Filtering and resizing of the activation matrix])[
    - *Inputs*: Activation matrix $bold(H)$, ...
    - *Output*: Activation matrix $bold(H)'$
    + #text(fill:red)[TODO]
  ]
) <algo:filter-and-resize>

== Implementation
#text(fill: red)[pytorch, discussion de cpu vs. gpu, github]

=== Use of a sparse matrix representation
#text(fill: red)[Matrice creuses, représentation BSR = mega speedup + baisse de conso mémoire]

== Résultats
=== Sur mix synthétiques
=== Sur unmixdb
=== Sur mix réels

== Extensions
=== Estimation des EQ
#text(fill: red)[Découpage des matrices en bandes + justification par le hardware]
=== Invariance à la transposition
#text(fill: red)[cf. divergence invariante à la transposition, ou NMFD, ou PLCA... ou etude quantitative que le binnage du melspec suffit. Tester aussi la CQT ?]

= Conclusion
