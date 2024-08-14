#import "@preview/lovelace:0.3.0": *
#import "@preview/ctheorems:1.1.2": *

#let title = "DJ mix reverse engineering using multi-pass non-negative matrix factorization"

#set page(header: context {
  if counter(page).get().first() > 2 [
    #align(center)[
      #smallcaps(title)
    ]
  ]
})

#set page(footer: context [
  #align(center)[
    #counter(page).display(
      "1/1",
      both: true,
    )
  ]
])
#set text(lang: "en", font: "Linux Libertine")
#show math.equation: set text(font: "New Computer Modern Math")

#set heading(numbering: "I.1.")
#set figure(placement: auto)
#let appendix(body) = {
  set heading(numbering: "A.1.", supplement: [Appendix])
  counter(heading).update(0)
  body
}
#show heading.where(level: 1): it => [
  #pagebreak(weak: false)

  #set text(size: 16pt, hyphenate: false)
  #block(smallcaps(it))
  #line(length: 100%)
  #v(1em)
]

#let my-lovelace-defaults = (
  booktabs: true,
)
#let pseudocode = pseudocode.with(..my-lovelace-defaults)
#let pseudocode-list = pseudocode-list.with(..my-lovelace-defaults)

#let proof = thmproof("proof", "Proof")
#show: thmrules.with(qed-symbol: $square$)

#set math.equation(numbering: "(1)")
#show math.equation.where(block: true): it => rect(width: 100%, fill: rgb("#def4ff"))[
  #v(0.5em)
  #it
  #v(0.5em)
]

#let logos = (
  "IRCAM.CP.jpg",
  "LOGO_SU_HORIZ_SEUL.jpg",
  "Logo_Télécom_Paris.jpg",
)
#align(center)[
  #text(hyphenate: false)[
    #smallcaps[
      #v(1fr)
      #image("ATIAM_logo_LONG_RVB.jpg", width: 50%)
      #text(20pt, "Internship report")
      #v(1fr)
      #line(length: 100%)
      #text(25pt, weight: "bold", title)
      #line(length: 100%)
      #v(1fr)
    ]
  ]
  #grid(
    columns: 2,
    column-gutter: 5em,
    [_Author_ \ Étienne ANDRÉ], [_Supervisors_ \ Diemo SCHWARZ \ Dominique FOURER],
  )
  #v(1fr)
  // TODO: vertical align + telecom paris too big
  #grid(
    columns: logos.len(),
    ..logos.map(i => box(image(i, width: 1 / logos.len() * 150%)))
  )

  #pagebreak()

  #v(1fr)
  #par(justify: false)[
    *Abstract* \
    #lorem(200)
  ]
  #v(1fr)
  #par(justify: false)[
    *Résumé* \
    #lorem(200)
  ]
  #v(1fr)
]


#show outline.entry.where(level: 1): it => {
  v(12pt, weak: true)
  strong(it)
}
#outline(indent: auto)

#set par(justify: true)

= Introduction

In this section, we introduce DJ culture and techniques, and the field of DJ-MIR. We discuss prior art and available datasets.

== The art of DJing

For decades, DJs (_Disc-Jockeys_) and the practice of DJing have played an integral role in shaping our musical landscape. While DJing can be interpreted in various ways, this thesis defines it as the continuous playback of recorded media (_tracks_) to an audience—whether live, via radio or other broadcast media, or through recorded formats like _mixtapes_ or streaming services.

A DJ's performance, often referred to as a _mix_, involves the careful selection and sequential playback of music tracks. However, DJing is not merely a passive process; it is a transformative art. DJs often overlap tracks, either to create entirely new musical compositions (_mashups_) or more commonly, to maintain a seamless flow between songs. The overlapping segments, known as _transitions_, are crucial to this process, with the simplest form being a cross-fade.

In the realm of dance music, these transitions are crafted to be as seamless as possible to maintain the energy and keep the audience dancing. DJs employ various techniques to achieve this, such as synchronizing the tempo and downbeats of overlapping tracks, manipulating EQ settings to highlight or diminish certain elements, and applying effects like reverb and delay.

Traditional vinyl DJs are typically limited to selecting _cue points_ — the specific time offset at which a track is introduced — and adjusting the playback speed, sometimes in extreme ways, as in the case of scratching. In contrast, modern digital DJ equipment offers a broader array of transformative tools, including transposition, time-warping, looping, and jumping between different sections of a track.

Moreover, contemporary DJ sets may include additional elements such as spoken word or vocals from an MC, rhythm machines, and sound effects like sirens or jingles, further enriching the auditory experience.

== DJ-MIR and prior art

Despite the pervasive influence of DJing in popular culture, it remains an understudied domain within academic research. A deeper understanding of DJ practices could significantly contribute to various fields, including musicology, cultural studies, music technology, and the automation of DJ techniques for both entertainment and commercial applications.

The field of DJ-MIR (DJ Music Information Retrieval) seeks to address this gap by developing methods to extract metadata from recorded DJ mixes. This field has traditionally been divided into several key subtasks:

/ Identification: Retrieving the playlist used to create the mix, as explored in @sixOlafLightweightPortable2023 @sonnleitnerLandmarkBasedAudioFingerprinting2016 @wangIndustrialStrengthAudioSearch;
/ Alignment: Determining the start and end times of each track within the mix, along with the time-scaling factors, as in @ewertHighResolutionAudio2009 @ramonaAutomaticAlignmentAudio2011 @werthen-brabantsGroundTruthExtraction2018 @kimComputationalAnalysisRealWorld2020 @schwarzMethodsDatasetsDJMix2021 @yangAligningUnsynchronizedPart2021 @sixDiscStitchAudiotoaudioAlignment2022;
/ Unmixing: Estimating and reversing the fade curves @ramonaSimpleEfficientFader2011 @werthen-brabantsGroundTruthExtraction2018 @schwarzExtractionGroundTruth @kimJointEstimationFader2022, the EQ parameters @kimJointEstimationFader2022, and any additional effects @barchiesiReverseEngineeringMix2010;
/ Content analysis: Derive metadata such as genre and indicators of various social and musical aspects of the mix, such as cue points @zehrenAutomaticDetectionCue2020, @schwarzHeuristicAlgorithmDJ2018 or drops @yadatiDetectingDropsElectronic2014 @kimAutomaticDjMix2017.
/ Mix generation: Automatically generating realistic mixes from a given playlist, as addressed in @bittnerAutomaticPlaylistSequencing2017 @cliffHangDJAutomatic2000 @fujioSystemMixingSongs2003 @kimAutomaticDjMix2017 @chenAutomaticDJTransitions2022 @liLooPyResearchFriendlyMix2023

While identification, mix generation, and content analysis have already seen some level of industrial application, the tasks of alignment and unmixing require further research.

Previous research has typically treated alignment and unmixing as sequential, independent tasks. Alignment is generally accomplished through feature extraction or fingerprinting combined with dynamic time warping (DTW). Subsequently, time-scaling factors are estimated, transition regions are segmented, and the unmixing process follows.

This approach has proven effective for broadcast mixes and simpler DJ sets, where tracks are often assumed to played monotonously (i.e. without jumps and loops) and at a fixed speed to mitigate alignment issues in transition regions. However, it is precisely in these transition regions that DJs are likely to employ the most creative transformations and effects.

Therefore, we propose an integrated approach that treats alignment and unmixing as a single, conjoint tanscription task, enabling a more general understanding of the complexities inherent in DJ mixes.

== DJ-MIR Datasets

In order to evaluate their methods, researchers are in the need of appropriate datasets. Despite the enormous amount of available mixes, thanks to decades of recorded mixes and specialized streaming services, they are rarely sufficiently and correctly annotated.

Appropriate datasets for DJ-MIR typically consist of mixes paired with the individual tracks used, accompanied by their respective time positions within the mix. Depending on the available annotations, these datasets may also include detailed information about the manipulations performed by the DJ during the creation of the mix, such as the volume levels for each track, time-stretching and transposition factors, the types and parameters of effects applied, and more. We operate under the assumption that having precise knowledge of all processing applied to the original tracks enables the accurate reconstruction of the mixed signal.

Such datasets can be separated in two broad categories:

/ Synthetic: These consist of mixes generated according to a set of predefined rules.
/ Real: These are derived from real-life mixes and are annotated either manually or automatically.

=== Synthetic datasets

Synthetic datasets offer the distinct advantage of precision and completeness, as the parameters used in the creation of the mixes are explicitly known and controlled. This allows for a high level of accuracy in analyses, as every aspect of the mix, from track selection to the application of effects, is documented by design. However, because these datasets are generated according to predefined rules, they may lack the diversity and complexity found in real-world DJ sets. This limitation can affect their ecological validity, meaning their ability to generalize to real-world scenarios may be compromised.

In the context of DJ-MIR research, UnmixDB @schwarzUnmixDBDatasetDJMix2018, which is based on the `mixotic` dataset from #cite(<sonnleitnerLandmarkBasedAudioFingerprinting2016>, form:"prose")), and the dataset from #cite(<werthen-brabantsGroundTruthExtraction2018>, form:"prose"), are among the few accessible examples of synthetic datasets. These datasets are derived from copyleft tracks released on the Mixotic netlabel #footnote(link("https://www.mixotic.net")), which primarily features electronic music. As a result, the datasets may lack stylistic diversity, potentially limiting their applicability to broader DJ-MIR research that encompasses a wider range of musical genres and DJ practices. However, because UnmixDB is easily accessible and features a large amount of data, we used it for evaluation of our method.

While synthetic datasets are invaluable for certain types of controlled experimentation and algorithm development, one must be mindful of their limitations when drawing conclusions that aim to reflect the complexities of real-world DJing.

=== Real datasets

Datasets derived from real DJ sets are valuable because they accurately reflect actual DJing practices, both past and present. Unlike synthetic datasets, they do not suffer from issues related to artificial generation, making them more representative of real-world scenarios. However, the use of real DJ sets presents several challenges. One of the primary obstacles is the need for time-intensive manual annotation or reliance on automatic annotation methods, which can be prone to errors. Additionally, the commercial nature of the mixes and the tracks within them raises significant legal concerns, particularly regarding their use in an academic setting.

To our knowledge, there is currently no freely available real dataset that meets the needs of DJ-MIR research. Existing datasets, such as those described in #cite(<kimComputationalAnalysisRealWorld2020>, form:"prose"), #cite(<scarfeLongRangeSelfsimilarityApproach2013>, form:"prose"), #cite(<kimJointEstimationFader2022>, form:"prose") and the `disco` dataset from #cite(<sonnleitnerLandmarkBasedAudioFingerprinting2016>, form:"prose"), are not publicly accessible due to commercial and copyright constraints.

The identification of tracks, or Track IDs, is a recurring topic of interest within DJ culture, engaging both DJs and their audiences. This has led to the emergence of online platforms that host crowdsourced or automatically detected track IDs, such as 1001tracklists#footnote(link("https://www.1001tracklists.com")), CueNation#footnote(link("https://cuenation.com/")), LiveTracklist#footnote(link("https://www.livetracklist.com/")), mixes.wiki (formerly known as MixesDB)#footnote(link("https://www.mixes.wiki/w/Main_Page")) and TrackId.net#footnote(link("https://trackid.net/")).

These platforms collectively provide a vast amount of data, which could potentially be leveraged for data-mining activities and may be particularly useful for the content analysis aspect of DJ-MIR. However, the information available on these sites typically includes only track identification and approximate track positions within the mix. Furthermore, the audio content is often copyrighted, and the data itself may be subject to legal restrictions, which complicates its use in research. As a result, researchers must independently obtain the associated audio content, adding another layer of complexity to the process.

During the course of the internship, we obtained an extract of the TrackId.net database with the assistance of its owners. However, we found that this dataset offered limited practical utility for our purposes, but that it could be valuable for future research. A more detailed analysis of this dataset is provided in @sec:trackidnet.

= DJ mix transcription using Non-Negative Matrix Factorization (NMF)

In this section, we present the use of the well-known NMF algorithm to perform DJ mix transcription.
We first study DJ hardware and software to justify the transcription task as a matrix factorization problem, and introduce the penalized Beta-NMF algorithm. We then show that the matrix factorization can yield an intuitive representation of DJ mix parameters. We propose a multi-pass extension of the NMF algorithm that greatly improves its performance. We then present reproducible results and discuss possible improvements.

== Objectives

*Given*:
- a recording of a DJ mix;
- the recordings of all the mix's constituting tracks;

*Estimate*:
- any temporal transformations applied to the tracks (play time, duration, speed, loops, jumps);
- the mixing gain;
- any effects applied to the tracks/and or mix (distortion, compression, pitch-shifting, filtering, EQing...)

== DJ mixing hardware

The DJs' field of expression is defined by its hardware and/or software: decks, mixing tables, and controllers. Despite the diversity in brands and models, which offer varying feature sets, the underlying workflow remains consistent across different setups. This consistency allows us to generalize the signal path from recorded tracks to the final mixed output, as depicted in @dj-signal-path.

#figure(
  image("dj-deck.svg"),
  caption: [Signal path of typical DJ hardware],
  placement: none,
) <dj-signal-path>

The signal path illustrated is directly derived from standard DJ setups, encompassing both hardware and software environments.#footnote[It is noteworthy that DJ software is typically designed to emulate the functionality of traditional DJ hardware, thereby preserving the validity of this signal path.] The process can be described as follows:

- Two or more DJ decks (in blue) are used as signal sources and play pre-recorded tracks and apply time-warping.
- The signal from the decks is routed to the DJ mixer, which perform a weighted sum of the input signals. The mixer may also apply various effects, the most prevalent being a 3- or 4-band equalizer (EQ). Additional elements, such as external audio sources or digital effects, are also integrated at this stage.
- Post-mixing, additional processing might be applied to the mixed output to meet specific distribution or venue requirements. This processing typically involves light modifications such as compression and equalization. However, given the minimal nature of these modifications, they will be considered negligible and thus omitted from further discussion in this report.

== Matrix representation

We now use this knowledge to introduce a matrix formulation of DJ mixing, by considering a spectrogram reprensentation of the signals. We achive this by grouping all non-time-based transformations, and modeling any additional elements and timbral effects as additive noise, as illustrated in @separate-boxes.

#figure(
  image("separate-boxes.svg"),
  caption: [Separated signal path],
  placement: none,
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
  image("nmf-djmix.svg"),
  caption: [Matrix form of the DJ mixing process],
  placement: none,
) <nmf-djmix>


Then, by defining two additional matrices $bold(W)_((a))$ and $bold(H)_((a))$ of compatible dimensions so that $bold(N) = bold(W)_((a)) bold(H)_((a))$, we can rewrite @eq:mix as a simple matrix multiplication of two large matrices by concatenation:

$
  bold(V) &= bold(W)_((a)) bold(H)_((a)) + sum_(i = 1)^M bold(W)_((i)) bold(H)_((i)) \
  &= underbrace(mat(bold(W)_((1)) bold(W)_((2)) ... bold(W)_((M))bold(W)_((a))), bold(W)) underbrace(mat( bold(H)_((1)); bold(H)_((2)); dots.v; bold(H)_((M)); bold(H)_((a))), bold(H))
$

Thus, estimating the gain and time-warping amounts to determining the coefficients of the $bold(H)$ matrix. Additionally, by determining $bold(W)_((a))$, any additional elements and timbral effects can be estimated. Such an estimation task is well-suited for the NMF family of algorithms, which has proven effective especially in audio source separation tasks.

== NMF Algorithm

=== Beta-NMF and Multiplicative Update rules
Let $bold(W)_(F times K)$, $bold(H)_(K times N)$ and $bold(V)_(F times N)$ non-negative matrices. The NMF algorithm in its most basic form aims to minimise a similarity measure $cal(D)$ between the _target matrix_ $bold(V)$ and the _estimated matrix_ $bold(W) bold(H)$, and amounts to solving the following optimization problem:

$
  min_(bold(W) , bold(H)) cal(D) (bold(V) | bold(W) bold(H)) "with" bold(W) >= 0 , bold(H) >= 0
$ <eq:optimization-problem>

The similarity measure we use is the beta-divergence, which is defined $forall beta in RR$ as follows:

$
  cal(D)_beta lr((bold(V) | bold(W) bold(H))) = sum_(f = 1)^F sum_(n = 1)^N d_beta lr((bold(V)_(f n) | lr((bold(W H)))_(f n)))
$
$
  d_beta lr((x | y)) = cases(
  frac(1, beta lr((beta - 1))) lr((x^beta + lr((beta - 1)) y^beta - beta x y^(beta - 1))) & "if" beta != {0, 1},
  x log x / y - x + y & "if" beta = 1,
  x / y - log x y - 1 & "if" beta = 0
)
$ <eq:beta-divergence>

It can be noted that the beta-divergence is equivalent to:
- the Euclidian distance if $beta = 2$;
- the Kullblack-Liebler divergence if $beta = 1$;
- the Itakura-Saito divergence if $beta = 0$.

As shown in @fevotteNonnegativeMatrixFactorization2009 and later extended in @fevotteAlgorithmsNonnegativeMatrix2011, an efficient and simple gradient descent algorithm for $bold(W)$ and $bold(H)$ can be derived if the gradient of the divergence w.r.t. a parameter $bold(theta)$ is separable into its positive and negative parts:

$
  gradient_bold(bold(theta)) cal(D)_beta (bold(V) | bold(W) bold(H)) = gradient_bold(bold(theta))^+ cal(D)_beta (
    bold(V) | bold(W) bold(H)
  ) - gradient_bold(bold(theta))^- cal(D)_beta (bold(V) | bold(W) bold(H))
$ <eq:gradient-separation>

Using the notation trick described in @fevotteAlgorithmsNonnegativeMatrix2011, the so-called _multiplicative update_ rules can be obtained#footnote[$dot.circle$ and $.../...$ stand respectively for Hadamard's (element-wise) product and division.]:
$
  bold(theta) <- bold(theta) dot.circle (gradient_bold(bold(theta))^- cal(D)_beta (
    bold(V) | bold(W) bold(H)
  )) / (gradient_bold(bold(theta))^+ cal(D)_beta (bold(V) | bold(W) bold(H)))
$ <eq:mu-gradient>

With the beta-divergence, this yields the update rules in @algo:mu-betadiv which can be very efficiently implemented, with strong monotonicity guarantees when $beta in [0,2]$.

#figure(
  kind: "algorithm",
  supplement: [Algorithm],

  pseudocode-list(numbered-title: [NMF Algorithm with Multiplicative Updates], line-gap: 1.4em)[
    + *Initialize* $bold(W) >= 0$ and $bold(H) >= 0$
    + *Until* convergence criterion is reached:
      + $bold(H) arrow.l bold(H) dot.circle (bold(W)^T ((bold(W H))^(beta - 2) dot.circle bold(V))) / (bold(W)^T (
            bold(W H)
          )^(beta - 1))$
      + $bold(W) arrow.l bold(W) dot.circle (((bold(W H))^(beta - 2) dot.circle bold(V)) bold(H)^T) / ((
            bold(W H)
          )^(beta - 1) bold(H)^T)$
  ],
) <algo:mu-betadiv>

An interesting property of this algorithm is that any zeroes in $bold(H)$ or $bold(W)$, by property of multiplication, remain zero troughout the optimization process. We will exploit this property in @sec:multi-pass.

=== Penalized Beta-NMF

More complex objective functions can be crafted by adding supplementary functions to the similarity measure, for penalization or regularization of the solutions. To illustrate, we will consider a new objective function $cal(C)$ comprised of the beta-divergence and an additional penalty function $cal(P)$ on $bold(H)$ weighted by $lambda in RR^+$:

$
  cal(C) = cal(D)_beta (bold(V) | bold(W H)) + lambda cal(P)(bold(H))
$

Supposing the gradients of $cal(C)$ w.r.t. $bold(H)$ and $bold(W)$ are separable into their positive and negative parts, new multiplicative update rules can be derived by following the procedure from the previous section. However, the monotonicity mentioned above is no longer guanranteed, and is dependent on the form of $cal(P)$ and the choice of $lambda$; which both can be validated through experimentation.

=== Choosing the divergence and the type of spectrograms

Previous research has demonstrated that the Kullback-Leibler divergence yields favorable results in source separation tasks when applied to magnitude spectrograms @virtanenMonauralSoundSource2007. Conversely, pairing the Itakura-Saito divergence with power spectrograms has been shown to perform well, supported by a robust statistical model @fevotteMajorizationminimizationAlgorithmSmooth2011.

Additionally, #cite(<fitzgeraldUseBetaDivergence2009>, form: "prose") explore the use of fractional values for the parameter beta in source separation and suggests that spectrograms could be raised to a fractional power. This approach is further refined by the introduction of the tempered beta-divergence, where beta acts as a temperature parameter that varies during the optimization process. However, despite these theoretical advancements, we found no conclusive evidence in the existing literature regarding the optimal approach for our specific task.

Ultimately, after conducting our own experiments, we selected the Itakura-Saito divergence applied to power spectrograms, as this combination consistently produced the best results in our trials.

== Characterization of warp and gain transformations

In this section, we present a model for time-warping and applying gain to a signal within the spectral domain. We demonstrate that a specific solution for the activation matrix exists, which reveals intuitive structural properties. Following this, we define estimators for both gain and time-warping, and conduct an analysis of their robustness in the presence of noise and other sources of uncertainty.

From now on, $t$ denotes the discrete time in the track, and $tau$ the discrete time in the mix.

=== The ideal kernel

Let $x[t]$ be a real-valued signal, and define the following:
- $f[tau]$ a time-warping injective function that maps a mix time step $tau$ to a track time step $t$;
- $g[tau]$ a gain factor signal;
- $y[tau]$ the time-warped and gain-modulated transformation of $x$;
- $w[n]$ an arbitrary window function of length $M$.

We define $bold(X) = (bold(X)_(m t))$ as the the power spectrogram matrix of $x$ ($M$ frequency bins $times T$ time steps):
$ bold(X)_(m t) = abs(sum_(n=0)^(M-1) x[n+t] w[n] e^(-j 2 pi n m / M))^2 $ <eq:stft>

Similarly, we define $bold(Y) = (bold(Y)_(m tau))$ as the power spectrogram mateix of $y$ ($M$ frequency bins $times K$ time steps). We show that $bold(Y)$ can be expressed in terms of $bold(X)$ as follows:

$
  bold(Y)_(m tau) &= abs(g[tau] sum_(n=0)^(M-1) x[n+f[tau]] w[n] e^(-j 2 pi n  m/ M))^2 \
  &= g[tau]^2 bold(X)_(m,f[tau])
$ <xy-relation>

We can then find a matrix $bold(H) = (bold(H)_(t tau))$ (of dimensions $T$ time steps $times K$ time steps) that satisfies:
$ bold(Y) &= bold(X) bold(H) <=> bold(Y)_(m tau) = sum_(t=0)^(T-1) bold(X)_(m t) bold(H)_(t tau) $ <matmul>

The _ideal kernel_ $bold(H)^"ideal"$, a solution to @matmul, is of particular interest. When viewed as an image, this matrix offers an intuitive understanding of the transformations applied to $x[t]$, as illustrated in @fig:time-time.

The ideal kernel is defined as:

$ bold(H)^"ideal"_(t tau) eq.def g[tau]^2 delta_(t,f[tau]) $ <ideal-kernel>
where $delta_(a,b)$ is the Kronecker delta function, defined by $delta_(a,b) = cases(1 "if" a = b, 0 "otherwise")$.

#figure(
  image("../2024-03-26/temps-temps.drawio.png"),
  caption: [Some examples of the structures emerging in $bold(H)^"ideal"$, with the associated DJ nomenclature.],
) <fig:time-time>

=== Estimation of the warp and gain values

We define the following estimators for the gain and time-warping functions:

$ tilde(g) [tau] = sqrt(sum_(t=1)^(T) bold(H)_(t tau) ) $ <gain_estimator_sum>
$ tilde(f) [tau] = "argmax"_(t in [1...T]) bold(H)_(t tau) $ <time_estimator_argmax>

Intuitively, $tilde(g)[tau]$ represents the energy of a column of $bold(H)$, while $tilde(f)[tau]$ corresponds to the position of its peak.

In the case of the ideal kernel (@ideal-kernel), it can be easily shown that these are exact estimators, meaning they perfectly recover the gain and time-warping functions. However, in practical scenarios, the optimization algorithm used to compute $bold(H)$ does not inherently guarantee convergence to this ideal solution.

In practice, the NMF tends to converge towards the similarity matrix between $y$ and $x$, rather than the idealized sparse solution with line features. This underscores the need to incorporate additional techniques that guide the algorithm towards convergence to the ideal solution, thereby ensuring that the estimators remain robust in the presence of noise and other uncertainties. We discuss briefly a few examples of such indeterminacies in the next section.

=== Sources of indeterminacies

==== Impact of self-similar input signals

#figure(
  [#image("../2024-05-30/longue-note.png")],
  caption: [Indeterminaciess in $bold(H)$ caused by spectrally similar frames.],
) <fig:indeterminacies>

#figure(
  [#image("../2024-05-17/image-2.png")],
  caption: [Parallel line structures in $bold(H)$ caused by loops in the reference tracks.],
) <fig:parallel-lines>

Given the nature of musical signals, two columns of $bold(X)$ could be almost identical (@fig:parallel-lines), for example in the presence of a loop in electronic music (@fig:indeterminacies).

Let $t_1$ and $t_2$ be the time steps at which this is true, and $tau_1=f^(-1)[t_1]$ and $tau_2=f^(-1)[t_2]$ their antecedents. We then have $forall m$:
$ bold(Y)_(m tau_1) = bold(Y)_(m tau_2) = g[tau_1]^2 bold(X)_(m t_1) = g[tau_2]^2 bold(X)_(m t_2) $

Visually, this corresponds to having multiple activations per column of $bold(H)$, with the energy of the activations being distributed arbitrarily between four points: ${(t_1, tau_1), (t_1, tau_2), (t_2, tau_1), (t_2, tau_2)}$. Fortunately, such indeterminacies do not invalidate $tilde(g)$, but the same can not be said of $tilde(f)$.

==== Impact of hop size discretization

Usually, the spectrogram is not calculated for every sample of a signal as in our earlier definition (@eq:stft), but at regular intervals defined by a hop size $h$. This effectively downsamples the time steps to $overline(t) = h t$ and $overline(tau) = h tau$, leading to the following expressions for the spectrograms:

$ bold(X)_(m overline(t)) = abs(sum_(n=0)^(M-1) x[n+h t] w[n] e^(-j 2 pi n m / M))^2 $
$ bold(Y)_(m overline(tau)) &= abs(g[tau] sum_(n=0)^(M-1) x[n+f[h tau]] w[n] e^(-j 2 pi n  m/ M))^2 $

Due to this discretization, there may not be an exact alignment between $overline(t)$ and $overline(tau)$. Consequently, the activations within $bold(H)$ could be distributed across neighboring cells.

#text(fill: red)[ajouter une illustration des pics étalés dans $bold(H)$]

== Multi-pass NMF Algorithm<sec:multi-pass>

DJ mixes consist of multiple tracks, each typically appearing only within a specific segment of the mix. Consequently, the activation matrix $bold(H)$ is expected to exhibit block-sparsity, as depicted in @fig:block-sparse.

While applying NMF directly to feature matrices with the desired hop size may yield the expected results, empirical observations suggest that an increased number of tracks and smaller window sizes exacerbate cross-track indeterminacies. This issue arises especially because tracks in a mix are often stylistically or tonally similar.

To address this we propose a multi-pass NMF algorithm, formally described in @algo:multipass, paired with a filter-threshold-resize procedure inbetween each pass.

The methodology involves initially performing NMF on feature matrices with a significantly large hop size (on the order of minutes) to obtain the approximate position of each track in the mix. The resultant activation matrix is then processed through filtering to reduce noise, followed by blurring and thresholding. This matrix is resized to match the dimensions corresponding to the next smaller hop size, resulting in a larger activation matrix that serves as the initialization for the subsequent NMF pass. This iterative process continues until the desired hop size is reached.

By property of the NMF with multiplicative updates, regions of the matrix set to zero during thresholding will remain zero in subsequent iterations, thereby avoiding spurious activations. However, careful selection of filtering and thresholding techniques is crucial to avoid the inadvertent elimination of valid activations.

When coupled with an appropriate block-sparse matrix representation, this approach significantly enhances processing efficiency and memory usage, which is particularly advantageous given the large size of feature matrices at smaller hop sizes.

#figure(
  image("../2024-05-17/creuse.svg"),
  caption: [Expected block-sparse form of the activation matrix in a 5-track mix, with transition regions annotated.],
) <fig:block-sparse>

#figure(
  kind: "algorithm",
  supplement: [Algorithm],

  pseudocode-list(numbered-title: [Multi-pass NMF])[
    - *Inputs*:
      - _hop_sizes_[]: list of decreasing hop sizes
      - _overlap_: overlap factor
    - *Output*: estimated activation matrix
    + *Let* _Hs_[] = empty list
    + *For* $i$ in _hop_sizes_.length()
      + _hlen_ $<-$ _hop_sizes_[$i$]
      + _wlen_ $<-$ _hop_ \* _overlap_
      + $bold(W) <-$ spectrogram(concatenate(reference tracks), _hlen_, _wlen_))
      + $bold(V) <-$ spectrogram(mix, _hlen_, _wlen_))
      + *If* $i = 0$
        + $bold(H) <-$ noise
      + *Else*
        + $bold(H) <-$ filter_threshold_resize(_Hs_[$i-1$], _hlen_)
      + *End If*
      + _Hs_.append($bold(H)$)
    + *End for*
    + *Return* _Hs_[$i$]
  ],
) <algo:multipass>

=== Filter-threshold-resize procedure

The filter-threshold-resize procedure is integral to the effectiveness of the multipass NMF algorithm.

/ Filtering: A line-enhancing filter is applied on each submatrix $bold(H)_((i))$ of $bold(H)$, inspired by #cite(<mullerEnhancingSimilarityMatrices2006>, form: "prose"). This filter is designed using fixed-length one-pixel-wide straight line kernels with slopes distributed between a minimum and maximum value. A morphological opening operation is performed on HH with these kernels, and the results are aggregated. This process eliminates activations shorter than the specified length and that do not meet the expected slope limits, effectively denoising the activation matrix.
/ Blurring: A gaussian blur is applied on each submatrix with a small kernel. This has the effect of smearing the activations in time.
/ Thresholding: Set activations below a specified threshold to zero.
/ Resizing: The thresholded activation matrix is then resized to a larger size, i.e. corresponding to a smaller hop size, and returned.

== Additional improvements <sec:improvements>

=== Information compression

DJ mixes are typically lengthy, ranging from 30 minutes to several hours, and as they consist of musical signals, their frequency bandwidth is notably extensive. When employing the Short-Time Fourier Transform (STFT) with standard hop durations and a typical number of frequency bins for musical signals, the resulting feature matrix can become exceedingly large. This leads to substantial memory usage and elevated resource consumption.

In order to mitigate these issues, we have opted to use relatively large hop durations, on the order of 100 milliseconds. This approach not only reduces the computational load but also offers an additional advantage: longer hop durations are better adapted to the temporal structures inherent in music. The hop duration however is not fixed, as is explained in @sec:multi-pass.

Additionally, we compress the frequency information using the mel-scale transform @stevensScaleMeasurementPsychological1937. This transform groups nearby frequencies into bins based on a perceptual model of human hearing, which is particularly well-suited for processing musical signals. Importantly, this transform has no effect on the ideal kernel and our estimators.

#proof[
  Let $bold(M)$ be a matrix of mel filterbank coefficients. The mel-spectrograms are calculated from the regular spectrograms: $bold(X)^"mel" = bold(M)bold(X)$ and $bold(Y)^"mel" = bold(M)bold(Y)$. Then we have:
  $
    bold(Y)^"mel"_(m tau) &= sum_i bold(M)_(m i) bold(Y)_(i tau) \
    &= g[tau]^2 sum_i bold(M)_(m i) bold(X)_(i,f[tau]) \
    &= g[tau]^2 bold(X)^"mel"_(m, f[tau])
  $

  So the ideal kernel $bold(H)^"ideal"$ is still clearly a solution of @matmul.
]

=== Analysis window overlap

A key parameter when working with spectrograms is the overlap factor of the analysis windows. In order to emphasize the temporal continuity of the musical signals, we use high overlap factors: our experiments have shown that a window size of 6 to 8 times the hop size give the best results. It has proven to be highly effective in reducing indeterminacies, but tends to smooth out the results as a side-effect, which can potentially obscure finer details in the signal.

Typically, such large window sizes would result in a substantial increase in the number of frequency bins, leading to higher computational demands. However, by applying the mel-scale transform, we effectively mitigate this issue.

#text(fill: red)[Ajouter figures de $bold(H)$ pour != valeurs d'overlap]

=== Normalization

To improve the numeric stability of the NMF, the columns of $bold(X)$ are usually normalized to sum to 1. We also normalize $bold(Y)$ by a single factor#footnote[Normalizing by column as for $bold(X)$ would cancel out the gain information in $bold(H)$.]. This results in a simple scaling factor for $bold(H)$ and therefore for the estimators.

#proof[
  Let the scaling factors $bold(k)_t eq.def sum_i bold(X)_(i t)$ and $kappa eq.def sum_i sum_t bold(Y)_(i t)$.

  The normalized spectrograms are:
  $ bold(X)^"norm"_(m t) eq.def bold(X)_(m t) / bold(k)_t $

  $ bold(Y)^"norm" eq.def bold(Y) / kappa $


  Using @xy-relation:
  $
    bold(Y)^"norm"_(m tau) = bold(k)_t / kappa g[tau]^2 bold(X)^"norm"_(m, f[tau])
  $

  We can then deduce the ideal normalized kernel $bold(H)^"norm"$ as a solution to @matmul:

  $
    bold(H)^"norm"_(t tau) &eq.def bold(k)_t / kappa g[tau]^2 delta_(t,f[tau]) \
    &= bold(k)_t / kappa bold(H)^"ideal"_(t tau)
  $
]

=== Thresholding of low-power frames

Recorded music tracks often contain moments of silence or faint noise at the beginning and end. It is also quite common to find fade-out endings at the end of tracks, or reverb tails. These sounds take the form of low-power frames in the feature matrices. These elements appear as low-power frames in the feature matrices. Reverb tails, in particular, are problematic due to their spectral similarity to the rest of the track, which can introduce indeterminacies in the analysis.

Additionally, low-power frames are unlikely to be present in a DJ mix, as DJs typically focus on playing the most musically significant portions of tracks, omitting the very beginning and end.

To address this, we detect and mark low-power frames in the input track spectrograms as unused and set the corresponding cells in the activation matrix to zero, effectively preventing these frames from contributing to the optimization process.

=== Penalty functions

#text(fill: red)[Définition des fonctions de régularisation utilisées et poids $lambda$ recommandés + warm-up/temperature/tempered. Section à garder même si non utilisée à la fin ? A reecrire mieux.]

#text(fill: red)[ajouter L1 (et L2 si on s'en sert ?)]

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
  partial / (partial bold(H)_(i j)) (
    bold(H)_(t tau) - bold(H)_(t, tau-1)
  )^2 = cases(
  2(bold(H)_(i j) - bold(H)_(i, j-1)) &"if" i=t "and" j=tau,
  -2(bold(H)_(i,j+1) - bold(H)_(i j)) &"if" i=t "and" j+1=tau,
  0 &"otherwise"
)
$

So:
$
  (partial cal(P)_g) / (partial bold(H)_(i j)) &= 2(bold(H)_(i j) - bold(H)_(i, j-1)) -2(
    bold(H)_(i,j+1) - bold(H)_(i j)
  ) \
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
  partial / (partial bold(H)_(i j)) (
    bold(H)_(t tau) - bold(H)_(t-1, tau-1)
  )^2 = cases(
  2(bold(H)_(i j) - bold(H)_(i-1, j-1)) &"if" i=t "and" j=tau,
  -2(bold(H)_(i+1,j+1) - bold(H)_(i j)) &"if" i+1=t "and" j+1=tau,
  0 &"otherwise"
)
$

So:
$
  (partial cal(P)_d) / (partial bold(H)_(i j)) &= 2(bold(H)_(i j) - bold(H)_(i-1, j-1)) -2(
    bold(H)_(i+1,j+1) - bold(H)_(i j)
  ) \
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
  cal(P)_l (bold(H)) &= sum_(t=0)^(T-2) sum_(tau=0)^(K-2) bold(H)_(t,tau) (
    bold(H)_(t,tau+1) bold(H)_(t+1,tau+1) + bold(H)_(t+1,tau) bold(H)_(t+1,tau+1) + bold(H)_(t+1,tau) bold(H)_(t,tau+1)
  )
$

*gradient calculation and separation*
$
  (partial cal(P)_l) / (partial bold(H)_(i j)) =gradient_bold(H)^+ cal(P)_l &= bold(H)_(i,j+1) bold(H)_(i+1,j+1) + bold(H)_(i+1,j) bold(H)_(i+1,j+1) + bold(H)_(i+1,j) bold(H)_(i,j+1) \
  &+ bold(H)_(i-1,j) bold(H)_(i,j+1) + bold(H)_(i-1,j) bold(H)_(i-1,j+1) \
  &+ bold(H)_(i,j-1) bold(H)_(i+1,j) + bold(H)_(i,j-1) bold(H)_(i+1,j-1) \
  &+ bold(H)_(i-1,j-1) bold(H)_(i-1,j) + bold(H)_(i-1,j-1) bold(H)_(i,j-1) \
  gradient_bold(H)^- cal(P)_l &= 0
$


== Implementation

The algorithm has been implemented in Python and is available at #text(fill:red)[github link].

By leveraging the pytorch#footnote(link("https://pytorch.org")) framework, the optimization process can run on CPU and GPU and benefit from parallel matrix multiplications.

== Results
#text(fill:red)[Présentation des résultats sur unmixdb, mix custom pour montrer là où ça marche bien, et qq mix réels]
=== UnmixDB
=== Bespoke mixes
=== Real-world mixes

== Extensions
=== EQ estimation
#text(fill: red)[justification par le hardware, découpage des matrics en bandes puis traiter le problème de la même manière]

=== Invariance à la transposition
#text(fill: red)[cf. divergence invariante à la transposition, ou NMFD, ou PLCA... mais en pratique le binnage du melspec suffit pour les "petites" transpo (justif quantitative nécessaire?).]

= Conclusion
#text(fill:red)[Résultats moins bons que l'état de l'art, mais méthode + générale et prend moins d'hypothèses restrictives sur la forme du mix.]

#lorem(200)

#show: appendix

= Dataset creation by ground truth measurement

== Suivi optique de disques vinyles
#text(fill: red)[Présentation, résultats]

== Extraction des métadonnées de fichiers de projet DAW
#text(fill: red)[Présentation, résultats]

== Instrumentation de logiciel de mix
#text(fill: red)[Idée]

= Analysis of the TrackId.net dataset <sec:trackidnet>
#text(fill: red)[Présentation, statistiques, données, problèmes de droits. En attente de la réponse de Luis (proprio de trackid.net)]

#bibliography("../../zotero.bib"),
