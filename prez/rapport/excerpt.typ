
= DJ mix transcription

In this section, we introduce a new application of NMF algorithm to perform DJ mix transcription.
We first study DJ hardware and software to justify the transcription task as a matrix factorization problem, and introduce the base Beta-NMF algorithm. We then show that the matrix factorization can yield an intuitive representation of DJ mix parameters. We propose a multi-pass extension of the NMF algorithm that greatly improves its performance, and discuss additional modifications. We then present example results and evaluate our method on a publicly available dataset.

== DJ mixing hardware

The DJs' field of expression is defined by its hardware and/or software: decks, mixing tables, and controllers. Despite the diversity in brands and models, which offer varying feature sets, the underlying workflow remains consistent across different setups. This consistency allows us to generalize the signal path from recorded tracks to the final mixed output, as depicted in @dj-signal-path.

#figure(
  image("dj-deck.svg"),
  caption: [Schematic view of the DJ mixing process],
  placement: none,
) <dj-signal-path>

The signal path illustrated is directly derived from standard DJ setups, encompassing both hardware and software environments.#footnote[It is noteworthy that DJ software is typically designed to emulate the functionality of traditional DJ hardware, thereby preserving the validity of this signal path.] The process can be described as follows:

- Two or more DJ decks (in blue) are used as signal sources and play pre-recorded tracks and apply time-warping.
- The signal from the decks is routed to the DJ mixer, which performs a weighted sum of the input signals. The mixer may also apply various effects, the most prevalent being a 3- or 4-band equalizer (EQ). Additional elements, such as external audio sources or audio effects, are also integrated at this stage.
- Post-mixing, additional processing might be applied to the mixed output to meet specific distribution or venue requirements. This processing typically involves light modifications such as compression and equalization. However, given the minimal nature of these modifications, they will be considered negligible and thus omitted from further discussion in this report.

== Matrix representation of the DJ mixing process

We now use this knowledge to introduce a matrix formulation of DJ mixing, by considering a spectrogram reprensentation of the signals. We achive this by grouping all non-time-based transformations, and modeling any additional elements and timbral effects as additive noise, as illustrated in @separate-boxes.

#figure(
  image("separate-boxes.svg"),
  caption: [Separated signal path],
  placement: none,
) <separate-boxes>

Assuming the $M$ constituting tracks of the mix are known, let $forall i in [1 ... M]$:
- $WW_((i))$ the spectrogram of track $i$;
- $HH_((i))$ the so-called _activation matrix_ of track $i$, representing both the time-warping operations and the gain applied at the mixing stage;
- $VV_((i)) = WW_((i)) HH_((i))$ the time-remapped spectrogram of track $i$ with gain appplied;
- $bold(upright(N))$ a noise matrix representing the timbral changes applied to the mix and/or tracks and any additional elements;
- $VV$ the spectrogram of the mix.

Using these notations, we can write:

$ VV &= bold(upright(N)) + sum_(i = 1)^M WW_((i)) HH_((i)) $ <eq:mix>

An computation diagram of @eq:mix is given @nmf-djmix.

#figure(
  image("nmf-djmix.svg"),
  caption: [Matrix form of the DJ mixing process],
  placement: none,
) <nmf-djmix>


Then, by defining two additional matrices $WW_((a))$ and $HH_((a))$ of compatible dimensions so that $bold(upright(N)) = WW_((a)) HH_((a))$, we can rewrite @eq:mix as a simple matrix multiplication of two large matrices by concatenation:

$
  VV &= WW_((a)) HH_((a)) + sum_(i = 1)^M WW_((i)) HH_((i)) \
  &= underbrace(mat(WW_((1)) WW_((2)) ... WW_((M))WW_((a))), WW) underbrace(mat( HH_((1)); HH_((2)); dots.v; HH_((M)); HH_((a))), HH)
$ <eq:matmul>

Because we assume that the constituent tracks of the mix are known, the ($WW_((1))$ to $WW_((M))$ submatrices are known. Transcribing the DJ mix then amounts to determining the other coefficients of @eq:matmul:
- If the noise matrix is assumed to be zero, estimating the gain and time-warping amounts to determining the coefficients of the $HH$ matrix while keeping the $WW$ matrix fixed.
- If not, only part of the $WW$ matrix ($WW_((1))$ to $WW_((M))$ submatrices) is kept fixed, while the $WW_((a))$, $HH_((1))$ to $HH_((M))$ and $HH_((a))$ submatrices are estimated.

This can be understood as a matrix factorization problem. It is well-suited to the NMF family of algorithms, which has proven especially effective in audio source separation tasks, which we present in the next section.

== NMF Algorithm

=== Beta-NMF and Multiplicative Update rules <sec:beta-nmf>

Let $WW in RR_+^(F times K)$, $HH in RR_+^(K times N)$ and $VV in RR_+^(F times N)$ non-negative matrices. The NMF algorithm in its most basic form aims to minimise a similarity measure $cal(D)$ between the _target matrix_ $VV$ and the _estimated matrix_ $WW HH$, and amounts to solving the following optimization problem:

$
  min_(WW , HH) cal(D) (VV | WW HH) "with" VV >= 0, WW >= 0 , HH >= 0
$ <eq:optimization-problem>

The similarity measure we use is the beta-divergence, which is defined $forall beta in RR$ as follows:

$
  cal(D)_beta lr((VV | WW HH)) = sum_(f = 1)^F sum_(n = 1)^N d_beta lr((VV_(f n) | lr((WW HH))_(f n)))
$
$
  d_beta lr((x | y)) = cases(
  frac(1, beta lr((beta - 1))) lr((x^beta + lr((beta - 1)) y^beta - beta x y^(beta - 1))) & "if" beta != {0, 1},
  x ln x / y - x + y & "if" beta = 1,
  x / y - ln x y - 1 & "if" beta = 0
)
$ <eq:beta-divergence>

It can be noted that the beta-divergence is equivalent to:
- the Euclidian distance if $beta = 2$;
- the Kullblack-Leibler divergence if $beta = 1$;
- the Itakura-Saito divergence if $beta = 0$.

As shown in @fevotteNonnegativeMatrixFactorization2009 and later extended in @fevotteAlgorithmsNonnegativeMatrix2011, an efficient and simple gradient descent algorithm for $WW$ and $HH$ can be derived if the gradient of the divergence w.r.t. a parameter $bold(theta)$ is separable into its positive and negative parts:

$
  delta_bold(bold(theta)) cal(D)_beta (VV | WW HH) = delta_bold(bold(theta))^+ cal(D)_beta (
    VV | WW HH
  ) - delta_bold(bold(theta))^- cal(D)_beta (VV | WW HH)
$ <eq:gradient-separation>

Using the notation trick described in @fevotteAlgorithmsNonnegativeMatrix2011, the so-called _multiplicative update_ rules can be obtained#footnote[$dot.circle$ and $.../...$ stand respectively for Hadamard's (element-wise) product and division.]:
$
  bold(theta) <- bold(theta) dot.circle (delta_bold(bold(theta))^- cal(D)_beta (
    VV | WW HH
  )) / (delta_bold(bold(theta))^+ cal(D)_beta (VV | WW HH))
$ <eq:mu-gradient>

With the beta-divergence, this yields the update rules of @algo:mu-betadiv which can be efficiently implemented, with strong monotonicity guarantees when $beta in [0,2]$.

#figure(
  kind: "algorithm",
  supplement: [Algorithm],

  [
    + *Initialize* $WW >= 0$ and $HH >= 0$
    + *Until* convergence criterion is reached:
      + $HH arrow.l HH dot.circle (WW^T ((WW HH)^(beta - 2) dot.circle VV)) / (WW^T (WW HH)^(beta - 1))$
      + $WW arrow.l WW dot.circle (((WW HH)^(beta - 2) dot.circle VV) HH^T) / ((WW HH)^(beta - 1) HH^T)$
  ],
) <algo:mu-betadiv>

An interesting property of this algorithm is that any zeroes in $HH$ or $WW$, by property of multiplication, remain zero troughout the optimization process. We will exploit this property in @sec:multi-pass.

More complex objective functions can be crafted by adding supplementary functions to the similarity measure, for penalization or regularization of the solutions. We detail the principle and some penalty functions considered for DJ mix transcription in @sec:penalty.

=== Choosing the divergence and the type of spectrograms

Previous research empirically shown that the Kullback-Leibler divergence yields favorable results in source separation tasks when applied to magnitude spectrograms @virtanenMonauralSoundSource2007. Conversely, pairing the Itakura-Saito divergence with power spectrograms has been shown to perform well, supported by a robust statistical model @fevotteMajorizationminimizationAlgorithmSmooth2011.

Additionally, #cite(<fitzgeraldUseBetaDivergence2009>, form: "prose") explore the use of fractional values for the parameter beta in source separation and suggests that spectrograms could be raised to a fractional power. This approach is further refined by the introduction of the tempered beta-divergence, where beta acts as a temperature parameter that varies during the optimization process. However, despite these theoretical advancements, we found no conclusive evidence in the existing literature regarding the optimal approach for our specific task.

Ultimately, after conducting our own experiments, we selected the Itakura-Saito divergence applied to power spectrograms, as this combination consistently produced the best results in our trials.

== Characterization of warp and gain transformations

In this section, we present a model for time-warping and applying gain to a signal within the spectral domain. We demonstrate that a specific solution for the activation matrix exists, which reveals intuitive structural properties. Following this, we define estimators for both gain and time-warping, and conduct an analysis of their robustness in the presence of noise and other sources of uncertainty.

=== The ideal kernel

Let $x[t]$ be a real-valued signal, and define the following:
- $f:tau |-> t$ a time-warping injective function that maps a mix time step $tau$ to a track time step $t$;
- $g[tau]$ a gain factor sequence;
- $y[tau]$ the time-warped (by $f$) and gain-modulated (by $g$) transformation of $x$;
- $w$ an arbitrary window function of length $M$.

We define $XX = (XX_(m t))$ as the spectrogram matrix of $x$ ($M$ frequency bins $times T$ time steps):
$ XX_(m t) = abs(sum_(n=1)^(M) x[n+t] w[n] e^(-j 2 pi n m / M))^2 $ <eq:stft>

Similarly, we define $YY = (YY_(m tau))$ as the power spectrogram matrix of $y$ ($M$ frequency bins $times K$ time steps). We show that $YY$ can be expressed in terms of $XX$ as follows:

$
  YY_(m tau) &= abs(g[tau] sum_(n=1)^(M) x[n+f[tau]] w[n] e^(-j 2 pi n  m/ M))^2 \
  &= g[tau]^2 XX_(m,f[tau])
$ <xy-relation>

We can then find a matrix $HH = (HH_(t tau))$ (of dimensions $T$ time steps $times K$ time steps) that satisfies:
$ YY &= XX HH <=> YY_(m tau) = sum_(t=0)^(T-1) XX_(m t) HH_(t tau) $ <matmul>

The _ideal kernel_ $HH^"ideal"$, a solution to @matmul, is of particular interest. When viewed as an image, this matrix offers an intuitive understanding of the transformations applied to $x[t]$, as illustrated in @fig:time-time.

The ideal kernel is defined as:

$ HH^"ideal"_(t tau) eq.def g[tau]^2 delta_(t,f[tau]) $ <ideal-kernel>
where $delta_(a,b)$ is the Kronecker delta function, defined by $delta_(a,b) = cases(1 "if" a = b, 0 "otherwise")$.

#figure(
  image("../2024-03-26/temps-temps.drawio.png"),
  caption: [Some examples of the structures emerging in $HH^"ideal"$, with the associated DJ nomenclature.],
) <fig:time-time>

=== Estimation of the warp and gain values

We define the following estimators for the gain and time-warping functions:

$ tilde(g) [tau] = sqrt(sum_(t=1)^(T) HH_(t tau) ) $ <gain_estimator_sum>
$ tilde(f) [tau] = "argmax"_(t in [1 ... T]) HH_(t tau) $ <time_estimator_argmax>

Intuitively, $tilde(g)[tau]$ represents the energy of a column of $HH$, while $tilde(f)[tau]$ corresponds to the position of its peak.

In the case of the ideal kernel (@ideal-kernel), it can be easily shown that $tilde(f)$ and $tilde(g)$ are exact estimators, meaning they perfectly recover the gain and time-warping functions. However, in practical scenarios, the optimization algorithm used to compute $HH$ does not inherently guarantee convergence to this ideal solution.

In practice, the NMF tends to converge towards the similarity matrix between $y$ and $x$, rather than the idealized sparse solution with line features. This underscores the need to incorporate additional techniques that guide the algorithm towards convergence to the ideal solution, thereby ensuring that the estimators remain robust in the presence of noise and other uncertainties. We discuss briefly a few examples of such indeterminacies in the next section.

=== Sources of indeterminate forms

==== Impact of self-similar input signals

#figure(
  [#image("../2024-05-30/longue-note.png")],
  caption: [Artifacts in $HH$ caused by spectrally similar frames.],
) <fig:indeterminacies>

#figure(
  [#image("../2024-05-17/image-2.png")],
  caption: [Parallel line structures in $HH$ caused by loops in the reference tracks.],
) <fig:parallel-lines>

Given the nature of musical signals, two columns of $XX$ could be almost identical (@fig:parallel-lines), for example in the presence of a loop in electronic music (@fig:indeterminacies).


Let $t_1$ and $t_2$ be the time steps at which this is true, and $tau_1=f^(-1)[t_1]$ and $tau_2=f^(-1)[t_2]$ their antecedents. We then have $forall m$:
$ YY_(m tau_1) = YY_(m tau_2) = g[tau_1]^2 XX_(m t_1) = g[tau_2]^2 XX_(m t_2) $

Visually, this corresponds to multiple activations per column of $HH$, with the energy of the activations being distributed arbitrarily between four points: ${(t_1, tau_1), (t_1, tau_2), (t_2, tau_1), (t_2, tau_2)}$. Fortunately, such indeterminacies do not invalidate $tilde(g)$, but the same can not be said of $tilde(f)$.

==== Impact of hop size discretization

Usually, the spectrogram is not computed for every sample of a signal as in our earlier definition (@eq:stft), but at uniformly sampled time instants spaced by a _hop size_ $h$. This effectively downsamples the time steps to $overline(t) = h t$ and $overline(tau) = h tau$, leading to the following expressions for the spectrograms:

$ XX_(m overline(t)) = abs(sum_(n=0)^(M-1) x[n+h t] w[n] e^(-j 2 pi n m / M))^2 $
$ YY_(m overline(tau)) &= abs(g[tau] sum_(n=0)^(M-1) x[n+f[h tau]] w[n] e^(-j 2 pi n  m/ M))^2 $

Due to this discretization, there may not be an exact alignment between $overline(t)$ and $overline(tau)$. Consequently, the activations within $HH$ could be distributed across neighboring cells.

== The Multi-pass NMF Algorithm <sec:multi-pass>

DJ mixes consist of multiple tracks, each typically appearing only within a specific segment of the mix. Consequently, the activation matrix $HH$ is expected to exhibit block-sparsity, as depicted in @fig:block-sparse.

#figure(
  image("block-sparse.svg", width: 80%),
  caption: [Expected block-sparse form of the activation matrix in a 5-track mix, with transition regions annotated.],
) <fig:block-sparse>

While applying NMF directly to spectrograms computed with the desired hop size may yield the expected results, our experiments suggest that an increased number of tracks and smaller window sizes exacerbate cross-track indeterminacies. This issue arises especially because tracks in a mix are often stylistically or tonally similar.

To address this we propose a multi-pass NMF algorithm, described in @fig:multipass-flow, paired with a filter-threshold-resize procedure inbetween each pass.

The methodology involves initially performing NMF on spectrograms computed with a significantly large hop size (on the order of minutes) to obtain the approximate position of each track in the mix. The resultant activation matrix is then processed through filtering to reduce noise, followed by blurring and thresholding. This matrix is resized to match the dimensions corresponding to the next smaller hop size, resulting in a larger activation matrix that serves as the initialization for the subsequent NMF pass. This iterative process continues until the desired hop size is reached.

#figure(
  image("multipass-flowchart.svg", width: 50%),
  caption: [Flow chart of the multipass NMF algorithm],
) <fig:multipass-flow>

By property of the NMF with multiplicative updates, regions of the matrix set to zero during thresholding will remain zero in subsequent iterations, thereby avoiding spurious activations. However, careful selection of filtering and thresholding techniques is crucial to avoid the inadvertent elimination of valid activations.

Moreover, when coupled with an appropriate block-sparse matrix representation, this approach significantly enhances processing efficiency and memory usage, which is particularly advantageous given the large size of spectrogram matrices at smaller hop sizes.

As an illustration, we ran the multipass NMF algorithm on a 3-track mix with gradually decreasing hop sizes. The @fig:multipass depicts the estimated activation matrices at the end of each pass. The first hop size (15 seconds) gives a rough estimation of the positions of the constituent tracks in the mix. and with each subsequent pass, the activation matrices are larger the activations more precise, and become less noisy and more sparse.

#figure(
  image("multipass.svg", width: 120%),
  caption: [Vizualization of successive estimated activation matrices during execution of the multipass NMF algorithm. Zero-valued cells are depicted in white.],
) <fig:multipass>

=== Filter-threshold-resize procedure

The filter-threshold-resize procedure is integral to the effectiveness of the multipass NMF algorithm. The steps of the procedure are described below, and illustrated @fig:interpass.

/ Morpohological filtering: A line-enhancing filter is applied on each submatrix $HH_((i))$ of $HH$, inspired by #cite(<mullerEnhancingSimilarityMatrices2006>, form: "prose"). This filter is designed using fixed-length one-pixel-wide straight line kernels with slopes distributed between a minimum and maximum value. A morphological opening operation is performed on $HH$ with these kernels, and the results are aggregated. This process eliminates activations shorter than the specified length and that do not meet the expected slope limits, effectively denoising the activation matrix.
/ Blurring: A gaussian blur is applied on each submatrix with a small kernel. This has the effect of smearing the activations in time.
/ Thresholding: Set activations below a specified threshold to zero.
/ Resizing: The thresholded activation matrix is then resized to a larger size, i.e. corresponding to a smaller hop size, and returned.
#figure(
  image("interpass.svg", width: 150%),
  caption: [Vizualisation of the steps of the filter-threshold-resize procedure on a 3-track mix. The input activation matrix corresponds to a hop size of 3 seconds, and the output corresponds to a hop size of 1 second. Zero-valued cells are depicted in white.],
) <fig:interpass>
== Downsampling and use of the mel transform

DJ mixes are typically lengthy, ranging from 30 minutes to several hours, and as they consist of musical signals, their frequency bandwidth is notably extensive. When employing the Short-Time Fourier Transform (STFT) with standard hop durations and a typical number of frequency bins for musical signals, the resulting feature matrix can become exceedingly large. This leads to substantial memory usage and elevated resource consumption.

In order to mitigate these issues, we have opted to use relatively large hop durations. This approach not only reduces the computational load but also offers an additional advantage: longer hop durations are better adapted to the temporal structures inherent in music. The hop duration however is not fixed, as is explained in @sec:multi-pass.

Additionally, we compress the frequency information using the mel-scale transform @stevensScaleMeasurementPsychological1937. This transform groups nearby frequencies into bins based on a perceptual model of human hearing, which is particularly well-suited for processing musical signals. Importantly, this transform has no effect on the ideal kernel and our estimators.


Let $bold(M)$ be a matrix of mel filterbank coefficients. The mel-spectrograms are calculated from the regular spectrograms: $XX^"mel" = bold(M)XX$ and $YY^"mel" = bold(M)YY$. Then we have:
$
  YY^"mel"_(m tau) &= sum_i bold(M)_(m i) YY_(i tau) \
  &= g[tau]^2 sum_i bold(M)_(m i) XX_(i,f[tau]) \
  &= g[tau]^2 XX^"mel"_(m, f[tau])
$

So the ideal kernel $HH^"ideal"$ is still clearly a solution of @matmul.


== Analysis window overlap

A key parameter when working with spectrograms is the overlap factor of the analysis windows. In order to emphasize the temporal continuity of the musical signals, we use high overlap factors: our experiments have shown that a window size of 6 to 8 times the hop size give the best results. It has experimentally shown to be highly effective in reducing indeterminacies, but tends to smooth out the results as a side-effect, which can potentially obscure finer details in the signal, as shown in @fig:overlap.

Typically, such large window sizes would result in a substantial increase in the number of frequency bins, leading to higher computational demands. However, by applying the mel-scale transform, we effectively mitigate this issue.

#figure(
  grid(
    columns: 3,
    image("overlap-1.svg"), image("overlap-8.svg"), image("overlap-16.svg"),
  ),
  caption: [Comparison of the influence of different overlap factors. Top row: estimated activation matrices. Bottom row: zooms on the middle activation. ],
) <fig:overlap>

== Spectrogram normalization

To improve the numeric stability of the NMF, the columns of $XX$ are typically normalized to sum to 1. We also normalize $YY$ by a single factor#footnote[Normalizing by column as for $XX$ would cancel out the gain information in $HH$.]. We show that this results in a simple scaling factor for $HH$ and therefore for the estimators.

Let the scaling factors $bold(k)_t eq.def sum_i XX_(i t)$ and $kappa eq.def sum_i sum_t YY_(i t)$.

The normalized spectrograms are:
$ XX^"norm"_(m t) eq.def XX_(m t) / bold(k)_t $

$ YY^"norm" eq.def YY / kappa $


Using @xy-relation:
$
  YY^"norm"_(m tau) = bold(k)_t / kappa g[tau]^2 XX^"norm"_(m, f[tau])
$

We can then deduce the ideal normalized kernel $HH^"norm"$ as a solution to @matmul:

$
  HH^"norm"_(t tau) &eq.def bold(k)_t / kappa g[tau]^2 delta_(t,f[tau]) \
  &= bold(k)_t / kappa HH^"ideal"_(t tau)
$

== Thresholding of low-power frames

Recorded music tracks often contain moments of silence or faint noise at the beginning and end of the signal. It is also quite common to find fade-out endings at the end of tracks, or reverberation tails. These elements appear as low-power frames in the feature matrices. Reverberation tails, in particular, are problematic due to their spectral similarity to the rest of the track, which can introduce indeterminacies in the analysis.

Additionally, low-power frames are unlikely to be present in a DJ mix, as DJs typically focus on playing the most musically significant portions of tracks, omitting the very beginning and end.

To address this, we detect and mark low-power frames in the input track spectrograms as unused and set the corresponding cells in the activation matrix to zero, effectively preventing these frames from contributing to the optimization process.


== Implementation

The algorithm has been implemented in Python. By leveraging the pytorch#footnote(link("https://pytorch.org")) framework, the optimization process can run on both CPU and GPU and benefit from parallel matrix multiplications on the latter.

We summarize the tunable parameters in @table:hyperparams along with their typical values. These can be further adjusted with prior knowledge of the mixes' characteristics.

#figure(
  table(
    columns: 4,
    [*Name*], [*Description*], [*Unit*], [*Typical value*],
    [`FS`], [Sampling rate], [Hz], [22050],
    [`HOP_SIZES`], [Decreasing list of hop durations], [s.], [`[20, 10, 2, 0.5, 0.1]`],
    [`OVERLAP`], [STFT analysis window overlap factor], [-], [6 to 8],
    [`NMELS`], [Number of mel bands], [-], [64 to 256],
    [`SPEC_POWER`], [STFT power], [-], [2],
    [`DIVERGENCE`], [Divergence function], [-], [$cal(D)_beta$ with $beta=0$],
    [`LOW_POWER_THRESHOLD`], [Threshold under which frames from input tracks are discarded], [dB], [-40],
    [`CARVE_THRESHOLD`], [Threshold under which a cell of the activation matrix is deemed zero], [dB], [-120],
    [`CARVE_BLUR_SIZE`], [Size of the gaussian blur kernel], [-], [3],
    [`CARVE_MIN_DURATION`], [Minimum line duration for morphological filtering], [s.], [10],
    [`CARVE_MAX_SLOPE`], [Maximum allowed deviation from original playing speed], [#sym.plus.minus %], [10 to 50],
    [`NOISE_DIM`], [Number of columns for noise estimation], [-], [0 to 50],
  ),
  caption: [Summary of tunable parameters],
) <table:hyperparams>

== Example results

We run the algorithm on several constructed mixes, that showcase mix scenarios with the typical transformations that we aim to be able to reverse engineer. All figures depict:
- On the left, the final estimated activation matrix, with zero-valued cells in gray;
- In the middle, the estimated gain (dots) and ground truth (dashed)
- On the right, the estimated warp function (dots) and ground truth (dashed).

Firstly, we focus on time transformations using "mixes" composed of a single track. The @fig:jumploop presents a single track that has been sliced and looped, and the @fig:timestretch corresponds to a single track that has been time-stretched by the use of the Rubberband#footnote(link("https://breakfastquay.com/rubberband/")) library. Although presenting some artifacts, especially in the gain estimation for the timestretched track, we can see that the time-warping function is estimated accurately.

#figure(
  image("jumploop.svg", width: 140%),
  caption: [Results for a sliced track with loops and jumps.],
) <fig:jumploop>

#figure(
  image("timestretch.svg", width: 140%),
  caption: [Results for a time-stretched track.],
) <fig:timestretch>


Secondly, we focus on overlapping tracks. The @fig:linear-mix depicts a simple mix of two tracks with a large transition region composed of a crossfade. Despite having no kwowledge of the form of cross-fading used, the method correctly estimates the gain for both tracks, and mostly correctly estimates the warp function.

It should be noted that the figure depict the raw warp function estimations, which is computed for all tracks for the whole duration of the mix. Hence, the points outside of the tracks' playing range are effectively noise and could be filtered, for example, using a simple threshold on the gain estimation.

#figure(
  image("linear-mix.svg", width: 140%),
  caption: [Beat-synchronous mix of two tracks with linear fades.],
) <fig:linear-mix>
