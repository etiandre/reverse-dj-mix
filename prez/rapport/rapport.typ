#import "@preview/lovelace:0.3.0": *
#import "@preview/ctheorems:1.1.2": *
#import "@preview/glossarium:0.4.1": make-glossary, print-glossary, gls, glspl
#show: make-glossary

#let title = "DJ mix reverse engineering using multi-pass non-negative matrix factorization"
#set page(margin: 3.3cm)
#set page(header: context {
  if counter(page).get().first() > 3 [
    #align(center)[
      #smallcaps(title)
    ]
  ]
})

#set page(footer: context {
  if counter(page).get().first() >= 6 [
    #align(center)[
      #counter(page).display(
        "1/1",
        both: true,
      )
    ]
  ]
})
#set par(leading: 0.65em, first-line-indent: 1em)
#show par: set block(spacing: 1em)
#show heading: set block(above: 2em, below: 1.3em)
// #set text(lang: "en", font: "New Computer Modern")
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

  #set text(size: 18pt, hyphenate: false)
  #set align(center)
  #block(smallcaps(it))
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
    box(image("IRCAM.CP.jpg", width: 50%)),
    box(image("LOGO_SU_HORIZ_SEUL.jpg", width: 50%)),
    box(image("Logo_Télécom_Paris.jpg", width: 30%)),
  )

  #pagebreak()

  #v(1fr)
  #par(justify: false)[
    *Abstract* \
    Disc jockeys (DJs) create mixes by creatively combining existing music tracks, while applying various transformations such as time-warping and audio effects. DJ-mix reverse engineering involves computationally analyzing these mixes to extract the parameters used in their creation. Previous approaches have treated this process as two separate tasks: alignment and unmixing, which are difficult to generalize to the wide range of transformations that DJs employ. This report introduces an integrated approach for both tasks using a multi-pass Non-negative Matrix Factorization (NMF) algorithm that is able to extract arbitrary time-warping transformations and the mixing gains while being robust to noise. The method's effectiveness is evaluated both qualitatively, through representative examples, and quantitatively, using a publicly available dataset. Additionally, the report explores the challenges of developing suitable datasets for DJ mix reverse engineering, proposing potential approaches for dataset creation by ground truth measurement.
  ]
  #v(1fr)
  #par(justify: false)[
    *Résumé* \
    Les disc-jockeys (DJ) créent des mixes en combinant des pistes musicales de manière créative, en appliquant diverses transformations telles que la manipulation temporelle ou des effets audio. La rétro-ingénierie de mix DJ consiste en l'analyse computationelle de ceux-ci afin d'en extraire les paramètres utilisés lors de leur création. Les approches antérieures ont traité ce processus comme deux tâches distinctes : l'alignement et le démixage, qui sont difficiles à généraliser à la large gamme de transformations employées par les DJ. Ce rapport introduit une approche intégrée pour les deux tâches en utilisant un algorithme de factorisation en matrices non-négatives (NMF) multi-passe capable d'extraire des manipulations temporelles arbitraires et le gain de mixage tout en étant robuste au bruit. L'efficacité de la méthode est évaluée à la fois qualitativement, à l'aide d'exemples représentatifs, et quantitativement, à l'aide d'un ensemble de données accessibles au public. En outre, le rapport explore les défis posés par le développement de jeux de données appropriés pour la rétro-ingénierie de mix DJ, en proposant de nouvelles méthodes pour la création de jeux de données par mesure de vérité terrain.
  ]
  #v(1fr)
]

#pagebreak()

#align(right)[
  #emph[
    I would like to express my gratitude to: \
    Diemo and Dominique, for their continued mentorship and support; \
    Rémi, Roland and Geoffroy for their help and valuable insights; \
    Luis for their enthusiasm and keenness to share; \
    The band of merry IRCAM interns for making office days enjoyable.
  ]
]

//////
#show outline.entry.where(level: 1): it => {
  v(12pt, weak: true)
  strong(it)
}
#outline(indent: auto)
///////
#set par(justify: true)
#show link: set text(fill: blue.darken(60%))

//////////////////////////////////////////
#let VV = math.bold(math.upright([V]))
#let HH = math.bold(math.upright([H]))
#let WW = math.bold(math.upright([W]))
#let XX = math.bold(math.upright([X]))
#let YY = math.bold(math.upright([Y]))

//////////////////////////////////////////

#heading(numbering: none)[Glossary and mathematical conventions]
/ Track: A recorded piece of music or song
/ Mix: A creative combination of overlapping consecutive tracks, potentially transformed using various effects and techniques
/ Transition: The overlapping segment where a track finishes and the next one starts playing
/ Fade-in, fade-out: A way to introduce and de introduce tracks using a continuous gain increase or decrease
/ Cross-fade: A transition made of the simultaneous use of a fade-out and a fade-in
/ DJ: Disc-Jockey
/ DJ Deck: A standalone unit that plays recorded media
/ DJ Mixer: A specialized mixing console for DJs
/ Cue point: the point of a track at which it is introduced in a mix
/ Time-warping, warping: time transformation of an audio signal
  / Resampling: time-warping with transposition, akin to speeding a vinyl record up or down
  / Time-stretching: time-warping without transposition
/ DTW: Dynamic Time Warping
/ NMF: Non-negative Matrix Factorization
/ STFT: Short-Time Fourier Transform
/ DJ-MIR: DJ Music Information Retrieval
/ DAW: Digital Audio Workstation
/ EQ: Equalizer
/ SIFT: Scale Invariant Feature Transform
  / Spectrogram: squared modulus of the STFT
  / Hop size: duration between two consecutive frames of the spectrogram
  / Window size: duration of the time segment used for a frame of the spectrogram
  / Overlap factor: ratio between the hop size and the window size

#v(5em)

#align(
  center,
  table(
    columns: 2,
    [$t in [1..T]$], [Discrete time step in track context],
    [$tau in [1..K]$], [Discrete time step in mix context],
    [$dot.circle$], [Hadamard (term-wise) matrix product],
    [$delta_(a,b)$], [Kronecker delta function: $delta_(a,b) = cases(1 "if" a = b, 0 "otherwise")$],
    [$VV$], [Mix spectrogram matrix],
    [$WW$], [Track spectrogram matrix],
    [$HH$], [Activation matrix],
    [$g$], [Mixing gain],
    [$f$], [Time-warping function],
  ),
)

= Introduction
== The art of DJing

For decades, DJs (_Disc-Jockeys_) and the practice of DJing have played an integral role in shaping our musical landscape. DJing can be interpreted in various ways, this thesis defines it as the continuous playback of recorded media (_tracks_) to an audience—whether live, via radio or other broadcast media, or through recorded formats like _mixtapes_ or streaming services.

A DJ's performance, often referred to as a _mix_, involves the careful selection and sequential playback of music tracks. However, DJing is not merely a passive process; it is a transformative art. DJs often overlap tracks, either to create entirely new musical compositions (_mashups_) or more commonly, to maintain a seamless flow between songs. The overlapping segments are known as _transitions_, with the simplest form being a cross-fade.

In the realm of dance music, these transitions are crafted to be as seamless as possible to maintain the energy and keep the audience dancing. DJs employ various techniques to achieve this, such as synchronizing the tempo and downbeats of overlapping tracks, manipulating EQ settings to highlight or diminish certain elements, and applying effects like reverberation and delay.

Traditional vinyl DJs are typically limited to selecting _cue points_ — the specific time offset at which a track is introduced — and adjusting the playback speed, sometimes in extreme ways, as in the case of scratching. In contrast, modern digital DJ equipment offers a broader array of transformative tools, including transposition, time-warping, looping, and jumping between different sections of a track.

Moreover, DJ sets may include additional elements such as spoken word or sung vocals, rhythm machines, and sound effects like sirens or jingles, further enriching the auditory experience.

== Prior art on DJ Music Information Retrieval

The pervasive influence of DJing in popular culture has made it a subject of interest in academic research. Indeed, a deeper understanding of DJ practices could significantly contribute to various fields, including musicology, cultural studies, music technology, and the automation of DJ techniques for both entertainment and commercial applications.

In particular, the field of DJ-MIR (DJ Music Information Retrieval) seeks to develop methods to extract metadata from recorded DJ mixes. This field has traditionally been divided into several key subtasks:

/ Identification: Retrieving the playlist used to create the mix, as explored in @sixOlafLightweightPortable2023 @sonnleitnerLandmarkBasedAudioFingerprinting2016 @wangIndustrialStrengthAudioSearch;
/ Alignment: Determining the start and end times of each track within the mix, along with the time-scaling factors, as in @ewertHighResolutionAudio2009 @ramonaAutomaticAlignmentAudio2011 @werthen-brabantsGroundTruthExtraction2018 @kimComputationalAnalysisRealWorld2020 @schwarzMethodsDatasetsDJMix2021 @yangAligningUnsynchronizedPart2021 @sixDiscStitchAudiotoaudioAlignment2022;
/ Unmixing: Estimating and reversing the fade curves @ramonaSimpleEfficientFader2011 @werthen-brabantsGroundTruthExtraction2018 @schwarzExtractionGroundTruth @kimJointEstimationFader2022, the EQ parameters @kimJointEstimationFader2022, and any additional effects @barchiesiReverseEngineeringMix2010;
/ Content analysis: Deriving metadata such as genre and indicators of various social and musical aspects of the mix, such as cue points @zehrenAutomaticDetectionCue2020, @schwarzHeuristicAlgorithmDJ2018 or drops @yadatiDetectingDropsElectronic2014 @kimAutomaticDjMix2017.
/ Mix generation: Automatically generating realistic mixes from a given playlist, as addressed in @bittnerAutomaticPlaylistSequencing2017 @cliffHangDJAutomatic2000 @fujioSystemMixingSongs2003 @kimAutomaticDjMix2017 @chenAutomaticDJTransitions2022 @liLooPyResearchFriendlyMix2023

== Problem overview of DJ-mix reverse engineering

Within these subtasks, we focus our interest on the task of _DJ-mix reverse engineering_, as defined by #cite(<schwarzMethodsDatasetsDJMix2021>, form: "prose"). This term encompasses the identification, alignment and unmixing subtasks described above. Given that identification is a well-explored problem space, we further narrow our focus on the alignment on unmixing tasks.

More formally, our goal is as follows:

*Given*:
- a recording of a DJ mix;
- the recordings of all the mix's constituting tracks;

*Estimate*:
- any temporal transformations applied to the tracks (cue points, play duration, time stretching, loops, jumps);
- the evolution of the mixing gain of each track
- any effects applied to the tracks/and or mix (distortion, compression, pitch-shifting, filtering, EQing...)

Previous research has typically treated alignment and unmixing as sequential, independent tasks. Alignment is generally accomplished through feature extraction or fingerprinting combined with dynamic time warping (DTW). Subsequently, time-scaling factors are estimated, transition regions are segmented, and the unmixing process follows.

This approach has shown to be effective for broadcast mixes and simpler DJ sets, where tracks are often assumed to played monotonously (i.e. without jumps and loops) and at a fixed speed to mitigate alignment issues in transition regions. However, it is precisely in these transition regions that DJs are likely to employ the most creative transformations and effects.

Therefore, we propose an integrated approach that treats alignment and unmixing as a single, conjoint tanscription task, enabling a more general understanding of the complexities inherent in DJ mixes.

= Datasets for DJ-mix reverse engineering

In order to evaluate their methods, researchers are in the need of appropriate datasets. Despite the large amount of available mixes, thanks to decades of recorded mixes and specialized streaming services, they are rarely sufficiently and correctly annotated.

Appropriate datasets for DJ mix reverse engineering typically consist of mixes paired with their constituent tracks, accompanied by their respective time positions within the mix. Depending on the available annotations, these datasets may also include detailed information about the manipulations performed by the DJ during the creation of the mix, such as the volume levels for each track, time-stretching and transposition factors, the types and parameters of effects applied, and more. We operate under the assumption that all the transformations applied to the original tracks, if known, enable to reconstruct the resulting mix signal accurately.

== Existing datasets

We identify two broad categories in which existing datasets can be classified:

/ Synthetic: These consist of mixes generated according to a set of predefined rules.
/ Real: These are derived from real-world mixes and are annotated either manually or automatically.

=== Synthetic datasets

Synthetic datasets offer the distinct advantage of precision and completeness, as the parameters used in the creation of the mixes are explicitly known and controlled. This allows for a high level of accuracy in analyses, as every aspect of the mix, from track selection to the application of effects, is documented by design. However, because these datasets are generated according to predefined rules, they may lack the diversity and complexity found in real-world DJ sets. This limitation can affect their ecological validity, meaning their ability to generalize to real-world scenarios may be compromised.

In the context of DJ-mix reverse engineering research, UnmixDB @schwarzUnmixDBDatasetDJMix2018, which is based on the `mixotic` dataset from #cite(<sonnleitnerLandmarkBasedAudioFingerprinting2016>, form:"prose")), and the dataset from #cite(<werthen-brabantsGroundTruthExtraction2018>, form:"prose"), are among the few accessible examples of synthetic datasets. These datasets are derived from copyleft tracks released on the Mixotic netlabel #footnote(link("https://www.mixotic.net")), which primarily features electronic music. As a result, the datasets may lack stylistic diversity, potentially limiting their applicability to broader DJ-MIR research that encompasses a wider range of musical genres and DJ practices. However, because UnmixDB is easily accessible and features a large amount of data, we used it for evaluation of our method.

=== Real datasets

Datasets derived from real DJ sets are valuable because they accurately reflect actual DJing practices, both past and present. Unlike synthetic datasets, they do not suffer from issues stemming from artificial generation, making them more representative of real-world scenarios. However, the use of real DJ sets presents several challenges. One of the primary obstacles is the need for time-intensive manual annotation or reliance on automatic annotation methods, which can be prone to errors.

The act of identifying a track from a DJ mix, known as colloquially as a _track ID_, is a recurring topic of interest within DJ culture, engaging both DJs and their audiences. This has led to the emergence of online platforms that host crowdsourced or automatically detected track IDs, such as 1001tracklists#footnote(link("https://www.1001tracklists.com")), CueNation#footnote(link("https://cuenation.com/")), LiveTracklist#footnote(link("https://www.livetracklist.com/")), mixes.wiki (formerly known as MixesDB)#footnote(link("https://www.mixes.wiki/w/Main_Page")) and TrackId.net#footnote(link("https://trackid.net/")).

These platforms collectively provide a vast amount of data, which can potentially be leveraged for data-mining activities and may be particularly useful for the content analysis aspect of DJ-MIR. However, the information available on these sites typically includes only track identification and approximate track positions within the mix. Furthermore, the audio content is often copyrighted, and the data itself may be subject to legal restrictions, which complicates its use in research. As a result, researchers must independently obtain the associated audio content, adding another layer of complexity to the process.

Nonetheless, this is how #cite(<kimComputationalAnalysisRealWorld2020>, form:"prose") and #cite(<kimJointEstimationFader2022>, form:"prose"); obtained large-scale datasets for evaluation of their methods, with the latter being available, albeit without audio data: the commercial nature of the mixes and the tracks within them raises legal concerns, particularly regarding their use in an academic setting. To circumvent this, audio data can be downloaded from streaming services, but then its random availability becomes a concern for study repetability.

Other noteworthy datasets are the one from #cite(<scarfeLongRangeSelfsimilarityApproach2013>, form:"prose"), which has been obtained from radio show archives, and the `disco` dataset from #cite(<sonnleitnerLandmarkBasedAudioFingerprinting2016>, form:"prose") which features measured bespoke mixes. Both of these are not publicly available.

During the course of the internship, we obtained an extract of the TrackId.net database with the assistance of its owners. However, we found that this dataset was of limited interest for our purposes, but that it could be valuable for future research. A more detailed analysis of this dataset is provided in @sec:trackidnet.

== Approaches for dataset creation

An ideal dataset for DJ-mix reverse engineering would consist of real-world DJ sets that are both diverse in the styles represented and annotated with precise and comprehensive ground truth data. Achieving such a dataset is challenging, as current automatic annotation methods fall short in providing the necessary accuracy and completeness.

With this in mind, we suggest that ground truth may be measured during the DJ mixing process. We detail some approaches that could be leveraged to this extent.

=== Optical tracking of vinyl records

We implemented an optical tracking method to measure the angular velocity of a vinyl record during playback, enabling the extraction of the time-warping ground truth from a filmed DJ performance. The method can be especially useful given the vast amount of videos of DJ performances online. By tracking a reference picture of a vinyl record to each frame of a video of a DJ turntable, the record's rotation speed can be estimated.

*Tracking:* The tracking is based on the scale-invariant feature transform (SIFT) algorithm @loweDistinctiveImageFeatures2004. The algorithm extracts keypoints from a reference image of the vinyl's label (@fig:vinyl-ref) and from each frame of a video of the vinyl being played. By matching these keypoints, a homography matrix is computed for each frame. This matrix represents the geometric transformation between the reference and the current frame, capturing the rotation, translation, and scale changes (@fig:vinyl-track). The sequence of homography matrices obtained across the frames is stored for further analysis.

#figure(
  image("vinyl-ref.jpg"),
  caption: [Reference image of the vinyl label.],
) <fig:vinyl-ref>

#figure(
  grid(
    image("vinyl-1.jpg", width: 100%),
    image("vinyl-2.jpg", width: 100%),
    image("vinyl-3.jpg", width: 100%),
    image("vinyl-4.jpg", width: 100%),
    columns: 4),
  caption: [Frames from the vinyl video. Matches' bounding boxes are shown as white rectangles.],
) <fig:vinyl-track>

*Rotation computation:* The stored homography matrices are then decomposed to extract the rotation angles of the vinyl record using the singular value decomposition method from #cite(<faugerasMOTIONSTRUCTUREMOTION1988>, form: "prose"). The rotation on the $z$ axis corresponds to the rotation of the vinyl record. The rotation angle sequence is unwrapped then derived to obtain the rotation speed. Then, with knowledge of the nominal speed of the record (e.g. 33 or 45 rpm), the time-warping function can be calculated. Example results are illustrated in @fig:vinyl-results.


#figure(
  image("rotato.svg"),
  caption: [Computed angle and rotation speed of the vinyl. In the experiment, the record was played at 33 rpm for the first 30 seconds, then was "scratched" by hand for the remaining of the experiment.],
) <fig:vinyl-results>

The accuracy of the extracted rotational speed is directly dependent on the temporal resolution of the video (usually 30 or 60 images per second), which limits the granularity of the measurements. Additionally, the method is susceptible to noise introduced by the tracking process, which can affect the precision of the rotation and speed calculations.

Similar object tracking techniques could be used to extract ground truth data from any visible physical control of the DJ equipment.

=== Metadata extraction from Digital Audio Workstation project files

DJ mixes are not only created in live settings but can also be constructed in studio environments using Digital Audio Workstations (DAWs). In such cases, an audio engineer can replicate a DJ performance offline by leveraging the DAW’s audio editing and automation features. The resulting DAW project files then contain the reference tracks along with all the ground truth data required to generate the final mix audio file. This data can be programmatically extracted and processed to suit the needs of DJ-MIR research.

This approach has been notably previously explored by #cite(<werthen-brabantsGroundTruthExtraction2018>, form: "prose") for the creation of their dataset. Their tool, the _Ableton Live Mix Extractor_#footnote(link("https://github.com/werthen/ableton-extraction-tool")), is designed to extract ground truth data from _Ableton Live_ project files, focusing on scenarios with fixed tempo, fixed stretch factors, and simple crossfades. But the _Ableton Live_ DAW supports many additional features that could be used to further imitate real DJ mixes, namely:
- Non-constant tempo curves;
- Non-constant time-warping;
- Complex effect chains with evolving parameters.

We have developed an enhanced version of this extraction tool. Our version supports the extraction of all aforementioned parameters and computes high-level ground truth data relevant to DJ-mix reverse engineering tasks. As a case study, we produced an example mix, illustrated in @fig:ableton-project. The mix features:
- Two tracks ("A" and "B") mixed together;
- Complex gain automation on both tracks, including:
  - Fade-in and fade-outs on both "A" and "B" (depicted with darker, curved backgrounds);
  - Multiple gain automations on "A" at both the mixer (first red line) and effect chain stages (second red line);
- Complex time manipulation:
  - Variable tempo curve (bottom red line);
  - Non-constant clip warping on "A" (yellow warp markers).

#figure(
  grid(
    image("ableton-project.png", width: 80%),
    image("ableton-warp.png", width: 80%),
    columns: 1,
    gutter: 1em
  ),
  caption: [Example mix created in Ableton Live. \ Top: session view with clips and automations. \ Bottom: warp markers of the first clip.],
) <fig:ableton-project>

The extracted gain and time-warp data are illustrated in @fig:ableton-results, demonstrating that all the applied transformations have been captured#footnote[The events in @fig:ableton-project do not visually align horizontally with @fig:ableton-results, as the time axis in the former is non-linear.].

Due to time constraints, our tool remains in a proof-of-concept stage, as its correctness has not been thouroughly evaluated.

#figure(
  image("ableton-results.svg"),
  caption: [Extracted ground truth data from the example mix.],
) <fig:ableton-results>

=== Instrumentation of DJ mixing software

Building on the approach outlined in the previous section, specialized DJ mixing software could similarly be exploited for dataset creation. DJ software is typically modeled after traditional DJ hardware, aiming to replicate the performance capabilities of physical equipment in a digital environment. Since all audio processing occurs within the software, it is theoretically possible to instrument the software to record ground truth data as a DJ mix is performed.

We identified _Mixxx_#footnote(link("https://mixxx.org")) as a suitable candidate for this approach due to its open-source nature, which allows for modification. By patching the software to include data-logging capabilities, it would be possible to simultaneously capture the mix, reference tracks, and associated ground truth data.

While we were unable to pursue this concept due to time and resource constraints, we believe that it holds significant potential for creating a real-world dataset with precise ground truth.

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

Assuming the $M$ constituting tracks of the mix are known, let $forall i in [1...M]$:
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
  gradient_bold(bold(theta)) cal(D)_beta (VV | WW HH) = gradient_bold(bold(theta))^+ cal(D)_beta (
    VV | WW HH
  ) - gradient_bold(bold(theta))^- cal(D)_beta (VV | WW HH)
$ <eq:gradient-separation>

Using the notation trick described in @fevotteAlgorithmsNonnegativeMatrix2011, the so-called _multiplicative update_ rules can be obtained#footnote[$dot.circle$ and $.../...$ stand respectively for Hadamard's (element-wise) product and division.]:
$
  bold(theta) <- bold(theta) dot.circle (gradient_bold(bold(theta))^- cal(D)_beta (
    VV | WW HH
  )) / (gradient_bold(bold(theta))^+ cal(D)_beta (VV | WW HH))
$ <eq:mu-gradient>

With the beta-divergence, this yields the update rules of @algo:mu-betadiv which can be efficiently implemented, with strong monotonicity guarantees when $beta in [0,2]$.

#figure(
  kind: "algorithm",
  supplement: [Algorithm],

  pseudocode-list(numbered-title: [NMF Algorithm with Multiplicative Updates], line-gap: 1.4em)[
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
$ tilde(f) [tau] = "argmax"_(t in [1...T]) HH_(t tau) $ <time_estimator_argmax>

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

#proof[
  Let $bold(M)$ be a matrix of mel filterbank coefficients. The mel-spectrograms are calculated from the regular spectrograms: $XX^"mel" = bold(M)XX$ and $YY^"mel" = bold(M)YY$. Then we have:
  $
    YY^"mel"_(m tau) &= sum_i bold(M)_(m i) YY_(i tau) \
    &= g[tau]^2 sum_i bold(M)_(m i) XX_(i,f[tau]) \
    &= g[tau]^2 XX^"mel"_(m, f[tau])
  $

  So the ideal kernel $HH^"ideal"$ is still clearly a solution of @matmul.
]

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

#proof[
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
]

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
    table.header[*Name*][*Description*][*Unit*][*Typical value*],
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

== Evaluation on UnmixDB

We applied our algorithm to the UnmixDB dataset version 1.1#footnote(link("https://github.com/Ircam-RnD/unmixdb-creation")), which includes excerpts from open-licensed dance tracks and their corresponding automatically generated mixes.

Each mix consists of three track excerpts, mixed in a beat-aligned manner with linear crossfades to simulate a realistic DJ context. Tracks are mixed with different combinations of audio effects (none, bass boost, compression, distortion) and time-scaling methods (none, resampling, time-stretching). The dataset provides complete ground truth for all mixes, and includes Python code for generating similar datasets. For further details, see #cite(<schwarzUnmixDBDatasetDJMix2018>, form: "prose").

The computation took about two hours on an Intel Xeon E5-2630 CPU with 12 threads. The tunable parameters used are summarized in @table:unmixdb-params.

#figure(
  table(
    columns: 2,
    table.header[*Name*][*Value*],
    [`FS`], [ 22050 Hz],
    [`HOP_SIZES`], [ \[ 4, 2, 1, 0.5 \] seconds],
    [`OVERLAP`], [ 8],
    [`NMELS`], [ 128],
    [`SPEC_POWER`], [ 2],
    [`DIVERGENCE`], [ $cal(D)_beta$ with $beta=0$],
    [`LOW_POWER_THRESHOLD`], [-40 dB],
    [`CARVE_THRESHOLD`], [-120 dB],
    [`CARVE_BLUR_SIZE`], [ 3],
    [`CARVE_MIN_DURATION`], [ 10 seconds],
    [`CARVE_MAX_SLOPE`], [ 1.5],
    [`NOISE_DIM`], [ 15],
  ),
  caption: [Tunable parameters for UnmixDB evaluation.],
) <table:unmixdb-params>

We define the followig evaluation metrics:
/ Gain error: mean absolute error between the estimated gain and ground truth gain: $1/K sum_(tau=1)^K abs(tilde(g)[tau] - g[tau])$
/ Warp error: mean absolute error in seconds between the estimated and ground truth warp: $1/K sum_(tau=1)^K abs(tilde(f)[tau] - f[tau])$

The mixes in UnmixDB are generated with fixed time-scale factors. To obtain comparable metrics to #cite(<schwarzMethodsDatasetsDJMix2021>, form: "prose"), we estimate the speed factor and the cue point by linear regression over the warp sequence, and define the following metrics:

/ Speed ratio: mean ratio between the estimated and the ground truth speed factors.
/ Cue point error: mean absolute error in seconds between the estimated and the ground truth cue points.

Box plots of these metrics are represented respectively in @fig:unmixdb-gain, @fig:unmixdb-warp, @fig:unmixdb-cue and @fig:unmixdb-speed.

These results demonstrate the validity of our method, particularly regarding the variants with time-stretching and without. The performance on resampled mixes, which feature transposition, is poorer. However, given that our mixing model doesn't include pitch-shifting in its assumptions, we find it is still acceptable performance.

The performance of cue point estimation is comparable to #cite(<schwarzMethodsDatasetsDJMix2021>, form:"prose"), but the same cannot be said of the speed ratio estimation. We explain this by the more lax assumptions of our approach regarding time-warping.

#figure(
  image("../../nmf/results-plots/best_gain_err.svg"),
  caption: [Box plot of the gain error per variant.],
) <fig:unmixdb-gain>
#figure(
  image("../../nmf/results-plots/best_warp_err.svg"),
  caption: [Box plot of the warp error per variant.],
) <fig:unmixdb-warp>
#figure(
  image("../../nmf/results-plots/best_track_start_err.svg"),
  caption: [Box plot of the cue point error per variant.],
) <fig:unmixdb-cue>
#figure(
  image("../../nmf/results-plots/best_speed_ratio.svg"),
  caption: [Box plot of the speed ratio per variant.],
) <fig:unmixdb-speed>


=== Impact of noise estimation

We conduct an additional experiment on UnmixDB to evaluate the effectiveness of the additive noise estimation. We compare in @fig:noise-gain and @fig:noise-warp two runs of the algorithm without noise (in blue) and with noise estimation (in orange). The results show that in the case of added audio effects, adding noise estimation to the optimization algorithm improves the estimation. The "dist" and especially the "bass" variants benefit the most.

#figure(
  image("../../nmf/results-plots/noise_gain_err.svg"),
  caption: [Box plot of the gain error per variant],
) <fig:noise-gain>
#figure(
  image("../../nmf/results-plots/noise_warp_err.svg"),
  caption: [Box plot of the warp error per variant],
) <fig:noise-warp>

= Conclusion

This internship report has discussed existing methods and datasets for DJ mix reverse engineering. The need for datasets with precise and complete ground truth annotations has been highlighted, emphasizing their importance for advancing research in this area, and explored potential methodologies for their creation.

In response to the challenges presented by DJ mixes with complex time-warping transformations, and to address limitations of previous work in this regard, a new integrated approach was proposed. This approach involves the use of Non-negative Matrix Factorization (NMF) with a multi-pass extension, supported by a mixing model grounded in the technical principles of DJ hardware. We demonstrated the effectiveness of the method on arbitrary time-warping transformations. While the results obtained in quantitative evaluation did not match the precision of previous methods, the proposed approach demonstrated potential in capturing a broader spectrum of DJ practices, offering a foundation for further refinement and exploration in future research.

///////////////////////////// BIBLIO

#bibliography("../../zotero.bib")

///////////////////////////// APPPENDIX


#show: appendix

= The TrackId.net dataset <sec:trackidnet>

Trackid.net#footnote(link("https://trackid.net")) is an automated track identification service, presented as a freely accessible website, which features a collection of mixes along with their associated playlists. Users request track identification by submitting a link to a mix from streaming services, and the website uses a fingerprinting method to identify the tracks played. Registered users can also amend the tracklist to correct identification errors or manually add tracks.

During our internship, we contacted the website's owners, who kindly provided a snapshot of their database. As of April 10, 2024, this snapshot contained metadata for 136,231 mixes for a cumulative duration of 7,896 days of audio, and 666,625 unique tracks.

== Dataset contents
For each mix, the available metadata includes:
- The title, duration, and source of the mix;
- The tracklist with titles, artists, and labels (when available);
- The start and end times for each track;
- The inferred style of the mix.

Notably, the database does not include any audio data. While the mixes' audio can be retrieved via the provided streaming service links (assuming they are still available), obtaining the associated tracks' audio is more challenging. Only fuzzy matching of the tracks' title, artist, and label is possible, making it difficult to ensure the retrieved tracks correspond exactly to those referenced in the database.

Additionally, the vast majority of the audio content is protected under copyright law, which presents challenges for its use in scientific research.

== Analysis

=== General observations

Given the substantial amount of mixes included, the dataset offers insights about the current landscape of recorded DJ mixes.

As shown in @fig:trackidnet-duration, most mixes range from 30 minutes to 2 hours in length, with an average duration of 83 minutes. Some mixes, however, are considerably longer. It is also noteworthy that the duration of many mixes is a multiple of 30 minutes.

#figure(
  image("../../datasets/trackidnet/trackidnet-duration.svg"),
  caption: [Distribution of the mixes' duration.],
) <fig:trackidnet-duration>

The distribution of the number of tracks per mix follows a similar pattern (@fig:trackidnet-trackspermix), with an average of 12.5 tracks per mix.

#figure(
  image("../../datasets/trackidnet/trackidnet-trackspermix.svg"),
  caption: [Distribution of the number of unique tracks per mix.],
) <fig:trackidnet-trackspermix>

The distribution of _play duration_ of the tacks in mixes (@fig:trackidnet-trackduration), which is defined as the cumulated time spans between the start and end of identified tracks, is also expected, with an average of 3 minutes and 30 seconds.


#figure(
  image("../../datasets/trackidnet/trackidnet-trackduration.svg"),
  caption: [Distribution of the play duration of tracks in mixes.],
) <fig:trackidnet-trackduration>

=== Styles

According to discussions with the owners of Trackid.net, the style(s) of a mix are inferred from the styles of its constituent tracks. The primary source for track styles is the Discogs API#footnote(link("https://www.discogs.com/developers")).

The most frequent styles, as depicted in @fig:trackidnet-styles, are predominantly electronic, with House and Techno leading. Interestingly, "Experimental" and "Ambient" are also popular, which may be due to the liberal application of these categories on Discogs.

#figure(
  image("../../datasets/trackidnet/trackidnet-styles.svg"),
  caption: [Most prevalent styles. Mixes can have more than one style.],
) <fig:trackidnet-styles>

=== Quality of the annotations

To assess the quality of the metadata, we calculated the _gap duration_ between tracks, defined as the difference between the detected start of track $n$ and the detected end of track $n-1$. A positive difference indicates a gap between tracks, while a negative difference indicates overlapping tracks.

The distribution of gap durations (@fig:trackidnet-gapduration) suggests a significant number of incompletely annotated mixes. However, it is important to note that gaps may not always signify poor annotations. For example, radio mixes may correctly present gaps for segments where people are talking between tracks.

#figure(
  image("../../datasets/trackidnet/trackidnet-gapduration.svg"),
  caption: [Distribution of the durations of gaps between tracks.],
) <fig:trackidnet-gapduration>

This observation is further supported by examining the proportion of identified time per mix (@fig:trackidnet-identified), defined as the ratio between the cumulative identified track time and the total duration of the mix. Very few mixes are completely or nearly completely annotated, with the average identification proportion being around 50%.

#figure(
  image("../../datasets/trackidnet/trackidnet-identified.svg"),
  caption: [Distribution of the proportion of identified time per mix.],
) <fig:trackidnet-identified>

== Conclusion

The trackid.net dataset stands out due to:
- A variety of styles represented;
- A large number of entries, making it suitable for big data and machine learning applications;
- High ecological validity.

However, there are some challenges that make it less suitable for research in its current form:
- Numerous false positives and negatives from the fingerprinting process;
- Imprecise timestamps and gaps in the data;
- Over- or under-representation of certain styles;
- Uncertain availability of audio.

We believe that all of these challenges can be curbed by curating a smaller, higher quality dataset from this one, by exploiting the statistical indicators used in the previous section. In particular, mixes should be filtered to favorize:
- A more uniform and diverse representation of musical styles;
- High identification proportion;
- Small gap durations.

= Penalized NMF for DJ mix transcription <sec:penalty>

As stated in @sec:beta-nmf, the objective function optimized by the NMF algorithm can be modified to include penalizations. To illustrate, we will consider a new objective function $cal(C)$ involving the beta-divergence and an additional penalty function $cal(P)$ on $HH$ weighted by $lambda in RR^+$:

$
  cal(C) = cal(D)_beta (VV | WW HH) + lambda cal(P)(HH)
$

Assuming the gradients of $cal(C)$ w.r.t. $HH$ and $WW$ are separable into their positive and negative parts, new multiplicative update rules can be derived by following the procedure from @sec:beta-nmf. However, the monotonicity of the gradient descent is no longer guaranteed, and is dependent on the form of $cal(P)$ and the choice of $lambda$; which both can be validated through experimentation.

We present the following penalty functions to favorize certain characteristics of the activation matrix:

/ L1 regularization: encourages sparsity of the activation matrix.
/ Gain smoothness: discourages abrupt changes in gain.
/ Diagonal smoothness: favorizes diagonal activations.
/ Lineness: discourages multiple horizontal, vertical or diagonal activations.

However, during our experimentation, we observed that these modifications resulted in only marginal improvements in transcription quality. Nevertheless, for completeness and reference, the rationale behind each penalty function, along with their gradient derivations, is provided in the following sections.

== L1 (Lasso) regularization

To promote sparsity in the activation matrix, the L1 regularization penalty can be added to the objective function. This penalty function favors solutions with many zero entries in the activation matrix, effectively encouraging a sparse representation of the data.

The penalty function for L1 regularization is defined as:

$
  cal(P)_"lasso" (HH) = sum_(i=0)^(T-1) sum_(j=0)^(K-1) |HH_(i j)|
$

Given that $HH$ is positive:

$
  cal(P)_"lasso" (HH) = sum_(i=0)^(T-1) sum_(j=0)^(K-1) HH_(i j)
$

*Gradient calculation:* The partial derivative of the penalty function with respect to $HH_(i j)$ is given by:

$
  (partial cal(P)_"lasso") / (partial HH_(i j)) = 1
$

Thus:
$
  gradient_HH^+ cal(P)_"lasso" &= 1 \
  gradient_HH^- cal(P)_"lasso" &= 0
$

== Gain smoothness penalty

The DJ is expected to perform smooth transitions, so we assume that the gain applied to each track is varying relatively slowly. Consequently, we want to penalize abrupt changes in the gain. Using the gain estimator (@gain_estimator_sum), we have:

$
  g[tau]^2 - g[tau-1]^2 &= sum_(t=0)^(T-1) HH_(t tau) - sum_(t=0)^(T-1) HH_(t, tau-1) \
  &= sum_(t=0)^(T-1) (HH_(t tau) - HH_(t, tau-1))
$

The gradient of this expression is not separable as-is, so we square it and define the following penalty function:

$
  cal(P)_g (HH) &= sum_(tau=1)^(K-1) sum_(t=0)^(T-1) (HH_(t tau) - HH_(t, tau-1))^2 \
$

*Gradient calculation*
$
  partial / (partial HH_(i j)) (
    HH_(t tau) - HH_(t, tau-1)
  )^2 = cases(
  2(HH_(i j) - HH_(i, j-1)) &"if" i=t "and" j=tau,
  -2(HH_(i,j+1) - HH_(i j)) &"if" i=t "and" j+1=tau,
  0 &"otherwise"
)
$

So:
$
  (partial cal(P)_g) / (partial HH_(i j)) &= 2(HH_(i j) - HH_(i, j-1)) -2(HH_(i,j+1) - HH_(i j)) \
  &= 4 HH_(i j) - 2 (HH_(i,j-1) + HH_(i,j+1))
$

Thus:

$
  gradient_HH^+ cal(P)_g = 4 HH \
  (gradient_HH^- cal(P)_g)_(i j) = 2 (HH_(i,j-1) + HH_(i,j+1))
$

== Diagonal smoothness penalty

We hypothesize the tracks to be played near their original speed, and that there will be significant time intervals without any loops or jumps. This results in diagonal line structures in $HH$. We define a *diagonal smoothness* penalty that minimises the difference between diagonal cells of $HH$:

$
  cal(P)_d (HH) = sum_(t=1)^(T-1) sum_(tau=1)^(K-1) (HH_(t,tau) - HH_(t-1, tau-1))^2
$

*Gradient calculation*:
Similarly to the the gain smoothness penalty:
$
  partial / (partial HH_(i j)) (
    HH_(t tau) - HH_(t-1, tau-1)
  )^2 = cases(
  2(HH_(i j) - HH_(i-1, j-1)) &"if" i=t "and" j=tau,
  -2(HH_(i+1,j+1) - HH_(i j)) &"if" i+1=t "and" j+1=tau,
  0 &"otherwise"
)
$

So:
$
  (partial cal(P)_d) / (partial HH_(i j)) &= 2(HH_(i j) - HH_(i-1, j-1)) -2(HH_(i+1,j+1) - HH_(i j)) \
  &= 4 HH_(i j) - 2 (HH_(i-1,j-1) + HH_(i+1,j+1))
$

Thus:

$
  gradient_HH^+ cal(P)_d = 4 HH \
  (gradient_HH^- cal(P)_d)_(i j) = 2 (HH_(i-1,j-1) + HH_(i+1,j+1))
$

== Lineness penalty

The time-warping function $f[tau]$ is expected to be injective and piecewise continuous. This means we can characterize the neighboring cells of a given activation in $HH$. Given an activated cell $(i,j)$, only the up direction $(i+1,j)$, right direction $(i,j+1)$, or upper-right diagonal direction $(i+1, j+1)$ should be activated, but not any combination of the three.

Thus we define the following, that increases when more than one of these directions are activated near an activated cell:

$
  cal(P)_l (HH) &= sum_(t=0)^(T-2) sum_(tau=0)^(K-2) HH_(t,tau) (
    HH_(t,tau+1) HH_(t+1,tau+1) + HH_(t+1,tau) HH_(t+1,tau+1) + HH_(t+1,tau) HH_(t,tau+1)
  )
$

*Gradient calculation*:
$
  (partial cal(P)_l) / (partial HH_(i j)) =gradient_HH^+ cal(P)_l &= HH_(i,j+1) HH_(i+1,j+1) + HH_(i+1,j) HH_(i+1,j+1) + HH_(i+1,j) HH_(i,j+1) \
  &+ HH_(i-1,j) HH_(i,j+1) + HH_(i-1,j) HH_(i-1,j+1) \
  &+ HH_(i,j-1) HH_(i+1,j) + HH_(i,j-1) HH_(i+1,j-1) \
  &+ HH_(i-1,j-1) HH_(i-1,j) + HH_(i-1,j-1) HH_(i,j-1) \
  gradient_HH^- cal(P)_l &= 0
$


