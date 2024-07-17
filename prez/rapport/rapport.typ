#import "@preview/ilm:1.1.1": *
#set text(lang: "fr")


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
  date-format: auto
  // table-of-contents: none
)

= Sujet

= État de l'art
== Les pratiques DJ
Présentation générale des pratiques DJ et intérêt de s'y intéresser
== Ingénierie inverse de mix DJ
Définition des différentes notions + état de l'art
== Jeux de données et _track ID_
Aspect culturel du _track ID_ et lien avec les jeux de données
=== Jeux de données réels
État de l'art + sources de données encore non exploitées
=== Jeux de données synthétiques
État de l'art

= Création de jeux de données
== Méthodes de mesure de vérité terrain
Mesurer les paramètres de mix sur un mix réel: que des avantages
=== Suivi optique de disques vinyles
Présentation, résultats, github
=== Extraction des métadonnées de fichiers de projet DAW
Présentation, résultats, github
=== Instrumentation de logiciel de mix
Présentation
== `trackid.net`
Présentation, statistiques, données, problèmes de droits.


= Transcription de mix DJ par factorisation de matrices non-négatives (NMF)
== Formulation matricielle du processus de mix
Justification de l'utilisation de la NMF
== Beta-NMF et règles MU
Présentation générale de la beta-nmf, avec les propriétés intéressantes pour la suite. Lien entre beta et puissance du spectrogramme
== Caractérisation des transformations temporelles
Justification de la forme attendue des résultats
== Estimation du gain et du _warp_
Justification de l'exactitude des estimateurs dans le cas idéal
== Améliorations algorithmiques
=== Compression de l'information
Réduction des dimensions des matrices $=>$ problème devient tractable
=== Recouvrement
Exploitation de la continuité temporelle des signaux
=== Normalisation
Normalisation des matrices et impact sur les estimateurs
=== Régularisation
Définition des fonctions de régularisation utilisées et poids $lambda$ recommandés
=== _Warm-up_
Warm-up de $beta$, $lambda$, puissance du spectrogramme $=>$ pas de compromis entre robustesse et performance
=== NMF multi-passes
Explication de l'algorithme, + creusage des matrices
=== Estimation des EQ
Découpage des matrices en bandes + justification matérielle
=== Invariance à la transposition
cf. divergence invariante à la transposition, ou NMFD, ou PLCA...
== Implémentation
pytorch, discussion de cpu vs. gpu
== Résultats
=== Sur mix synthétiques
=== Sur unmixdb
=== Sur mix réels

= Conclusion
