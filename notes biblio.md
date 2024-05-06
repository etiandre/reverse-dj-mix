---
bibliography: Stage.bib
---
# Questions

Définition de track / mashup / sample ? Un mashup est-il 2 tracks ou 1 track ?

Peut on étendre la définition: mix->track et track->sample/loop

EDM only ? (reggae/dancehall/dub: dj=selecta; hip hop)

On a de quoi accéder aux papiers en pas open access (genre IEEE) ?

Niveaux de confiance: Synthétique > Expert > **Droits d'auteur** > Crowd-source modéré >  Crowd-source volontaire > Crowd-source involontaire ?

Definition adhoc: broadcast mix, lounge mix, artistic mix



Regarder chez les gens qui ont besoin de droits d'auteur: radio dj, etc

Définitions de mix/track/EDM: c'est nous qu'on choisit

Faire un dataset => c'est ok de faire un dataset "difficile"

Tester methode diemo sur un vrai mix, pour voir

Critère objectif de performance: mix - mix estimé devrait être 0

Instrumentation de platine + mixer un peu moderne => super cool ! midi + timecode => faders, eq, speed curve.



time alignment : faire d'abord source separation puis le faire sur basse/batterie => moins de bruits dans les infos

réseau type unet pour la séparation de source

détection des moments de transition => via théorie de l'information (entropie +++ quand y'a 2 tracks en même temps + période de grand changement). Utilisation du papier de dominique qui n'a rien à voir

regarder chez antescofo pour le audio-to-audio alignment

blind extract basse/batterie => time unwarping => loop detection with (convolutive ?) nmf => time curve

regarder les hardware courants: cutoff des filtres ? valeur q ? type de filtre ? c'est les mêmes partout ?

Blind estimation of audio effects using auto encoder