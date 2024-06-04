postprocessing avec la matrice d'autosimilarité pour éviter les ripples: filtrage morphologique, identifier les templates non voulues...
retirer les boucles du morceau original avant processing

initialiser avec des diago plutôt que du bruit

l2: favorise smoothness sur colonne et lignes
l1: favorise sparsity

envoyer papier smooth NMF

fenetre beat-synchrone approximatives
tester spectre tonnetz
dérivée du spectro ? peuetre pitch (ou au moins freqshift invariant ?)

NMF source **filtre**: maxime bouvier

mettre des variantes des reftracks pour être plus robuste aux effets supplémentaires

tester sur unmixdb

mesurer disto entre V et Vhat