# Pistes de recherche

## Faits établis (pilote, 1000 positions/case, Lichess 2026-10)
- T (Boltzmann, cp) décroît avec l'Elo : ~110 cp (800-1800) -> ~45 cp (2400-2600). Rapid un peu plus froid que Blitz au-dessus de 1400.
- Coups hors des 5 candidats : 36 % (800) -> 16 % (2500). Sous un logit (propriété IIA), les exclure ne biaise pas T ;
  ils servent à TESTER le modèle (il faut alors évaluer tous les coups légaux).
- Piste 1 (forme de la loi) : Boltzmann rejeté partout (rapport de vraisemblance 15 à 106, seuil 3,8).
  Exponentielle étirée exp(-(ΔE/s)^c) avec c ≈ 0,5 dans TOUTES les cases, sans tendance nette avec l'Elo :
  la forme serait universelle, seule l'échelle s change avec le niveau.
  f(ΔE) non paramétrique : saut d'environ 1 nat entre 0 et 25 cp (prime au meilleur coup), puis croissance lente.

## Contrôles de la piste 1 (faits)
- B, bruit : réévaluation à profondeur 20 de 1000 positions -> c = 0,48 dans les deux cas,
  Δc = -0,003 [-0,07 ; +0,07]. Le c ≈ 0,5 n'est PAS un artefact de bruit (meilleur coup changé dans 23 % des cas).
- A, spectre complet (tous les coups légaux, profondeur 12, 17 542 positions) : c = 0,44 à 0,58 (IC ±0,05),
  Boltzmann rejeté (LR 108 à 290). Légère hausse possible de c avec l'Elo (0,45 à 900 -> 0,55 à 2500), à confirmer.
  f(ΔE) ≈ ΔE^0,6 entre 10 et 700 cp (le point à 1000 cp est déformé par l'écrêtage).
- Test IIA : un modèle ajusté sur les 5 premiers coups SURPRÉDIT le taux de coups hors top-5
  (préd. 0,25-0,57 contre obs. 0,17-0,37), pour Boltzmann comme pour l'exponentielle étirée.
  Les ~27 coups « mauvais » ne sont pas « thermiquement accessibles » comme le voudrait un logit :
  les joueurs n'envisagent qu'un sous-ensemble de coups (déficit d'entropie).
  Piste nouvelle (7) : modèle à deux étages, ensemble de coups envisagés + choix ; mesurer la taille
  effective de cet ensemble (une « entropie ») en fonction de l'Elo et du temps.

## Pistes
1. **Forme de la loi de choix** (en cours) : c ≈ 0,5 universel ? À vérifier : bruit d'évaluation à petit ΔE,
   spectre complet de tous les coups légaux, modèle de mélange « voit le meilleur coup / Boltzmann ».
2. **Temps comme recuit simulé** : T (ou s, c) en fonction du temps restant ([%clk] déjà dans
   data/sample_positions.csv, colonne clock_before) ; chercher une superposition T(Elo, t) = f(t/τ(Elo)).
3. **Champ extérieur** : T selon que le joueur mène, est à égalité ou est mené.
4. **Fluctuation-dissipation** : tester dU/dT = Var(E)/T² entre tranches ; une violation signale
   qu'une seule température ne suffit pas.
5. **Désordre** : distribution des T individuels dans une tranche (plusieurs positions par joueur,
   nouveau tirage, ≈1-2 h de calcul).
6. **Type de position** : T dans une position calme ou tactique (écart entre les deux meilleurs coups) : est-ce le joueur ou la position qui « chauffe » ?

## Références
- K. Regan, G. Haworth, *Intrinsic Chess Ratings*, AAAI 2011 (modèle à sensibilité s et consistance c).
- R. McIlroy-Young et al., *Aligning Superhuman AI with Human Behavior: Chess as a Model System* (Maia), KDD 2020.

## Coûts mesurés (Stockfish 18, 3 workers)
- MultiPV 5, profondeur 16 : 0,37 s/position.
- Tous les coups légaux (~32) : prof. 10 : 0,21 s ; prof. 12 : 0,29 s ; prof. 16 : 2,18 s par position.
