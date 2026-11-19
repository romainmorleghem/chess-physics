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

## Piste 7 (faite) : modèle à deux étages
- K coups les mieux classés envisagés (K aléatoire, loi libre), puis choix ∝ exp(-(ΔE/s)^c).
- Bien meilleur que le logit à un étage (LR 38 à 167, 6 ddl) ; reproduit le taux de coups hors top-5.
- Ω_eff (coups envisagés en moyenne, sur ~33 légaux) : ~20 (Elo 900) -> ~14 (Elo 2300-2500), Blitz et Rapid.
  Les joueurs forts « élaguent » davantage (cohérent avec de Groot). c reste ≈ 0,5-0,6.
- Limite : la saillance est approchée par le rang Stockfish ; Ω_eff dépend de ce choix.

## Piste 2 (faite) : temps restant
- c fixé à 0,5 ; échelle s par (groupe d'Elo x fraction du temps initial restant).
- s augmente quand le temps diminue (« chauffe » sous la pression) : x ~2 en Blitz entre >75 % et <10 % du temps.
- Superposition s(Elo, t) = A(Elo) * g(t) acceptée : Blitz p = 0,48 ; Rapid p = 0,056 (limite).
- Limite : confusion avec la phase de jeu (peu de temps = fin de partie) -> contrôler par le numéro de coup.

## Contrôle de la piste 2 (fait) : temps restant à phase de jeu fixée
- Modèle additif log s = a(tranche Elo) + b(temps) + d(phase), c = 0,5 ; phase = demi-coup (0-30-50-70-fin)
  ou matériel non-pion (24 au départ ; >20, 14-20, 8-14, <8). `python run.py phase`, results/phase*.
- Confusion forte : corrélation de Spearman demi-coup / temps restant = -0,84.
- s(<10 %) / s(>75 %) : sans contrôle x2,2 (Blitz), x1,6 (Rapid) ;
  à demi-coup fixé x1,3 [IC 1,0-1,6] et x1,1 [0,85-1,5], effet du temps NON significatif (LR p = 0,22 et 0,57) ;
  à matériel fixé x1,9 et x1,6, toujours significatif (p < 1e-7).
- Effet du demi-coup à temps fixé, fort (p < 1e-12) : s x2,1 (Blitz) / x1,65 (Rapid) entre <30 et >70 demi-coups.
- Conclusion : la « chauffe » de la piste 2 suit l'avancement de la partie (demi-coup), pas l'horloge en soi ;
  le matériel est un moins bon contrôle. Réserve : colinéarité forte, un effet résiduel du temps <= x1,5 n'est pas exclu.
  Le demi-coup peut porter fatigue, complexité ou temps accumulé : à démêler (piste 6, type de position).

## Piste 3 (faite) : champ extérieur (mène / égalité / mené)
- log s = a(Elo) + e(score) + d(demi-coup), c ajusté ; score = éval. du meilleur coup pour le joueur au trait
  (prof. 12), tranches <-300, -300..-100, ±100, 100..300, >300 cp. `python run.py lead`, results/lead*.
- En centipions : effet faible et SYMÉTRIQUE (p = 1e-5 Blitz, 1e-3 Rapid) : s x1,1 à x1,3 dès qu'on s'éloigne
  de l'égalité, dans les deux sens ; asymétrie mène/mené non significative (rapport 1,15 p = 0,22 ; 0,99 p = 0,98).
- En probabilité de gain : effet fort et MONOTONE (p < 1e-17) : s(mené >300) / s(égal) = 0,59 [0,53-0,67],
  s(mène >300) / s(égal) = 1,23 (Blitz) ; asymétrie x2 (p < 0,01) dans les deux cadences.
- Les deux énergies ne se valent pas : avec le score en covariable, la vraisemblance favorise la proba. de gain
  (Δnll = -45 Blitz, -32 Rapid, même nombre de paramètres). Lecture : en enjeu réel (chances de gain),
  le joueur mené joue « froid » (se bat), le joueur qui mène accepte de petites pertes (simplifie).
  L'asymétrie vient en partie de la courbure de la sigmoïde ; elle dépend donc du choix de l'énergie.
- Le demi-coup reste très significatif à score fixé (p < 1e-23) : la chauffe avec la phase n'est pas un effet du score.

## Piste 6 (faite) : type de position — le joueur ou la position qui « chauffe » ?
- log s = a(Elo) + g(écart 1er-2e coup) + d(demi-coup) + n(coups légaux) + e(score) + échec, c ajusté.
  `python run.py postype`, results/postype*. Écart : <10, 10-30, 30-100, 100-300, >300 cp (prof. 12).
- Écart : effet modeste (p = 7e-4 à 0,02) : s x1,15-1,25 quand un coup se détache (écart > 30 cp) vs position calme.
- En échec : s x4,5 (Blitz) / x2,7 (Rapid), p < 1e-8. Le fort effet apparent « peu de coups légaux » (s x2 sous 20 coups)
  était entièrement dû à l'échec (52-58 % de ces positions) ; à échec fixé, coups légaux ~sans effet.
  Robuste à la restriction du choix aux 8 ou 12 meilleurs coups (pas un artefact du logit à un étage, cf. piste 7).
- Demi-coup à type de position fixé : toujours fort (p < 1e-13) : s x2,0 (Blitz) / x1,5 (Rapid) entre <30 et >70.
  => c'est surtout le JOUEUR (avancement de la partie) qui chauffe, pas la composition de la position.
- Forme : c croît avec l'écart, 0,46 [0,44-0,48] (calme) -> 0,62-0,66 (écart 30-300), Blitz et Rapid
  (profil de vraisemblance). c ≈ 0,5 n'est donc pas strictement universel : la loi étirée moyenne la « prime au
  meilleur coup » (saut à petit ΔE) et la croissance ~ΔE^0,6 ; le poids relatif change avec le type de position.
- Limite : écart et ΔE viennent de la même évaluation (prof. 12), le bruit peut les corréler.

## Pistes
1. **Forme de la loi de choix** (en cours) : c ≈ 0,5 universel ? À vérifier : bruit d'évaluation à petit ΔE,
   spectre complet de tous les coups légaux, modèle de mélange « voit le meilleur coup / Boltzmann ».
2. **Temps comme recuit simulé** (fait, puis contrôlé : effet dû surtout à la phase) : T (ou s, c) en fonction du temps restant ([%clk] déjà dans
   data/sample_positions.csv, colonne clock_before) ; chercher une superposition T(Elo, t) = f(t/τ(Elo)).
3. **Champ extérieur** (fait) : T selon que le joueur mène, est à égalité ou est mené.
4. **Fluctuation-dissipation** : tester dU/dT = Var(E)/T² entre tranches ; une violation signale
   qu'une seule température ne suffit pas.
5. **Désordre** : distribution des T individuels dans une tranche (plusieurs positions par joueur,
   nouveau tirage, ≈1-2 h de calcul).
6. **Type de position** (fait) : T dans une position calme ou tactique (écart entre les deux meilleurs coups) : est-ce le joueur ou la position qui « chauffe » ?

## Références
- K. Regan, G. Haworth, *Intrinsic Chess Ratings*, AAAI 2011 (modèle à sensibilité s et consistance c).
- R. McIlroy-Young et al., *Aligning Superhuman AI with Human Behavior: Chess as a Model System* (Maia), KDD 2020.

## Coûts mesurés (Stockfish 18, 3 workers)
- MultiPV 5, profondeur 16 : 0,37 s/position.
- Tous les coups légaux (~32) : prof. 10 : 0,21 s ; prof. 12 : 0,29 s ; prof. 16 : 2,18 s par position.
