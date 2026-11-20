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

## Piste 4 (faite) : fluctuation-dissipation
- Logit p ∝ exp(-β H) : le maximum de vraisemblance égale Σ H_obs et Σ <H>_β (énergie moyenne U) ;
  la relation dU/dβ = -Var(H) prédit alors les fluctuations. Deux mesures par case (spectre complet, prof. 12) :
  T_V/T_U (β du 2e moment contre β du 1er ; global) et R = Σ(H_obs - <H>)² / Σ Var(H) (position par position).
  Une seule température <=> T_V/T_U = 1 et R = 1. `python run.py fdt`, results/fdt*.
- Boltzmann (H = ΔE) : violation nette partout : T_V/T_U = 1,3-1,7, R = 2,9 [2,75-3,0] (moyenne pondérée des 18 cases).
- Loi étirée, H = ΔE^c : T_V/T_U = 1,02 (IC inclut 1 dans la plupart des cases) : dans la variable ΔE^0,5 une seule
  « température » rend compte des fluctuations globales. Mais R (en H) = 1,15-1,30.
- Comparaison des modèles en ΔE (même grandeur) : R = 2,9 (Boltzmann), 1,54 [1,43-1,64] (étirée),
  1,26 [1,18-1,34] (deux étages). Le modèle à deux étages capte l'essentiel ; il reste ~25 % de fluctuations
  en excès, homogène entre cases (χ²/ddl = 0,4) : hétérogénéité non modélisée (T dépendant de la phase, de l'échec,
  du joueur : pistes 2, 6, 5). R > 1 avec T_V ≈ T_U signifie une pente < 1 de ΔE observé sur ΔE prédit
  (positions prédites « chaudes » plus froides que prévu et inversement).
- Limite : moments d'ordre 2 dominés par les grosses gaffes (écrêtage à 1000 cp) ; résultats similaires en proba. de gain.

## Piste 5 (faite) : désordre (températures individuelles)
- Nouvel échantillon : préfixe de 1 Go du même mois (~45 h), Blitz, 160 joueurs actifs (>= 40 parties dans le préfixe),
  20 par tranche d'Elo 800-2400 (Elo médian du joueur), 25 parties par joueur, une position par partie (joueur au trait),
  tous les coups légaux à prof. 10 (6 min). 3320 positions après filtre |éval| <= 500 cp. Noms hachés.
  `python run.py players_sample | players_eval | players`, results/players*.
- Modèle : log s = a(tranche) + d(demi-coup) + échec + u_joueur, u ~ N(0, σ²), c = 0,51 fixé, Gauss-Hermite.
  Validé sur simulation (mêmes tailles) : σ = 0,3 retrouvé (0,31, LR = 19) ; σ = 0 -> 0,08 non significatif.
- Résultat : σ = 0,00, IC 95 % [0 ; 0,12] (vraisemblance profilée), LR = 0. Joueurs permutés : σ = 0 aussi.
  Fiabilité moitié/moitié nulle (Spearman -0,06) ; effet joueur sans lien avec l'Elo dans la tranche (-0,02).
  Pour comparaison, s varie d'un facteur 2,8 entre 800 et 2200 (29 -> 10 cp) et x2 avec la phase.
- Conclusion : à Elo, phase et échec fixés, les joueurs actifs ont la même « température » à ±12 % près :
  l'Elo est (presque) une statistique suffisante de T. L'excès de fluctuations de la piste 4 (R ≈ 1,26)
  ne vient donc pas des joueurs mais des positions (forme de f, saillance).
- Limites : joueurs très actifs seulement, Blitz, effet joueur sur s seulement (pas sur c ni sur Ω_eff).

## Chaleur spécifique (faite)
- C = β² Var(H), H = ΔE^c, β = s^-c (loi étirée par case) : sensibilité du choix à la température
  (C = -β d<H>/dβ, k_B = 1). `python run.py heat`, results/heat*.
- C(s) sur l'ensemble des positions : pic à s* ≈ 10 (Blitz) / 13 cp (Rapid), juste sous les s humains (12-31 cp) :
  tous les niveaux sont sur le plateau près du maximum (C ≈ 1,4-1,6, sans tendance avec l'Elo).
- Par type de position, le pic se déplace avec l'écart entre les deux meilleurs coups : s* ≈ 3-5 cp (écart < 30),
  10 (30-100), 19-23 (100-300), 33-41 (> 300). Les positions où un coup se détache de 100 à 300 cp ont leur pic
  DANS la plage humaine : ce sont elles qui « discriminent » les niveaux (lien : points de bascule de Barthelemy 2023,
  difficulté d'Anderson et al. 2016, c croissant avec l'écart, piste 6). Les positions calmes sont « saturées »
  (choix quasi aléatoire entre coups équivalents, sans conséquence).
- C observé / C modèle = R de la piste 4 : excès concentré dans les positions à coup unique (écart > 300 :
  ×2,1 Blitz, ×2,1 Rapid ; ailleurs ×1,0-1,3). On y trouve le coup ou on gaffe lourdement : comportement
  bimodal que la loi continue ne capte pas (cf. modèle de mélange « voit / ne voit pas », piste 1).

## Piste 8 (pilote fait) : énergie contre entropie (tactique / stratégie)
- 1008 positions Blitz (112 par tranche), 8 meilleurs coups ; positions filles (adversaire au trait) et petites-filles
  (après la meilleure réponse) évaluées sur tous les coups, prof. 10 (25 min). `run.py entropy_eval | entropy`.
- S = log N_bons (coups à < 50 cp du meilleur) ; L = <ΔE> sous la loi de la tranche (perte attendue).
- (A) p ∝ exp[-(ΔE/s)^c + λ_moi S_moi + λ_adv S_adv] : LR = 55 vs énergie seule.
  λ_moi = +0,27 ± 0,06 (garder ses options), λ_adv = -0,37 ± 0,06 (priver l'adversaire des siennes) ;
  ~10-15 cp par doublement d'options. Robuste : capture/échec contrôlés (capture +1,1 : saillance),
  énergie non paramétrique, c = 0,35-0,7. Corrélation ΔE / S_adv (-0,45) joue contre l'effet.
  Par Elo : λ_moi 0,40 -> 0,27 -> 0,15 (faibles = tiennent plus à leurs options, ~2σ) ; λ_adv ≈ -0,3 à -0,5.
- (B) énergie libre « rationnelle » E_prat = ΔE + κ_moi L_moi - κ_adv L_adv (profil sur grille, la nll n'est pas
  dérivable en κ) : κ ≈ 0,2 [0 ; 0,3] pour les deux ; LR = 8,5 seulement ; κ = 1 rejeté (LR = 91).
  => les joueurs réagissent au NOMBRE de bonnes options (heuristique), pas à un calcul de leurs chances d'erreur.
- À faire : échantillon complet (~3 h), λ selon le score (lien piste 3 : « simplifier » = réduire l'entropie),
  seuil de 50 cp à faire varier, Rapid, entropie comme saillance de l'étage 1 (piste 7).

## Recherche d'une vraie transition (faite, tests 1 et 2, sans Stockfish)
- `python run.py transition`, results/transition*. Spectre complet, prof. 12, loi étirée par case.
- Test 1, trouve / gaffe (analogue du 1er ordre ?) : en coup unique (écart > 300 cp) la perte est forcément 0 ou >= 300 :
  la bimodalité est STRUCTURELLE (spectre), déjà prédite par le modèle. Gaffes x1,5 (Blitz) / x1,6 (Rapid) [IC 1,2-2,0].
  Raideur : logit P(trouve) obs. = a + b logit P(prédit) : b < 1 partout (0,5-0,7 positions calmes ; 0,2 [0,1-0,4]
  en coup unique). P(trouve) ≈ 0,85 presque plate quand le modèle prédit 0,68 -> 0,998 :
  pas de bascule brutale, au contraire un PLANCHER de gaffes (« inattention » ~10-15 %) indépendant de la difficulté.
  Par Elo : gaffes en coup unique 25 % (900) -> ~5 % (2500), décroissance graduelle, excès surtout au-dessus de 1500
  (1900 : 16 % contre 7 % prédit). Réserve : le bruit de p_modèle atténue b (erreur sur la variable).
  => modèle de mélange « distraction » (taux de lapsus fixe + loi étirée) plutôt que transition du 1er ordre.
- Test 2, effet de taille (hors échec, quintiles de coups légaux 21 -> 44) : pic s* stable (8-11 cp Blitz, 10-14 Rapid),
  C_max 1,54 -> 1,95, largeur 1,63 -> 1,54 décade (repondéré écart + phase).
  Témoin (positions >= 40 coups réduites au hasard) : C_max 1,65 -> 1,98, largeur 1,48 constante.
  La hausse de C_max est celle du simple comptage des niveaux (~ ln N) ; l'affinement résiduel (0,1 décade pour N x2,
  ~N^-0,08) ne tend pas vers 0 : bosse, pas de transition. Test peu puissant (plage de N étroite).
- Reste : test 3 (partie entière, points de bascule hors équilibre, > 1 h de Stockfish, à confirmer).

## Synthèse (après les pistes 1 à 7)
- Loi de choix p ∝ exp(-(ΔE/s)^c), c ≈ 0,5 (0,45 positions calmes -> 0,65 quand un coup se détache),
  avec un étage d'attention (Ω_eff ~ 20 -> 14 coups envisagés).
- s est fixé par l'Elo (x2,8), l'avancement de la partie (x1,5-2), l'échec (x2,7-4,5), et, en proba. de gain,
  le score (mené : plus froid). Pas d'effet propre de l'horloge à phase fixée ; pas de désordre entre joueurs.
- Pistes ouvertes : (a) saillance mieux que le rang Stockfish (Maia) pour l'étage 1 ; (b) effets de phase/échec
  sur Ω_eff plutôt que sur s ; (c) démêler demi-coup = fatigue / complexité / temps consommé (temps par coup,
  positions de même structure à demi-coups différents) ; (d) désordre en Rapid et sur c.

## Pistes
1. **Forme de la loi de choix** (en cours) : c ≈ 0,5 universel ? À vérifier : bruit d'évaluation à petit ΔE,
   spectre complet de tous les coups légaux, modèle de mélange « voit le meilleur coup / Boltzmann ».
2. **Temps comme recuit simulé** (fait, puis contrôlé : effet dû surtout à la phase) : T (ou s, c) en fonction du temps restant ([%clk] déjà dans
   data/sample_positions.csv, colonne clock_before) ; chercher une superposition T(Elo, t) = f(t/τ(Elo)).
3. **Champ extérieur** (fait) : T selon que le joueur mène, est à égalité ou est mené.
4. **Fluctuation-dissipation** (fait) : tester dU/dT = Var(E)/T² entre tranches ; une violation signale
   qu'une seule température ne suffit pas.
5. **Désordre** (fait) : distribution des T individuels dans une tranche (plusieurs positions par joueur,
   nouveau tirage, ≈1-2 h de calcul).
6. **Type de position** (fait) : T dans une position calme ou tactique (écart entre les deux meilleurs coups) : est-ce le joueur ou la position qui « chauffe » ?

## Références
- K. Regan, G. Haworth, *Intrinsic Chess Ratings*, AAAI 2011 (modèle à sensibilité s et consistance c).
- R. McIlroy-Young et al., *Aligning Superhuman AI with Human Behavior: Chess as a Model System* (Maia), KDD 2020.

## Coûts mesurés (Stockfish 18, 3 workers)
- Tous les coups légaux, prof. 10, échantillon joueurs : 0,09 s/position (4 cœurs). Modèle additif (gradient analytique) : 0,2 s/ajustement.
- MultiPV 5, profondeur 16 : 0,37 s/position.
- Tous les coups légaux (~32) : prof. 10 : 0,21 s ; prof. 12 : 0,29 s ; prof. 16 : 2,18 s par position.
