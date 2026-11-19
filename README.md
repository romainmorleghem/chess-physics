# Température des coups d'échecs en fonction de l'Elo

Modèle de Boltzmann : pour une position, avec e_i l'évaluation Stockfish (point de vue du
joueur au trait) des k = 5 meilleurs coups, p_i = exp(e_i/T) / Σ_j exp(e_j/T).
De façon équivalente, avec l'énergie E_i = e_best − e_i (perte par rapport au meilleur coup),
p_i ∝ exp(−E_i/T). T est estimé par maximum de vraisemblance, pour chaque cadence × tranche d'Elo.

## Lancer

```
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
sh scripts/install_stockfish.sh
.venv/bin/python run.py all          # ou : download | sample | timing | evaluate | fit
```
Stockfish 18 (binaire officiel `stockfish-ubuntu-x86-64-avx2`) dans `bin/stockfish`.
Tous les paramètres sont dans `config.yaml` (graine fixe).

## Choix méthodologiques
- Données : les 2 premiers Go du mois Lichess 2026-10 (PGN standard rated), parties Blitz/Rapid,
  hors BOT, terminaison « Normal » ou « Time forfeit ». Les deux joueurs dans la même tranche de 200.
- Une seule position par partie (tirage par réservoir puis position au hasard) : les positions
  sont indépendantes, donc le bootstrap sur les positions est un bootstrap sur les parties.
- Exclus : les 10 premiers demi-coups, le premier et le dernier coup, les positions à coup forcé,
  les positions où la meilleure ligne est un mat ou |éval| > 500 cp.
- Évaluations : Stockfish local, profondeur 16, MultiPV 5, 1 thread par moteur, hash vidé
  à chaque position. Le résultat est déterministe (vérifié : 200/200 identiques).
  On n'utilise ni les `[%eval]` (environ 6 % des parties, une seule PV, sélection) ni la base
  d'évaluations cloud (biais vers les positions analysées et la théorie).
- Coups hors des 5 candidats : comptés par tranche et exclus de la vraisemblance
  (vraisemblance conditionnelle au fait que le coup joué est parmi les candidats).
- Variantes : (1) centipions écrêtés à ±1000, mat → ±1000 ; (2) probabilité de gain Lichess
  en % : 100 / (1 + exp(−0.00368208·cp)).

## Sorties (`results/`)
`summary.csv` (T, IC 95 %, taux hors candidats, taux de meilleur coup par case), `T_vs_elo.png`.
