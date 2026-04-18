# Travaux futurs — Cycle 2 (Paper 1, brouillon S19.2)

**Longueur cible** : ~0,5-1 page markdown (≈ 700 mots)

---

## 9.1 Substrat E-SNN (thalamocortical Loihi-2)

L'extension la plus directe du cycle 1 consiste à valider le
Critère de Conformité DR-3 sur un second substrat : un réseau
neuronal spiking thalamocortical (E-SNN) déployé sur le matériel
neuromorphique Intel Loihi-2. Ceci a été différé du cycle 1 selon
la décision de SCOPE-DOWN (master spec §7.3) pour assurer que le
cycle 1 se clôture à temps avec une validation sur substrat
unique.

Le substrat E-SNN testerait si les axiomes exécutables du framework
restent opérationnels lorsque les opérations sont réalisées comme
dynamiques de taux de spike plutôt que comme mises à jour par
gradient sur matrices denses. Une conformité réussie apporterait
la preuve d'indépendance du substrat que le Paper 1 revendique
comme propriété théorique mais ne démontre pas encore
empiriquement sur deux substrats.

## 9.2 Câblage réel du profil P_max

Le cycle 1 n'implémente P_max qu'en squelette (`status="skeleton"`,
`unimplemented_ops=["recombine_full"]`). Le cycle 2 câblera les
composants restants :

- **Traces brutes du flux α** canal d'entrée (actuellement déclaré
  P_max-uniquement mais non consommé) — requiert un tampon
  circulaire firehose avec rétention bornée
- **Canal de sortie canal-4 ATTENTION_PRIOR** — requiert
  l'invariant de bornage de l'attention prior (S4) et le câblage
  en aval vers les modules consommateurs
- **Variante d'opération `recombine_full`** — paire complète
  d'encodeur / décodeur VAE au-delà du squelette d'interpolation
  allégée C-Hobson

Avec P_max réellement câblé, l'hypothèse H2 (équivalence P_max vs
P_equ dans ±5 %) devient une comparaison réelle plutôt que le test
de fumée d'auto-équivalence du cycle 1.

## 9.3 Partenariat réel avec un laboratoire IRMf

Le cycle 1 verrouille Studyforrest comme repli IRMf (G1 Branche
A). Le cycle 2 poursuit un partenariat actif avec un ou plusieurs
laboratoires IRMf identifiés via la campagne de recrutement de
relecteurs T-Col :

- **Huth Lab** (UT Austin) : jeu de données Narratives
- **Norman Lab** (Princeton) : études de mémoire épisodique
- **Gallant Lab** (UC Berkeley) : BOLD piloté par stimuli
  naturalistes

Un partenariat réel avec un laboratoire permettrait la RSA sur
des stimuli linguistiques **contrôlés par tâche** plutôt que sur
le repli de compréhension narrative fourni par Studyforrest. Ceci
renforcerait H3 (alignement représentationnel monotone) qui n'a
atteint qu'une significativité limite dans la validation du
pipeline synthétique du cycle 1 (run_id
`syn_s15_3_g4_synthetic_pipeline_v1`, dump
`docs/milestones/ablation-results.json`).

## 9.4 Validation multi-substrat du Critère de Conformité

L'assertion théorique la plus forte du Framework C-v0.5.0+STABLE
— l'indépendance du substrat via le Critère de Conformité DR-3 —
nécessite une validation empirique sur plus de deux substrats pour
être défendable en relecture par les pairs. Le cycle 2 établit la
matrice de validation : pour chaque substrat candidat
(implémentation de référence du cycle 1 ✅, E-SNN, instance
hypothétique basée sur transformer), vérifier les trois conditions
de conformité (typage des signatures, tests de propriété
axiomatiques passants, invariants BLOCKING applicables).

Une suite de tests de conformité réutilisable (brouillonnée au
cycle 1 sous `tests/conformance/`) constitue le fondement. Le
cycle 2 l'étendra avec des adaptateurs spécifiques au substrat et
exécutera la suite complète contre chaque substrat candidat,
produisant un rapport de conformité publiable comme artefact
supplémentaire pour le Paper 1 (ou comme contribution principale
de l'article d'ablation ingénierie du Paper 2).

---

## Notes pour révision

- Resserrer à ≤700 mots pour la discipline Nature HB
- Renvoyer aux documents du plan cycle-2 une fois ceux-ci rédigés
  (post-G6 rapport de décision cycle-2 S28.1)
- Réordonner les sous-sections par priorité une fois le périmètre
  cycle-2 verrouillé
