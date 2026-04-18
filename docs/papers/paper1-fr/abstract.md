# Résumé (Paper 1, brouillon)

**Objectif de mots** : 250 mots

---

## Brouillon v0.1 (S17.2, 2026-04-18)

L'oubli catastrophique demeure un obstacle central pour les systèmes
cognitifs artificiels apprenant séquentiellement une succession de
tâches. La consolidation mnésique inspirée du sommeil a été proposée
comme remède, les travaux antérieurs ayant exploré la réactivation
(Walker, van de Ven), l'homéostasie synaptique (Tononi), la
recombinaison créative (Hobson) et le codage prédictif (Friston) —
mais aucun framework formel unifié n'intègre ces quatre piliers en
opérations composables et indépendantes du substrat.

Nous introduisons **dreamOfkiki**, un framework formel à axiomes
exécutables (DR-0 redevabilité, DR-1 conservation épisodique, DR-2
compositionnalité sur un semi-groupe libre d'opérations oniriques,
DR-3 indépendance du substrat via un Critère de Conformité, DR-4
inclusion en chaîne des profils). Le framework définit 8 primitives
typées (entrées α, β, γ, δ ; 4 canaux de sortie), 4 opérations
canoniques (replay, downscale, restructure, recombine) et une
ontologie de Dream Episode en quintuplet. Le framework admet
plusieurs substrats conformes ; des implémentations exemplaires
valident la conception et sont rapportées séparément (voir Paper 2).

Des hypothèses pré-enregistrées (DOI OSF : en attente) sont évaluées
via le test t de Welch, l'équivalence TOST, la tendance de
Jonckheere-Terpstra et un test t à un échantillon contre un budget
de calcul, sous correction de Bonferroni.

**Validation du pipeline (substitution synthétique, pilote G2).** Le
pipeline de mesure et statistique de bout en bout est exercé avec
des prédicteurs mock aux niveaux de précision scriptés ; les
chiffres sont rapportés au §7 conjointement avec leur run_id
enregistré et leur dump JSON sous `docs/milestones/`. L'inférence
mega-v2 réelle et toute analyse de similarité représentationnelle
IRMf sont repoussées au cycle 2 (Paper 2). L'ensemble du code, des
spécifications et du pré-enregistrement est ouvert sous
MIT/CC-BY-4.0.

---

## Notes pour révision

- Remplacer les résultats synthétiques par des chiffres d'ablation
  réels post-S20+
- Insérer le DOI OSF une fois verrouillé (action en cours)
- Insérer le DOI Zenodo pour les artefacts code+modèle au tag de
  soumission
- Resserrer à ≤250 mots (actuellement ~265)
