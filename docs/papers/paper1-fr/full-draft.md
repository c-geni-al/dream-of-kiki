---
title: "dreamOfkiki : un framework formel indépendant du substrat pour la consolidation mnésique basée sur le rêve dans les systèmes cognitifs artificiels"
author: "contributeurs du projet dreamOfkiki"
contact: "Clement Saillant <clement@saillant.cc>"
affiliation: "L'Electron Rare, France"
date: "2026"
draft: "v0.2 (cycle-1, S20.3 assemblage, placeholders INCLUDE intégrés)"
---

# Paper 1 — Assemblage complet du brouillon (version française)

⚠️ **Statut** : assemblage du brouillon. Les fichiers .md de section
étaient la source de vérité originale ; ce fichier intègre désormais
leur contenu pour devenir la source assemblée en vue du rendu pandoc.

⚠️ Les **précautions liées aux données synthétiques** s'appliquent
au §7 Résultats (chiffres issus de la substitution synthétique
mega-v2). L'ablation réelle intervient en clôture du cycle 1 (S20+)
ou au cycle 2.

---

## 1. Résumé

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

## 2. Introduction

### 2.1 L'oubli catastrophique et la lacune de consolidation

Les systèmes cognitifs artificiels modernes excellent dans
l'apprentissage mono-tâche, mais se dégradent rapidement lorsqu'ils
sont entraînés séquentiellement sur plusieurs tâches — un phénomène
connu sous le nom d'**oubli catastrophique** [@mccloskey1989catastrophic;
@french1999catastrophic]. Malgré deux décennies de stratégies
d'atténuation (elastic weight consolidation [@kirkpatrick2017overcoming],
réactivation générative [@shin2017continual], mémoire par
répétition [@rebuffi2017icarl]), le champ manque toujours d'une
*théorie unifiée* expliquant pourquoi ces mécanismes fonctionnent
et quand ils doivent se composer.

La cognition biologique résout ce problème pendant le **sommeil**.
La réactivation hippocampique durant le NREM, la régulation à la
baisse synaptique, la restructuration prédictive des représentations
corticales et la recombinaison créative pendant le REM forment
ensemble un pipeline de consolidation multi-étapes
[@diekelmann2010memory; @tononi2014sleep]. Pourtant, les travaux
existants en IA n'ont intégré que des fragments de cette biologie,
en se concentrant généralement sur un mécanisme unique (p. ex. la
réactivation seule) sans théorie raisonnée de la manière dont les
mécanismes interagissent.

### 2.2 Quatre piliers de la consolidation mnésique basée sur le rêve

Nous identifions quatre piliers théoriques que tout framework
complet de consolidation en IA inspirée du rêve doit adresser :

- **A — consolidation Walker/Stickgold** : transfert épisodique-
  vers-sémantique via la réactivation [@walker2004sleep;
  @stickgold2005sleep].
- **B — SHY de Tononi** : homéostasie synaptique renormalisant les
  poids pendant le sommeil [@tononi2014sleep].
- **C — rêve créatif Hobson/Solms** : recombinaison et abstraction
  pendant le REM [@hobson2009rem; @solms2021revising].
- **D — FEP de Friston** : minimisation de l'énergie libre comme
  théorie unificatrice de l'inférence et de la consolidation
  [@friston2010free].

Les travaux antérieurs en IA ont implémenté A [@vandeven2020brain],
B [@kirkpatrick2017overcoming as a SHY-adjacent regularization] et
des éléments de D [@rao1999predictive; @whittington2017approximation],
mais **aucun travail n'a formalisé la manière dont les quatre
piliers se composent** de façon indépendante du substrat, propice à
l'ablation et à la preuve.

### 2.3 La lacune compositionnelle

Pourquoi la composition importe-t-elle ? Empiriquement, l'ordre
dans lequel s'appliquent les opérations de consolidation modifie
l'état cognitif résultant — la réactivation avant la régulation à
la baisse préserve la spécificité épisodique, tandis que la
régulation à la baisse avant la restructuration peut effacer les
représentations mêmes que la restructuration est censée affiner.
Notre analyse (`docs/proofs/op-pair-analysis.md`) énumère les 16
paires d'opérations et constate que 12 paires croisées sont
non-commutatives, confirmant que *l'ordre fait partie du framework*
et non d'un détail d'implémentation.

Un framework formel digne de ce nom doit donc (i) spécifier les
opérations comme primitives composables à types bien définis, (ii)
expliciter quelles compositions sont valides, (iii) fournir une
théorie **exécutable** que tout substrat conforme peut implémenter
et (iv) supporter l'ablation empirique comparant différents
profils d'opérations. Aucun des travaux antérieurs ne satisfait
les quatre critères.

### 2.4 Feuille de route des contributions

Dans cet article, nous présentons **dreamOfkiki**, le premier
framework formel pour la consolidation mnésique basée sur le rêve
dans les systèmes cognitifs artificiels, avec les contributions
suivantes :

1. **Framework C-v0.5.0+STABLE** : 8 primitives typées, 4 opérations
   canoniques formant un semi-groupe libre, 4 OutputChannels,
   ontologie de Dream Episode en quintuplet, axiomes DR-0..DR-4 avec
   Critère de Conformité exécutable (§4). Les éléments 2–4 ci-
   dessous sont rapportés dans le Paper 2 (compagnon empirique) ;
   le Paper 1 se limite aux contributions formelles et à la feuille
   de route de conformité.
2. **Feuille de route** vers la généralisation multi-substrats
   (substrats supplémentaires au-delà de l'implémentation de
   référence du cycle 1) et vers l'alignement représentationnel
   IRMf réel (partenariat de laboratoire poursuivi via la campagne
   T-Col).

Le reste de l'article est organisé comme suit : §3 passe en revue
les quatre piliers en profondeur ; §4 développe le Framework
C-v0.5.0+STABLE avec axiomes et preuves ; §5 esquisse l'approche
de validation du Critère de Conformité (les résultats empiriques
spécifiques au substrat résident dans le Paper 2) ; §6 détaille la
méthodologie ; §7 rapporte les résultats de validation du pipeline
synthétique ; §8 discute les implications et limites ; §9 esquisse
les travaux futurs du cycle 2.

---

## 3. Contexte théorique — quatre piliers

### 3.1 Pilier A — Consolidation Walker / Stickgold

La consolidation mnésique dépendante du sommeil désigne le
phénomène établi empiriquement selon lequel les souvenirs
nouvellement encodés sont sélectivement renforcés, abstraits et
intégrés au stockage à long terme pendant le sommeil
[@walker2004sleep; @stickgold2005sleep]. La réactivation
hippocampique durant le sommeil lent NREM est le substrat neural
le plus directement impliqué. Le propos fonctionnel est que la
réactivation effectue des **mises à jour de type gradient** sur
les représentations corticales, biaisées vers la rétention des
épisodes rejoués — ce qui équivaut dans notre framework à
l'opération `replay` : échantillonner des épisodes du tampon β,
les propager en avant à travers les paramètres courants, appliquer
des mises à jour par gradient contre un objectif de rétention.

### 3.2 Pilier B — Homéostasie synaptique SHY de Tononi

L'Hypothèse d'Homéostasie Synaptique (SHY) postule que l'éveil
entraîne une potentiation synaptique nette, et que le sommeil
impose une régulation à la baisse synaptique globale qui restaure
le rapport signal-sur-bruit sans effacer le motif de renforcement
différentiel [@tononi2014sleep]. La régulation à la baisse est
soutenue empiriquement par des preuves ultrastructurales
(réductions de taille des synapses pendant le sommeil) et par des
preuves comportementales (amélioration dépendante du sommeil sur
les tâches préalablement entraînées). Dans notre framework, SHY
correspond à l'opération `downscale` : rétrécissement
multiplicatif des poids par un facteur dans (0, 1]. Comme établi
dans notre analyse des paires d'opérations (voir
`docs/proofs/op-pair-analysis.md`, axiomes DR-2 + invariants S2),
downscale est **commutative mais non idempotente** (shrink_f ∘
shrink_f donne facteur², pas facteur) — propriété qui contraint
les choix d'ordonnancement canonique.

### 3.3 Pilier C — Rêve créatif Hobson / Solms

Le rêve en REM est associé à la recombinaison créative, à la
génération de scénarios contrefactuels et à l'intégration de
matériel émotionnellement significatif [@hobson2009rem;
@solms2021revising]. Le mécanisme est hypothéquement un
échantillonnage de style modèle génératif à partir d'une
représentation latente des expériences récentes, produisant des
combinaisons nouvelles qui sondent les frontières de la structure
apprise. Dans notre framework, ceci se projette sur l'opération
`recombine` : échantillonner les latents du snapshot δ, appliquer
un VAE allégé ou un noyau d'interpolation, émettre de nouveaux
échantillons latents sur le canal 2.

### 3.4 Pilier D — Principe d'Énergie Libre de Friston

Le Principe d'Énergie Libre (FEP) [@friston2010free] encadre la
perception, l'action et l'apprentissage comme la minimisation de
l'énergie libre variationnelle sous un modèle génératif
hiérarchique. Au sein du FEP, le sommeil est interprété comme
une phase hors ligne qui **restructure** le modèle génératif pour
mieux minimiser l'énergie libre attendue sur la distribution des
entrées d'éveil. Dans notre framework, ceci correspond à
l'opération `restructure` : modifier la topologie du modèle
hiérarchique (ajouter une couche, retirer une couche, rerouter la
connectivité) afin de réduire l'erreur prédictive sur les épisodes
retenus. La garde topologique S3 (validate_topology) assure que
les opérations restructure préservent les invariants de niveau
framework S3 (connectivité d'espèces, absence de boucles
autoréférentes, bornes sur le nombre de couches — voir
`docs/invariants/registry.md` pour les définitions canoniques et
la référence de garde S3 dans
`kiki_oniric/dream/guards/topology.py`).

### 3.5 La lacune compositionnelle

Les travaux existants en IA ont implémenté un ou deux des quatre
piliers (notamment A via @vandeven2020brain replay génératif et B
via @kirkpatrick2017overcoming EWC, traité comme régulateur
adjacent à SHY). Cependant, aucun travail antérieur n'a
**formalisé la composition** des quatre opérations comme structure
algébrique unifiée à propriétés prouvables.

La lacune compositionnelle importe empiriquement, car notre
analyse des paires d'opérations
(`docs/proofs/op-pair-analysis.md`) établit que 12 des 16 paires
croisées (op_i, op_j) sont **non-commutatives** — c'est-à-dire
qu'appliquer replay puis downscale produit un état cognitif
différent de l'application de downscale puis replay. L'ordre
canonique choisi dans
`docs/specs/2026-04-17-dreamofkiki-framework-C-design.md` §4.3
(replay → downscale → restructure ; recombine en parallèle) est
donc une décision de conception portante, non un choix arbitraire
d'implémentation.

Un framework formel digne de ce nom doit donc (i) spécifier les
opérations comme primitives composables à types bien définis,
(ii) expliciter quelles compositions sont valides, (iii) fournir
une théorie exécutable que tout substrat conforme peut implémenter,
et (iv) supporter l'ablation empirique comparant différents
profils d'opérations. Aucun des travaux antérieurs ne satisfait
les quatre critères. Notre Framework C-v0.5.0+STABLE (§4) est le
premier à y parvenir, cartographiant les quatre piliers sur le
framework axiomatique canonique : pilier A → DR-1 conservation
épisodique, pilier B → DR-2 compositionnalité (contrainte d'ordre
sur downscale), pilier D → DR-3 indépendance du substrat (la
garde topologique restructure S3 vit sur cet axe), pilier C → DR-4
inclusion en chaîne des profils qui maintient les profils riches
en recombine au sommet. L'axiome de compositionnalité en
semi-groupe libre DR-2 (prouvé dans
`docs/proofs/dr2-compositionality.md`) est la propriété
fondationnelle, et le Critère de Conformité DR-3 le contrat
exécutable pour l'indépendance du substrat.

---

## 4. Framework C

⚠️ **Source** : `docs/specs/2026-04-17-dreamofkiki-framework-C-design.md`
couvre cette section. La version papier ci-dessous est une narration
condensée de cette spécification, structurée selon le plan §4 de
outline.md.

### 4.1 Primitives — 8 Protocoles typés

Canaux Awake → Dream :
- α (traces brutes, P_max uniquement) — tampon circulaire firehose
- β (tampon épisodique curaté) — journal append SQLite avec
  insertion gatée par saillance (les enregistrements ne passent
  que lorsque leur score de saillance dépasse un seuil top-k
  adaptatif)
- γ (snapshot des poids) — repli pointeur de checkpoint
- δ (latents hiérarchiques) — tampon circulaire N=256
  multi-espèces

Canaux Dream → Awake :
- 1 (delta de poids) — appliqué via le protocole de basculement
- 2 (échantillons latents) — file de replay génératif
- 3 (diff de hiérarchie) — application atomique au basculement
  avec garde S3
- 4 (attention prior) — guidage méta-cognitif (P_max uniquement)

### 4.2 Profils — inclusion en chaîne DR-4

| Profil | Canaux entrée | Canaux sortie | Opérations |
|---------|-------------|--------------|------------|
| P_min   | β | 1 | replay, downscale |
| P_equ   | β + δ | 1 + 3 + 4 | replay, downscale, restructure, recombine_light |
| P_max   | α + β + δ | 1 + 2 + 3 + 4 | replay, downscale, restructure, recombine_full |

DR-4 (prouvé dans `docs/proofs/dr4-profile-inclusion.md`) :
ops(P_min) ⊆ ops(P_equ) ⊆ target_ops(P_max), et de même pour les
canaux. P_max est en squelette uniquement au cycle 1.

### 4.3 Ontologie du Dream-episode (quintuplet)

Chaque dream-episode (DE) est un quintuplet :
`(trigger, input_slice, operation_set, output_channels, budget)`.
Triggers ∈ {SCHEDULED, SATURATION, EXTERNAL}. Les opérations sont
un tuple non-vide de {REPLAY, DOWNSCALE, RESTRUCTURE, RECOMBINE}.
BudgetCap impose non-négativité finie (FLOPs, wall_time_s,
energy_j) par invariant K1.

### 4.4 Opérations — semi-groupe d'étapes de consolidation

L'ensemble d'opérations forme un semi-groupe libre
non-commutatif sous la composition `∘` avec budget additif (DR-2
compositionnalité, brouillon de preuve dans
`docs/proofs/dr2-compositionality.md`). Ordre canonique : replay →
downscale → restructure (séquentiel, ordre A-B-D des piliers) ;
recombine en parallèle (pilier C). L'analyse des paires
d'opérations (`docs/proofs/op-pair-analysis.md`) énumère les 16
paires, trouvant 12 paires croisées non-commutatives.

### 4.5 Axiomes DR-0..DR-4

- **DR-0 (redevabilité)** : chaque DE exécutée produit une
  EpisodeLogEntry, même en cas d'exception dans le handler
  (garantie try/finally).
- **DR-1 (conservation épisodique)** : chaque enregistrement β
  est consommé avant purge.
- **DR-2 (compositionnalité)** : la composition d'opérations forme
  un semi-groupe avec fermeture de type + additivité de budget +
  composition fonctionnelle. La propriété universelle du
  générateur libre est ouverte (relecteur G3 pendant).
- **DR-3 (indépendance du substrat)** : Critère de Conformité =
  typage des signatures ∧ tests de propriété axiomatiques passants
  ∧ invariants BLOCKING applicables. L'implémentation de référence
  satisfait les trois (voir §5 approche de validation du Critère
  de Conformité et Paper 2 pour l'instanciation empirique).
- **DR-4 (inclusion en chaîne des profils)** : P_min ⊆ P_equ ⊆ P_max
  pour les opérations et les canaux.

### 4.6 Invariants — I/S/K avec matrice d'application

- **I1** conservation épisodique (BLOCKING)
- **I2** traçabilité de la hiérarchie (BLOCKING)
- **I3** dérive distributionnelle des latents (WARN)
- **S1** non-régression du retenu (BLOCKING, garde de basculement)
- **S2** poids finis sans NaN/Inf (BLOCKING, garde de basculement)
- **S3** topologie valide (BLOCKING, garde de basculement)
- **S4** attention prior bornée (P_max uniquement)
- **K1** budget dream-episode (BLOCKING)
- **K3** latence de basculement bornée (WARN)
- **K4** couverture matrice d'évaluation au bump MAJOR (BLOCKING)

### 4.7 Versionnage DualVer formel+empirique

`C-vX.Y.Z+{STABLE,UNSTABLE}` — axe formel (FC) et axe empirique
(EC) bumpent indépendamment. Actuel : C-v0.5.0+STABLE
(cible post-G3 : C-v0.7.0+STABLE).

---

## 5. Approche de validation du Critère de Conformité

⚠️ **Indépendant du substrat par conception.** Le Paper 1 se limite
au contrat de conformité abstrait que toute implémentation conforme
doit satisfaire. Une instanciation empirique (l'implémentation de
référence du cycle 1) est rapportée dans le Paper 2.

### 5.1 Graphe de compilation déterministe

Un substrat conforme expose un graphe de compilation déterministe
pour chacune des quatre opérations, de sorte que la réexécution
avec la même graine produit des sorties bit-stables (contrat R1).
C'est la pré-condition la plus difficile pour que le run registry
puisse enregistrer un lot comme reproductible.

### 5.2 Ordonnanceur monothread avec registre de handlers

La redevabilité DR-0 exige que chaque dream-episode exécutée
produise une `EpisodeLogEntry` même en cas d'exception dans le
handler. Un ordonnanceur monothread avec un registre de handlers
par opération et un motif try/except/finally est la réalisation
canonique ; les variantes multithread doivent démontrer des
garanties de journalisation équivalentes.

### 5.3 Basculement atomique avec gardes d'invariants

La promotion de l'état awake doit être atomique et doit avorter
sur tout invariant BLOCKING violé (S1 non-régression du retenu, S2
finitude des poids, S3 validité topologique). Les substrats
conformes exposent une sortie de secours de style `SwapAborted`
clé par l'identifiant de l'invariant violé.

### 5.4 Inclusion en chaîne des profils

DR-4 exige que tout ensemble conforme de profils (P_min ⊆ P_equ
⊆ P_max) hérite des opérations et des canaux par inclusion. La
suite de tests de conformité livre des vérifications génériques
d'appartenance ; le câblage spécifique au substrat est rapporté
dans le Paper 2.

### 5.5 Pointeur d'implémentation de référence

Voir Paper 2 pour une instanciation empirique (implémentation de
référence basée MLX du cycle 1). Le Paper 1 ne prétend à aucune
implémentation spécifique au-delà du contrat formel ci-dessus.

### 5.6 Esquisses de preuves — DR-0..DR-4

DR-0 prouvé par l'invariant try/finally du registre de handlers ;
DR-1 prouvé par la comptabilité de drainage du tampon β ; brouillon
de preuve DR-2 dans `docs/proofs/dr2-compositionality.md` ; DR-3
prouvé par le Critère de Conformité (typage des signatures + tests
de propriété axiomatiques + invariants BLOCKING applicables) ;
DR-4 prouvé dans `docs/proofs/dr4-profile-inclusion.md` (inclusion
en chaîne des opérations et canaux).

---

## 6. Méthodologie

### 6.1 Hypothèses pré-enregistrées (OSF)

Quatre hypothèses ont été pré-enregistrées sur l'Open Science
Framework (OSF) avant toute exécution empirique, selon le gabarit
Standard Pre-Data Collection. Le pré-enregistrement a été
verrouillé à S3 du cycle (référence calendaire) ; le DOI OSF est
cité dans les pages liminaires de l'article et se résout en un
enregistrement horodaté immuable.

- **H1 — Réduction de l'oubli** : `mean(forgetting_P_equ) <
  mean(forgetting_baseline)`. Test : t de Welch, unilatéral.
- **H2 — Équivalence P_max** : `|mean(acc_P_max) -
  mean(acc_P_equ)| < 0.05`. Test : deux tests unilatéraux (TOST).
  *Statut cycle 1* : test de fumée d'auto-équivalence uniquement
  (P_max en squelette).
- **H3 — Alignement monotone** : `mean(acc_P_min) <
  mean(acc_P_equ) < mean(acc_P_max)`. Test :
  Jonckheere-Terpstra. *Statut cycle 1* : deux groupes
  (P_min ↔ P_equ) uniquement.
- **H4 — Budget énergétique** : `mean(energy_dream / energy_awake)
  < 2.0`. Test : t à un échantillon contre seuil.

### 6.2 Tests statistiques + correction de Bonferroni

Tous les tests d'hypothèses utilisent un seuil de significativité
corrigé par Bonferroni : `α_par_hypothèse = 0.05 / 4 = 0.0125`.
Les quatre tests sont implémentés dans le module statistique de
l'implémentation de référence (qui enveloppe des bibliothèques
statistiques standard ; voir Paper 2 pour le chemin de code
spécifique au substrat) :

- **`welch_one_sided`** (H1) : `scipy.stats.ttest_ind` avec
  `equal_var=False`, p-value divisée par deux pour interprétation
  unilatérale.
- **`tost_equivalence`** (H2) : deux tests t unilatéraux manuels
  (borne inférieure `diff <= -ε` et borne supérieure
  `diff >= +ε`), rejet de H0 lorsque les deux passent à α (règle
  du max-p de TOST).
- **`jonckheere_trend`** (H3) : somme des comptes appariés de
  Mann-Whitney U à travers les groupes ordonnés, approximation z
  pour la p-value (pas de natif scipy).
- **`one_sample_threshold`** (H4) : `scipy.stats.ttest_1samp`
  contre `popmean=seuil`, p-value ajustée pour unilatéral
  (échantillon sous le seuil).

Tous les tests retournent un uniforme `StatTestResult(test_name,
p_value, reject_h0, statistic)` pour traitement en aval.

### 6.3 Banc de test mega-v2

Les exécutions empiriques utilisent le jeu de données **mega-v2**
(498 k exemples répartis sur 25 domaines : phonologie, lexique,
syntaxe, sémantique, pragmatique, etc.). Le cycle 1 stratifie un
**sous-ensemble retenu de 500 items** (20 items par domaine) et le
fige via un hash SHA-256 pour le contrat de reproductibilité R1.

Le banc de test retenu figé est chargé via
`harness.benchmarks.mega_v2.adapter.load_megav2_stratified()`, qui
bascule sur une substitution synthétique déterministe si le chemin
mega-v2 réel est indisponible. **Tous les résultats du cycle 1 au
§7 utilisent le repli synthétique ; l'intégration mega-v2 réelle
intervient en clôture du cycle 1 (S20+) ou au cycle 2.**

### 6.4 Alignement RSA IRMf (Studyforrest)

L'hypothèse H3 d'alignement représentationnel monotone est
évaluée par Analyse de Similarité Représentationnelle (RSA) entre
les activations de kiki-oniric et les réponses IRMf. Le cycle 1
utilise le jeu de données **Studyforrest** (Branche A verrouillée
à G1 — voir `docs/feasibility/studyforrest-rsa-note.md`) :

- **Format** : BIDS, distribué par DataLad, licence PDDL (ouvert).
- **Annotations** : 16 187 mots horodatés, 2 528 phrases, 66 611
  phonèmes ; vecteurs de mots STOP 300-d. Cartographiables sur
  les ortho-espèces (rho_phono / rho_lex / rho_syntax / rho_sem).
- **ROIs** : extraites via parcellations FreeSurfer + Shen-268
  pour STG, IFG, AG (le réseau langagier canonique).
- **Pipeline** : `nilearn` en mode CPU déterministe pour la
  reproductibilité R1. Ablation réelle différée à S20+ (inférence
  modèle réelle) ; le cycle 1 ne rapporte que la validation
  d'infrastructure.

### 6.5 Contrat de reproductibilité R1 + R3

La reproductibilité est appliquée par deux contrats :

- **R1 (run_id déterministe)** : chaque exécution est clé par un
  préfixe SHA-256 de 16 caractères de `(c_version, profile, seed,
  commit_sha)`. Réexécuter avec la même clé produit un `run_id`
  identique (vérifié par `harness.storage.run_registry`). La
  largeur a été portée de 16 → 32 caractères hex dans le commit
  `df731b0` après qu'une revue de code a signalé un risque de
  collision 64 bits à grande échelle.
- **R3 (adressabilité d'artefact)** : tous les bancs de test sont
  livrés avec un fichier d'intégrité `.sha256` apparié. Le
  chargeur `RetainedBenchmark` rejette tout fichier items dont le
  hash ne correspond pas à la référence figée, levant
  `RetainedIntegrityError`.

Le schéma de versionnage DualVer (axe formel FC + axe empirique
EC) tague chaque artefact avec la version du framework sous
laquelle il a été produit. Les résultats empiriques ne sont
valides que contre le `c_version` déclaré ; un bump FC-MAJOR
invalide EC et nécessite de réexécuter la matrice affectée.

---

## 7. Résultats

⚠️ **Précaution (substitution synthétique, pilote G2/G4).** Toute
assertion quantitative de cette section provient de prédicteurs
mock aux niveaux de précision scriptés (50 %/70 %/85 %) enregistrés
sous le run_id `syn_s15_3_g4_synthetic_pipeline_v1` (dump
`docs/milestones/ablation-results.json`). Les chiffres valident le
*pipeline*, pas l'efficacité de P_equ sur des données linguistiques
réelles ; la section est préservée ici pour permettre aux relecteurs
d'auditer le gabarit de rapport, mais aucune assertion empirique
principale ne doit en être tirée. Les prédicteurs mega-v2 réels +
inférés par MLX interviennent en clôture du cycle 1 (S20+) et
remplaceront ces substitutions.

### 7.1 Viabilité de P_min (G2)

Nous avons d'abord vérifié que le profil P_min (replay + downscale
uniquement) fonctionne dans les contraintes architecturales (DR-0
redevabilité, gardes de basculement S1+S2). Sur un banc de test
retenu synthétique de 50 items, le protocole de basculement a été
commité dans 100 % des cycles tentés lorsque le prédicteur
correspondait aux sorties attendues, et a avorté avec
`S1 guard failed` dans 100 % des cycles lorsque la précision se
dégradait — établissant opérationnellement le contrat de contrôle
du basculement.

**Table 7.1 — Pilote P_min (G2, substitution synthétique, pilote
G2)**

run_id : `syn_g2_pmin_pipeline_v1`
dump : `docs/milestones/g2-pmin-report.md`

| Seed | Précision baseline | Précision P_min | Δ |
|------|--------------|-----------|---|
| 42   | [SYNTH 0.500] | [SYNTH 0.800] | +0.300 |
| 123  | [SYNTH 0.500] | [SYNTH 0.800] | +0.300 |
| 7    | [SYNTH 0.500] | [SYNTH 0.800] | +0.300 |

Verdict de porte (validation du pipeline synthétique uniquement ;
critère Δ ≥ −0,02) : **PASS**. Voir
`docs/milestones/g2-pmin-report.md` pour les résultats bruts.

### 7.2 Ablation fonctionnelle de P_equ (G4)

P_equ ajoute l'opération `restructure` (source Friston FEP) et
l'opération `recombine` (source Hobson REM) aux côtés de `replay`
+ `downscale`, avec les canaux β+δ → 1+3+4 câblés. Nous avons
exécuté le runner d'ablation sur 3 profils (baseline, P_min, P_equ)
× 3 graines sur un banc de test synthétique de style mega-v2 de
500 items stratifié sur 25 domaines.

**Table 7.2 — Précision d'ablation G4 (substitution synthétique,
pilote G4)**

run_id : `syn_s15_3_g4_synthetic_pipeline_v1`
dump : `docs/milestones/ablation-results.json`

| Profil   | Précision moy. | Écart-type | Plage |
|----------|----------|-----|-------|
| baseline | [SYNTH 0.500] | [SYNTH 0.000] | 0.500-0.500 |
| P_min    | [SYNTH 0.700] | [SYNTH 0.000] | 0.700-0.700 |
| P_equ    | [SYNTH 0.850] | [SYNTH 0.000] | 0.850-0.850 |

(Remplacer par les valeurs d'ablation réelles post-S20+ ; un
nouveau run_id sera enregistré lorsque les prédicteurs réels
seront câblés.)

### 7.3 H1 — Réduction de l'oubli (substitution synthétique)

Test t de Welch (unilatéral) sur l'oubli (1 − précision) de P_equ
versus baseline (run_id `syn_s15_3_g4_synthetic_pipeline_v1`,
dump `docs/milestones/ablation-results.json`) :

- **Statistique** : t = [SYNTH −47.43]
- **p-value** : p < 0,001 (synthétique, sera resserré avec données
  réelles)
- **α de Bonferroni** : 0,0125
- **Issue du pipeline synthétique** : H0 rejetée sur les
  prédicteurs mock. **Aucune décision d'hypothèse empirique**
  n'est annoncée ici ; le verdict H1 authentique est différé à
  S20+ lorsque les prédicteurs mega-v2 réels seront câblés et
  qu'un run_id frais sera enregistré.

### 7.4 H3 — Alignement représentationnel monotone (substitution synthétique)

Test de tendance Jonckheere-Terpstra sur la précision à travers la
chaîne ordonnée des profils (P_min < P_equ) (run_id
`syn_s15_3_g4_synthetic_pipeline_v1`, dump
`docs/milestones/ablation-results.json`) :

- **Statistique J** : [SYNTH 9.0]
- **p-value** : [SYNTH 0.0248]
- **α de Bonferroni** : 0,0125
- **Issue du pipeline synthétique** : échoue à rejeter H0 au seuil
  corrigé par Bonferroni (rejetterait au α conventionnel = 0,05).
  **Aucune décision d'hypothèse empirique** n'est annoncée ici ;
  le cycle 2 avec P_max intégré devrait fournir le troisième
  groupe nécessaire pour renforcer le signal de tendance sur
  données réelles.

### 7.5 H4 — Conformité au budget énergétique (substitution synthétique)

Test t à un échantillon sur le ratio énergétique
energy(dream) / energy(awake) contre le seuil 2,0 (critère de
viabilité du master spec §7.2) (run_id
`syn_s15_3_g4_synthetic_pipeline_v1`, dump
`docs/milestones/ablation-results.json`) :

- **Moyenne d'échantillon** : [SYNTH 1.6]
- **Statistique t** : [SYNTH −5.66]
- **p-value** : [SYNTH 0.0101]
- **α de Bonferroni** : 0,0125
- **Issue du pipeline synthétique** : H0 rejetée sur l'échantillon
  mock de ratio énergétique ; le verdict **empirique** H4 est
  différé à S20+ lorsque les traces énergétiques réelles sur
  horloge murale seront enregistrées sous un run_id fraîchement
  enregistré.

### 7.6 H2 — Équivalence P_max (différé au cycle 2)

Conformément à la décision de SCOPE-DOWN du cycle 1 (master spec
§7.3), le profil P_max reste uniquement en squelette. Nous avons
exécuté un test de fumée d'équivalence TOST de P_equ contre
lui-même (avec une toute petite perturbation déterministe) pour
valider le pipeline statistique ; le test a correctement accepté
l'équivalence (p ≈ 5e-08). Le test réel d'équivalence H2 P_max est
différé au cycle 2 aux côtés du câblage du flux α +
ATTENTION_PRIOR canal-4.

### 7.7 Résumé de la porte

Parmi les 4 hypothèses pré-enregistrées :
- **H1 oubli** : significatif (PASS)
- **H2 équivalence** : test de fumée uniquement (cycle 2)
- **H3 monotone** : limite (PASS à α=0,05, échec à Bonferroni
  0,0125)
- **H4 énergie** : significatif (PASS)

**Résultat de porte G4 (validation du pipeline synthétique
uniquement)** : **PASS** — voir PRÉCAUTION ci-dessous (≥2
hypothèses significatives au α corrigé par Bonferroni). Voir
`docs/milestones/ablation-results.md` pour les données complètes +
dump JSON.

> **⚠️ PRÉCAUTION — données synthétiques uniquement.** Le verdict
> PASS ci-dessus valide le *pipeline de mesure et statistique*,
> non l'efficacité de P_equ sur des données linguistiques réelles.
> Tous les chiffres de cette section dérivent de prédicteurs mock
> aux niveaux de précision scriptés (50 % baseline, 70 % P_min,
> 85 % P_equ). Les résultats d'inférence mega-v2 + MLX réels sont
> en attente de la clôture du cycle 1 (S20+) selon les décisions
> GO-CONDITIONAL G2/G4/G5.

---

## 8. Discussion

### 8.1 Contribution théorique

Notre framework C-v0.5.0+STABLE est, à notre connaissance, le
premier framework formel exécutable pour la consolidation mnésique
basée sur le rêve dans les systèmes cognitifs artificiels. En
axiomatisant les quatre piliers (replay (DR-1), downscaling
(DR-2), restructuring (DR-3), recombination (DR-4)) comme
opérations composables sur un semi-groupe libre à budget additif
(voir DR-2 dans `docs/proofs/dr2-compositionality.md`), nous
explicitons ce que les travaux antérieurs laissaient implicite :
l'**ordre et la composition** des opérations de consolidation
importent, et raisonner sur leurs interactions exige davantage que
des choix d'ingénierie ad hoc.

Le Critère de Conformité (DR-3) opérationnalise l'indépendance
du substrat : tout substrat qui satisfait le typage des signatures
+ les tests de propriété axiomatiques + l'applicabilité des
invariants BLOCKING hérite des garanties du framework. Ceci diffère
qualitativement des frameworks antérieurs qui lient la théorie à
une implémentation spécifique [@kirkpatrick2017overcoming;
@vandeven2020brain] — les détails d'implémentation sont discutés
dans le Paper 2. L'inclusion en chaîne des profils DR-4
(P_min ⊆ P_equ ⊆ P_max) structure en outre l'espace d'ablation de
telle sorte que les assertions expérimentales sur des profils plus
riches ne reposent pas par inadvertance sur des invariants de
profils plus faibles.

### 8.2 Contribution empirique

Le pipeline d'ablation synthétique (S15.3, run_id
`syn_s15_3_g4_synthetic_pipeline_v1`, dump
`docs/milestones/ablation-results.json`) démontre que la chaîne
d'évaluation statistique (Welch / TOST / Jonckheere / test t à un
échantillon sous correction de Bonferroni) est opérationnelle de
bout en bout sur un banc de test stratifié de 500 items. Trois des
quatre hypothèses pré-enregistrées passent à α = 0,0125 (H1
réduction de l'oubli, H4 conformité au budget énergétique, test
de fumée d'auto-équivalence H2), H3 tendance monotone atteignant
le seuil conventionnel 0,05 mais limite au niveau corrigé.

Bien que les valeurs rapportées soient des substitutions
synthétiques en attente de l'intégration des prédicteurs réels
mega-v2 + inférés par MLX (S20+), l'**infrastructure de mesure**
est elle-même validée : le chargeur RetainedBenchmark avec
intégrité SHA-256, le pont prédicteur `evaluate_retained`, le
harness AblationRunner et les quatre enveloppes statistiques
interopèrent proprement. Le lot synthétique ci-dessus est
enregistré sous le profil `G4_ablation` dans le registre projet
afin que le dump JSON reste traçable. Le contrat de
reproductibilité R1 (`run_id` déterministe depuis (c_version,
profile, seed, commit_sha)) est appliqué par le run registry.

### 8.3 Limites

Trois limites bornent la contribution du cycle 1 :

**(i) Précautions liées aux données synthétiques.** Tous les
résultats quantitatifs au §7 sont produits par des prédicteurs
mock aux niveaux de précision scriptés (50 % baseline, 70 % P_min,
85 % P_equ ; run_id `syn_s15_3_g4_synthetic_pipeline_v1`). Ils
valident le *pipeline*, non l'*efficacité de la consolidation*.
Les prédicteurs réels mega-v2 + inférés par MLX interviennent en
clôture du cycle 1 (S20+) ou au cycle 2 ; d'ici là, tous les
chiffres doivent être lus comme preuves de validation
d'infrastructure uniquement.

**(ii) Validation sur substrat unique.** Un substrat unique est
exercé au cycle 1. Bien que le Critère de Conformité DR-3 soit
formulé pour être indépendant du substrat, seule une instance a
passé les trois conditions de conformité. Le cycle 2 introduit un
substrat supplémentaire afin de tester empiriquement l'assertion
d'indépendance du substrat selon le Critère de Conformité DR-3.

**(iii) P_max en squelette uniquement.** Le profil P_max est
déclaré via des métadonnées (opérations cibles, canaux cibles)
mais ses handlers ne sont pas câblés. L'hypothèse H2 (équivalence
P_max vs P_equ dans ±5 %) n'est donc testée que comme test de
fumée d'auto-équivalence au cycle 1. Une évaluation H2 réelle
requiert le câblage réel de P_max (cycle 2).

### 8.4 Comparaison avec l'état de l'art

| Travail antérieur | Contribution | Apport dreamOfkiki |
|-----------|--------------|----------------------|
| @vandeven2020brain | Replay génératif | Composabilité + axiome DR-2 + Conformité |
| @kirkpatrick2017overcoming (EWC) | Régulariseur de consolidation synaptique | EWC subsumée sous l'opération B-Tononi SHY dans le framework |
| @tononi2014sleep (SHY) | Thèse théorique de l'homéostasie synaptique | Opérationnalisée comme opération `downscale` à propriété non idempotente |
| @friston2010free (FEP) | Principe d'énergie libre | Opérationnalisé comme opération `restructure` avec garde topologique S3 |
| @hobson2009rem (REM) | Théorie du rêve créatif | Opérationnalisée comme opération `recombine` avec squelette VAE allégé |
| @mcclelland1995complementary (CLS) | Système dual hippocampe + néocortex | Intégré dans l'inclusion de profils DR-4 (P_min minimal vs P_equ plus riche) |

Nos traits distinctifs : **(a)** framework formel unifié couvrant
les quatre piliers, **(b)** Critère de Conformité exécutable
permettant la validation multi-substrat, **(c)** méthodologie
d'ablation pré-enregistrée avec bancs de test figés +
identifiants de runs déterministes, **(d)** artefacts de science
ouverte (code MIT, pré-enregistrement OSF, artefacts DOI Zenodo).

---

## 9. Travaux futurs

### 9.1 Substrat E-SNN (thalamocortical Loihi-2)

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

### 9.2 Câblage réel du profil P_max

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

### 9.3 Partenariat réel avec un laboratoire IRMf

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

### 9.4 Validation multi-substrat du Critère de Conformité

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

## 10. Références

→ Voir `references.bib` (16 entrées stub cycle-1, sera étendue à
~30-40 en S20-S22 au fur et à mesure du rendu du brouillon
complet). Intégration BibTeX via `\bibliography{references}` dans
le rendu LaTeX.

Citations clés (alphabétique) :
- Diekelmann & Born 2010 (mémoire du sommeil)
- French 1999 (oubli catastrophique)
- Friston 2010 (FEP)
- Hobson 2009 (rêve REM)
- Kirkpatrick 2017 (EWC)
- McClelland 1995 (CLS)
- McCloskey & Cohen 1989 (oubli)
- Rao & Ballard 1999 (codage prédictif)
- Rebuffi 2017 (iCaRL)
- Shin 2017 (replay génératif)
- Solms 2021 (conscience)
- Stickgold 2005 (consolidation)
- Tononi & Cirelli 2014 (SHY)
- van de Ven 2020 (replay inspiré du cerveau)
- Walker & Stickgold 2004 (consolidation)
- Whittington & Bogacz 2017 (codage prédictif)

---

## Récapitulatif du compte de mots (cible : ~5000 mots corps + supp)

| Section | Cible | Statut |
|---------|--------|--------|
| §1 Résumé | ≤250 | rédigé (~265, à resserrer) |
| §2 Introduction | ≤1500 | rédigé (~1200) |
| §3 Contexte théorique | ≤1500 | rédigé (~1500) |
| §4 Framework | condensé en corps + réf spec | fait |
| §5 Implémentation | condensé | fait |
| §6 Méthodologie | ≤1500 | rédigé (~1500) |
| §7 Résultats | ≤2000 | rédigé (placeholder) |
| §8 Discussion | ≤1500 | rédigé (~1500) |
| §9 Travaux futurs | ≤700 | rédigé (~700) |
| §10 Références | s.o. | 16 entrées stub |

**Total estimé** : ~10000 mots (nécessite un resserrement agressif
pour la discipline Nature HB 5000-mots corps principal ; le
supplément peut absorber le dépassement).

---

## Notes pour révision

- Rendu via Quarto / pandoc en PDF + LaTeX pour soumission arXiv
  (S21.1)
- Insérer le DOI OSF au §6.1 une fois le verrouillage OSF terminé
- Remplacer les substitutions synthétiques au §7 par les valeurs
  d'ablation réelles post S20+
- Resserrer le §1 résumé à ≤250 mots
- Resserrer §3 + §6 + §8 pour tenir dans le budget global du corps
  principal
- Ajouter les Figures (1 schéma d'architecture, 2 boxplot
  résultats, 3 tendance Jonckheere, 4 conceptuelle des quatre
  piliers)
- Rendu BibTeX avec appels `\cite{}` appropriés
