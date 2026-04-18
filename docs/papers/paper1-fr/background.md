# Contexte théorique — quatre piliers (Paper 1, brouillon S20.2)

**Longueur cible** : ~1,5 page markdown (≈ 1500 mots)

---

## 3.1 Pilier A — Consolidation Walker / Stickgold

La consolidation mnésique dépendante du sommeil désigne le
phénomène établi empiriquement selon lequel les souvenirs
nouvellement encodés sont sélectivement renforcés, abstraits et
intégrés au stockage à long terme pendant le sommeil [Walker &
Stickgold 2004, Stickgold 2005]. La réactivation hippocampique
durant le sommeil lent NREM est le substrat neural le plus
directement impliqué. Le propos fonctionnel est que la réactivation
effectue des **mises à jour de type gradient** sur les
représentations corticales, biaisées vers la rétention des épisodes
rejoués — ce qui équivaut dans notre framework à l'opération
`replay` : échantillonner des épisodes du tampon β, les propager
en avant à travers les paramètres courants, appliquer des mises à
jour par gradient contre un objectif de rétention.

## 3.2 Pilier B — Homéostasie synaptique SHY de Tononi

L'Hypothèse d'Homéostasie Synaptique (SHY) postule que l'éveil
entraîne une potentiation synaptique nette, et que le sommeil
impose une régulation à la baisse synaptique globale qui restaure
le rapport signal-sur-bruit sans effacer le motif de renforcement
différentiel [Tononi & Cirelli 2014]. La régulation à la baisse
est soutenue empiriquement par des preuves ultrastructurales
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

## 3.3 Pilier C — Rêve créatif Hobson / Solms

Le rêve en REM est associé à la recombinaison créative, à la
génération de scénarios contrefactuels et à l'intégration de
matériel émotionnellement significatif [Hobson 2009, Solms 2021].
Le mécanisme est hypothéquement un échantillonnage de style
modèle génératif à partir d'une représentation latente des
expériences récentes, produisant des combinaisons nouvelles qui
sondent les frontières de la structure apprise. Dans notre
framework, ceci se projette sur l'opération `recombine` :
échantillonner les latents du snapshot δ, appliquer un VAE allégé
ou un noyau d'interpolation, émettre de nouveaux échantillons
latents sur le canal 2.

## 3.4 Pilier D — Principe d'Énergie Libre de Friston

Le Principe d'Énergie Libre (FEP) [Friston 2010] encadre la
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

## 3.5 La lacune compositionnelle

Les travaux existants en IA ont implémenté un ou deux des quatre
piliers (notamment A via van de Ven 2020 replay génératif et B
via Kirkpatrick 2017 EWC, traité comme régulateur adjacent à SHY).
Cependant, aucun travail antérieur n'a **formalisé la composition**
des quatre opérations comme structure algébrique unifiée à
propriétés prouvables.

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

## Notes pour révision

- Insérer les citations bibtex appropriées (S19.3 references.bib)
  en utilisant `\cite{walker2004sleep}` etc. une fois le brouillon
  complet rendu
- Resserrer §3.5 à ~300 mots pour la discipline du corps principal
  Nature HB
- Ajouter une figure supplémentaire de Contexte théorique :
  schéma conceptuel des quatre piliers avec leurs correspondances
  opérationnelles dreamOfkiki
