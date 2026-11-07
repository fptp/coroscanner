"""CoroSeg — segmentation et analyse des artères coronaires sur coroscanner.

Outil de recherche / pédagogie. NON destiné au diagnostic clinique.
"""

__version__ = "0.1.0"

# Étiquettes de la carte de labels produite par le post-traitement
LABELS = {
    1: ("TC", "Tronc commun", (255, 196, 0)),
    2: ("IVA", "Interventriculaire antérieure", (230, 40, 40)),
    3: ("Cx", "Circonflexe", (40, 110, 240)),
    4: ("CD", "Coronaire droite", (30, 180, 70)),
    5: ("Diag", "Diagonales", (255, 140, 120)),
    6: ("Marg", "Marginales", (120, 190, 255)),
    7: ("Br. CD", "Branches de la CD", (150, 230, 140)),
    8: ("Bissec.", "Branche intermédiaire (bissectrice)", (180, 100, 220)),
    9: ("?", "Non attribué", (160, 160, 160)),
}
