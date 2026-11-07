# CoroSeg ↔ Orthanc : segmentation automatique des coroscanners

Un **worker** tourne sur la VM à côté d'Orthanc. Quand un coroscanner arrive dans Orthanc, il :

1. choisit la meilleure série CT de l'étude : coupes les plus fines (≤ 1,5 mm), au moins 80 coupes, en excluant
   score calcique, topogramme, bolus, MIP/MPR… Si la série contient plusieurs phases, il garde la plus proche de 75 % ;
2. la télécharge, lance **TotalSegmentator** (`coronary_arteries` + aorte), puis l'étiquetage CoroSeg
   (TC, IVA, Cx, CD, branches) ;
3. **renvoie les résultats dans la même étude Orthanc** :

| Série | Contenu | Visible dans |
|---|---|---|
| 9001 `CoroSeg - SEG coronaires` | **DICOM SEG**, un segment coloré par vaisseau, superposé au scanner | OHIF, Stone Web Viewer, 3D Slicer, Weasis… |
| 9002 `CoroSeg - fusion coronaires` | coupes axiales avec les coronaires colorées (RGB) | **toutes** les visionneuses / PACS |
| 9003 `CoroSeg - CPR` | une image par vaisseau principal : CPR redressée à 0/45/90/135° + profil de calibre | toutes |
| 9004 `CoroSeg - rapport` | tableau longueurs / calibres / rétrécissement max, avertissements, vue antérieure de l'arbre | toutes |

Il pose ensuite le label `coroseg-ok` (ou `coroseg-erreur`) sur l'étude. Les fichiers NIfTI, STL et `rapport.json` restent
sur la VM dans `/data/coroseg/<étude>/<série>/resultats/`.

> ⚠️ Outil de recherche, non marqué CE : chaque image envoyée porte la mention « non validé pour le diagnostic ».

## Installation sur la VM (Docker, recommandé)

Prérequis : Docker. Pour le GPU : pilote NVIDIA + `nvidia-container-toolkit`. Sans GPU, ça marche aussi, mais compter
plusieurs minutes par examen.

```bash
cd coroseg
cp deploy/coroseg.env.example deploy/coroseg.env
nano deploy/coroseg.env          # ORTHANC_URL, identifiants, TOTALSEG_LICENSE=aca_..., COROSEG_DEVICE
docker compose -f deploy/docker-compose.yml up -d --build
docker compose -f deploy/docker-compose.yml logs -f
```

VM **sans GPU** : dans `deploy/docker-compose.yml`, supprimer le bloc `deploy:`, décommenter la ligne `args` (index PyTorch
CPU) et mettre `COROSEG_DEVICE=cpu`.

Au premier examen, TotalSegmentator télécharge ses poids (quelques centaines de Mo, conservés dans le volume `/data`).
La VM doit donc pouvoir joindre Internet, au moins à ce moment-là.

### Tester sans examen ni licence

```bash
docker compose -f deploy/docker-compose.yml run --rm coroseg demo
```

Cette commande envoie un fantôme synthétique dans Orthanc et le traite (segmentation simulée). L'étude
« FANTOME^CoroSeg » doit apparaître avec les 4 séries ci-dessus.

## Déclenchement

- `COROSEG_TRIGGER=auto` (défaut) : toute étude **stable** (aucune image reçue depuis `StableAge` secondes, 60 s par défaut
  dans Orthanc) qui contient une série CT éligible est traitée. Pour limiter aux coroscanners, utiliser par exemple
  `COROSEG_STUDY_REGEX=coro|cardi`.
- `COROSEG_TRIGGER=label` : seules les études auxquelles on ajoute le label **`coroseg`** sont traitées (dans
  Orthanc Explorer 2 ou via l'API).
- Dans les deux modes, **pour relancer une étude**, il suffit de lui ajouter le label `coroseg`. Par l'API :

```bash
curl -X PUT -u user:mdp http://localhost:8042/studies/<ID>/labels/coroseg
```

- En ligne de commande :

```bash
docker compose -f deploy/docker-compose.yml run --rm coroseg series <ID étude>
```

```bash
docker compose -f deploy/docker-compose.yml run --rm coroseg process <ID étude> --serie <ID série>
```

`series` montre la série qui serait choisie. `process … --serie` force une série précise.

Pour renvoyer aussi les résultats au PACS : `COROSEG_FORWARD_MODALITY=<nom de la modalité déclarée dans Orthanc>`.

## Sans Docker (venv + systemd)

```bash
sudo mkdir -p /opt/coroseg && sudo cp -r . /opt/coroseg && cd /opt/coroseg
python3 -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu124   # ou .../whl/cpu
pip install -r requirements-orthanc.txt
totalseg_set_license -l aca_XXXXXXXXXXXX
cp deploy/coroseg.env.example deploy/coroseg.env && nano deploy/coroseg.env
sudo cp deploy/coroseg-orthanc.service /etc/systemd/system/ && sudo systemctl enable --now coroseg-orthanc
```

## Code

```
coroseg_orthanc/config.py        variables d'environnement
coroseg_orthanc/orthanc.py       client REST Orthanc
coroseg_orthanc/worker.py        surveillance (/changes, labels), choix de série, traitement, envoi
coroseg_orthanc/dicom_export.py  DICOM SEG (highdicom), séries fusion / CPR / rapport (Secondary Capture)
coroseg_orthanc/demo.py          fantôme exporté en DICOM pour les tests
tests/test_orthanc_bridge.py     aller-retour DICOM (alignement SEG ↔ coupes source)
```
