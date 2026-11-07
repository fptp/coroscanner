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
| 9005 `CoroSeg - lignes centrales` | DICOM SR 3D : une polyligne par vaisseau (coordonnées patient LPS, 1 point/mm, de l'ostium vers la distalité) + rayons du masque ; codes `99COROSEG` `CL-ROOT`, `CL-VESSEL`, `CL-NAME`, `CL-LABEL`, `CL-LINE`, `CL-RADIUS`, `CL-LENGTH` | viewer (CPR interactive) |

Il pose ensuite le label `coroseg-ok` (ou `coroseg-erreur`) sur l'étude. Les fichiers NIfTI, STL et `rapport.json` restent
sur la VM dans `/data/coroseg/<étude>/<série>/resultats/`.

> ⚠️ Outil de recherche, non marqué CE : chaque image envoyée porte la mention « non validé pour le diagnostic ».

## Installation sur la VM (sans Docker, recommandé)

Même principe que SegBrainMRI : un environnement Python dans `/opt/coroseg`, un utilisateur `coroseg`, la configuration
dans `/etc/coroseg/coroseg.env` et un service systemd `coroseg-orthanc`.

```bash
cd ~/coroscanner && git pull
./deploy/install.sh              # VM sans GPU : PyTorch CPU
nano /etc/coroseg/coroseg.env    # ORTHANC_URL, ORTHANC_USER, ORTHANC_PASSWORD, TOTALSEG_LICENSE
./deploy/install.sh              # relancer : démarre le service et teste la connexion à Orthanc
journalctl -u coroseg-orthanc -f
```

Dans `coroseg.env`, ne mettez pas de guillemets autour des valeurs, même si le mot de passe contient un `!`.

### CPU aujourd'hui, GPU demain

`COROSEG_DEVICE=auto` (défaut) : le worker utilise le GPU si PyTorch en voit un, sinon le CPU. Le jour où le GPU
arrive, il suffit d'installer le pilote NVIDIA (`nvidia-smi` doit répondre), puis :

```bash
./deploy/install.sh --gpu
```

Rien à changer dans le code ni dans la configuration. Sur CPU, compter 5 à 15 minutes par examen. Pour limiter la charge
de la VM, régler `COROSEG_THREADS`.

### Tester sans examen ni licence

```bash
coroseg check     # connexion à Orthanc, CPU ou GPU, licence
coroseg demo
```

`coroseg demo` envoie un fantôme synthétique dans Orthanc et le traite
(segmentation simulée, sans TotalSegmentator). L'étude « FANTOME^CoroSeg » doit apparaître avec les 4 séries
ci-dessus.

Au premier vrai examen, TotalSegmentator télécharge ses poids (quelques centaines de Mo, dans
`/var/lib/coroseg/totalseg`). La VM doit donc pouvoir joindre Internet, au moins à ce moment-là.

### Variante Docker

`deploy/Dockerfile` et `deploy/docker-compose.yml` restent disponibles si Docker est installé :

```bash
cp deploy/coroseg.env.example deploy/coroseg.env
docker compose -f deploy/docker-compose.yml up -d --build
```

## Déclenchement

### Envoi manuel en DICOM (`COROSEG_TRIGGER=dicom`)

CoroSeg ne surveille plus Orthanc : il écoute en DICOM, comme SegBrainMRI, et ne traite que ce qu'on lui envoie.

| | |
|---|---|
| AET | `COROSEG` |
| Hôte | 192.168.10.145 (VM IA) |
| Port | `11113` (11112 est déjà pris par SegBrainMRI) |

Dans Orthanc (192.168.10.172), déclarer la modalité, par exemple dans le fichier de configuration :

```json
"DicomModalities": {
  "coroseg": { "AET": "COROSEG", "Host": "192.168.10.145", "Port": 11113 }
}
```

Ensuite, dans Orthanc, ouvrir l'examen, « Envoyer vers une modalité DICOM », choisir `coroseg`. On peut envoyer
l'étude entière : CoroSeg choisit lui-même la bonne série CT. Le traitement démarre 20 s après la dernière image
reçue. Les résultats reviennent dans Orthanc par l'API REST (ORTHANC_URL et identifiants toujours nécessaires) et
l'étude reçoit le label `coroseg-ok`. En cas d'échec, les images reçues et `erreur.txt` sont gardées dans
`/var/lib/coroseg/echecs/`.

### Surveillance d'Orthanc (`auto` / `label`)


- `COROSEG_TRIGGER=auto` (défaut) : toute étude **stable** (aucune image reçue depuis `StableAge` secondes, 60 s par défaut
  dans Orthanc) qui contient une série CT éligible est traitée. Pour limiter aux coroscanners : `COROSEG_STUDY_REGEX=c(oe|œ)ur`
  (description d'étude contenant « coeur » ou « cœur », majuscules ou non). Les autres études sont ignorées
  sans label ; le label `coroseg` force quand même le traitement.
- `COROSEG_TRIGGER=label` : seules les études auxquelles on ajoute le label **`coroseg`** sont traitées (dans
  Orthanc Explorer 2 ou via l'API).
- Dans les deux modes, **pour relancer une étude**, il suffit de lui ajouter le label `coroseg`. Par l'API :

```bash
curl -X PUT -u user:mdp http://localhost:8042/studies/<ID>/labels/coroseg
```

- En ligne de commande :

```bash
coroseg series <ID étude>
```

```bash
coroseg process <ID étude> --serie <ID série>
```

`series` montre la série qui serait choisie. `process … --serie` force une série précise.

Pour renvoyer aussi les résultats au PACS : `COROSEG_FORWARD_MODALITY=<nom de la modalité déclarée dans Orthanc>`.

## Code

```
coroseg_orthanc/config.py        variables d'environnement
coroseg_orthanc/device.py        choix CPU / GPU (auto)
coroseg_orthanc/orthanc.py       client REST Orthanc
coroseg_orthanc/worker.py        surveillance (/changes, labels), choix de série, traitement, envoi
coroseg_orthanc/dicom_export.py  DICOM SEG (highdicom), séries fusion / CPR / rapport (Secondary Capture)
coroseg_orthanc/demo.py          fantôme exporté en DICOM pour les tests
tests/test_orthanc_bridge.py     aller-retour DICOM (alignement SEG ↔ coupes source)
```
