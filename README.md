# Blitzkugel

Desktop-Anwendung (Python, PyQt6) für das **Blitzkugelverfahren nach DIN EN 62305-3
(VDE 0185-305-3)**: Eine Blitzkugel wird rechnerisch über ein 3D-Modell aus einer
STEP-Datei gerollt. Alle Punkte, die sie berührt (mögliche Einschlagpunkte), werden
in einer 3D-Ansicht markiert und lassen sich als PNG und CSV exportieren.

## Installation

Empfohlen ist **Python 3.11** (oder 3.12). Für neuere Python-Versionen gibt es
`open3d` bzw. `cadquery-ocp` unter Umständen noch nicht als fertiges Paket; dann
scheitert `pip install`.

### pip (empfohlen)

macOS / Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows (PowerShell oder Eingabeaufforderung):

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

`cadquery-ocp` (OpenCascade) wird hier über pip installiert. Dafür ist kein conda nötig.

**Windows:**
- Python von python.org installieren; der Launcher `py` ist dann vorhanden.
  `py --list` zeigt die installierten Versionen.
- Blockiert PowerShell das Aktivierungsskript, einmalig
  `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` ausführen oder die
  Eingabeaufforderung (`cmd`) verwenden.
- Die 3D-Ansicht benötigt OpenGL 3.2. In Remote-Desktop-Sitzungen oder virtuellen
  Maschinen ohne Grafiktreiber bleibt sie unter Umständen schwarz.
- VS Code erkennt das `.venv` im Projektordner automatisch als Interpreter.

**macOS:**
- open3d ab 0.19 benötigt `brew install libusb`. Ohne lauffähiges open3d rechnet
  das Programm mit einer exakten, aber etwa 100–500× langsameren NumPy-Variante.
  Die Statusleiste zeigt an, welches Backend aktiv ist.
- Liegt das Projekt in einem iCloud-synchronisierten Ordner (z. B. `~/Documents`),
  das venv als `.venv.nosync` anlegen und `.venv` darauf verlinken. Sonst erzeugt
  iCloud Konfliktkopien („… 2“) und markiert Dateien als versteckt; Qt findet dann
  sein Plattform-Plugin `cocoa` nicht.
  ```bash
  python3.11 -m venv .venv.nosync && ln -s .venv.nosync .venv
  ```

### conda

```bash
conda env create -f environment.yml
conda activate blitzkugel
```

## Start

```bash
python -m blitzkugel                 # leeres Fenster
python -m blitzkugel modell.step     # Datei direkt laden
python examples/make_demo_step.py    # erzeugt demo_gebaeude.step zum Ausprobieren
```

Tests: `python -m pytest`

## Bedienung

1. **Modell**: STEP-Datei öffnen (oder *Demo*).
   - Die Einheit wird aus der Datei gelesen, intern wird in Metern gerechnet.
   - Bei falscher Einheitenangabe in der Datei kann sie manuell überschrieben werden.
   - *Lineare Abweichung* und *Winkelabweichung* steuern die Tessellierung.
   - Die Bodenhöhe ist standardmäßig das z-Minimum des Modells.
2. **Blitzkugel**: Schutzklasse I–IV (D = 40/60/90/120 m) oder freien Durchmesser
   wählen. Die *Rasterweite* bestimmt die Rechengenauigkeit.
3. **Verfahren** wählen und *Berechnen* (läuft im Hintergrund, abbrechbar).
   - **A – Richtungen**: Die Kugel rollt in parallelen vertikalen Ebenen je
     gewählter Richtung (0° = +x, gegen den Uhrzeigersinn) über Boden und Struktur.
   - **B – 3D-Hülle**: Alle Kugellagen im Raum werden ausgewertet.
   - **C – 3D adaptiv**: Ergebnis wie B, aber zuerst mit grober Rasterweite; nur
     mögliche Kontaktbereiche werden stufenweise bis zur Rasterweite verfeinert,
     an Ecken, Kanten und ebenen Flächen direkt ausgewertet. Damit sind feine
     Raster bis 1 cm auch bei großen Kugeln in Sekunden möglich.
4. **Darstellung**:
   - Farbe je Richtung und Legende
   - Heatmap: Anzahl der Kugellagen, die einen Bereich berühren. Kanten und Ecken
     sind am stärksten exponiert.
   - Kugeln als Drahtgitter
   - Standardansichten (Iso perspektivisch, die übrigen in Parallelprojektion;
     *Vorne* = Blick in +y-Richtung)
5. **Export**:
   - PNG der aktuellen Ansicht oder aller sechs Standardansichten, in wählbarer
     Auflösung, mit Legende und Kugeldaten
   - CSV mit `x_m;y_m;z_m;richtung;dreieck_id`, Trennzeichen `;`, wahlweise mit
     Dezimalkomma (Standard, für deutsches Excel) oder Dezimalpunkt

## Verfahren

1. Regelmäßiges Raster möglicher Kugelmittelpunkte (Abstand *h*) um das Modell,
   beginnend bei z = Boden + r
2. Ungerichteter Abstand *d* jedes Mittelpunkts zum Dreiecksnetz (open3d-BVH)
3. Gültig, wenn d ≥ r (keine Durchdringung). Der Boden ist durch z ≥ Boden + r
   erfüllt.
4. Nur gültige Lagen, die mit der Oberseite des Rasters zusammenhängen, zählen
   (Flood-Fill). Kugeln in geschlossenen Hohlräumen oder zu engen Spalten fallen weg.
   - Modus B: 3D-Nachbarschaft
   - Modus A: Nachbarschaft nur innerhalb der jeweiligen Rollebene
5. Lagen mit d < r + Toleranz sind Kontaktlagen. Ihr nächster Netzpunkt ist ein
   Einschlagpunkt.

Die Tests prüfen unter anderem:

- Würfel 10 m / r = 20 m: Kontakt nur an Dachfläche und Oberkanten
- Turm 60 m: Seiteneinschläge erst ab Höhe r
- Zwei Fangstangen: Eindringtiefe p = r − √(r² − (d/2)²)
- Geschlossener Hohlraum wird ignoriert
- Innenhof wird nur erreicht, wenn die Kugel hineinpasst
- Modus C liefert ohne Abkürzung exakt dieselben Kontaktlagen wie Modus B,
  mit Abkürzung dieselben Einschlagbereiche (bis auf eine Rasterweite)

### Modus C (adaptiv)

Der Raum wird in Würfel der groben Rasterweite zerlegt (aufgerundet auf
Rasterweite · 2ⁿ, automatisch etwa r/4). Je Würfel wird nur der Abstand d seines
Mittelpunkts berechnet. Da sich der Abstand höchstens so stark ändert wie die
Entfernung, liegen alle Abstände im Würfel (halbe Diagonale e) zwischen d − e und
d + e:

- d − e ≥ r + Toleranz: Würfel **frei**, keine Durchdringung und keine Berührung
- d + e < r: Würfel **blockiert**
- sonst: in 8 Teilwürfel halber Größe teilen und auf der nächsten Stufe prüfen

Bevor ein Würfel geteilt wird, prüft das Verfahren, ob seine nächste Geometrie ein
einzelner **Eckpunkt, eine Kante oder eine ebene Fläche** ist. Dazu werden die
nächsten Modellpunkte seiner 8 Ecken und seiner Mitte bestimmt. Liegen sie auf
einem Punkt, einer Geraden oder einer Ebene (Abweichung ≤ h/2) und erklärt dieses
Element alle 9 Abstände, wird der Würfel nicht weiter geteilt. Die Kugellagen
werden dann direkt im Abstand r vom Element gesetzt: eine für einen Eckpunkt, eine
je Rasterweite entlang einer Kante, ein flaches Raster für eine Fläche.

Große Kugeln liegen fast überall auf Ecken und Kanten. Ohne diese Abkürzung
wächst der Aufwand mit der Hüllfläche der Kugelmittelpunkte, also etwa mit r²;
mit ihr nur noch mit der tatsächlich berührten Modelloberfläche. Damit sind 1 cm
Rasterweite auch bei Schutzklasse IV möglich.

Demo-Gebäude, 1 cm Rasterweite:

| Kugelradius | Abstandsberechnungen | Einschlagpunkte | Rechenzeit |
| --- | --- | --- | --- |
| 60 m | 2,9 Mio. | 0,64 Mio. | 1,5 s |
| 20 m | 14 Mio. | 5,1 Mio. | 8 s |
| 10 m | 23 Mio. | 9,5 Mio. | 12 s |

Das volle feine Raster hätte etwa 1 Mrd. Punkte.

Freie Würfel aller Größen, die Kanten- und Flächenwürfel und die gültigen feinen
Rasterpunkte werden über gemeinsame Seitenflächen zu einem Graphen verbunden.
Erreichbar ist, was mit der Oberseite zusammenhängt.

Abweichungen von Modus B:

- Die Einschlagpunkte stimmen bis auf etwa eine Rasterweite überein.
- Eine Erhebung, die zwischen den 9 Prüfpunkten eines Würfels verborgen liegt, wird
  nur übersehen, wenn sie weniger als die Kontakttoleranz herausragt. Dafür ist die
  Würfelgröße für diese Abkürzung auf √(8·r·Toleranz) begrenzt.
- Die Kugellagen liegen nicht mehr alle auf dem feinen Raster. Für die Heatmap
  werden sie entsprechend gewichtet.

Mit `SphereParams(feature_blocks=False)` rechnet Modus C exakt wie Modus B (ohne
diese Abkürzung); die Tests prüfen das.

## Grenzen

- **Rasterauflösung**: Einschlagbereiche sind nur auf etwa *h* genau. Die
  Toleranz (Standard: halbe Zellendiagonale) kann in Innenecken Punkte liefern,
  an die eine exakte Kugel nicht ganz heranreicht. Spalte, die schmaler als
  etwa *h* sind, werden nicht sicher aufgelöst. Kleinere Rasterweite ist genauer,
  kostet aber mehr: halbe Rasterweite ergibt in Modus B etwa 8-fache Punktzahl.
- **Tessellierung**: Gekrümmte Flächen werden durch Dreiecke angenähert (Fehler
  ≤ lineare Abweichung). Sehr dünne Teile wie Fangstangen und Seile brauchen eine
  ausreichend feine Tessellierung.
- **Modus A**: θ und θ + 180° ergeben dieselben Berührpunkte. Gegenrichtungen
  werden nicht neu berechnet, sondern übernommen. Überlagerte Punkte verdecken
  sich in der Darstellung.
- Gelände wird nur als horizontale Bodenebene berücksichtigt. Umgebungsbebauung muss
  im STEP-Modell enthalten sein.
- Das Programm ist eine Rechenhilfe und ersetzt nicht die Planung und Prüfung durch
  eine Blitzschutz-Fachkraft.

## Struktur

```
blitzkugel/
  core/          Berechnung ohne GUI (mesh, distance, rolling_sphere, progress)
  geometry/      STEP-Import und Tessellierung (OCP)
  export/        CSV-Export
  viz/           pyvista-Szene, Standardansichten, Off-Screen-Bildexport
  gui/           PyQt6-Hauptfenster, QThread-Worker
tests/           pytest
examples/        Demo-STEP-Generator
```
