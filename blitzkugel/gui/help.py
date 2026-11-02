"""Help texts for the settings and the small "?" button that shows them."""

from __future__ import annotations

from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtWidgets import QHBoxLayout, QLayout, QToolButton, QWhatsThis, QWidget

HELP_TEXTS: dict[str, str] = {
    "unit": (
        "<b>Einheit</b><br>Einheit der Koordinaten in der STEP-Datei. <i>Automatisch</i> übernimmt "
        "die Angabe aus der Datei. Nur ändern, wenn das Modell offensichtlich falsch skaliert ist, "
        "z. B. ein Gebäude mit 30 km Länge. Intern wird immer in Metern gerechnet.<br>"
        "Wirkt nach <i>Neu tessellieren</i>."
    ),
    "lin_defl": (
        "<b>Lineare Abweichung</b><br>Die CAD-Flächen werden für die Berechnung in Dreiecke zerlegt "
        "(Tessellierung). Dieser Wert ist der größte erlaubte Abstand zwischen exakter Fläche und "
        "Dreiecksnetz.<br>Kleinere Werte bilden Rundungen, Rohre und dünne Fangstangen genauer ab, "
        "erzeugen aber mehr Dreiecke. Sinnvoll ist ein Wert kleiner als die Rasterweite.<br>"
        "Wirkt nach <i>Neu tessellieren</i>."
    ),
    "ang_defl": (
        "<b>Winkelabweichung</b><br>Größter Winkel zwischen benachbarten Dreiecken auf gekrümmten "
        "Flächen. Kleinere Werte ergeben rundere Zylinder und Kuppeln, unabhängig von deren Größe.<br>"
        "Wirkt nach <i>Neu tessellieren</i>."
    ),
    "ground": (
        "<b>Bodenhöhe</b><br>Höhe der Erdoberfläche. Die Blitzkugel kann den Boden berühren, aber "
        "nicht in ihn eindringen: Ihr Mittelpunkt liegt immer mindestens r über dem Boden.<br>"
        "Standard ist die tiefste Stelle des Modells (<i>= Modell-Min.</i>). Höher setzen, wenn das "
        "Modell z. B. Fundamente oder Keller unter Geländeniveau enthält."
    ),
    "class": (
        "<b>Schutzklasse</b><br>Die Blitzschutzklasse nach DIN EN 62305-3 legt die Größe der "
        "Blitzkugel fest:<br>"
        "I: D = 40 m (r = 20 m)<br>II: D = 60 m (r = 30 m)<br>"
        "III: D = 90 m (r = 45 m)<br>IV: D = 120 m (r = 60 m)<br>"
        "Klasse I ist am strengsten: Die kleinere Kugel dringt tiefer zwischen Fangeinrichtungen "
        "ein und erreicht mehr Stellen. <i>benutzerdefiniert</i> erlaubt einen freien Durchmesser."
    ),
    "diameter": (
        "<b>Durchmesser</b><br>Durchmesser D der Blitzkugel, der Radius ist r = D/2. Wird durch die "
        "Schutzklasse gesetzt; ein anderer Wert schaltet die Schutzklasse auf "
        "<i>benutzerdefiniert</i>."
    ),
    "grid": (
        "<b>Rasterweite</b><br>Abstand der untersuchten Kugelmittelpunkte und damit die "
        "Rechengenauigkeit: Einschlagbereiche und Eindringtiefen stimmen auf etwa eine Rasterweite. "
        "Spalte, die schmaler als etwa die Rasterweite sind, werden nicht sicher erkannt.<br>"
        "Aufwand bei halber Rasterweite: Verfahren A und B etwa 8-fach, Verfahren C etwa 4-fach. "
        "Für feine Raster (bis 1 cm) Verfahren C verwenden.<br>"
        "Der geschätzte Aufwand steht unter den Eingabefeldern."
    ),
    "tolerance": (
        "<b>Kontakttoleranz</b><br>Eine Kugellage zählt als Berührung, wenn ihr Abstand zum Modell "
        "zwischen r und r + Toleranz liegt. Da die Kugelmittelpunkte nur auf einem Raster liegen, "
        "berühren sie das Modell fast nie exakt; die Toleranz gleicht das aus.<br>"
        "<i>auto</i>: halbe Diagonale einer Rasterzelle (≈ 0,87 × Rasterweite).<br>"
        "Größer: breitere Einschlagbereiche, auch Stellen, die die Kugel knapp verfehlt. "
        "Kleiner: Lücken zwischen den Einschlagpunkten möglich."
    ),
    "mode_a": (
        "<b>Verfahren A – Richtungen</b><br>Die Kugel rollt in parallelen senkrechten Ebenen in der "
        "gewählten Richtung über Boden und Bauwerk, wie beim Abrollen auf Ansichten und Schnitten. "
        "Jede Ebene wird für sich betrachtet.<br>"
        "0° = Rollen in +x-Richtung, Winkel gegen den Uhrzeigersinn (von oben gesehen). θ und θ + 180° "
        "ergeben dieselben Berührpunkte. Weitere Winkel ins Textfeld schreiben, getrennt durch "
        "<i>;</i> oder Leerzeichen.<br>"
        "Stellen, die die Kugel nur schräg zu allen gewählten Richtungen erreicht, können fehlen."
    ),
    "mode_b": (
        "<b>Verfahren B – 3D-Hülle</b><br>Die Kugel bewegt sich frei im Raum. Alle Kugellagen auf "
        "dem Raster werden ausgewertet, das ergibt alle möglichen Einschlagpunkte unabhängig von "
        "einer Richtung.<br>"
        "Rechnet das volle Raster; der Aufwand steigt bei halber Rasterweite auf das 8-Fache."
    ),
    "mode_c": (
        "<b>Verfahren C – 3D adaptiv</b><br>Gleiches Prinzip wie B, aber zuerst mit grober "
        "Rasterweite. Nur Bereiche, in denen die Kugel das Modell berühren kann, werden stufenweise "
        "bis zur Rasterweite verfeinert. An Ecken, Kanten und ebenen Flächen werden die Kugellagen "
        "direkt gesetzt.<br>"
        "Damit sind Rasterweiten bis 1 cm auch mit großen Kugeln in Sekunden möglich. Die "
        "Einschlagpunkte stimmen mit B bis auf etwa eine Rasterweite überein."
    ),
    "coarse": (
        "<b>Grobe Rasterweite</b> (Verfahren C)<br>Rasterweite der ersten Stufe. Jede weitere Stufe "
        "halbiert sie, bis die Rasterweite erreicht ist; deshalb wird sie auf Rasterweite × 2ⁿ "
        "aufgerundet.<br><i>auto</i>: etwa r/4. Beeinflusst vor allem die Rechenzeit, kaum das "
        "Ergebnis."
    ),
    "color_dir": (
        "<b>Farbe je Rollrichtung</b><br>Nur bei Verfahren A: Die Einschlagpunkte jeder Richtung "
        "erhalten eine eigene Farbe. Aus: alle Punkte rot."
    ),
    "legend": (
        "<b>Legende</b><br>Zeigt oben rechts die Anzahl der Einschlagpunkte (je Richtung). Wird auch "
        "in die PNG-Bilder übernommen."
    ),
    "info": (
        "<b>Kugeldaten einblenden</b><br>Textblock oben links mit Kugeldurchmesser, Schutzklasse, "
        "Verfahren, Rasterweite, Toleranz und Bodenhöhe. Wird auch in die PNG-Bilder übernommen."
    ),
    "heatmap": (
        "<b>Heatmap</b><br>Färbt die Modelloberfläche nach der Anzahl der Kugellagen, die einen "
        "Bereich berühren (logarithmische Skala). Kanten und Ecken werden am häufigsten berührt und "
        "sind am stärksten gefährdet. Nicht berührte Flächen bleiben grau.<br>"
        "Bei vielen Einschlagpunkten die Punktgröße auf 1–2 px verkleinern, sonst verdecken sie die "
        "Heatmap."
    ),
    "ground_plane": "<b>Bodenebene</b><br>Zeigt den Boden als grüne Fläche in der eingestellten Bodenhöhe.",
    "edges": (
        "<b>Dreieckskanten</b><br>Zeigt die Kanten des Dreiecksnetzes. Hilfreich, um die "
        "Tessellierung zu prüfen, z. B. ob Rundungen fein genug zerlegt sind."
    ),
    "spheres": (
        "<b>Kugelpositionen</b><br>Zeichnet einige Lagen der Blitzkugel, in denen sie das Modell "
        "berührt, als Drahtgitter. Sie werden gleichmäßig verteilt ausgewählt, die höchsten zuerst. "
        "Die Zahl legt fest, wie viele Kugeln gezeichnet werden."
    ),
    "opacity": (
        "<b>Modell-Deckkraft</b><br>Niedrige Werte machen das Modell durchscheinend, sodass auch "
        "Einschlagpunkte auf der Rückseite sichtbar sind."
    ),
    "point_size": (
        "<b>Punktgröße</b><br>Größe der Einschlagpunkte in Pixeln. Beim Bildexport wird sie passend "
        "zur Bildbreite vergrößert."
    ),
    "views": (
        "<b>Standardansichten</b><br>Iso ist perspektivisch, die übrigen Ansichten verwenden "
        "Parallelprojektion. <i>Vorne</i> = Blick in +y-Richtung.<br>"
        "Maus in der 3D-Ansicht: linke Taste drehen, Mausrad oder rechte Taste zoomen, "
        "Umschalt + linke Taste verschieben."
    ),
    "resolution": (
        "<b>Auflösung</b><br>Breite × Höhe der exportierten PNG-Bilder in Pixeln, unabhängig von der "
        "Fenstergröße. Legende und Kugeldaten werden mit exportiert, wenn sie eingeblendet sind."
    ),
}


class HelpButton(QToolButton):
    """Small round "?" button; a click shows ``text`` as a What's-This popup."""

    def __init__(self, text: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._text = text
        self.setText("?")
        self.setFixedSize(18, 18)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("Hilfe anzeigen")
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setStyleSheet(
            "QToolButton { border: 1px solid #8a8a8a; border-radius: 9px; color: #505050;"
            " font-weight: bold; font-size: 11px; padding: 0; }"
            "QToolButton:hover { background: #dde8fb; border-color: #4a78c2; color: #1d4c9c; }"
        )
        self.clicked.connect(self._show)

    def _show(self) -> None:
        QWhatsThis.showText(self.mapToGlobal(QPoint(self.width(), self.height())), self._text, self)


def with_help(item: QWidget | QLayout, key: str) -> QHBoxLayout:
    """Row with ``item`` (stretched) and a help button for ``HELP_TEXTS[key]`` on the right."""
    row = QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    if isinstance(item, QLayout):
        row.addLayout(item, 1)
    else:
        row.addWidget(item, 1)
    row.addWidget(HelpButton(HELP_TEXTS[key]))
    return row
