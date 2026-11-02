"""Main window of the Blitzkugel application."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from PyQt6.QtCore import Qt, QTimer, pyqtSlot
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor

from .. import __version__
from ..core.distance import open3d_available
from ..core.mesh import TriMesh, make_box, make_rod
from ..core.progress import CancelCallback, ProgressCallback, ProgressScope
from ..core.rolling_sphere import (
    Mode,
    ProtectionClass,
    SphereParams,
    StrikeResult,
    estimate_grid_points,
    roll_3d,
    roll_adaptive,
    roll_directions,
)
from ..export.csv_export import export_strike_points_csv
from ..geometry.step_import import StepModel, TessellationSettings, load_step, tessellate
from ..viz.scene import STANDARD_VIEWS, DisplayOptions, SceneData, apply_view, build_scene, focus_bounds
from ..viz.scene import export_camera_view, export_views
from .help import with_help
from .workers import TaskRunner

log = logging.getLogger(__name__)

UNIT_CHOICES: list[tuple[str, str | None]] = [
    ("Automatisch (aus STEP)", None),
    ("Millimeter", "mm"),
    ("Zentimeter", "cm"),
    ("Meter", "m"),
    ("Zoll", "in"),
    ("Fuß", "ft"),
]
DEFAULT_DIRECTIONS = [0, 45, 90, 135, 180, 225, 270, 315]
#: Warn before evaluations with more grid points than this.
WARN_GRID_POINTS = 40_000_000


def _spin(lo: float, hi: float, value: float, decimals: int = 2, step: float = 1.0, suffix: str = "") -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setRange(lo, hi)
    box.setDecimals(decimals)
    box.setSingleStep(step)
    box.setValue(value)
    if suffix:
        box.setSuffix(suffix)
    box.setKeyboardTracking(False)
    return box


def parse_angles(text: str) -> list[float]:
    """Parse free angle input such as ``"22,5; 67.5 100"`` (separators ``;`` or blanks)."""
    angles: list[float] = []
    for token in re.split(r"[;\s]+", text.strip()):
        if token:
            angles.append(float(token.replace(",", ".")) % 360.0)
    return angles


class MainWindow(QMainWindow):
    """Main window: settings panel on the left, 3D view on the right."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"Blitzkugel {__version__} – Blitzkugelverfahren nach DIN EN 62305-3")
        self.resize(1500, 950)

        self.step_model: StepModel | None = None
        self.scene = SceneData()
        self.runner = TaskRunner(self)
        self._running_label = ""

        self.plotter = QtInteractor(self)
        self.plotter.set_background("white")

        panel = self._build_panel()
        scroll = QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(panel.minimumSizeHint().width() + scroll.verticalScrollBar().sizeHint().width() + 4)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(scroll)
        splitter.addWidget(self.plotter.interactor)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([450, 1050])
        self.setCentralWidget(splitter)
        self._build_menu()

        self.statusBar().showMessage(
            "Distanz-Backend: open3d" if open3d_available()
            else "Hinweis: open3d nicht verfügbar – langsame NumPy-Berechnung (nur für kleine Modelle)."
        )
        self._update_enabled()
        self._refresh_scene()

    # ------------------------------------------------------------------ UI construction

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("&Datei")
        act_open = QAction("STEP-Datei &öffnen …", self)
        act_open.setShortcut(QKeySequence.StandardKey.Open)
        act_open.triggered.connect(self.open_step_dialog)
        file_menu.addAction(act_open)
        act_demo = QAction("&Demo-Modell laden", self)
        act_demo.triggered.connect(self.load_demo)
        file_menu.addAction(act_demo)
        file_menu.addSeparator()
        for text, slot in (("Aktuelle Ansicht als PNG …", self.export_current_png),
                           ("Alle Standardansichten als PNG …", self.export_all_png),
                           ("Einschlagpunkte als CSV …", self.export_csv)):
            act = QAction(text, self)
            act.triggered.connect(slot)
            file_menu.addAction(act)
        file_menu.addSeparator()
        act_quit = QAction("&Beenden", self)
        act_quit.setShortcut(QKeySequence.StandardKey.Quit)
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

        view_menu = self.menuBar().addMenu("&Ansicht")
        for key, (label, _, _) in STANDARD_VIEWS.items():
            act = QAction(label, self)
            act.triggered.connect(lambda _=False, k=key: self.set_view(k))
            view_menu.addAction(act)

        help_menu = self.menuBar().addMenu("&Hilfe")
        act_about = QAction("Über Blitzkugel", self)
        act_about.triggered.connect(self._about)
        help_menu.addAction(act_about)

    def _build_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.addWidget(self._build_model_group())
        layout.addWidget(self._build_sphere_group())
        layout.addWidget(self._build_method_group())
        layout.addWidget(self._build_display_group())
        layout.addWidget(self._build_export_group())
        layout.addStretch(1)
        return panel

    def _build_model_group(self) -> QGroupBox:
        group = QGroupBox("1  Modell")
        lay = QVBoxLayout(group)
        btn_row = QHBoxLayout()
        self.btn_open = QPushButton("STEP-Datei öffnen …")
        self.btn_open.clicked.connect(self.open_step_dialog)
        btn_row.addWidget(self.btn_open)
        self.btn_demo = QPushButton("Demo")
        self.btn_demo.setToolTip("Einfaches Testmodell (Gebäude mit Fangstangen und Turm) laden")
        self.btn_demo.clicked.connect(self.load_demo)
        btn_row.addWidget(self.btn_demo)
        lay.addLayout(btn_row)

        self.lbl_model = QLabel("Kein Modell geladen.")
        self.lbl_model.setWordWrap(True)
        self.lbl_model.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        lay.addWidget(self.lbl_model)

        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.cmb_unit = QComboBox()
        for text, _ in UNIT_CHOICES:
            self.cmb_unit.addItem(text)
        self.cmb_unit.setToolTip("Einheit der Koordinaten in der Datei. Nur ändern, wenn die "
                                 "Einheitenangabe der STEP-Datei falsch ist.")
        form.addRow("Einheit:", with_help(self.cmb_unit, "unit"))
        self.spn_lin_defl = _spin(0.01, 10_000, 20.0, 2, 5.0, " mm")
        self.spn_lin_defl.setToolTip("Maximale Sehnenabweichung zwischen CAD-Fläche und Dreiecksnetz")
        form.addRow("Lineare Abweichung:", with_help(self.spn_lin_defl, "lin_defl"))
        self.spn_ang_defl = _spin(1.0, 90.0, 15.0, 1, 1.0, " °")
        self.spn_ang_defl.setToolTip("Maximaler Winkel zwischen benachbarten Dreiecken")
        form.addRow("Winkelabweichung:", with_help(self.spn_ang_defl, "ang_defl"))
        lay.addLayout(form)
        self.btn_retess = QPushButton("Neu tessellieren")
        self.btn_retess.clicked.connect(self.retessellate)
        lay.addWidget(self.btn_retess)

        ground_row = QHBoxLayout()
        self.spn_ground = _spin(-100_000, 100_000, 0.0, 3, 0.5, " m")
        self.spn_ground.valueChanged.connect(self._on_ground_changed)
        ground_row.addWidget(QLabel("Bodenhöhe z:"))
        ground_row.addWidget(self.spn_ground, 1)
        self.btn_ground_min = QPushButton("= Modell-Min.")
        self.btn_ground_min.setToolTip("Bodenhöhe auf die tiefste z-Koordinate des Modells setzen")
        self.btn_ground_min.clicked.connect(self._reset_ground)
        ground_row.addWidget(self.btn_ground_min)
        lay.addLayout(with_help(ground_row, "ground"))
        return group

    def _build_sphere_group(self) -> QGroupBox:
        group = QGroupBox("2  Blitzkugel")
        form = QFormLayout(group)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.cmb_class = QComboBox()
        for pc in ProtectionClass:
            self.cmb_class.addItem(f"Schutzklasse {pc.name}  (D = {pc.diameter:g} m)", pc)
        self.cmb_class.addItem("benutzerdefiniert", None)
        self.cmb_class.currentIndexChanged.connect(self._on_class_changed)
        form.addRow("Schutzklasse:", with_help(self.cmb_class, "class"))

        self.spn_diameter = _spin(0.1, 2000.0, 40.0, 2, 5.0, " m")
        self.spn_diameter.valueChanged.connect(self._on_diameter_changed)
        form.addRow("Durchmesser:", with_help(self.spn_diameter, "diameter"))

        self.spn_grid = _spin(0.01, 50.0, 0.5, 2, 0.1, " m")
        self.spn_grid.setToolTip("Rasterweite der Kugelmittelpunkte (Rechengenauigkeit).\n"
                                 "Halbierung → etwa 8-fache Rechenzeit (Modus B).")
        self.spn_grid.valueChanged.connect(self._update_estimate)
        form.addRow("Rasterweite:", with_help(self.spn_grid, "grid"))

        tol_row = QHBoxLayout()
        self.chk_tol_auto = QCheckBox("auto")
        self.chk_tol_auto.setChecked(True)
        self.chk_tol_auto.setToolTip("Automatisch: halbe Rasterzellen-Diagonale")
        self.spn_tol = _spin(0.001, 50.0, 0.5, 3, 0.05, " m")
        self.spn_tol.setEnabled(False)
        self.chk_tol_auto.toggled.connect(lambda on: self.spn_tol.setEnabled(not on))
        tol_row.addWidget(self.spn_tol, 1)
        tol_row.addWidget(self.chk_tol_auto)
        form.addRow("Kontakttoleranz:", with_help(tol_row, "tolerance"))

        self.lbl_estimate = QLabel("")
        self.lbl_estimate.setWordWrap(True)
        form.addRow(self.lbl_estimate)
        return group

    def _build_method_group(self) -> QGroupBox:
        group = QGroupBox("3  Verfahren")
        lay = QVBoxLayout(group)
        self.rad_a = QRadioButton("A – Rollen entlang von Richtungen")
        self.rad_b = QRadioButton("B – 3D-Hülle (alle Richtungen)")
        self.rad_b.setChecked(True)
        mode_group = QButtonGroup(group)
        mode_group.addButton(self.rad_a)
        mode_group.addButton(self.rad_b)
        self.rad_a.toggled.connect(self._update_enabled)
        self.rad_a.toggled.connect(self._update_estimate)
        lay.addLayout(with_help(self.rad_a, "mode_a"))

        self.dir_box = QWidget()
        grid = QGridLayout(self.dir_box)
        grid.setContentsMargins(20, 0, 0, 0)
        self.chk_dirs: list[QCheckBox] = []
        for i, angle in enumerate(DEFAULT_DIRECTIONS):
            chk = QCheckBox(f"{angle}°")
            chk.setChecked(True)
            chk.toggled.connect(self._update_estimate)
            grid.addWidget(chk, i // 4, i % 4)
            self.chk_dirs.append(chk)
        self.edt_angles = QLineEdit()
        self.edt_angles.setPlaceholderText("weitere Winkel, z. B. 22,5; 67,5")
        self.edt_angles.editingFinished.connect(self._update_estimate)
        grid.addWidget(self.edt_angles, 2, 0, 1, 4)
        hint = QLabel("0° = +x, gegen den Uhrzeigersinn. θ und θ+180° liefern dieselben Berührpunkte.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: gray; font-size: 11px;")
        grid.addWidget(hint, 3, 0, 1, 4)
        lay.addWidget(self.dir_box)
        lay.addLayout(with_help(self.rad_b, "mode_b"))

        self.rad_c = QRadioButton("C – 3D adaptiv (grob → fein)")
        self.rad_c.setToolTip("Sucht mögliche Kontaktbereiche mit grober Rasterweite und verfeinert nur dort\n"
                              "stufenweise bis zur Rasterweite. Ergebnis wie B, aber deutlich schneller.")
        mode_group.addButton(self.rad_c)
        self.rad_c.toggled.connect(self._update_enabled)
        self.rad_c.toggled.connect(self._update_estimate)
        lay.addLayout(with_help(self.rad_c, "mode_c"))
        self.coarse_box = QWidget()
        coarse_row = QHBoxLayout(self.coarse_box)
        coarse_row.setContentsMargins(20, 0, 0, 0)
        coarse_row.addWidget(QLabel("Grobe Rasterweite:"))
        self.spn_coarse = _spin(0.01, 200.0, 4.0, 2, 1.0, " m")
        self.spn_coarse.setToolTip("Rasterweite der ersten Stufe. Wird auf Rasterweite · 2ⁿ aufgerundet.")
        self.spn_coarse.setEnabled(False)
        self.spn_coarse.valueChanged.connect(self._update_estimate)
        self.chk_coarse_auto = QCheckBox("auto")
        self.chk_coarse_auto.setChecked(True)
        self.chk_coarse_auto.setToolTip("Automatisch: etwa r/4")
        self.chk_coarse_auto.toggled.connect(lambda on: self.spn_coarse.setEnabled(not on))
        self.chk_coarse_auto.toggled.connect(self._update_estimate)
        coarse_row.addWidget(self.spn_coarse, 1)
        coarse_row.addWidget(self.chk_coarse_auto)
        lay.addLayout(with_help(self.coarse_box, "coarse"))

        btn_row = QHBoxLayout()
        self.btn_run = QPushButton("Berechnen")
        self.btn_run.setDefault(True)
        self.btn_run.clicked.connect(self.start_calculation)
        self.btn_cancel = QPushButton("Abbrechen")
        self.btn_cancel.clicked.connect(self.runner.cancel)
        btn_row.addWidget(self.btn_run)
        btn_row.addWidget(self.btn_cancel)
        lay.addLayout(btn_row)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setTextVisible(True)
        self.progress.setFormat("")
        lay.addWidget(self.progress)
        self.lbl_progress = QLabel("")
        self.lbl_progress.setWordWrap(True)
        lay.addWidget(self.lbl_progress)
        return group

    def _build_display_group(self) -> QGroupBox:
        group = QGroupBox("4  Darstellung")
        lay = QVBoxLayout(group)
        defaults = DisplayOptions()

        def check(text: str, value: bool) -> QCheckBox:
            chk = QCheckBox(text)
            chk.setChecked(value)
            chk.toggled.connect(self._refresh_scene)
            return chk

        self.chk_color_dir = check("Farbe je Rollrichtung", defaults.color_by_direction)
        self.chk_legend = check("Legende", defaults.show_legend)
        self.chk_info = check("Kugeldurchmesser / Schutzklasse einblenden", defaults.show_info)
        self.chk_heatmap = check("Berührte Bereiche als Heatmap", defaults.show_heatmap)
        self.chk_ground = check("Bodenebene", defaults.show_ground)
        self.chk_edges = check("Dreieckskanten", defaults.show_edges)
        for chk, key in ((self.chk_color_dir, "color_dir"), (self.chk_legend, "legend"), (self.chk_info, "info"),
                         (self.chk_heatmap, "heatmap"), (self.chk_ground, "ground_plane"),
                         (self.chk_edges, "edges")):
            lay.addLayout(with_help(chk, key))

        sph_row = QHBoxLayout()
        self.chk_spheres = check("Kugelpositionen (Drahtgitter):", defaults.show_spheres)
        self.spn_n_spheres = QSpinBox()
        self.spn_n_spheres.setRange(1, 200)
        self.spn_n_spheres.setValue(defaults.n_spheres)
        self.spn_n_spheres.valueChanged.connect(self._refresh_scene)
        sph_row.addWidget(self.chk_spheres)
        sph_row.addWidget(self.spn_n_spheres)
        lay.addLayout(with_help(sph_row, "spheres"))

        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.sld_opacity = QSlider(Qt.Orientation.Horizontal)
        self.sld_opacity.setRange(5, 100)
        self.sld_opacity.setValue(int(defaults.model_opacity * 100))
        self.sld_opacity.sliderReleased.connect(self._refresh_scene)
        form.addRow("Modell-Deckkraft:", with_help(self.sld_opacity, "opacity"))
        self.spn_point = _spin(1.0, 50.0, defaults.point_size, 0, 1.0, " px")
        self.spn_point.valueChanged.connect(self._refresh_scene)
        form.addRow("Punktgröße:", with_help(self.spn_point, "point_size"))
        lay.addLayout(form)

        views = QGridLayout()
        for i, (key, (label, _, _)) in enumerate(STANDARD_VIEWS.items()):
            btn = QPushButton(label)
            btn.clicked.connect(lambda _=False, k=key: self.set_view(k))
            views.addWidget(btn, i // 3, i % 3)
        lay.addLayout(with_help(QLabel("Standardansichten:"), "views"))
        lay.addLayout(views)
        return group

    def _build_export_group(self) -> QGroupBox:
        group = QGroupBox("5  Export")
        lay = QVBoxLayout(group)
        res_row = QHBoxLayout()
        self.spn_img_w = QSpinBox()
        self.spn_img_h = QSpinBox()
        for box, val in ((self.spn_img_w, 1920), (self.spn_img_h, 1080)):
            box.setRange(200, 8000)
            box.setValue(val)
            box.setSuffix(" px")
        res_row.addWidget(QLabel("Auflösung:"))
        res_row.addWidget(self.spn_img_w)
        res_row.addWidget(QLabel("×"))
        res_row.addWidget(self.spn_img_h)
        lay.addLayout(with_help(res_row, "resolution"))
        self.btn_png = QPushButton("Aktuelle Ansicht als PNG …")
        self.btn_png.clicked.connect(self.export_current_png)
        self.btn_png_all = QPushButton("Alle Standardansichten als PNG …")
        self.btn_png_all.clicked.connect(self.export_all_png)
        self.chk_decimal_comma = QCheckBox("CSV mit Dezimalkomma (deutsches Excel)")
        self.chk_decimal_comma.setChecked(True)
        lay.addLayout(with_help(self.chk_decimal_comma, "decimal_comma"))
        self.btn_csv = QPushButton("Einschlagpunkte als CSV …")
        self.btn_csv.clicked.connect(self.export_csv)
        for b in (self.btn_png, self.btn_png_all, self.btn_csv):
            lay.addWidget(b)
        return group

    # ------------------------------------------------------------------ state helpers

    def _display_options(self) -> DisplayOptions:
        return DisplayOptions(
            model_opacity=self.sld_opacity.value() / 100.0,
            show_edges=self.chk_edges.isChecked(),
            point_size=self.spn_point.value(),
            color_by_direction=self.chk_color_dir.isChecked(),
            show_legend=self.chk_legend.isChecked(),
            show_info=self.chk_info.isChecked(),
            show_heatmap=self.chk_heatmap.isChecked(),
            show_spheres=self.chk_spheres.isChecked(),
            n_spheres=self.spn_n_spheres.value(),
            show_ground=self.chk_ground.isChecked(),
        )

    def _sphere_params(self) -> SphereParams:
        return SphereParams(
            radius=self.spn_diameter.value() / 2.0,
            grid_step=self.spn_grid.value(),
            ground_z=self.spn_ground.value(),
            contact_tolerance=None if self.chk_tol_auto.isChecked() else self.spn_tol.value(),
            coarse_step=None if self.chk_coarse_auto.isChecked() else self.spn_coarse.value(),
        )

    def _mode(self) -> Mode:
        if self.rad_a.isChecked():
            return Mode.DIRECTIONS
        return Mode.ADAPTIVE if self.rad_c.isChecked() else Mode.HULL_3D

    def _angles(self) -> list[float]:
        angles = [float(a) for a, chk in zip(DEFAULT_DIRECTIONS, self.chk_dirs) if chk.isChecked()]
        angles += parse_angles(self.edt_angles.text())
        unique: list[float] = []
        for a in angles:
            if all(abs(a - b) > 1e-9 for b in unique):
                unique.append(a)
        return unique

    def _update_enabled(self) -> None:
        busy = self.runner.busy
        has_mesh = self.scene.mesh is not None
        has_result = self.scene.result is not None
        self.dir_box.setEnabled(self.rad_a.isChecked() and not busy)
        self.coarse_box.setEnabled(self.rad_c.isChecked() and not busy)
        for w in (self.btn_open, self.btn_demo, self.cmb_unit, self.spn_lin_defl, self.spn_ang_defl):
            w.setEnabled(not busy)
        self.btn_retess.setEnabled(not busy and self.step_model is not None)
        self.btn_run.setEnabled(not busy and has_mesh)
        self.btn_cancel.setEnabled(busy)
        self.btn_ground_min.setEnabled(has_mesh)
        for w in (self.btn_png, self.btn_png_all):
            w.setEnabled(has_mesh and not busy)
        self.btn_csv.setEnabled(has_result and not busy)

    def _model_info(self) -> str:
        mesh = self.scene.mesh
        if mesh is None:
            return "Kein Modell geladen."
        lo, hi = mesh.bounds
        dims = hi - lo
        lines = []
        if self.step_model is not None:
            lines.append(f"<b>{self.step_model.path.name}</b>")
            lines.append(f"Einheit in Datei: {self.step_model.file_unit}")
        else:
            lines.append("<b>Demo-Modell</b>")
        n_faces = f"{mesh.n_faces:,}".replace(",", ".")
        lines.append(f"{len(mesh.body_names)} Körper, {n_faces} Dreiecke")
        lines.append(f"Abmessungen: {dims[0]:.2f} × {dims[1]:.2f} × {dims[2]:.2f} m")
        lines.append(f"z: {lo[2]:.3f} … {hi[2]:.3f} m")
        return "<br>".join(lines)

    @pyqtSlot()
    def _update_estimate(self) -> None:
        mesh = self.scene.mesh
        if mesh is None:
            self.lbl_estimate.setText("")
            return
        try:
            angles = self._angles() if self._mode() is Mode.DIRECTIONS else [0.0]
            if not angles:
                self.lbl_estimate.setText("Keine Richtung gewählt.")
                return
            n = estimate_grid_points(mesh, self._sphere_params(), self._mode(), angles)
        except ValueError as exc:
            self.lbl_estimate.setText(str(exc))
            return
        if self.chk_tol_auto.isChecked():
            self.spn_tol.blockSignals(True)
            self.spn_tol.setValue(self._sphere_params().tolerance(self._mode()))
            self.spn_tol.blockSignals(False)
        color = "#b00000" if n > WARN_GRID_POINTS else "gray"
        text = f"≈ {n / 1e6:.1f} Mio. Rasterpunkte"
        if self._mode() is Mode.ADAPTIVE:
            p = self._sphere_params()
            text = (f"≈ {n / 1e6:.1f} Mio. Abstandsberechnungen, "
                    f"{p.refinement_levels + 1} Stufen {p.effective_coarse_step:g} m → {p.grid_step:g} m")
        self.lbl_estimate.setText(f"<span style='color:{color}'>{text}</span>")

    # ------------------------------------------------------------------ scene

    @pyqtSlot()
    def _refresh_scene(self) -> None:
        self.scene.ground_z = self.spn_ground.value()
        first = not self.plotter.camera_set
        build_scene(self.plotter, self.scene, self._display_options())
        if first and self.scene.mesh is not None:
            self.set_view("iso")
        else:
            self.plotter.render()

    def set_view(self, name: str) -> None:
        apply_view(self.plotter, name, focus_bounds(self.scene))

    @pyqtSlot()
    def _on_ground_changed(self) -> None:
        self._update_estimate()
        self._refresh_scene()

    @pyqtSlot()
    def _reset_ground(self) -> None:
        if self.scene.mesh is not None:
            self.spn_ground.setValue(float(self.scene.mesh.bounds[0][2]))

    @pyqtSlot(int)
    def _on_class_changed(self, index: int) -> None:
        pc = self.cmb_class.itemData(index)
        if pc is not None:
            self.spn_diameter.blockSignals(True)
            self.spn_diameter.setValue(pc.diameter)
            self.spn_diameter.blockSignals(False)
            self._update_estimate()

    @pyqtSlot(float)
    def _on_diameter_changed(self, value: float) -> None:
        pc = ProtectionClass.from_diameter(value)
        idx = self.cmb_class.findData(pc) if pc is not None else self.cmb_class.count() - 1
        self.cmb_class.blockSignals(True)
        self.cmb_class.setCurrentIndex(idx)
        self.cmb_class.blockSignals(False)
        self._update_estimate()

    def _set_mesh(self, mesh: TriMesh) -> None:
        self.scene.mesh = mesh
        self.scene.result = None
        self.scene.heatmap_cache = None
        self.scene.title = self.step_model.path.name if self.step_model else "Demo-Modell"
        self.lbl_model.setText(self._model_info())
        self.spn_ground.blockSignals(True)
        self.spn_ground.setValue(float(mesh.bounds[0][2]))
        self.spn_ground.blockSignals(False)
        build_scene(self.plotter, self.scene, self._display_options())
        self.set_view("iso")
        self._update_estimate()
        self._update_enabled()

    # ------------------------------------------------------------------ loading

    @pyqtSlot()
    def open_step_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "STEP-Datei öffnen", "", "STEP-Dateien (*.step *.stp *.STEP *.STP);;Alle Dateien (*)"
        )
        if path:
            self.load_step_file(path)

    def load_step_file(self, path: str | Path) -> None:
        """Load and tessellate a STEP file in the background."""
        settings = self._tess_settings()
        unit = UNIT_CHOICES[self.cmb_unit.currentIndex()][1]

        def task(progress: ProgressCallback, cancel: CancelCallback) -> tuple[StepModel, TriMesh]:
            scope = ProgressScope(progress, 0.0, 1.0, cancel)
            scope(0.02, "STEP-Datei lesen …")
            model = load_step(path)
            mesh = tessellate(model, settings, unit, scope.sub(0.3, 1.0), cancel)
            return model, mesh

        self._start(task, self._on_step_loaded, f"Lade {Path(path).name}")

    def _tess_settings(self) -> TessellationSettings:
        return TessellationSettings(self.spn_lin_defl.value(), self.spn_ang_defl.value())

    @pyqtSlot(object)
    def _on_step_loaded(self, payload: tuple[StepModel, TriMesh]) -> None:
        self._finish_task()
        self.step_model, mesh = payload
        self._set_mesh(mesh)
        self.statusBar().showMessage(f"{self.step_model.path.name} geladen: {mesh.n_faces:,} Dreiecke", 10_000)

    @pyqtSlot()
    def retessellate(self) -> None:
        if self.step_model is None:
            return
        model, settings = self.step_model, self._tess_settings()
        unit = UNIT_CHOICES[self.cmb_unit.currentIndex()][1]

        def task(progress: ProgressCallback, cancel: CancelCallback) -> tuple[StepModel, TriMesh]:
            return model, tessellate(model, settings, unit, progress, cancel)

        self._start(task, self._on_step_loaded, "Tesselliere")

    @pyqtSlot()
    def load_demo(self) -> None:
        """Small building with two air-termination rods next to a tower."""
        self.step_model = None
        mesh = TriMesh.merge([
            make_box(-10, -5, 0, 10, 5, 12, "Gebäude"),
            make_rod(-9, 0, 12, 3, 0.1, "Fangstange 1"),
            make_rod(9, 0, 12, 3, 0.1, "Fangstange 2"),
            make_box(15, -3, 0, 25, 3, 25, "Turm"),
        ])
        self._set_mesh(mesh)

    # ------------------------------------------------------------------ calculation

    @pyqtSlot()
    def start_calculation(self) -> None:
        mesh = self.scene.mesh
        if mesh is None:
            return
        try:
            params = self._sphere_params()
            mode = self._mode()
            angles = self._angles() if mode is Mode.DIRECTIONS else []
        except ValueError as exc:
            QMessageBox.warning(self, "Ungültige Eingabe", str(exc))
            return
        if mode is Mode.DIRECTIONS and not angles:
            QMessageBox.warning(self, "Keine Richtung", "Bitte mindestens eine Rollrichtung wählen.")
            return
        if params.ground_z > mesh.bounds[1][2]:
            QMessageBox.warning(self, "Bodenhöhe", "Die Bodenhöhe liegt über dem Modell.")
            return

        n = estimate_grid_points(mesh, params, mode, angles or [0.0])
        warnings = []
        if n > WARN_GRID_POINTS:
            warnings.append(f"Die Berechnung umfasst ca. {n / 1e6:.0f} Mio. Rasterpunkte und kann "
                            "lange dauern bzw. viel Speicher benötigen.")
        if not open3d_available() and mesh.n_faces > 2000:
            warnings.append("open3d ist nicht verfügbar; die NumPy-Ersatzberechnung ist für "
                            f"{mesh.n_faces:,} Dreiecke sehr langsam.")
        if warnings and QMessageBox.question(
            self, "Rechenaufwand", "\n\n".join(warnings) + "\n\nTrotzdem starten?"
        ) != QMessageBox.StandardButton.Yes:
            return

        def task(progress: ProgressCallback, cancel: CancelCallback) -> StrikeResult:
            if mode is Mode.HULL_3D:
                return roll_3d(mesh, params, progress, cancel)
            if mode is Mode.ADAPTIVE:
                return roll_adaptive(mesh, params, progress, cancel)
            return roll_directions(mesh, params, angles, progress, cancel)

        self._start(task, self._on_calc_finished, "Berechnung")

    @pyqtSlot(object)
    def _on_calc_finished(self, result: StrikeResult) -> None:
        self._finish_task()
        self.scene.result = result
        self.scene.heatmap_cache = None
        self._refresh_scene()
        msg = (f"{result.n_strike_points:,} Einschlagpunkte in {result.elapsed_s:.1f} s "
               f"(Backend {result.backend})").replace(",", ".")
        self.lbl_progress.setText(msg + ("<br>" + "<br>".join(result.notes) if result.notes else ""))
        self.statusBar().showMessage(msg, 15_000)
        self._update_enabled()

    # ------------------------------------------------------------------ task plumbing

    def _start(self, task, on_finished, label: str) -> None:  # noqa: ANN001
        self._running_label = label
        self.progress.setValue(0)
        self.progress.setFormat(f"{label} … %p %")
        self.lbl_progress.setText("")
        self.runner.start(task, on_finished, self._on_progress, self._on_failed, self._on_cancelled)
        self._update_enabled()

    @pyqtSlot(float, str)
    def _on_progress(self, fraction: float, message: str) -> None:
        self.progress.setValue(int(round(fraction * 1000)))
        self.lbl_progress.setText(message)

    @pyqtSlot(str, str)
    def _on_failed(self, message: str, details: str) -> None:
        self._finish_task()
        box = QMessageBox(QMessageBox.Icon.Critical, "Fehler", f"{self._running_label} fehlgeschlagen:\n\n{message}",
                          parent=self)
        box.setDetailedText(details)
        box.exec()
        self.lbl_progress.setText("Fehler: " + message)

    @pyqtSlot()
    def _on_cancelled(self) -> None:
        self._finish_task()
        self.lbl_progress.setText(f"{self._running_label} abgebrochen.")

    def _finish_task(self) -> None:
        self.progress.setValue(self.progress.maximum())
        self.progress.setFormat("")
        # The worker thread finishes asynchronously; refresh once it is gone.
        QTimer.singleShot(50, self._update_enabled)

    # ------------------------------------------------------------------ export

    def _image_size(self) -> tuple[int, int]:
        return self.spn_img_w.value(), self.spn_img_h.value()

    def _default_basename(self) -> str:
        stem = self.step_model.path.stem if self.step_model else "demo"
        res = self.scene.result
        if res is None:
            return stem
        suffix = {Mode.DIRECTIONS: "Richtungen", Mode.HULL_3D: "3D", Mode.ADAPTIVE: "3D-adaptiv"}[res.mode]
        return f"{stem}_D{res.params.diameter:g}m_{suffix}"

    @pyqtSlot()
    def export_current_png(self) -> None:
        if self.scene.mesh is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Ansicht speichern", f"{self._default_basename()}.png",
                                              "PNG-Bild (*.png)")
        if not path:
            return
        try:
            export_camera_view(self.scene, self._display_options(), self.plotter.camera, path, self._image_size())
        except Exception as exc:  # noqa: BLE001
            log.exception("PNG-Export fehlgeschlagen")
            QMessageBox.critical(self, "Export fehlgeschlagen", str(exc))
            return
        self.statusBar().showMessage(f"Bild gespeichert: {path}", 10_000)

    @pyqtSlot()
    def export_all_png(self) -> None:
        if self.scene.mesh is None:
            return
        folder = QFileDialog.getExistingDirectory(self, "Zielordner für Standardansichten")
        if not folder:
            return
        try:
            files = export_views(self.scene, self._display_options(), folder, self._default_basename(),
                                 list(STANDARD_VIEWS), self._image_size())
        except Exception as exc:  # noqa: BLE001
            log.exception("PNG-Export fehlgeschlagen")
            QMessageBox.critical(self, "Export fehlgeschlagen", str(exc))
            return
        QMessageBox.information(self, "Export", f"{len(files)} Bilder gespeichert in\n{folder}")

    @pyqtSlot()
    def export_csv(self) -> None:
        if self.scene.result is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Einschlagpunkte speichern",
                                              f"{self._default_basename()}.csv", "CSV (*.csv)")
        if not path:
            return
        try:
            n = export_strike_points_csv(self.scene.result, path,
                                          decimal_comma=self.chk_decimal_comma.isChecked())
        except OSError as exc:
            QMessageBox.critical(self, "Export fehlgeschlagen", str(exc))
            return
        self.statusBar().showMessage(f"{n} Einschlagpunkte gespeichert: {path}", 10_000)

    # ------------------------------------------------------------------ misc

    def _about(self) -> None:
        QMessageBox.about(
            self, "Über Blitzkugel",
            f"<b>Blitzkugel {__version__}</b><br>"
            "Blitzkugelverfahren nach DIN EN 62305-3 (VDE 0185-305-3) für STEP-Modelle.<br><br>"
            "Die Ergebnisse sind eine rechnerische Hilfe und ersetzen nicht die Prüfung "
            "durch eine Blitzschutz-Fachkraft.",
        )

    def closeEvent(self, event) -> None:  # noqa: ANN001, N802
        self.runner.wait()
        self.plotter.close()
        super().closeEvent(event)
