import sys
import os
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Callable
from dataclasses import dataclass
import warnings

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget,
    QFileDialog, QLineEdit, QLabel, QHBoxLayout, QMessageBox, QTabWidget,
    QGroupBox, QComboBox, QGridLayout, QScrollArea, QFrame, QSplitter,
    QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QPalette, QColor
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from scipy.interpolate import interp1d
from scipy.signal import find_peaks, savgol_filter
from scipy import stats
import matplotlib.pyplot as plt

# Suppress warnings
warnings.filterwarnings('ignore')


class CustomToolbar(NavigationToolbar):
    """Custom toolbar with only specific buttons"""
    def __init__(self, canvas, parent):
        super().__init__(canvas, parent)
        
        tools_to_remove = ['Save', 'Pan', 'Subplots', 'Configure']
        
        for child in self.children():
            if hasattr(child, 'text') and child.text() in tools_to_remove:
                child.setParent(None)
            elif hasattr(child, 'toolTip'):
                tooltip = child.toolTip()
                if any(remove_tool in tooltip for remove_tool in ['Save', 'Pan', 'Subplots', 'Configure']):
                    child.setParent(None)


@dataclass
class ThicknessResult:
    """Container for thickness estimation results with metadata"""
    thickness_nm: float
    optical_thickness_nm: float
    confidence_score: float
    n_pairs: int
    std_dev: Optional[float]
    n_peaks: int
    method: str
    individual_estimates: List[float]
    refractive_index: float


class OpticalEnvelopeApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OpticalSimulatorApp (Swanepole method")
        self.setGeometry(100, 100, 1600, 900)
        
        # Suppress numpy warnings
        np.seterr(divide='ignore', invalid='ignore')
        
        # Data storage
        self.data = None
        self.wavelength = None
        self.transmittance = None

        # Manual Tmax/Tmin points & envelopes
        self.tmax = []
        self.tmin = []
        self.envelope_upper = None
        self.envelope_lower = None

        # Auto-selected Tmax/Tmin with prominence scores
        self.auto_tmax = []
        self.auto_tmin = []

        # Optical parameters
        self.thickness_nm = 100.0
        self.simple_thickness_nm = 100.0
        self.substrate_refractive_index = 1.4585
        self.save_folder = os.getcwd()
        self.min_points_required = 5
        self.refractive_index_dispersion = None

        # Envelope method grids
        self.xx = None
        self.TM = None
        self.Tm = None
        self.n2 = None
        self.alpha_env = None
        self.T_simulated = None
        self.T_exp_xx = None
        self.interference_free_T = None

        # Simple analysis results
        self.alpha_calc = None
        self.E = None
        
        # Band gap results
        self.band_gap_direct = None
        self.band_gap_indirect = None
        self.band_gap_confidence = None
        
        # Calculated optical properties
        self.n_final = None
        self.k_final = None
        self.e1 = None
        self.e2 = None
        self.tan_delta = None
        self.skin_depth = None
        self.sigma = None
        self.optical_density = None
        self.alpha_display = None
        self.alphaE = None
        self.alphaE_sq = None
        self.alphaE_sqrt = None
        self.alphaE_23 = None
        self.d_alpha_dE = None
        self.ln_alpha = None

        # Manual selection/editing
        self.selection_mode = None
        self.dragging = False
        self.drag_point_type = None
        self.drag_point_index = None
        
        # Flags
        self.updating_thickness = False
        self.auto_sim_run = False
        self._has_plotted = False
        self._thickness_manually_set = False
        
        # Thickness estimation settings
        self.min_peak_distance = 25.0
        self.min_thickness = 10.0
        self.max_thickness = 5000.0
        
        # Curvature selection
        self.curvature_type = 'cubic'

        # Horizontal-only snap-to-data settings
        self.snap_to_data_x_range_nm = 10.0

        # Store individual plot tabs
        self.param_tabs = {}
        self.main_ax = None
        self.main_canvas = None

        self.init_ui()

    # ----------------------------------------------------------------------
    # UI SETUP WITH INDIVIDUAL SMALL TABS
    # ----------------------------------------------------------------------
    def init_ui(self):
        # Set application font
        app_font = QFont("Times New Roman", 14)
        QApplication.setFont(app_font)
        
        # Main splitter for 3 panels (25% : 50% : 25%)
        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #bdc3c7;
                width: 2px;
            }
        """)
        
        # Left panel (25%)
        left_panel = QWidget()
        left_panel.setMinimumWidth(300)
        left_panel.setMaximumWidth(400)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(4)
        left_layout.setContentsMargins(8, 5, 8, 5)
        
        # Middle panel (50%) - Tab widget for parameters
        middle_panel = QWidget()
        middle_layout = QVBoxLayout(middle_panel)
        middle_layout.setContentsMargins(5, 5, 5, 5)
        middle_layout.setSpacing(5)
        
        # Right panel (25%)
        right_panel = QWidget()
        right_panel.setMinimumWidth(300)
        right_panel.setMaximumWidth(400)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(12)
        right_layout.setContentsMargins(10, 10, 10, 10)
        
        # ==================== LEFT PANEL GROUP BOXES ====================
        
        # 1) Data Control GroupBox
        self.data_group = QGroupBox("📂 DATA CONTROL")
        self.data_group.setStyleSheet(self.get_groupbox_style())
        data_layout = QVBoxLayout(self.data_group)
        data_layout.setSpacing(4)
        
        self.import_btn = QPushButton("📥 Import Data (TXT/CSV/Excel)")
        self.import_btn.setStyleSheet(self.get_button_style())
        self.import_btn.clicked.connect(self.load_data)
        
        self.export_btn = QPushButton("📤 Export Data (Excel/TXT)")
        self.export_btn.setStyleSheet(self.get_button_style())
        self.export_btn.clicked.connect(self.extract_optical_properties)
        self.export_btn.setEnabled(False)
        
        self.clear_btn = QPushButton("🗑️ Clear All")
        self.clear_btn.setStyleSheet(self.get_button_style())
        self.clear_btn.clicked.connect(self.clear_all)
        
        data_layout.addWidget(self.import_btn)
        data_layout.addWidget(self.export_btn)
        data_layout.addWidget(self.clear_btn)
        
        left_layout.addWidget(self.data_group)
        
        # 2) Method Selection GroupBox
        self.method_group = QGroupBox("📊 METHOD SELECTION")
        self.method_group.setStyleSheet(self.get_groupbox_style())
        method_layout = QVBoxLayout(self.method_group)
        method_layout.setSpacing(4)
        
        self.method_combo = QComboBox()
        self.method_combo.addItems([
            "🌊 Envelope Method (with fringes)",
            "📏 Simple Analysis (no fringes)"
        ])
        self.method_combo.setStyleSheet("""
            QComboBox {
                min-height: 32px;
                font-size: 12px;
                font-family: 'Times New Roman';
                padding: 3px;
                border: 2px solid #4A90E2;
                border-radius: 6px;
                background-color: white;
                color: #2c3e50;
            }
            QComboBox QAbstractItemView {
                color: #2c3e50;
                background-color: white;
            }
        """)
        self.method_combo.currentIndexChanged.connect(self.on_method_changed)
        
        method_layout.addWidget(self.method_combo)
        
        left_layout.addWidget(self.method_group)
        
        # 3) Refractive Index GroupBox
        self.ri_group = QGroupBox("🔬 REFRACTIVE INDEX")
        self.ri_group.setStyleSheet(self.get_groupbox_style())
        ri_layout = QVBoxLayout(self.ri_group)
        ri_layout.setSpacing(4)
        
        ri_input_layout = QHBoxLayout()
        ri_label = QLabel("Substrate n:")
        ri_label.setStyleSheet("font-weight: bold; font-size: 12px; color: #2c3e50;")
        ri_label.setFixedWidth(90)
        
        self.substrate_input = QLineEdit("1.4585")
        self.substrate_input.setStyleSheet("""
            QLineEdit {
                min-height: 30px;
                font-size: 12px;
                font-family: 'Times New Roman';
                padding: 3px;
                border: 2px solid #4A90E2;
                border-radius: 6px;
                background-color: white;
                color: #2c3e50;
            }
        """)
        self.substrate_input.textChanged.connect(self.update_substrate_index)
        
        ri_input_layout.addWidget(ri_label)
        ri_input_layout.addWidget(self.substrate_input)
        ri_layout.addLayout(ri_input_layout)
        
        left_layout.addWidget(self.ri_group)
        
        # 4) Envelope Controls GroupBox
        self.envelope_group = QGroupBox("📈 ENVELOPE CONTROLS")
        self.envelope_group.setStyleSheet(self.get_groupbox_style())
        envelope_layout = QVBoxLayout(self.envelope_group)
        envelope_layout.setSpacing(4)

        self.upper_env_btn = QPushButton("📈 Upper Envelope")
        self.upper_env_btn.setCheckable(True)
        self.upper_env_btn.setChecked(False)
        self.upper_env_btn.setStyleSheet(self.get_button_style())
        self.upper_env_btn.clicked.connect(self.toggle_upper_envelope_mode)
        envelope_layout.addWidget(self.upper_env_btn)

        self.lower_env_btn = QPushButton("📉 Lower Envelope")
        self.lower_env_btn.setCheckable(True)
        self.lower_env_btn.setChecked(False)
        self.lower_env_btn.setStyleSheet(self.get_button_style())
        self.lower_env_btn.clicked.connect(self.toggle_lower_envelope_mode)
        envelope_layout.addWidget(self.lower_env_btn)

        self.interp_btn = QPushButton("🔗 Interpolate Envelopes")
        self.interp_btn.setStyleSheet(self.get_button_style())
        self.interp_btn.clicked.connect(self.update_envelopes)
        self.interp_btn.setEnabled(False)
        envelope_layout.addWidget(self.interp_btn)
        
        left_layout.addWidget(self.envelope_group)
        
        # ==================== MIDDLE PANEL - INDIVIDUAL SMALL TABS ====================
        
        # Main tab widget for all parameters
        self.main_tab_widget = QTabWidget()
        self.main_tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 2px solid #4A90E2;
                border-radius: 8px;
                background-color: white;
            }
            QTabBar::tab {
                font-size: 14px;
                font-family: 'Times New Roman';
                min-width: 55px;
                min-height: 30px;
                padding: 4px 6px;
                font-weight: bold;
                background-color: #ecf0f1;
                color: #2c3e50;
                border: 1px solid #4A90E2;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #4A90E2;
                color: white;
            }
            QTabBar::tab:hover {
                background-color: #357ABD;
                color: white;
            }
        """)
        
        # Define all parameter tabs with short names
        param_tabs_config = [
            ("Main", "Wavelength (nm)", "Transmittance"),
            ("n", "Wavelength (nm)", "n"),
            ("k", "Wavelength (nm)", "k"),
            ("ε₁", "Wavelength (nm)", "ε₁"),
            ("ε₂", "Wavelength (nm)", "ε₂"),
            ("tanδ", "Wavelength (nm)", "tan δ"),
            ("SD", "Wavelength (nm)", "δ (nm)"),
            ("σ", "Wavelength (nm)", "σ (S/cm)"),
            ("OD", "Wavelength (nm)", "OD"),
            ("α", "Energy (eV)", "α (cm⁻¹)"),
            ("(αE)²", "Energy (eV)", "(αE)² (cm⁻²·eV²)"),
            ("(αE)^½", "Energy (eV)", "(αE)^½ (cm^{-½}·eV^{½})"),
            ("dα/dE", "Energy (eV)", "dα/dE (cm⁻¹·eV⁻¹)"),
            ("lnα", "Energy (eV)", "ln(α)")
        ]
        
        # Create each tab
        for tab_name, xlabel, ylabel in param_tabs_config:
            tab_widget = QWidget()
            tab_layout = QVBoxLayout(tab_widget)
            tab_layout.setContentsMargins(5, 5, 5, 5)
            
            # Create figure and canvas
            fig = Figure(figsize=(8, 6), dpi=100, facecolor='white')
            canvas = FigureCanvas(fig)
            toolbar = CustomToolbar(canvas, self)
            
            ax = fig.add_subplot(111)
            ax.set_xlabel(xlabel, fontsize=11, fontweight='bold')
            ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
            ax.tick_params(axis='both', which='major', labelsize=9)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_facecolor('#fafafa')
            
            # Store for later updates
            self.param_tabs[tab_name] = {
                'widget': tab_widget,
                'canvas': canvas,
                'ax': ax,
                'toolbar': toolbar,
                'xlabel': xlabel,
                'ylabel': ylabel
            }
            
            # Store main spectrum axes reference separately for mouse events
            if tab_name == "Main":
                self.main_ax = ax
                self.main_canvas = canvas
            
            tab_layout.addWidget(toolbar)
            tab_layout.addWidget(canvas)
            
            self.main_tab_widget.addTab(tab_widget, tab_name)
        
        middle_layout.addWidget(self.main_tab_widget)
        
        # Redraw canvas when switching tabs (hidden tabs don't repaint)
        self.main_tab_widget.currentChanged.connect(self.on_tab_changed)
        
        # Connect mouse events to main canvas
        if self.main_canvas:
            self.main_canvas.mpl_connect('button_press_event', self.on_click)
            self.main_canvas.mpl_connect('motion_notify_event', self.on_motion)
            self.main_canvas.mpl_connect('button_release_event', self.on_release)
        
        # ==================== RIGHT PANEL GROUP BOXES ====================
        
        # 5) Simulation Control GroupBox
        self.sim_group = QGroupBox("🔧 SIMULATION CONTROL")
        self.sim_group.setStyleSheet(self.get_groupbox_style())
        sim_layout = QVBoxLayout(self.sim_group)
        sim_layout.setSpacing(4)
        
        self.auto_select_btn = QPushButton("🎯 Auto Select Tmax & Tmin")
        self.auto_select_btn.setStyleSheet(self.get_button_style())
        self.auto_select_btn.clicked.connect(self.auto_select_both)
        self.auto_select_btn.setEnabled(False)
        sim_layout.addWidget(self.auto_select_btn)

        auto_manual_layout = QHBoxLayout()
        self.add_auto_tmax_btn = QPushButton("+ Auto Tmax")
        self.add_auto_tmax_btn.setCheckable(True)
        self.add_auto_tmax_btn.setChecked(False)
        self.add_auto_tmax_btn.setStyleSheet(self.get_button_style('#8E44AD', '#7D3C98'))
        self.add_auto_tmax_btn.clicked.connect(self.toggle_auto_tmax_mode)
        auto_manual_layout.addWidget(self.add_auto_tmax_btn)

        self.add_auto_tmin_btn = QPushButton("+ Auto Tmin")
        self.add_auto_tmin_btn.setCheckable(True)
        self.add_auto_tmin_btn.setChecked(False)
        self.add_auto_tmin_btn.setStyleSheet(self.get_button_style('#16A085', '#138D75'))
        self.add_auto_tmin_btn.clicked.connect(self.toggle_auto_tmin_mode)
        auto_manual_layout.addWidget(self.add_auto_tmin_btn)
        sim_layout.addLayout(auto_manual_layout)

        thickness_layout = QHBoxLayout()
        thickness_label = QLabel("Thickness (nm):")
        thickness_label.setStyleSheet("font-weight: bold; font-size: 12px; color: #2c3e50;")
        thickness_label.setFixedWidth(110)
        
        self.thickness_input = QLineEdit("100")
        self.thickness_input.setStyleSheet("""
            QLineEdit {
                min-height: 30px;
                font-size: 12px;
                font-family: 'Times New Roman';
                padding: 3px;
                border: 2px solid #4A90E2;
                border-radius: 6px;
                background-color: white;
                color: #2c3e50;
            }
        """)
        self.thickness_input.textChanged.connect(self.update_thickness)
        
        thickness_layout.addWidget(thickness_label)
        thickness_layout.addWidget(self.thickness_input)
        sim_layout.addLayout(thickness_layout)
        
        self.simulate_btn = QPushButton("🚀 Simulate Spectra")
        self.simulate_btn.setStyleSheet(self.get_button_style('#27AE60', '#229954'))
        self.simulate_btn.clicked.connect(self.auto_simulate)
        self.simulate_btn.setEnabled(False)
        sim_layout.addWidget(self.simulate_btn)
        
        left_layout.addWidget(self.sim_group)
        left_layout.addStretch()
        
        # 6) Status Box GroupBox
        self.status_group = QGroupBox("📋 STATUS")
        self.status_group.setStyleSheet(self.get_groupbox_style())
        status_layout = QVBoxLayout(self.status_group)
        status_layout.setSpacing(8)
        
        self.selection_status = QLabel("⚫ Envelope Selection: INACTIVE")
        self.selection_status.setStyleSheet("""
            QLabel {
                font-size: 12px;
                font-family: 'Times New Roman';
                padding: 6px;
                background-color: #f0f0f0;
                border-radius: 6px;
                font-weight: bold;
            }
        """)
        status_layout.addWidget(self.selection_status)
        
        self.point_counter = QLabel("📌 Manual Points: Tmax=0, Tmin=0")
        self.point_counter.setStyleSheet("""
            QLabel {
                font-size: 12px;
                font-family: 'Times New Roman';
                padding: 6px;
                background-color: #e8f0fe;
                border-radius: 6px;
                font-weight: bold;
            }
        """)
        status_layout.addWidget(self.point_counter)
        
        self.auto_point_counter = QLabel("🤖 Auto Points: Tmax=0, Tmin=0")
        self.auto_point_counter.setStyleSheet("""
            QLabel {
                font-size: 12px;
                font-family: 'Times New Roman';
                padding: 6px;
                background-color: #e8daef;
                border-radius: 6px;
                font-weight: bold;
            }
        """)
        status_layout.addWidget(self.auto_point_counter)
        
        self.thickness_result_label = QLabel("📐 Estimated Thickness: -- nm")
        self.thickness_result_label.setStyleSheet("""
            QLabel {
                font-size: 12px;
                font-family: 'Times New Roman';
                padding: 6px;
                background-color: #d5f5e3;
                border-radius: 6px;
                font-weight: bold;
                color: #27AE60;
            }
        """)
        status_layout.addWidget(self.thickness_result_label)
        
        self.fit_percentage_label = QLabel("📊 Fit Quality: --%")
        self.fit_percentage_label.setStyleSheet("""
            QLabel {
                font-size: 12px;
                font-family: 'Times New Roman';
                padding: 6px;
                background-color: #fef5e7;
                border-radius: 6px;
                font-weight: bold;
            }
        """)
        status_layout.addWidget(self.fit_percentage_label)
        
        self.cursor_label = QLabel("🎯 Cursor: λ = -, T = -")
        self.cursor_label.setStyleSheet("""
            QLabel {
                font-size: 11px;
                font-family: 'Times New Roman';
                padding: 5px;
                background-color: #f0f0f0;
                border-radius: 6px;
            }
        """)
        status_layout.addWidget(self.cursor_label)
        
        right_layout.addWidget(self.status_group)
        
        # 7) Simple Analysis GroupBox
        self.simple_group = QGroupBox("📏 SIMPLE ANALYSIS")
        self.simple_group.setStyleSheet(self.get_groupbox_style())
        self.simple_group.setVisible(False)
        simple_layout = QVBoxLayout(self.simple_group)
        simple_layout.setSpacing(10)
        
        info_label = QLabel("For spectra WITHOUT interference fringes")
        info_label.setStyleSheet("font-size: 12px; color: #7f8c8d; padding: 5px;")
        simple_layout.addWidget(info_label)
        
        simple_thickness_layout = QHBoxLayout()
        simple_thickness_label = QLabel("Thickness (nm):")
        simple_thickness_label.setStyleSheet("font-weight: bold; font-size: 12px; color: #2c3e50;")
        simple_thickness_label.setFixedWidth(100)
        
        self.simple_thickness_input = QLineEdit("100")
        self.simple_thickness_input.setStyleSheet("""
            QLineEdit {
                min-height: 40px;
                font-size: 12px;
                font-family: 'Times New Roman';
                padding: 5px;
                border: 2px solid #F39C12;
                border-radius: 8px;
                background-color: white;
                color: #2c3e50;
            }
        """)
        self.simple_thickness_input.textChanged.connect(self.update_simple_thickness)
        
        simple_thickness_layout.addWidget(simple_thickness_label)
        simple_thickness_layout.addWidget(self.simple_thickness_input)
        simple_layout.addLayout(simple_thickness_layout)
        
        self.calc_simple_btn = QPushButton("🧮 Calculate Parameters")
        self.calc_simple_btn.setStyleSheet(self.get_button_style('#F39C12', '#E67E22'))
        self.calc_simple_btn.clicked.connect(self.calculate_simple_parameters)
        self.calc_simple_btn.setEnabled(False)
        simple_layout.addWidget(self.calc_simple_btn)
        
        right_layout.addWidget(self.simple_group)
        right_layout.addStretch()
        
        # Add panels to main splitter
        main_splitter.addWidget(left_panel)
        main_splitter.addWidget(middle_panel)
        main_splitter.addWidget(right_panel)
        
        main_splitter.setSizes([250, 500, 250])
        
        self.setCentralWidget(main_splitter)
        
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QWidget {
                background-color: #f5f5f5;
                color: #2c3e50;
            }
            QLabel {
                color: #2c3e50;
            }
            QLineEdit {
                color: #2c3e50;
            }
            QComboBox {
                color: #2c3e50;
            }
        """)
        
        self.showMaximized()
    
    def get_button_style(self, bg_color='#4A90E2', hover_color='#357ABD'):
        return f"""
            QPushButton {{
                background-color: {bg_color};
                color: white;
                border: none;
                border-radius: 6px;
                min-height: 30px;
                max-height: 30px;
                font-size: 11px;
                font-family: 'Times New Roman';
                font-weight: bold;
                padding: 4px 8px;
            }}
            QPushButton:hover {{
                background-color: {hover_color};
            }}
            QPushButton:pressed {{
                background-color: {hover_color};
            }}
            QPushButton:disabled {{
                background-color: #bdc3c7;
                color: #7f8c8d;
            }}
        """
    
    def get_groupbox_style(self):
        return """
            QGroupBox {
                font-size: 11px;
                font-family: 'Times New Roman';
                font-weight: bold;
                color: #2c3e50;
                border: 2px solid #4A90E2;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 8px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 10px;
                top: 2px;
                padding: 0 6px 0 6px;
                color: #4A90E2;
                background-color: white;
            }
        """
    
    # ----------------------------------------------------------------------
    # SELECTION MODE HANDLING
    # ----------------------------------------------------------------------
    def set_selection_mode(self, mode):
        """Set or clear the selection mode and keep buttons synchronized"""
        self.selection_mode = mode

        # Block signals on all toggle buttons
        self.upper_env_btn.blockSignals(True)
        self.lower_env_btn.blockSignals(True)
        self.add_auto_tmax_btn.blockSignals(True)
        self.add_auto_tmin_btn.blockSignals(True)

        self.upper_env_btn.setChecked(mode == 'Tmax')
        self.lower_env_btn.setChecked(mode == 'Tmin')
        self.add_auto_tmax_btn.setChecked(mode == 'AutoTmax')
        self.add_auto_tmin_btn.setChecked(mode == 'AutoTmin')

        self.upper_env_btn.blockSignals(False)
        self.lower_env_btn.blockSignals(False)
        self.add_auto_tmax_btn.blockSignals(False)
        self.add_auto_tmin_btn.blockSignals(False)

        if mode == 'Tmax':
            self.selection_status.setText("🔴 Envelope Selection: ACTIVE (Upper Envelope)")
            self.selection_status.setStyleSheet("""
                QLabel {
                    font-size: 12px;
                    font-family: 'Times New Roman';
                    padding: 6px;
                    background-color: #fadbd8;
                    border-radius: 6px;
                    font-weight: bold;
                    color: #c0392b;
                }
            """)
        elif mode == 'Tmin':
            self.selection_status.setText("🔵 Envelope Selection: ACTIVE (Lower Envelope)")
            self.selection_status.setStyleSheet("""
                QLabel {
                    font-size: 12px;
                    font-family: 'Times New Roman';
                    padding: 6px;
                    background-color: #d6eaf8;
                    border-radius: 6px;
                    font-weight: bold;
                    color: #2980b9;
                }
            """)
        elif mode == 'AutoTmax':
            self.selection_status.setText("🟣 Auto Tmax: ACTIVE (click to add)")
            self.selection_status.setStyleSheet("""
                QLabel {
                    font-size: 12px;
                    font-family: 'Times New Roman';
                    padding: 6px;
                    background-color: #e8daef;
                    border-radius: 6px;
                    font-weight: bold;
                    color: #8E44AD;
                }
            """)
        elif mode == 'AutoTmin':
            self.selection_status.setText("🟢 Auto Tmin: ACTIVE (click to add)")
            self.selection_status.setStyleSheet("""
                QLabel {
                    font-size: 12px;
                    font-family: 'Times New Roman';
                    padding: 6px;
                    background-color: #d1f2eb;
                    border-radius: 6px;
                    font-weight: bold;
                    color: #16A085;
                }
            """)
        else:
            self.selection_status.setText("⚫ Selection: INACTIVE")
            self.selection_status.setStyleSheet("""
                QLabel {
                    font-size: 12px;
                    font-family: 'Times New Roman';
                    padding: 6px;
                    background-color: #f0f0f0;
                    border-radius: 6px;
                    font-weight: bold;
                    color: #7f8c8d;
                }
            """)

    def toggle_upper_envelope_mode(self):
        if self.upper_env_btn.isChecked():
            self.set_selection_mode('Tmax')
        else:
            self.set_selection_mode(None)

    def toggle_lower_envelope_mode(self):
        if self.lower_env_btn.isChecked():
            self.set_selection_mode('Tmin')
        else:
            self.set_selection_mode(None)

    def toggle_auto_tmax_mode(self):
        if self.add_auto_tmax_btn.isChecked():
            self.set_selection_mode('AutoTmax')
        else:
            self.set_selection_mode(None)

    def toggle_auto_tmin_mode(self):
        if self.add_auto_tmin_btn.isChecked():
            self.set_selection_mode('AutoTmin')
        else:
            self.set_selection_mode(None)

    def is_navigation_mode_active(self):
        """Return True when matplotlib toolbar zoom/pan mode is active"""
        toolbar = getattr(self.main_canvas, 'toolbar', None) if self.main_canvas else None
        if toolbar is None:
            return False
        mode = getattr(toolbar, 'mode', '')
        return mode != '' and mode is not None

    def snap_x_only(self, x, y):
        """Snap ONLY the x-coordinate to the nearest measured x-value within the allowed range.
        The snap range adapts to the current zoom level of the main plot."""
        if self.wavelength is None or x is None or y is None or len(self.wavelength) == 0:
            return x, y

        wl = np.asarray(self.wavelength, dtype=float)
        if wl.ndim != 1:
            wl = wl.ravel()

        # Adaptive snap range: use 5% of visible x-range, at least the configured minimum
        snap_range = float(self.snap_to_data_x_range_nm)
        if self.main_ax is not None:
            xlim = self.main_ax.get_xlim()
            visible_range = abs(xlim[1] - xlim[0])
            snap_range = max(snap_range, visible_range * 0.05)

        diffs = np.abs(wl - float(x))
        idx = int(np.nanargmin(diffs))
        nearest_x = float(wl[idx])
        dx = float(diffs[idx])

        if dx <= snap_range:
            return nearest_x, y

        return x, y
    
    def get_envelope_wavelength_range(self):
        """Get the common wavelength range where both upper and lower envelopes exist."""
        if self.envelope_upper is None or self.envelope_lower is None:
            return None, None
        
        upper_min = self.envelope_upper[0].min() if len(self.envelope_upper[0]) > 0 else None
        upper_max = self.envelope_upper[0].max() if len(self.envelope_upper[0]) > 0 else None
        lower_min = self.envelope_lower[0].min() if len(self.envelope_lower[0]) > 0 else None
        lower_max = self.envelope_lower[0].max() if len(self.envelope_lower[0]) > 0 else None
        
        if None in [upper_min, upper_max, lower_min, lower_max]:
            return None, None
        
        min_wl = max(upper_min, lower_min)
        max_wl = min(upper_max, lower_max)
        
        if min_wl >= max_wl:
            return None, None
            
        return min_wl, max_wl
    
    def on_curvature_changed(self, text):
        """Handle curvature type change"""
        if text == "Linear":
            self.curvature_type = 'linear'
        elif text == "Quadratic":
            self.curvature_type = 'quadratic'
        else:
            self.curvature_type = 'cubic'
        
        if self.envelope_upper is not None or self.envelope_lower is not None:
            self.update_envelopes()
    
    def on_method_changed(self, index):
        if index == 0:  # Envelope Method
            self.envelope_group.setVisible(True)
            self.sim_group.setVisible(True)
            self.simple_group.setVisible(False)
        else:  # Simple Method
            self.envelope_group.setVisible(False)
            self.sim_group.setVisible(False)
            self.simple_group.setVisible(True)
            self.calc_simple_btn.setEnabled(self.data is not None)
    
    def update_button_states(self):
        has_data = self.data is not None
        has_pts = len(self.tmax) >= 2 and len(self.tmin) >= 2
        has_env = self.envelope_upper is not None and self.envelope_lower is not None
        has_auto_points = (len(self.auto_tmax) + len(self.auto_tmin)) >= 2
        has_sim = self.T_simulated is not None or self.alpha_calc is not None

        self.interp_btn.setEnabled(has_data and has_pts)
        self.auto_select_btn.setEnabled(has_data)
        self.simulate_btn.setEnabled(has_env and has_auto_points)
        self.calc_simple_btn.setEnabled(has_data)
        self.export_btn.setEnabled(has_sim)
    
    def update_thickness(self):
        if self.updating_thickness:
            return
            
        self.updating_thickness = True
        
        text = self.thickness_input.text()
        if text == "" or text == "0":
            self.thickness_nm = 0
            self.updating_thickness = False
            return
            
        try:
            t = float(text)
            if t >= 0:
                self.thickness_nm = t
                self._thickness_manually_set = True
        except ValueError:
            pass
            
        self.updating_thickness = False
    
    def update_simple_thickness(self):
        try:
            t = float(self.simple_thickness_input.text())
            if t > 0:
                self.simple_thickness_nm = t
            else:
                self.simple_thickness_nm = 100.0
                self.simple_thickness_input.setText("100")
        except ValueError:
            self.simple_thickness_nm = 100.0
            self.simple_thickness_input.setText("100")
    
    def update_substrate_index(self):
        try:
            s = float(self.substrate_input.text())
            if s <= 0:
                raise ValueError
            self.substrate_refractive_index = s
        except ValueError:
            self.substrate_input.setText("1.4585")
            self.substrate_refractive_index = 1.4585
    
    def update_fit_quality(self, rmse=None):
        if rmse is not None:
            max_possible_rmse = 0.5
            fit_pct = max(0, min(100, (1 - rmse / max_possible_rmse) * 100))
            
            color = '#27AE60' if fit_pct > 80 else '#F39C12' if fit_pct > 60 else '#E74C3C'
            self.fit_percentage_label.setText(f"📊 Fit Quality: {fit_pct:.1f}%")
            self.fit_percentage_label.setStyleSheet(f"""
                QLabel {{
                    font-size: 12px;
                    font-family: 'Times New Roman';
                    padding: 6px;
                    background-color: #fef5e7;
                    border-radius: 6px;
                    font-weight: bold;
                    color: {color};
                }}
            """)
        else:
            self.fit_percentage_label.setText("📊 Fit Quality: --%")
    
    # ----------------------------------------------------------------------
    # DATA LOADING AND CLEARING
    # ----------------------------------------------------------------------
    def load_data(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open File", "",
            "CSV/TXT/Excel (*.csv *.txt *.xlsx *.xls)"
        )
        if not path:
            return

        try:
            if path.endswith(".csv"):
                df = pd.read_csv(path, header=None)
            elif path.endswith(".txt"):
                df = pd.read_csv(path, sep=r"\s+", header=None)
            else:
                df = pd.read_excel(path, header=None)

            if df.shape[1] < 2:
                QMessageBox.warning(self, "Error", "File must contain at least 2 columns.")
                return

            wl = df.iloc[:, 0].astype(float).to_numpy()
            T = df.iloc[:, 1].astype(float).to_numpy()

            if np.any(T > 1.0):
                T = T / 100.0

            max_val = np.nanmax(T)
            if max_val <= 0:
                QMessageBox.warning(self, "Error", "Transmittance values are zero or negative.")
                return

            T = T / max_val

            if np.any(T < 0) or np.any(T > 1.01):
                QMessageBox.warning(self, "Error", "Transmittance must be between 0 and 1 (or 0–100).")
                return

            self.data = df.values
            self.wavelength = wl
            self.transmittance = T

            self.clear_all()
            self._has_plotted = False
            self.update_main_plot()
            self.update_button_states()

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
    
    def clear_all(self):
        self.tmax = []
        self.tmin = []
        self.auto_tmax = []
        self.auto_tmin = []
        self.envelope_upper = None
        self.envelope_lower = None
        self.xx = None
        self.TM = None
        self.Tm = None
        self.n2 = None
        self.alpha_env = None
        self.T_simulated = None
        self.T_exp_xx = None
        self.interference_free_T = None
        self.alpha_calc = None
        self.E = None
        self.band_gap_direct = None
        self.band_gap_indirect = None
        self.n_final = None
        self.k_final = None
        self.e1 = None
        self.e2 = None
        self.tan_delta = None
        self.skin_depth = None
        self.sigma = None
        self.optical_density = None
        self.alpha_display = None
        self.alphaE = None
        self.alphaE_sqrt = None
        self.alphaE_sq = None
        self.alphaE_23 = None
        self.d_alpha_dE = None
        self.ln_alpha = None
        self.dragging = False
        self.drag_point_type = None
        self.drag_point_index = None
        self.auto_sim_run = False
        self.simple_thickness_nm = 100.0
        
        self.set_selection_mode(None)
        self._has_plotted = False
        
        self.point_counter.setText("📌 Manual Points: Tmax=0, Tmin=0")
        self.auto_point_counter.setText("🤖 Auto Points: Tmax=0, Tmin=0")
        self.thickness_result_label.setText("📐 Estimated Thickness: -- nm")
        self.fit_percentage_label.setText("📊 Fit Quality: --%")
        
        if hasattr(self, 'simple_thickness_input'):
            self.simple_thickness_input.setText("100")
        
        self.update_main_plot()
        
        # Clear all parameter tab plots
        for tab_name, tab in self.param_tabs.items():
            if tab_name == "Main":
                continue
            ax = tab['ax']
            ax.clear()
            ax.set_xlabel(tab['xlabel'], fontsize=11, fontweight='bold')
            ax.set_ylabel(tab['ylabel'], fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_facecolor('#fafafa')
            tab['canvas'].draw()
    
    # ----------------------------------------------------------------------
    # ENVELOPE METHODS
    # ----------------------------------------------------------------------
    def interpolate_env(self, pts):
        if len(pts) < 2:
            return None, None
            
        x, y = zip(*sorted([(p[0], p[1]) for p in pts]))
        x = np.array(x)
        y = np.array(y)

        ux, idx = np.unique(x, return_index=True)
        uy = y[idx]

        if self.curvature_type == 'linear':
            kind = 'linear'
        elif self.curvature_type == 'quadratic':
            kind = 'quadratic' if len(ux) >= 3 else 'linear'
        else:
            kind = 'cubic' if len(ux) >= 4 else 'linear'

        f = interp1d(ux, uy, kind=kind, fill_value="extrapolate")
        xx = np.linspace(ux.min(), ux.max(), 200)
        yy = f(xx)
        return xx, yy
    
    def update_envelopes(self):
        if len(self.tmax) < 2 or len(self.tmin) < 2:
            QMessageBox.warning(
                self, "Error",
                "Need at least 2 points each for Tmax and Tmin."
            )
            return

        try:
            self.envelope_upper = self.interpolate_env(self.tmax)
            self.envelope_lower = self.interpolate_env(self.tmin)
            self.update_main_plot()
            self.update_button_states()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Envelope interpolation failed:\n{e}")
    
    # ----------------------------------------------------------------------
    # AUTO SELECTION AND THICKNESS ESTIMATION
    # ----------------------------------------------------------------------
    def auto_select_both(self):
        if self.wavelength is None or self.transmittance is None:
            QMessageBox.warning(self, "Error", "Load data first.")
            return

        try:
            T = self.transmittance
            wl = self.wavelength

            win = min(51, max(5, (len(T) // 20) * 2 + 1))
            T_smooth = savgol_filter(T, win, 3)

            peaks, peak_props = find_peaks(T_smooth, distance=20, prominence=0.005)
            
            if len(peaks) > 0:
                prominences = peak_props['prominences']
                self.auto_tmax = [(wl[i], T[i], float(prom)) for i, prom in zip(peaks, prominences)]
            else:
                self.auto_tmax = []

            valleys, valley_props = find_peaks(-T_smooth, distance=20, prominence=0.005)
            
            if len(valleys) > 0:
                prominences = valley_props['prominences']
                self.auto_tmin = [(wl[i], T[i], float(prom)) for i, prom in zip(valleys, prominences)]
            else:
                self.auto_tmin = []

            self.auto_point_counter.setText(f"🤖 Auto Points: Tmax={len(self.auto_tmax)}, Tmin={len(self.auto_tmin)}")
            
            if len(self.auto_tmax) + len(self.auto_tmin) >= 2:
                result = self.estimate_thickness_from_auto_peaks()
                self.update_thickness_display(result)
                
                if result.confidence_score > 0.5:
                    self.thickness_nm = result.thickness_nm
                    self.thickness_input.setText(f"{result.thickness_nm:.2f}")
            
            self.update_main_plot()
            self.update_button_states()

            if len(self.auto_tmax) == 0 and len(self.auto_tmin) == 0:
                QMessageBox.warning(self, "Auto Select", "No peaks or valleys found in spectrum.")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Auto selection failed:\n{e}")
    
    def estimate_thickness_from_auto_peaks(self, refractive_index: float = None) -> ThicknessResult:
        if refractive_index is None:
            if self.n2 is not None and len(self.n2) > 0:
                refractive_index = np.nanmean(self.n2)
            else:
                refractive_index = 1.5
        
        combined = []
        for x, y, prominence in self.auto_tmin:
            combined.append((x, y, prominence, 'Tmin'))
        for x, y, prominence in self.auto_tmax:
            combined.append((x, y, prominence, 'Tmax'))
        
        if len(combined) < 2:
            return ThicknessResult(
                thickness_nm=self.thickness_nm,
                optical_thickness_nm=self.thickness_nm * refractive_index,
                confidence_score=0.0,
                n_pairs=0,
                std_dev=None,
                n_peaks=len(combined),
                method='none',
                individual_estimates=[],
                refractive_index=refractive_index
            )
        
        combined.sort(key=lambda p: p[0])
        
        alt_pairs = []
        for i in range(len(combined) - 1):
            x1, y1, prom1, t1 = combined[i]
            x2, y2, prom2, t2 = combined[i + 1]
            if t1 != t2 and abs(x2 - x1) >= self.min_peak_distance:
                quality_score = (prom1 + prom2) / 2.0
                alt_pairs.append((x1, x2, quality_score))
        
        ds = []
        weights = []
        
        if alt_pairs:
            for lam1, lam2, quality in alt_pairs:
                if lam1 == lam2:
                    continue
                d_optical = abs((lam1 * lam2) / (2 * abs(lam1 - lam2)))
                d_physical = d_optical / refractive_index
                
                if self.min_thickness <= d_physical <= self.max_thickness:
                    ds.append(d_physical)
                    weights.append(quality)
        else:
            xs = sorted([p[0] for p in self.auto_tmax + self.auto_tmin])
            for i in range(len(xs) - 1):
                lam1, lam2 = xs[i], xs[i + 1]
                if lam1 == lam2:
                    continue
                if abs(lam2 - lam1) < self.min_peak_distance:
                    continue
                d_optical = abs((lam1 * lam2) / (2 * abs(lam1 - lam2)))
                d_physical = d_optical / refractive_index
                if self.min_thickness <= d_physical <= self.max_thickness:
                    ds.append(d_physical)
                    weights.append(1.0)
        
        if not ds:
            return ThicknessResult(
                thickness_nm=self.thickness_nm,
                optical_thickness_nm=self.thickness_nm * refractive_index,
                confidence_score=0.0,
                n_pairs=0,
                std_dev=None,
                n_peaks=len(combined),
                method='none',
                individual_estimates=[],
                refractive_index=refractive_index
            )
        
        weights = np.array(weights)
        ds = np.array(ds)
        weighted_avg = np.average(ds, weights=weights)
        std_dev = np.std(ds)
        
        n_pairs_score = min(1.0, len(ds) / 5.0)
        consistency_score = max(0.0, 1.0 - (std_dev / weighted_avg) if weighted_avg > 0 else 0)
        method_score = 1.0 if alt_pairs else 0.5
        
        confidence_score = (n_pairs_score * 0.3 + consistency_score * 0.5 + method_score * 0.2)
        method = 'alternating_pairs' if alt_pairs else 'adjacent_peaks'
        
        return ThicknessResult(
            thickness_nm=float(weighted_avg),
            optical_thickness_nm=float(weighted_avg * refractive_index),
            confidence_score=float(confidence_score),
            n_pairs=len(ds),
            std_dev=float(std_dev) if len(ds) > 1 else None,
            n_peaks=len(combined),
            method=method,
            individual_estimates=ds.tolist(),
            refractive_index=refractive_index
        )
    
    def update_thickness_display(self, result: ThicknessResult):
        confidence_pct = result.confidence_score * 100
        confidence_color = '#27AE60' if confidence_pct > 70 else '#F39C12' if confidence_pct > 40 else '#E74C3C'
        
        self.thickness_result_label.setText(
            f"📐 Thickness: {result.thickness_nm:.2f} nm (±{result.std_dev:.2f})"
        )
        self.thickness_result_label.setStyleSheet(f"""
            QLabel {{
                font-size: 12px;
                font-family: 'Times New Roman';
                padding: 6px;
                background-color: #d5f5e3;
                border-radius: 6px;
                font-weight: bold;
                color: {confidence_color};
            }}
        """)
    
    # ----------------------------------------------------------------------
    # SIMULATION METHODS
    # ----------------------------------------------------------------------
    def prepare_envelope_TM_Tm(self):
        min_wl, max_wl = self.get_envelope_wavelength_range()
        
        if min_wl is None or max_wl is None:
            QMessageBox.warning(self, "Error", "No overlapping envelope region found.")
            return False
        
        self.xx = np.linspace(min_wl, max_wl, 200)
        
        TM_interp = interp1d(*self.envelope_upper, fill_value="extrapolate")
        Tm_interp = interp1d(*self.envelope_lower, fill_value="extrapolate")
        
        self.TM = TM_interp(self.xx)
        self.Tm = Tm_interp(self.xx)
        
        self.interference_free_T = np.sqrt(self.TM * self.Tm)
        
        return True
    
    def run_simulation_for_thickness(self, d):
        s = self.substrate_refractive_index
        TM = self.TM
        Tm = self.Tm
        xx = self.xx

        N1 = (2 * s * (TM - Tm) / (TM * Tm)) + (s**2 + 1) / 2
        inner = np.maximum(N1**2 - s**2, 0)
        outer = N1 + np.sqrt(inner)
        n2 = np.sqrt(np.maximum(outer, 0))

        Ti = np.clip((2 * TM * Tm) / (TM + Tm), 1e-10, 1)
        d_cm = d * 1e-7
        alpha = (1 / d_cm) * np.log(1 / Ti)

        phi = 4 * np.pi * n2 * d / xx
        A = 16 * n2**2 * s
        B = (n2 + 1)**3 * (n2 + s**2)
        C = 2 * (n2**2 - 1) * (n2**2 - s**2)
        D = (n2 - 1)**3 * (n2 - s**2)
        F = (8 * n2**2 * s) / Ti

        disc = np.maximum(F**2 - (n2**2 - 1)**3 * (n2**2 - s**4), 0)
        x1 = (F - np.sqrt(disc)) / D
        Tsim = (A * x1) / (B - C * x1 * np.cos(phi) + D * x1**2)

        return n2, alpha, Tsim
    
    def auto_simulate(self):
        if self.envelope_upper is None or self.envelope_lower is None:
            QMessageBox.warning(self, "Error", "Interpolate envelopes first.")
            return

        if len(self.auto_tmax) + len(self.auto_tmin) < 2:
            QMessageBox.warning(self, "Error", "Use Select Tmax & Tmin first.")
            return

        try:
            if self.xx is None:
                success = self.prepare_envelope_TM_Tm()
                if not success:
                    return

            # Prepare T_exp_xx if not yet done
            if self.T_exp_xx is None:
                exp_interp = interp1d(self.wavelength, self.transmittance, 
                                     bounds_error=False, fill_value=np.nan)
                self.T_exp_xx = exp_interp(self.xx)
                
                valid_mask = ~np.isnan(self.T_exp_xx)
                if not np.all(valid_mask):
                    self.xx = self.xx[valid_mask]
                    self.TM = self.TM[valid_mask]
                    self.Tm = self.Tm[valid_mask]
                    self.T_exp_xx = self.T_exp_xx[valid_mask]
                    if self.interference_free_T is not None:
                        self.interference_free_T = self.interference_free_T[valid_mask]

            if self._thickness_manually_set:
                # Use the manually entered thickness directly
                d = self.thickness_nm
                self._thickness_manually_set = False
                
                self.n2, self.alpha_env, self.T_simulated = self.run_simulation_for_thickness(d)
                
                rmse = np.sqrt(np.nanmean((self.T_exp_xx - self.T_simulated)**2))
                self.update_fit_quality(rmse)
                
                self.auto_sim_run = True
                
                self.calculate_all_parameters()
                
                self.update_main_plot()
                self.update_all_parameter_plots()
                self.update_button_states()

            elif not self.auto_sim_run:
                # First auto run: estimate thickness automatically
                thickness_result = self.estimate_thickness_from_auto_peaks()
                d_est = thickness_result.thickness_nm
                
                self.thickness_nm = d_est
                self.updating_thickness = True
                self.thickness_input.setText(f"{d_est:.2f}")
                self.updating_thickness = False
                self._thickness_manually_set = False
                self.update_thickness_display(thickness_result)
                
                self.n2, self.alpha_env, self.T_simulated = self.run_simulation_for_thickness(d_est)
                
                self.auto_sim_run = True
                
                rmse = np.sqrt(np.nanmean((self.T_exp_xx - self.T_simulated)**2))
                self.update_fit_quality(rmse)
                
                self.calculate_all_parameters()
                
                self.update_main_plot()
                self.update_all_parameter_plots()
                self.update_button_states()
                
            else:
                # Subsequent auto run: optimize around current value
                center = self.thickness_nm
                best_t = self.optimize_full_curve(center)
                
                self.thickness_nm = best_t
                self.updating_thickness = True
                self.thickness_input.setText(f"{best_t:.2f}")
                self.updating_thickness = False
                self._thickness_manually_set = False
                
                self.n2, self.alpha_env, self.T_simulated = self.run_simulation_for_thickness(best_t)
                
                rmse = np.sqrt(np.nanmean((self.T_exp_xx - self.T_simulated)**2))
                self.update_fit_quality(rmse)
                
                self.calculate_all_parameters()
                
                self.update_main_plot()
                self.update_all_parameter_plots()
                self.update_button_states()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Simulate failed:\n{e}")
    
    def optimize_full_curve(self, center_thickness):
        def clip_range(a, b):
            return max(a, 1.0), min(b, 5000.0)

        best = center_thickness

        for pct, n in [(0.40, 120), (0.15, 200), (0.05, 300), (0.01, 400)]:
            lo, hi = clip_range(best * (1 - pct), best * (1 + pct))
            arr = np.linspace(lo, hi, n)
            best = self.evaluate_thickness_list(arr)

        return best
    
    def evaluate_thickness_list(self, thickness_array):
        best_d = thickness_array[0]
        best_err = 1e99

        for d in thickness_array:
            try:
                n2, alpha, Tsim = self.run_simulation_for_thickness(d)
                valid_mask = ~np.isnan(self.T_exp_xx)
                diff = self.T_exp_xx[valid_mask] - Tsim[valid_mask]
                if len(diff) > 0:
                    rmse = np.sqrt(np.nanmean(diff**2))
                    if rmse < best_err:
                        best_err = rmse
                        best_d = d
            except Exception:
                continue

        return best_d
    
    # ----------------------------------------------------------------------
    # PLOTTING METHODS
    # ----------------------------------------------------------------------
    def update_main_plot(self):
        """Update main spectrum tab"""
        main_tab = self.param_tabs.get("Main")
        if not main_tab:
            return
            
        ax = main_tab['ax']
        # Preserve zoom state across redraws
        _restore_zoom = self._has_plotted
        _xlim = ax.get_xlim()
        _ylim = ax.get_ylim()
        ax.clear()
        
        if self.wavelength is not None and self.transmittance is not None:
            ax.plot(self.wavelength, self.transmittance, 'k-', 
                   label="Experimental", linewidth=2.5, alpha=0.9)

        if self.tmax:
            x_tmax, y_tmax = zip(*[(p[0], p[1]) for p in self.tmax])
            ax.plot(x_tmax, y_tmax, 'ro-', label="Upper Envelope (manual)", 
                   markersize=8, linewidth=2, markerfacecolor='red', markeredgecolor='darkred')

        if self.tmin:
            x_tmin, y_tmin = zip(*[(p[0], p[1]) for p in self.tmin])
            ax.plot(x_tmin, y_tmin, 'bo-', label="Lower Envelope (manual)", 
                   markersize=8, linewidth=2, markerfacecolor='blue', markeredgecolor='darkblue')

        if self.auto_tmax:
            x_auto_tmax, y_auto_tmax = zip(*[(p[0], p[1]) for p in self.auto_tmax])
            ax.plot(x_auto_tmax, y_auto_tmax, 'm*', markersize=10, 
                   label="Auto Tmax", linewidth=0, markeredgecolor='purple')

        if self.auto_tmin:
            x_auto_tmin, y_auto_tmin = zip(*[(p[0], p[1]) for p in self.auto_tmin])
            ax.plot(x_auto_tmin, y_auto_tmin, 'c*', markersize=10, 
                   label="Auto Tmin", linewidth=0, markeredgecolor='teal')

        if self.envelope_upper is not None and self.envelope_upper[0] is not None:
            ax.plot(self.envelope_upper[0], self.envelope_upper[1], 'r--', 
                   label="Upper Envelope", linewidth=2.5, alpha=0.8)

        if self.envelope_lower is not None and self.envelope_lower[0] is not None:
            ax.plot(self.envelope_lower[0], self.envelope_lower[1], 'b--', 
                   label="Lower Envelope", linewidth=2.5, alpha=0.8)

        if self.interference_free_T is not None and self.xx is not None:
            T_if_interp = interp1d(self.xx, self.interference_free_T, 
                                  bounds_error=False, fill_value=np.nan)(self.wavelength)
            ax.plot(self.wavelength, T_if_interp, 'c--', 
                   label="Interference-Free T", linewidth=2, alpha=0.8)

        if self.T_simulated is not None and self.xx is not None:
            T_sim_interp = interp1d(self.xx, self.T_simulated, 
                                   bounds_error=False, fill_value=np.nan)(self.wavelength)
            ax.plot(self.wavelength, T_sim_interp, 'g-', 
                   label="Simulated", linewidth=2.5, alpha=0.8)

        ax.set_xlabel(main_tab['xlabel'], fontsize=11, fontweight='bold')
        ax.set_ylabel(main_tab['ylabel'], fontsize=11, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor('#fafafa')

        handles, labels = ax.get_legend_handles_labels()
        uniq = dict(zip(labels, handles))
        if uniq:
            ax.legend(uniq.values(), uniq.keys(), fontsize=9, loc='best', framealpha=0.9)

        if _restore_zoom:
            ax.set_xlim(_xlim)
            ax.set_ylim(_ylim)

        if self.wavelength is not None:
            self._has_plotted = True

        main_tab['canvas'].draw()
        self.update_button_states()
    
    def update_all_parameter_plots(self):
        """Update all individual parameter tabs"""
        if self.wavelength is None:
            return
        
        # Get valid data mask
        if self.xx is not None and len(self.xx) > 0:
            min_wl, max_wl = self.get_envelope_wavelength_range()
            if min_wl is not None and max_wl is not None:
                valid_mask = (self.wavelength >= min_wl) & (self.wavelength <= max_wl)
            else:
                valid_mask = np.ones_like(self.wavelength, dtype=bool)
        else:
            valid_mask = np.ones_like(self.wavelength, dtype=bool)
        
        energy_valid_mask = valid_mask if self.E is not None else np.ones_like(self.wavelength, dtype=bool)
        
        # Define color mapping for each tab
        colors = {
            'n': 'blue', 'k': 'red', 'ε₁': 'green', 'ε₂': 'orange',
            'tanδ': 'purple', 'SD': 'cyan', 'σ': 'brown', 'OD': 'gray',
            'α': 'blue', 'αE': 'red', '(αE)²': 'green', '(αE)^½': 'magenta',
            'dα/dE': 'cyan', 'lnα': 'orange'
        }
        
        # Update n tab
        if "n" in self.param_tabs and self.n_final is not None:
            tab = self.param_tabs["n"]
            ax = tab['ax']
            ax.clear()
            if not np.all(np.isnan(self.n_final[valid_mask])):
                ax.plot(self.wavelength[valid_mask], self.n_final[valid_mask], colors.get('n', 'b-'), linewidth=2)
            ax.set_xlabel(tab['xlabel'], fontsize=11, fontweight='bold')
            ax.set_ylabel(tab['ylabel'], fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_facecolor('#fafafa')
            tab['canvas'].draw()
        
        # Update k tab
        if "k" in self.param_tabs and self.k_final is not None:
            tab = self.param_tabs["k"]
            ax = tab['ax']
            ax.clear()
            if not np.all(np.isnan(self.k_final[valid_mask])):
                ax.plot(self.wavelength[valid_mask], self.k_final[valid_mask], colors.get('k', 'r-'), linewidth=2)
            ax.set_xlabel(tab['xlabel'], fontsize=11, fontweight='bold')
            ax.set_ylabel(tab['ylabel'], fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_facecolor('#fafafa')
            tab['canvas'].draw()
        
        # Update ε₁ tab
        if "ε₁" in self.param_tabs and self.e1 is not None:
            tab = self.param_tabs["ε₁"]
            ax = tab['ax']
            ax.clear()
            if not np.all(np.isnan(self.e1[valid_mask])):
                ax.plot(self.wavelength[valid_mask], self.e1[valid_mask], colors.get('ε₁', 'g-'), linewidth=2)
            ax.set_xlabel(tab['xlabel'], fontsize=11, fontweight='bold')
            ax.set_ylabel(tab['ylabel'], fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_facecolor('#fafafa')
            tab['canvas'].draw()
        
        # Update ε₂ tab
        if "ε₂" in self.param_tabs and self.e2 is not None:
            tab = self.param_tabs["ε₂"]
            ax = tab['ax']
            ax.clear()
            if not np.all(np.isnan(self.e2[valid_mask])):
                ax.plot(self.wavelength[valid_mask], self.e2[valid_mask], colors.get('ε₂', 'orange'), linewidth=2)
            ax.set_xlabel(tab['xlabel'], fontsize=11, fontweight='bold')
            ax.set_ylabel(tab['ylabel'], fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_facecolor('#fafafa')
            tab['canvas'].draw()
        
        # Update tanδ tab
        if "tanδ" in self.param_tabs and self.tan_delta is not None:
            tab = self.param_tabs["tanδ"]
            ax = tab['ax']
            ax.clear()
            if not np.all(np.isnan(self.tan_delta[valid_mask])):
                ax.plot(self.wavelength[valid_mask], self.tan_delta[valid_mask], colors.get('tanδ', 'purple'), linewidth=2)
            ax.set_xlabel(tab['xlabel'], fontsize=11, fontweight='bold')
            ax.set_ylabel(tab['ylabel'], fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_facecolor('#fafafa')
            tab['canvas'].draw()
        
        # Update SD (Skin Depth) tab
        if "SD" in self.param_tabs and self.skin_depth is not None:
            tab = self.param_tabs["SD"]
            ax = tab['ax']
            ax.clear()
            if not np.all(np.isnan(self.skin_depth[valid_mask])):
                ax.plot(self.wavelength[valid_mask], self.skin_depth[valid_mask], colors.get('SD', 'cyan'), linewidth=2)
            ax.set_xlabel(tab['xlabel'], fontsize=11, fontweight='bold')
            ax.set_ylabel(tab['ylabel'], fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_facecolor('#fafafa')
            tab['canvas'].draw()
        
        # Update σ tab
        if "σ" in self.param_tabs and self.sigma is not None:
            tab = self.param_tabs["σ"]
            ax = tab['ax']
            ax.clear()
            if not np.all(np.isnan(self.sigma[valid_mask])):
                ax.plot(self.wavelength[valid_mask], self.sigma[valid_mask], colors.get('σ', 'brown'), linewidth=2)
            ax.set_xlabel(tab['xlabel'], fontsize=11, fontweight='bold')
            ax.set_ylabel(tab['ylabel'], fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_facecolor('#fafafa')
            tab['canvas'].draw()
        
        # Update OD tab
        if "OD" in self.param_tabs and self.optical_density is not None:
            tab = self.param_tabs["OD"]
            ax = tab['ax']
            ax.clear()
            if not np.all(np.isnan(self.optical_density[valid_mask])):
                ax.plot(self.wavelength[valid_mask], self.optical_density[valid_mask], colors.get('OD', 'gray'), linewidth=2)
            ax.set_xlabel(tab['xlabel'], fontsize=11, fontweight='bold')
            ax.set_ylabel(tab['ylabel'], fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_facecolor('#fafafa')
            tab['canvas'].draw()
        
        # Update α tab
        if "α" in self.param_tabs and self.alpha_display is not None and self.E is not None:
            tab = self.param_tabs["α"]
            ax = tab['ax']
            ax.clear()
            if not np.all(np.isnan(self.alpha_display[energy_valid_mask])):
                ax.plot(self.E[energy_valid_mask], self.alpha_display[energy_valid_mask], colors.get('α', 'b-'), linewidth=2)
                ax.set_yscale('log')
            ax.set_xlabel(tab['xlabel'], fontsize=11, fontweight='bold')
            ax.set_ylabel(tab['ylabel'], fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_facecolor('#fafafa')
            tab['canvas'].draw()
        
        # Update (αE)² tab
        if "(αE)²" in self.param_tabs and self.alphaE_sq is not None and self.E is not None:
            tab = self.param_tabs["(αE)²"]
            ax = tab['ax']
            ax.clear()
            if not np.all(np.isnan(self.alphaE_sq[energy_valid_mask])):
                ax.plot(self.E[energy_valid_mask], self.alphaE_sq[energy_valid_mask], colors.get('(αE)²', 'g-'), linewidth=2)
                if self.band_gap_direct and self.band_gap_direct > 0:
                    ax.axvline(x=self.band_gap_direct, color='k', linestyle='--', linewidth=2,
                              label=f'Eg={self.band_gap_direct:.3f}eV')
                    ax.legend(fontsize=9)
            ax.set_xlabel(tab['xlabel'], fontsize=11, fontweight='bold')
            ax.set_ylabel(tab['ylabel'], fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_facecolor('#fafafa')
            tab['canvas'].draw()
        
        # Update (αE)^½ tab
        if "(αE)^½" in self.param_tabs and self.alphaE_sqrt is not None and self.E is not None:
            tab = self.param_tabs["(αE)^½"]
            ax = tab['ax']
            ax.clear()
            if not np.all(np.isnan(self.alphaE_sqrt[energy_valid_mask])):
                ax.plot(self.E[energy_valid_mask], self.alphaE_sqrt[energy_valid_mask], colors.get('(αE)^½', 'm-'), linewidth=2)
                if self.band_gap_indirect and self.band_gap_indirect > 0:
                    ax.axvline(x=self.band_gap_indirect, color='k', linestyle='--', linewidth=2,
                              label=f'Eg={self.band_gap_indirect:.3f}eV')
                    ax.legend(fontsize=9)
            ax.set_xlabel(tab['xlabel'], fontsize=11, fontweight='bold')
            ax.set_ylabel(tab['ylabel'], fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_facecolor('#fafafa')
            tab['canvas'].draw()
        
        # Update dα/dE tab
        if "dα/dE" in self.param_tabs and self.d_alpha_dE is not None and self.E is not None:
            tab = self.param_tabs["dα/dE"]
            ax = tab['ax']
            ax.clear()
            if not np.all(np.isnan(self.d_alpha_dE[energy_valid_mask])):
                ax.plot(self.E[energy_valid_mask], self.d_alpha_dE[energy_valid_mask], colors.get('dα/dE', 'c-'), linewidth=2)
            ax.set_xlabel(tab['xlabel'], fontsize=11, fontweight='bold')
            ax.set_ylabel(tab['ylabel'], fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_facecolor('#fafafa')
            tab['canvas'].draw()
        
        # Update lnα tab
        if "lnα" in self.param_tabs and self.ln_alpha is not None and self.E is not None:
            tab = self.param_tabs["lnα"]
            ax = tab['ax']
            ax.clear()
            if not np.all(np.isnan(self.ln_alpha[energy_valid_mask])):
                ax.plot(self.E[energy_valid_mask], self.ln_alpha[energy_valid_mask], colors.get('lnα', 'orange'), linewidth=2)
            ax.set_xlabel(tab['xlabel'], fontsize=11, fontweight='bold')
            ax.set_ylabel(tab['ylabel'], fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_facecolor('#fafafa')
            tab['canvas'].draw()
    
    def update_all_plots(self):
        """Update all plots including main and parameters"""
        self.update_main_plot()
        self.update_all_parameter_plots()
    
    def on_tab_changed(self, index):
        """Redraw the canvas of the newly selected tab"""
        tab_text = self.main_tab_widget.tabText(index)
        tab = self.param_tabs.get(tab_text)
        if tab:
            tab['canvas'].draw_idle()
    
    # ----------------------------------------------------------------------
    # MOUSE EVENT HANDLERS
    # ----------------------------------------------------------------------
    def on_click(self, event):
        if event.inaxes != self.main_ax:
            return

        if self.is_navigation_mode_active():
            return

        if self.selection_mode is None:
            return

        if event.xdata is not None and event.ydata is not None:
            self.cursor_label.setText(f"🎯 Cursor: λ = {event.xdata:.1f} nm, T = {event.ydata:.3f}")

        if self.wavelength is None or self.transmittance is None:
            return

        x, y = event.xdata, event.ydata
        snapped_x, snapped_y = self.snap_x_only(x, y)
        
        if snapped_x != x:
            self.cursor_label.setText(f"🎯 Cursor: λ = {snapped_x:.1f} nm (snapped), T = {snapped_y:.3f}")
        
        x, y = snapped_x, snapped_y
        
        if x is None or y is None:
            return

        # Adaptive tolerance based on visible range (zoom level)
        if self.main_ax is not None:
            xlim = self.main_ax.get_xlim()
            ylim = self.main_ax.get_ylim()
            tolx = abs(xlim[1] - xlim[0]) * 0.01
            toly = abs(ylim[1] - ylim[0]) * 0.01
        else:
            tolx = (self.wavelength.max() - self.wavelength.min()) * 0.01
            toly = 0.02

        if event.button == 3:
            # Right-click removes nearest auto or manual point
            for i, (px, py, *_rest) in enumerate(self.auto_tmax):
                if abs(px - x) < tolx and abs(py - y) < toly:
                    del self.auto_tmax[i]
                    self.auto_point_counter.setText(f"🤖 Auto Points: Tmax={len(self.auto_tmax)}, Tmin={len(self.auto_tmin)}")
                    self.update_main_plot()
                    return

            for i, (px, py, *_rest) in enumerate(self.auto_tmin):
                if abs(px - x) < tolx and abs(py - y) < toly:
                    del self.auto_tmin[i]
                    self.auto_point_counter.setText(f"🤖 Auto Points: Tmax={len(self.auto_tmax)}, Tmin={len(self.auto_tmin)}")
                    self.update_main_plot()
                    return

            for i, (px, py) in enumerate(self.tmax):
                if abs(px - x) < tolx and abs(py - y) < toly:
                    del self.tmax[i]
                    self.point_counter.setText(f"📌 Manual Points: Tmax={len(self.tmax)}, Tmin={len(self.tmin)}")
                    self.update_main_plot()
                    return
            
            for i, (px, py) in enumerate(self.tmin):
                if abs(px - x) < tolx and abs(py - y) < toly:
                    del self.tmin[i]
                    self.point_counter.setText(f"📌 Manual Points: Tmax={len(self.tmax)}, Tmin={len(self.tmin)}")
                    self.update_main_plot()
                    return
            return

        if event.button == 1:
            for i, (px, py) in enumerate(self.tmax):
                if abs(px - x) < tolx and abs(py - y) < toly:
                    self.dragging = True
                    self.drag_point_type = 'Tmax'
                    self.drag_point_index = i
                    return

            for i, (px, py) in enumerate(self.tmin):
                if abs(px - x) < tolx and abs(py - y) < toly:
                    self.dragging = True
                    self.drag_point_type = 'Tmin'
                    self.drag_point_index = i
                    return

            new_point = (x, y)
            
            if self.selection_mode == 'Tmax':
                if not self.tmax:
                    self.tmax.append(new_point)
                else:
                    inserted = False
                    for i, (px, py) in enumerate(self.tmax):
                        if x < px:
                            self.tmax.insert(i, new_point)
                            inserted = True
                            break
                    if not inserted:
                        self.tmax.append(new_point)
                self.point_counter.setText(f"📌 Manual Points: Tmax={len(self.tmax)}, Tmin={len(self.tmin)}")
            elif self.selection_mode == 'Tmin':
                if not self.tmin:
                    self.tmin.append(new_point)
                else:
                    inserted = False
                    for i, (px, py) in enumerate(self.tmin):
                        if x < px:
                            self.tmin.insert(i, new_point)
                            inserted = True
                            break
                    if not inserted:
                        self.tmin.append(new_point)
                self.point_counter.setText(f"📌 Manual Points: Tmax={len(self.tmax)}, Tmin={len(self.tmin)}")
            elif self.selection_mode == 'AutoTmax':
                self.auto_tmax.append((x, y, 1.0))
                self.auto_tmax.sort(key=lambda p: p[0])
                self.auto_point_counter.setText(f"🤖 Auto Points: Tmax={len(self.auto_tmax)}, Tmin={len(self.auto_tmin)}")
            elif self.selection_mode == 'AutoTmin':
                self.auto_tmin.append((x, y, 1.0))
                self.auto_tmin.sort(key=lambda p: p[0])
                self.auto_point_counter.setText(f"🤖 Auto Points: Tmax={len(self.auto_tmax)}, Tmin={len(self.auto_tmin)}")

            self.update_main_plot()
    
    def on_motion(self, event):
        if event.inaxes == self.main_ax and event.xdata is not None and event.ydata is not None:
            self.cursor_label.setText(f"🎯 Cursor: λ = {event.xdata:.1f} nm, T = {event.ydata:.3f}")

        if self.is_navigation_mode_active():
            return

        if not self.dragging or self.selection_mode is None:
            return
            
        if event.inaxes != self.main_ax:
            return
            
        if event.xdata is None or event.ydata is None:
            return

        x, y = event.xdata, event.ydata
        snapped_x, snapped_y = self.snap_x_only(x, y)
        
        if self.drag_point_type == 'Tmax' and self.drag_point_index is not None:
            self.tmax[self.drag_point_index] = (snapped_x, y)
        elif self.drag_point_type == 'Tmin' and self.drag_point_index is not None:
            self.tmin[self.drag_point_index] = (snapped_x, y)

        self.update_main_plot()
    
    def on_release(self, event):
        dragging_was_active = self.dragging
        
        self.dragging = False
        
        if dragging_was_active and self.drag_point_type is not None and self.selection_mode is not None:
            if self.drag_point_type == 'Tmax' and self.tmax:
                self.tmax.sort(key=lambda p: p[0])
            elif self.drag_point_type == 'Tmin' and self.tmin:
                self.tmin.sort(key=lambda p: p[0])
            
            self.update_main_plot()
        
        self.drag_point_type = None
        self.drag_point_index = None
    
    # ----------------------------------------------------------------------
    # SIMPLE ANALYSIS
    # ----------------------------------------------------------------------
    def calculate_simple_parameters(self):
        if self.wavelength is None:
            QMessageBox.warning(self, "Error", "Load data first.")
            return

        try:
            d = self.simple_thickness_nm
            
            if d <= 0:
                QMessageBox.warning(self, "Error", "Thickness must be greater than 0.")
                return
                
            self.E = 1240.0 / self.wavelength
            
            d_cm = d * 1e-7
            T_clean = np.clip(self.transmittance, 1e-6, 1.0)
            self.alpha_calc = -np.log(T_clean) / d_cm
            
            self.thickness_nm = d
            self.thickness_input.setText(f"{d:.2f}")
            
            self.calculate_all_parameters()
            
            if self.alphaE is not None:
                ahv_direct = self.alphaE_sq
                ahv_indirect = self.alphaE_sqrt
                
                self.band_gap_direct = self.extract_band_gap(self.E, ahv_direct)
                self.band_gap_indirect = self.extract_band_gap(self.E, ahv_indirect)
            
            self.update_main_plot()
            self.update_all_parameter_plots()
            self.update_button_states()

            QMessageBox.information(self, "Success", 
                f"Simple analysis completed successfully!\n\n"
                f"Thickness used: {d:.2f} nm\n"
                f"Energy range: {self.E.min():.2f} - {self.E.max():.2f} eV\n"
                f"Direct band gap: {self.band_gap_direct:.3f} eV\n"
                f"Indirect band gap: {self.band_gap_indirect:.3f} eV")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Calculation failed:\n{str(e)}")
    
    def calculate_all_parameters(self):
        if self.wavelength is None:
            return
        
        self.E = 1240.0 / self.wavelength
        
        if self.alpha_env is not None and self.xx is not None:
            self.alpha_display = interp1d(self.xx, self.alpha_env, 
                                         bounds_error=False, fill_value="extrapolate")(self.wavelength)
            self.n_final = interp1d(self.xx, self.n2, 
                                   bounds_error=False, fill_value="extrapolate")(self.wavelength)
            self.k_final = self.alpha_display * self.wavelength * 1e-7 / (4 * np.pi)
        elif self.alpha_calc is not None:
            self.alpha_display = self.alpha_calc
            self.n_final = np.full_like(self.wavelength, np.nan)
            self.k_final = np.full_like(self.wavelength, np.nan)
        else:
            self.alpha_display = None
            self.n_final = np.full_like(self.wavelength, np.nan)
            self.k_final = np.full_like(self.wavelength, np.nan)
        
        if self.alpha_display is not None:
            if self.n_final is not None and self.k_final is not None and not np.all(np.isnan(self.n_final)):
                self.e1 = self.n_final**2 - self.k_final**2
                self.e2 = 2.0 * self.n_final * self.k_final
                self.tan_delta = np.abs(self.e2 / (self.e1 + 1e-10))
            else:
                self.e1 = np.full_like(self.wavelength, np.nan)
                self.e2 = np.full_like(self.wavelength, np.nan)
                self.tan_delta = np.full_like(self.wavelength, np.nan)
            
            self.skin_depth = 1e7 / np.maximum(self.alpha_display, 1e-10)
            
            c_cm_s = 3e10
            n_avg = np.nanmean(self.n_final) if not np.all(np.isnan(self.n_final)) else 1.5
            self.sigma = (self.alpha_display * n_avg * c_cm_s) / (4.0 * np.pi)
            
            d_cm = self.thickness_nm * 1e-7
            self.optical_density = self.alpha_display * d_cm
            
            self.alphaE = self.alpha_display * self.E
            self.alphaE_sqrt = np.sqrt(np.maximum(self.alphaE, 0))
            self.alphaE_sq = self.alphaE**2
            self.alphaE_23 = self.alphaE**(2/3)
            
            self.d_alpha_dE = np.gradient(self.alpha_display, self.E)
            self.ln_alpha = np.log(np.maximum(self.alpha_display, 1e-10))
    
    def extract_band_gap(self, energy, y_values):
        try:
            derivative = np.gradient(y_values, energy)
            
            positive_idx = np.where(derivative > np.max(derivative)*0.1)[0]
            
            if len(positive_idx) < 5:
                return 0.0
            
            start_idx = positive_idx[0]
            end_idx = positive_idx[-1]
            
            region_len = end_idx - start_idx
            fit_start = start_idx + int(region_len * 0.2)
            fit_end = end_idx - int(region_len * 0.2)
            
            if fit_end - fit_start < 5:
                fit_start = start_idx
                fit_end = end_idx
            
            x_fit = energy[fit_start:fit_end]
            y_fit = y_values[fit_start:fit_end]
            
            mask = np.isfinite(x_fit) & np.isfinite(y_fit)
            x_fit = x_fit[mask]
            y_fit = y_fit[mask]
            
            if len(x_fit) < 3:
                return 0.0
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_fit, y_fit)
            
            band_gap = -intercept / slope if slope != 0 else 0
            
            return max(0, band_gap)
            
        except Exception:
            return 0.0
    
    # ----------------------------------------------------------------------
    # EXPORT METHODS
    # ----------------------------------------------------------------------
    def extract_optical_properties(self):
        if self.wavelength is None:
            QMessageBox.warning(self, "Error", "No data to export.")
            return

        try:
            default_filename = f"Optical_Properties_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
            
            file_path, selected_filter = QFileDialog.getSaveFileName(
                self,
                "Save Optical Properties As",
                os.path.join(self.save_folder, default_filename),
                "CSV Files (*.csv);;All Files (*.*)"
            )
            
            if not file_path:
                return
            
            if not file_path.endswith('.csv'):
                file_path += '.csv'
            
            self.save_folder = os.path.dirname(file_path)
            
            df = pd.DataFrame()
            
            df["Wavelength_nm"] = self.wavelength
            df["Transmittance"] = self.transmittance
            
            if self.T_simulated is not None and self.xx is not None:
                T_sim_interp = interp1d(self.xx, self.T_simulated, 
                                       bounds_error=False, fill_value="extrapolate")(self.wavelength)
                T_sim_interp = np.clip(T_sim_interp, 0, 1)
                wl_min = self.xx.min()
                wl_max = self.xx.max()
                T_sim_interp[(self.wavelength < wl_min) | (self.wavelength > wl_max)] = np.nan
                df["Simulated_Transmittance"] = T_sim_interp
            else:
                df["Simulated_Transmittance"] = np.nan
            
            if self.interference_free_T is not None and self.xx is not None:
                T_if_interp = interp1d(self.xx, self.interference_free_T, 
                                      bounds_error=False, fill_value="extrapolate")(self.wavelength)
                wl_min = self.xx.min()
                wl_max = self.xx.max()
                T_if_interp[(self.wavelength < wl_min) | (self.wavelength > wl_max)] = np.nan
                df["Interference_Free_T"] = T_if_interp
            else:
                df["Interference_Free_T"] = np.nan
            
            df["n"] = self.n_final if self.n_final is not None else np.nan
            df["k"] = self.k_final if self.k_final is not None else np.nan
            df["e1"] = self.e1 if self.e1 is not None else np.nan
            df["e2"] = self.e2 if self.e2 is not None else np.nan
            df["tan_delta"] = self.tan_delta if self.tan_delta is not None else np.nan
            df["Skin_Depth_nm"] = self.skin_depth if self.skin_depth is not None else np.nan
            df["Sigma_S_per_cm"] = self.sigma if self.sigma is not None else np.nan
            df["Optical_Density"] = self.optical_density if self.optical_density is not None else np.nan
            
            df["Energy_eV"] = self.E if self.E is not None else 1240.0/self.wavelength
            df["Alpha_cm-1"] = self.alpha_display if self.alpha_display is not None else np.nan
            df["Alpha_E"] = self.alphaE if self.alphaE is not None else np.nan
            df["(AlphaE)^2"] = self.alphaE_sq if self.alphaE_sq is not None else np.nan
            df["(AlphaE)^0.5"] = self.alphaE_sqrt if self.alphaE_sqrt is not None else np.nan
            df["dAlpha_dE"] = self.d_alpha_dE if self.d_alpha_dE is not None else np.nan
            df["ln_Alpha"] = self.ln_alpha if self.ln_alpha is not None else np.nan

            column_order = [
                "Wavelength_nm", "Transmittance", "Simulated_Transmittance", "Interference_Free_T",
                "n", "k", "e1", "e2", "tan_delta",
                "Skin_Depth_nm", "Sigma_S_per_cm", "Optical_Density",
                "Energy_eV", "Alpha_cm-1", "Alpha_E",
                "(AlphaE)^2", "(AlphaE)^0.5", "dAlpha_dE", "ln_Alpha"
            ]
            
            existing_columns = [col for col in column_order if col in df.columns]
            df = df[existing_columns]

            for col in df.columns:
                if len(df[col]) != len(self.wavelength):
                    if len(df[col]) == 200 and self.xx is not None:
                        col_data = df[col].values
                        finite_mask = np.isfinite(col_data)
                        if np.any(finite_mask):
                            x_finite = self.xx[finite_mask] if len(col_data) == 200 else self.wavelength[finite_mask]
                            y_finite = col_data[finite_mask]
                            if len(x_finite) > 3:
                                f_interp = interp1d(x_finite, y_finite, 
                                                  bounds_error=False, 
                                                  fill_value="extrapolate")
                                df[col] = f_interp(self.wavelength)

            df.to_csv(file_path, index=False, encoding='utf-8')
            
            QMessageBox.information(self, "Success", f"Data exported to:\n{file_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save file:\n{str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    font = QFont("Times New Roman", 12)
    app.setFont(font)
    window = OpticalEnvelopeApp()
    sys.exit(app.exec_())
