import sys
import os
import numpy as np
import pyqtgraph as pg
import hashlib
import time
import threading
import queue
import logging
from collections import deque
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QLineEdit,
                             QVBoxLayout, QHBoxLayout, QWidget, QPushButton,
                             QComboBox, QTabWidget, QGroupBox, QSizePolicy,
                             QGridLayout, QSlider, QFrame, QScrollArea, QTextEdit,
                             QTableWidget, QTableWidgetItem, QHeaderView, QListWidget)
from PyQt5.QtCore import pyqtSlot, Qt, QTimer, QRectF
from PyQt5.QtGui import QFont, QPainter, QColor, QPen, QBrush, QLinearGradient, QPixmap

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec

# --- APRENDIZAJE AUTOMÁTICO ---
from sklearn.ensemble import IsolationForest

# --- MATEMÁTICAS AVANZADAS ---
from scipy.signal import hilbert, freqz

from hilo_modbus import TrabajoModbus
from db_manager    import DBManager
from firma_arranque import AnalizadorFirma, DetectorTendencia

# --- CONFIGURACIÓN DE AUDITORÍA ---
logging.basicConfig(
    filename='hmi_edge_errores.log', 
    level=logging.WARNING, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ══════════════════════════════════════════════════════════════════════════════
# FILTRO PREDICTIVO KALMAN 1D
# ══════════════════════════════════════════════════════════════════════════════
class Kalman1D:
    def __init__(self, q=0.01, r=1.0):
        self.x = 0.0 
        self.p = 1.0 
        self.q = q   
        self.r = r   

    def predict(self):
        self.p += self.q
        return self.x

    def update(self, z):
        k = self.p / (self.p + self.r)
        self.x = self.x + k * (z - self.x)
        self.p = (1 - k) * self.p
        return self.x

# ══════════════════════════════════════════════════════════════════════════════
# VISOR DE IMÁGENES SEGURO
# ══════════════════════════════════════════════════════════════════════════════
class VisorImagen(QWidget):
    def __init__(self):
        super().__init__()
        self.pixmap_original = None
        self.mensaje = "Seleccione una imagen del menú lateral para proyectarla.\n\nAsegúrese de que las imágenes estén en la misma carpeta que este script."
        self.setStyleSheet("background:#050811; border:1px solid #1A3060;")
        
    def set_imagen(self, ruta):
        if os.path.exists(ruta):
            self.pixmap_original = QPixmap(ruta)
            self.mensaje = ""
        else:
            self.pixmap_original = None
            self.mensaje = f"⚠ No se encontró el archivo:\n{ruta}\n\nDebe estar en la misma carpeta que el programa."
        self.update() 
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        if self.pixmap_original and not self.pixmap_original.isNull():
            scaled_pixmap = self.pixmap_original.scaled(
                self.width(), self.height(), 
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            x = (self.width() - scaled_pixmap.width()) // 2
            y = (self.height() - scaled_pixmap.height()) // 2
            painter.drawPixmap(x, y, scaled_pixmap)
        else:
            painter.setPen(QPen(QColor('#556688')))
            painter.setFont(QFont("Consolas", 12))
            painter.drawText(self.rect(), Qt.AlignCenter | Qt.TextWordWrap, self.mensaje)

# ══════════════════════════════════════════════════════════════════════════════
# PALETA GLOBAL INDUSTRIAL
# ══════════════════════════════════════════════════════════════════════════════
C_BG        = '#0A0E1A'
C_AX_BG     = '#0D1525'
C_PANEL     = '#0F1830'
C_BORDER    = '#1A3060'
C_GRID      = '#152040'
C_VIB       = '#00FFCC'
C_CORR      = '#FF8C00'
C_VIB_RAW   = '#FF00FF'
C_CORR_RAW  = '#FFE000'
C_REF       = '#39FF14'
C_ALARM     = '#FF3333'
C_EMA       = '#00BFFF'
C_IIR       = '#39FF14'
C_RAW_DIM   = '#FF6B6B'
C_TEXT      = '#C8D8F0'
C_TEXT_DIM  = '#556688'

pg.setConfigOption('background', C_BG)
pg.setConfigOption('foreground', C_TEXT)
pg.setConfigOptions(antialias=True)

def _estilizar_ax(ax, titulo, ylabel_izq, color_izq):
    ax.set_facecolor(C_AX_BG)
    ax.set_title(titulo, color='#AACCFF', fontsize=9, fontweight='bold', pad=6)
    ax.tick_params(colors='#6688AA', labelsize=7)
    ax.set_ylabel(ylabel_izq, color=color_izq, fontsize=8)
    ax.yaxis.label.set_color(color_izq)
    ax.tick_params(axis='y', labelcolor=color_izq)
    for spine in ax.spines.values():
        spine.set_edgecolor(C_BORDER)
    ax.grid(True, color=C_GRID, linestyle='--', linewidth=0.5, alpha=0.8)
    ax.set_xlabel("Muestras", color=C_TEXT_DIM, fontsize=7)

def _bloquear_grafica_pg(plot_item):
    plot_item.setMouseEnabled(x=False, y=False)
    if hasattr(plot_item, 'hideButtons'):
        plot_item.hideButtons()
    plot_item.setMenuEnabled(False)

class DisplayDigital(QWidget):
    def __init__(self, titulo, unidad, color_acento, min_val=0, max_val=100, parent=None):
        super().__init__(parent)
        self.titulo     = titulo
        self.unidad     = unidad
        self.color      = QColor(color_acento)
        self.min_val    = min_val
        self.max_val    = max_val
        self.valor      = 0.0
        self.setMinimumSize(160, 110)
        self.setMaximumSize(220, 130)

    def set_valor(self, v):
        self.valor = float(v)
        self.update()

    def set_color_acento(self, color_hex):
        self.color = QColor(color_hex)
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        p.setPen(QPen(self.color.darker(180), 1))
        grad = QLinearGradient(0, 0, 0, h)
        grad.setColorAt(0, QColor('#0D1830'))
        grad.setColorAt(1, QColor('#070C18'))
        p.setBrush(QBrush(grad))
        p.drawRoundedRect(1, 1, w - 2, h - 2, 8, 8)

        pct = max(0.0, min(1.0, (self.valor - self.min_val) / max(1, self.max_val - self.min_val)))
        bar_w = int((w - 16) * pct)
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(self.color.darker(200)))
        p.drawRoundedRect(8, h - 14, w - 16, 8, 3, 3)
        if bar_w > 0:
            bar_color = self.color if pct < 0.85 else QColor('#FF3333')
            p.setBrush(QBrush(bar_color))
            p.drawRoundedRect(8, h - 14, bar_w, 8, 3, 3)

        p.setPen(QPen(self.color.lighter(130)))
        p.setFont(QFont("Consolas", 8, QFont.Bold))
        p.drawText(8, 18, self.titulo.upper())

        texto_val = f"{self.valor:.1f}"
        p.setPen(QPen(QColor('white')))
        p.setFont(QFont("Consolas", 26, QFont.Bold))
        fm = p.fontMetrics()
        tw = fm.horizontalAdvance(texto_val)
        p.drawText(w - tw - 40, 62, texto_val)

        p.setPen(QPen(self.color))
        p.setFont(QFont("Consolas", 10))
        p.drawText(w - 36, 62, self.unidad)
        p.end()

class PanelDatosVivo(QWidget):
    def __init__(self, campos, parent=None):
        super().__init__(parent)
        self._campos = campos
        self._valores = {c[0]: 0.0 for c in campos}
        self.setFixedHeight(52)
        self.setStyleSheet(f"background:{C_PANEL};border:1px solid {C_BORDER};border-radius:5px;")
        
        layout = QHBoxLayout()
        layout.setContentsMargins(10, 4, 10, 4)
        layout.setSpacing(16)

        self._labels = {}
        for clave, etiqueta, unidad, color in campos:
            col_layout = QVBoxLayout()
            col_layout.setSpacing(0)

            lbl_titulo = QLabel(etiqueta.upper())
            lbl_titulo.setFont(QFont("Consolas", 7, QFont.Bold))
            lbl_titulo.setStyleSheet(f"color:{C_TEXT_DIM};background:transparent;border:none;")
            lbl_titulo.setAlignment(Qt.AlignCenter)

            lbl_valor = QLabel("—")
            lbl_valor.setFont(QFont("Consolas", 14, QFont.Bold))
            lbl_valor.setStyleSheet(f"color:{color};background:transparent;border:none;")
            lbl_valor.setAlignment(Qt.AlignCenter)
            lbl_valor.setMinimumWidth(90)

            col_layout.addWidget(lbl_titulo)
            col_layout.addWidget(lbl_valor)
            self._labels[clave] = (lbl_valor, unidad, color)
            layout.addLayout(col_layout)

            if campos.index((clave, etiqueta, unidad, color)) < len(campos) - 1:
                sep = QFrame()
                sep.setFrameShape(QFrame.VLine)
                sep.setStyleSheet(f"color:{C_BORDER};background:{C_BORDER};border:none;max-width:1px;")
                layout.addWidget(sep)
        
        self.setLayout(layout)

    def actualizar(self, datos: dict):
        for clave, (lbl, unidad, original_color) in self._labels.items():
            val = datos.get(clave, 0.0)
            
            if clave == 'ml_status':
                if "ANOMALÍA" in str(val):
                    lbl.setStyleSheet(f"color:{C_ALARM};background:transparent;border:none;")
                elif "CALIBRANDO" in str(val):
                    lbl.setStyleSheet(f"color:#FFD700;background:transparent;border:none;")
                else:
                    lbl.setStyleSheet(f"color:{original_color};background:transparent;border:none;")
            
            if isinstance(val, float): 
                lbl.setText(f"{val:.1f} {unidad}")
            else: 
                lbl.setText(f"{val} {unidad}")

class DashboardEjecutivo(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SALA DE CONTROL CENTRADA — SUPERVISIÓN HMI")
        self.resize(1000, 550)
        self.setStyleSheet(f"background-color: #050811; color: {C_TEXT};")
        
        layout = QVBoxLayout()
        self.lbl_titulo_dash = QLabel("CENTRAL HMI — DIAGNÓSTICO AVANZADO DE ACTIVOS")
        self.lbl_titulo_dash.setFont(QFont("Consolas", 12, QFont.Bold))
        self.lbl_titulo_dash.setStyleSheet("color: #7AAFFF; padding: 10px;")
        self.lbl_titulo_dash.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.lbl_titulo_dash)
        
        grid = QGridLayout()
        grid.setSpacing(15)
        
        self.box_carga = QGroupBox("ÍNDICE DE CARGA SOBERANO")
        self.box_carga.setFont(QFont("Consolas", 10, QFont.Bold))
        v_c = QVBoxLayout()
        self.lbl_carga_val = QLabel("0.0 %")
        self.lbl_carga_val.setFont(QFont("Consolas", 36, QFont.Bold))
        self.lbl_carga_val.setStyleSheet("color: #00FFCC;")
        self.lbl_carga_val.setAlignment(Qt.AlignCenter)
        self.lbl_carga_txt = QLabel("ESTADO: CALCULANDO")
        self.lbl_carga_txt.setFont(QFont("Consolas", 12, QFont.Bold))
        self.lbl_carga_txt.setAlignment(Qt.AlignCenter)
        v_c.addWidget(self.lbl_carga_val)
        v_c.addWidget(self.lbl_carga_txt)
        self.box_carga.setLayout(v_c)
        
        self.box_rul = QGroupBox("PRONÓSTICO DE VIDA RUL")
        self.box_rul.setFont(QFont("Consolas", 10, QFont.Bold))
        v_r = QVBoxLayout()
        self.lbl_rul_val = QLabel("ESPERANDO...")
        self.lbl_rul_val.setFont(QFont("Consolas", 24, QFont.Bold))
        self.lbl_rul_val.setStyleSheet("color: #FFD700;")
        self.lbl_rul_val.setAlignment(Qt.AlignCenter)
        v_r.addWidget(self.lbl_rul_val)
        self.box_rul.setLayout(v_r)
        
        self.box_eta = QGroupBox("EFICIENCIA ESTIMADA η")
        self.box_eta.setFont(QFont("Consolas", 10, QFont.Bold))
        v_e = QVBoxLayout()
        self.lbl_eta_val = QLabel("— %")
        self.lbl_eta_val.setFont(QFont("Consolas", 28, QFont.Bold))
        self.lbl_eta_val.setStyleSheet("color: #39FF14;")
        self.lbl_eta_val.setAlignment(Qt.AlignCenter)
        v_e.addWidget(self.lbl_eta_val)
        self.box_eta.setLayout(v_e)

        self.box_salud = QGroupBox("SALUD GLOBAL")
        self.box_salud.setFont(QFont("Consolas", 10, QFont.Bold))
        v_salud = QVBoxLayout()
        self.lbl_salud_val = QLabel("— %")
        self.lbl_salud_val.setFont(QFont("Consolas", 36, QFont.Bold))
        self.lbl_salud_val.setStyleSheet("color: #00FFCC;")
        self.lbl_salud_val.setAlignment(Qt.AlignCenter)
        v_salud.addWidget(self.lbl_salud_val)
        self.box_salud.setLayout(v_salud)

        self.box_confianza = QGroupBox("CONFIANZA DEL SISTEMA")
        self.box_confianza.setFont(QFont("Consolas", 10, QFont.Bold))
        v_conf = QVBoxLayout()
        self.lbl_confianza_val = QLabel("— %")
        self.lbl_confianza_val.setFont(QFont("Consolas", 28, QFont.Bold))
        self.lbl_confianza_val.setStyleSheet("color: #AADDFF;")
        self.lbl_confianza_val.setAlignment(Qt.AlignCenter)
        v_conf.addWidget(self.lbl_confianza_val)
        self.box_confianza.setLayout(v_conf)

        self.box_disp = QGroupBox("DISPONIBILIDAD TÉCNICA Ai")
        self.box_disp.setFont(QFont("Consolas", 10, QFont.Bold))
        v_disp = QVBoxLayout()
        self.lbl_disp_val = QLabel("100.0 %")
        self.lbl_disp_val.setFont(QFont("Consolas", 28, QFont.Bold))
        self.lbl_disp_val.setStyleSheet("color: #39FF14;")
        self.lbl_disp_val.setAlignment(Qt.AlignCenter)
        v_disp.addWidget(self.lbl_disp_val)
        self.box_disp.setLayout(v_disp)

        self.box_deriva = QGroupBox("DERIVA OPERACIONAL")
        self.box_deriva.setFont(QFont("Consolas", 10, QFont.Bold))
        v_deriva = QVBoxLayout()
        self.lbl_deriva_val = QLabel("— %")
        self.lbl_deriva_val.setFont(QFont("Consolas", 28, QFont.Bold))
        self.lbl_deriva_val.setStyleSheet("color: #FF8C00;")
        self.lbl_deriva_val.setAlignment(Qt.AlignCenter)
        v_deriva.addWidget(self.lbl_deriva_val)
        self.box_deriva.setLayout(v_deriva)

        self.box_diag = QGroupBox("DIAGNÓSTICO ML")
        self.box_diag.setFont(QFont("Consolas", 10, QFont.Bold))
        v_diag = QVBoxLayout()
        self.lbl_diag_val = QLabel("EN ESPERA")
        self.lbl_diag_val.setFont(QFont("Consolas", 16, QFont.Bold))
        self.lbl_diag_val.setStyleSheet("color: #39FF14;")
        self.lbl_diag_val.setAlignment(Qt.AlignCenter)
        v_diag.addWidget(self.lbl_diag_val)
        self.box_diag.setLayout(v_diag)

        grid.addWidget(self.box_carga, 0, 0, 2, 1)
        grid.addWidget(self.box_salud, 0, 1, 2, 1)
        grid.addWidget(self.box_rul, 2, 0)
        grid.addWidget(self.box_confianza, 2, 1)
        grid.addWidget(self.box_eta, 3, 0)
        grid.addWidget(self.box_disp, 3, 1)
        grid.addWidget(self.box_deriva, 4, 0)
        grid.addWidget(self.box_diag, 4, 1)
        
        layout.addLayout(grid)
        self.setLayout(layout)

    def actualizar_dashboard(self, carga, estado_lbl, color, rul_str, eta, disp, activo_id):
        self.lbl_carga_val.setText(f"{carga}%")
        self.lbl_carga_val.setStyleSheet(f"color: {color};")
        self.lbl_carga_txt.setText(f"ESTADO: {estado_lbl}")
        self.lbl_carga_txt.setStyleSheet(f"color: {color};")
        self.lbl_rul_val.setText(rul_str)
        self.lbl_eta_val.setText(f"{eta:.1f} %")
        self.lbl_disp_val.setText(f"{disp:.1f} %")
        if disp < 95.0: self.lbl_disp_val.setStyleSheet("color: #FF3333;")
        elif disp < 98.0: self.lbl_disp_val.setStyleSheet("color: #FFD700;")
        else: self.lbl_disp_val.setStyleSheet("color: #39FF14;")

    def actualizar_avanzado(self, salud, conf, deriva, diag):
        self.lbl_salud_val.setText(f"{salud:.1f} %")
        if salud < 50: self.lbl_salud_val.setStyleSheet("color: #FF3333;")
        elif salud < 80: self.lbl_salud_val.setStyleSheet("color: #FFD700;")
        else: self.lbl_salud_val.setStyleSheet("color: #00FFCC;")

        self.lbl_confianza_val.setText(f"{conf:.1f} %")
        self.lbl_deriva_val.setText(f"{deriva:.1f} %")
        
        self.lbl_diag_val.setText(diag)
        if diag == "OPERACION NOMINAL": self.lbl_diag_val.setStyleSheet("color: #39FF14;")
        else: self.lbl_diag_val.setStyleSheet("color: #FF3333;")

class PanelHMI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HMI Edge Computing v2.3 CERT ISO 10816 NEMA MG1 IEC 60034 IEEE 1099")
        self.resize(1280, 860)
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{ background-color: {C_BG}; color: {C_TEXT}; }}
            QGroupBox {{ border: 1px solid {C_BORDER}; border-radius: 6px; margin-top: 10px; font-weight: bold; color: #7AAFFF; font-size: 9pt; }}
            QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 5px; }}
            QComboBox, QLineEdit {{ background-color: #0D1525; border: 1px solid {C_BORDER}; border-radius: 4px; padding: 4px 8px; color: {C_TEXT}; }}
            QTabWidget::pane {{ border: 1px solid {C_BORDER}; background: {C_BG}; }}
            QTabBar::tab {{ background: #0D1525; padding: 7px 14px; color: #6090B0; border-top-left-radius: 4px; border-top-right-radius: 4px; font-size: 8pt; }}
            QTabBar::tab:selected {{ background: {C_PANEL}; color: #00FFCC; font-weight: bold; }}
            QTabBar::tab:hover   {{ background: #142035; color: #AADDFF; }}
            QSlider::groove:horizontal {{ background: #0D1525; border: 1px solid {C_BORDER}; height: 6px; border-radius: 3px; }}
            QSlider::handle:horizontal {{ background: #00BFFF; width: 14px; height: 14px; margin: -4px 0; border-radius: 7px; }}
            QSlider::sub-page:horizontal {{ background: #004466; border-radius: 3px; }}
        """)

        # --- HILO TRABAJADOR DE BASE DE DATOS ---
        self.db_queue = queue.Queue()
        self.db_worker = threading.Thread(target=self._db_worker_loop, daemon=True)
        self.db_worker.start()

        # --- HILO GENERADOR DE PRUEBAS ---
        self.timer_pruebas = QTimer()
        self.timer_pruebas.timeout.connect(self._generar_datos_prueba)
        self.sim_fase = 0.0

        self.dash_secundario = DashboardEjecutivo()
        self.dash_secundario.show()

        self.kalman_corr = Kalman1D(0.02, 4)
        self.kalman_vib  = Kalman1D(0.02, 1)
        self.kalman_rpm  = Kalman1D(0.05, 4)

        self.health_score = 100.0
        self.hist_sim = deque(maxlen=30)
        self.drift_operacional = 0.0
        self.impacto_actual = 0.0
        self.confianza_sistema = 100.0
        self.disponibilidad_tecnica = 100.0
        self.MTTR_TEORICO = 2.0 

        self.MAX_DATOS = 60
        self.MAX_DATOS_LARGO = 512 
        self.LIM_VIB   = 50
        self.LIM_CORR  = 800

        self.x_data      = np.arange(self.MAX_DATOS)
        self.hist_v_p1   = np.zeros(self.MAX_DATOS)
        self.hist_iir_p1 = np.zeros(self.MAX_DATOS)
        self.hist_c_p1   = np.zeros(self.MAX_DATOS)
        self.hist_v_p2   = np.zeros(self.MAX_DATOS)
        self.hist_c_p2   = np.zeros(self.MAX_DATOS)
        self.matriz_fft  = np.zeros((32, self.MAX_DATOS))

        self.envolvente = np.zeros(self.MAX_DATOS)
        self.buffer_vib_largo = deque(maxlen=self.MAX_DATOS_LARGO)

        self.hist_raw_didact  = [0.0] * self.MAX_DATOS
        self.hist_ema_didact  = [0.0] * self.MAX_DATOS
        self.hist_iir_didact  = [0.0] * self.MAX_DATOS 
        
        self.hist_etime       = [0.0] * self.MAX_DATOS
        self.hist_efreq       = [0.0] * self.MAX_DATOS

        self._iir_x = [0.0, 0.0, 0.0]
        self._iir_y = [0.0, 0.0, 0.0]

        self.ema_sim          = 0.0
        self.alpha_didact     = 0.125
        self.hist_rms_didact  = [0.0] * self.MAX_DATOS
        self.fft_bins         = np.zeros(32)

        self._motor_id = "UNSET"
        self._activo_id = "UNSET"
        self._eta_pct  = 0.0
        self._carga_pct = 0.0

        # --- VARIABLES PARA PESTAÑA ARQUITECTURA DSP ---
        self.ptr_circular = 0
        self.n_puntos_anillo = 64

        # --- VARIABLES PARA PESTAÑA MODULACIÓN AM ---
        self.fase_modulacion = 0.0

        # --- CONFIGURACIÓN APRENDIZAJE AUTOMÁTICO ---
        self._ml_modelo = IsolationForest(n_estimators=50, contamination='auto', random_state=42)
        self._ml_entrenado = False
        self._ml_entrenando = False
        self._ml_buffer = deque(maxlen=500)
        self._ML_SAMPLES_TRAIN = 150 
        self._muestras_desde_entrenamiento = 0

        # --- CONFIGURACIÓN RUL (PHM) ---
        self._placa = {'rpm_nom': 2200.0, 'i_vacio': 350.0, 'i_nom': 2400.0, 'i_bloq': 5600.0, 'vib_nom': 20.0, 'vida_nom': 17520.0}
        self._placa_cargada   = False
        self._rul_activo      = False
        self._calibrando_rul  = False
        self._buffer_calibracion = []
        self._MUESTRAS_CALIBRACION = 200
        
        self._MAX_HIST_RUL   = 500
        self._hist_di        = []
        self._hist_di_t      = []
        self._t0_vida_motor  = None 
        
        self._ref_sano       = None
        self._cov_sano       = None
        self._inv_cov_sano   = None
        self._rul_horas      = None
        self._rul_std_horas  = None
        self._di_umbral      = 6.0 

        self._historial_eventos  = []
        self._sobrecarga_count   = 0
        self._desbalance_count   = 0
        self._horas_acum         = 0.0
        self._timer_horas        = QTimer()
        self._timer_horas.timeout.connect(self._tick_horas)

        self._INTERVALO_BASE = {'rodamientos': 4000, 'lubricacion': 2000, 'escobillas': 1000, 'aislamiento': 8760, 'limpieza': 500, 'correas': 3000}
        self._prox_mant = dict(self._INTERVALO_BASE)

        self._db              = DBManager()
        self._firma           = None
        self._firma_ref_vec   = None
        self._similitud_sesion= None
        self._tendencia       = DetectorTendencia()
        self._muestra_sesion  = 0
        self._di_sesion       = []
        self._cargar_tendencia_bd()

        self.hilo_trabajo = None
        self._construir_interfaz()

    def closeEvent(self, event):
        """Sobrescritura segura del evento de cierre para liberar recursos físicos y lógicos."""
        if self.hilo_trabajo:
            self.hilo_trabajo.detener()
            self.hilo_trabajo.wait(2000)
        self._timer_horas.stop()
        self.dash_secundario.close()
        # Enviar señal de terminación limpia al hilo de base de datos
        if hasattr(self, 'db_queue'):
            self.db_queue.put(None)
        event.accept()

    def _db_worker_loop(self):
        """Consumidor de tareas asíncronas para operaciones SQLite."""
        while True:
            tarea = self.db_queue.get()
            if tarea is None:
                break
            func, args, kwargs = tarea
            try:
                func(*args, **kwargs)
            except Exception as e:
                logging.warning(f"Excepción en BD worker [{func.__name__}]: {e}")
            finally:
                self.db_queue.task_done()

    def analizar_salud_motor(self, carga, vib, impacto):
        if vib > 20 and carga < 50:
            return "DESBALANCE MECANICO"
        if vib > 20 and carga > 90:
            return "SOBRECARGA MECANICA"
        if vib < 10 and carga > 90:
            return "FRICCION INTERNA"
        if impacto > 0.5:
            return "IMPACTOS EN RODAMIENTO"
        return "OPERACION NOMINAL"

    def _calcular_pct(self, c_proc, rpm_val):
        if not self._placa_cargada:
            return 0.0, ("SIN PLACA", "#556688")
            
        p = self._placa
        i_vacio = p['i_vacio']
        i_nom   = p['i_nom']
        rpm_nom = p['rpm_nom']
        
        i_rango = max(i_nom - i_vacio, 1.0)
        carga_elec = (c_proc - i_vacio) / i_rango
        
        caida_rpm = max(0.0, 1.0 - (rpm_val / max(rpm_nom, 1.0)))
        carga_rpm = caida_rpm / 0.15 
        
        carga_total = np.clip(0.85 * carga_elec + 0.15 * carga_rpm, 0.0, 1.5)
        
        if carga_total < 0.25:   estado = ("VACÍO",        "#556688")
        elif carga_total < 0.60: estado = ("LIGERA",       "#00FFCC")
        elif carga_total < 0.85: estado = ("NOMINAL",    "#39FF14")
        elif carga_total < 1.10: estado = ("PLENA",      "#FFD700")
        elif carga_total < 1.30: estado = ("SOBRECARGA", "#FF8C00")
        else:                    estado = ("BLOQUEO",    "#FF3333")
        
        return round(carga_total * 100, 1), estado

    def _rul_tiempo_real_str(self):
        h = self._rul_horas
        if h is None: return "RUL: calculando..."
        if h > 8760: return f"RUL: {h/8760:.1f} años"
        elif h > 720: return f"RUL: {h/720:.1f} meses"
        elif h > 24: return f"RUL: {h/24:.0f} días"
        else: return f"RUL: {h:.0f} h  ⚠ CRÍTICO"

    def _toggle_actuador(self, checked):
        if checked:
            self.btn_actuador.setText("⚙ CARGA EXTERNA: ON ")
        else:
            self.btn_actuador.setText("⚙ CARGA EXTERNA: OFF")
        if self.hilo_trabajo is not None:
            self.hilo_trabajo.cambiar_actuador(1 if checked else 0)

    def _actualizar_reporte_texto(self, carga, vib, impacto, health, conf, diag):
        if health >= 80:
            estado_html = "<span style='color:#00FFCC;'>ÓPTIMO</span>"
            sugerencia = "Continuar monitoreo nominal."
        elif health >= 50:
            estado_html = "<span style='color:#FFD700;'>CON ADVERTENCIAS</span>"
            sugerencia = "Revisar posibles desajustes mecánicos en el siguiente mantenimiento preventivo."
        else:
            estado_html = "<span style='color:#FF3333;'>CRÍTICO</span>"
            sugerencia = "<b style='color:#FF3333;'>Detener el equipo e inspeccionar inmediatamente riesgo de falla catastrófica.</b>"

        if impacto > 0.5:
            cepstrum_txt = "Se detectan picos periódicos en la Envolvente y el Cepstrum. Esto indica <span style='color:#FF3333;'>golpeteo repetitivo posible falla en pistas bolas</span>."
        else:
            cepstrum_txt = "El filtro Cepstral muestra una señal limpia, <span style='color:#39FF14;'>sin periodicidades anómalas ni impactos</span>."

        if conf < 70:
            kalman_txt = "El filtro de Kalman detecta inestabilidad en las métricas base, existiendo una <span style='color:#FF8C00;'>alta deriva operacional</span> respecto a la firma de fábrica."
        else:
            kalman_txt = "Las métricas suavizadas por Kalman coinciden con la firma base histórica, confirmando estabilidad."

        report = f"""
        <b style='color:#AADDFF; font-size:11pt;'>INFORME DIAGNÓSTICO EN TIEMPO REAL:</b><br><br>
        • <b>Estado Global:</b> Se observa que el sistema electromecánico opera en un régimen {estado_html} Salud: {health:.1f}%.<br>
        • <b>Análisis de Vibración Hilbert Cepstrum:</b> A una carga del {carga:.1f}%, la vibración RMS es de {vib:.1f} u.a. {cepstrum_txt}<br>
        • <b>Comportamiento Algoritmo ML y Filtros Kalman:</b> El modelo de aprendizaje automático clasifica el patrón actual como <b style='color:#FFD700;'>{diag}</b>. {kalman_txt}<br><br>
        • <b>💡 Recomendación:</b> {sugerencia}
        """
        self.txt_analisis.setHtml(report)

    def _cambiar_modo(self, modo):
        if not self.btn_iniciar.isEnabled():
            if "Hardware" in modo:
                self.timer_pruebas.stop()
            else:
                if self.hilo_trabajo:
                    self.hilo_trabajo.detener()
                    self.hilo_trabajo.wait()
                    self.hilo_trabajo = None
                self.timer_pruebas.start(150)

    def _generar_datos_prueba(self):
        modo = self.combo_modo.currentText()
        if "Hardware" in modo: return

        self.sim_fase += 0.8
        ruido_c = float(np.random.normal(0, 8))
        ruido_v = float(np.random.normal(0, 0.8))
        ruido_rpm = float(np.random.normal(0, 15))
        
        # 1. Generamos un piso de ruido blanco para el espectro
        espectro_base = np.abs(np.random.normal(0.5, 0.2, 32))

        # 2. Diccionario base (Escenario: Motor Sano)
        d = {
            'c_proc': 350.0 + ruido_c,
            'v_proc': 5.0 + ruido_v,
            'rpm': 2200.0 + ruido_rpm,
            'c_crud': 360.0 + ruido_c * 1.5,
            'v_crud': 5.0 + np.sin(self.sim_fase)*0.5 + ruido_v,
            'ref_sana': 350.0,
            'alertas': 0,
            '_predicho': False
        }
        
        # En estado nominal, la vibración natural del giro es muy baja
        espectro_base[4] = 1.2 + ruido_v * 0.2
        # Si la amplitud del bin 4 es menor al ruido, el bin dominante será aleatorio
        d['bin_dom'] = 4 if espectro_base[4] > np.max(espectro_base[5:]) else int(np.argmax(espectro_base[1:]) + 1)

        # 3. Modificadores según la falla inducida
        if "Desbalance" in modo:
            onda_desb = np.sin(self.sim_fase * 1.5) * 22.0
            d['v_proc'] = 28.0 + ruido_v * 3
            d['v_crud'] = 28.0 + onda_desb + ruido_v * 4
            d['alertas'] = 0b10
            
            # ¡Aquí es donde ocurre la física! El bin 4 (1X RPM) explota en amplitud
            espectro_base[4] = 27.0 + ruido_v * 2
            d['bin_dom'] = 4
            
            # Simulación de un impacto de rodamiento ocasional
            if int(self.sim_fase * 10) % 15 == 0:
                d['v_crud'] += 45.0
                
        elif "Sobrecarga" in modo:
            d['c_proc'] = 2800.0 + ruido_c * 5
            d['c_crud'] = 3200.0 + np.sin(self.sim_fase)*100.0 + ruido_c * 6
            d['rpm'] = 1750.0 + ruido_rpm
            d['v_proc'] = 8.0 + ruido_v
            d['v_crud'] = 8.0 + np.sin(self.sim_fase)*3.0 + ruido_v
            d['alertas'] = 0b01
            
            # El esfuerzo eleva ligeramente la vibración a 1X, pero no es un desbalance puro
            espectro_base[4] = 3.5 + ruido_v
            # Podría saltar a otros armónicos debido al estrés magnético
            d['bin_dom'] = 4 if np.random.rand() > 0.3 else 8

        # 4. Asignar el espectro inyectado
        d['fft_vib'] = espectro_base.tolist()

        # Evitar valores negativos sin sentido físico
        for k in ['c_proc', 'v_proc', 'rpm', 'c_crud', 'v_crud']:
            d[k] = max(0.0, float(d[k]))

        # Enviar al flujo principal de la HMI
        self._datos(d)

    def _construir_interfaz(self):
        central = QWidget()
        layout  = QVBoxLayout()
        layout.setSpacing(5)
        layout.setContentsMargins(8, 8, 8, 8)

        grp = QGroupBox("Panel de Control del Nodo HMI")
        row = QHBoxLayout()
        row.setSpacing(8)

        self.combo_modo = QComboBox()
        self.combo_modo.addItems(["Adquisición Hardware", "Prueba: Estado Nominal", "Prueba: Desbalance Físico", "Prueba: Sobrecarga o Fricción"])
        self.combo_modo.setStyleSheet("background:#0D1525;border:1px solid #1A3060;border-radius:4px;padding:4px 8px;color:#00FFCC;")
        self.combo_modo.currentTextChanged.connect(self._cambiar_modo)

        # Configurador de Puerto Físico
        self.input_puerto = QLineEdit("/dev/ttyUSB0")
        self.input_puerto.setFixedWidth(120)
        self.input_puerto.setStyleSheet(f"background:#0D1525;border:1px solid {C_BORDER};border-radius:4px;padding:4px 8px;color:#FFD700;font-family:Consolas;")
        
        self.btn_actuador = QPushButton("⚙ CARGA EXTERNA: OFF")
        self.btn_actuador.setCheckable(True)
        self.btn_actuador.setStyleSheet("""
            QPushButton {background:#1E1414;color:#AA8888;border:1px solid #443333;border-radius:4px;padding:6px;font-weight:bold;} 
            QPushButton:checked {background:#003D14;color:#00FFCC;border:1px solid #00AA33;}
        """)
        self.btn_actuador.toggled.connect(self._toggle_actuador)

        self.input_rpm = QLineEdit("2200")
        self.input_rpm.setPlaceholderText("RPM 10–10000")
        self.input_rpm.setFixedWidth(100)

        self.input_delgas = QLineEdit("3")
        self.input_delgas.setPlaceholderText("Delgas")
        self.input_delgas.setFixedWidth(50)
        self.input_delgas.setStyleSheet("background:#0D1525;border:1px solid #1A3060;border-radius:4px;padding:4px 8px;color:#FFD700;")
        
        self.input_reductora = QLineEdit("0")
        self.input_reductora.setPlaceholderText("Caja 1:X")
        self.input_reductora.setFixedWidth(60)
        self.input_reductora.setStyleSheet("background:#0D1525;border:1px solid #1A3060;border-radius:4px;padding:4px 8px;color:#00FFCC;")

        self.btn_iniciar = QPushButton("▶  Iniciar")
        self.btn_iniciar.setStyleSheet("QPushButton{background:#003D14;color:#00FF66;border:1px solid #00AA33;border-radius:5px;font-weight:bold;padding:6px 14px;} QPushButton:hover{background:#005020;} QPushButton:disabled{background:#141E14;color:#334433;border-color:#223322;}")
        
        self.btn_detener = QPushButton("■  Detener HMI")
        self.btn_detener.setStyleSheet("QPushButton{background:#3D0000;color:#FF4444;border:1px solid #AA2222;border-radius:5px;font-weight:bold;padding:6px 14px;} QPushButton:hover{background:#550000;} QPushButton:disabled{background:#1E1414;color:#443333;border-color:#332222;}")
        self.btn_detener.setEnabled(False)

        self.btn_parada = QPushButton("🛑 APAGAR MOTOR")
        self.btn_parada.setCheckable(True)
        self.btn_parada.setStyleSheet("""
            QPushButton {background:#880000;color:#FFFFFF;border:1px solid #FF0000;border-radius:5px;font-weight:bold;padding:6px 14px;} 
            QPushButton:checked {background:#003D14;color:#00FFCC;border:1px solid #00AA33;}
            QPushButton:checked:hover {background:#005020;}
        """)
        self.btn_parada.toggled.connect(self._toggle_motor_remoto)

        self.btn_limpiar = QPushButton("🧹 Limpiar Búfer")
        self.btn_limpiar.setStyleSheet("QPushButton{background:#332200;color:#FFD700;border:1px solid #AA8800;border-radius:5px;font-weight:bold;padding:6px 14px;} QPushButton:hover{background:#553300;}")
        self.btn_limpiar.clicked.connect(self._limpiar_graficas)

        elementos_control = [
            ("Modo:", self.combo_modo),
            ("Puerto:", self.input_puerto),
            ("Salida Actuador:", self.btn_actuador), 
            ("Setpoint RPM:", self.input_rpm),
            ("Delgas:", self.input_delgas),
            ("Reductora 1:X:", self.input_reductora)
        ]

        for lbl, w in elementos_control:
            label = QLabel(lbl)
            label.setStyleSheet("color:#AACCFF; font-size:9pt;")
            row.addWidget(label)
            row.addWidget(w)
            
        row.addStretch()
        row.addWidget(self.btn_iniciar)
        row.addWidget(self.btn_detener)
        row.addWidget(self.btn_parada)
        row.addWidget(self.btn_limpiar)
        grp.setLayout(row)
        layout.addWidget(grp)

        tel_row = QHBoxLayout()
        tel_row.setSpacing(8)

        self.disp_vib  = DisplayDigital("Vibración", "u.a", C_VIB,  0, self.LIM_VIB * 1.5)
        self.disp_corr = DisplayDigital("Corriente", "mA",  C_CORR, 0, self.LIM_CORR * 1.5)
        self.disp_rpm  = DisplayDigital("RPM Motor", "rpm", "#AA88FF", 0, 10000)
        self.disp_carga = DisplayDigital("Carga NEMA", "%", "#556688", 0, 150)

        self.label_tel = QLabel("⏹  ESTADO: IDLE ESPERANDO")
        self.label_tel.setAlignment(Qt.AlignCenter)
        self.label_tel.setFont(QFont("Consolas", 11, QFont.Bold))
        self.label_tel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.label_tel.setStyleSheet(f"background:{C_BG};color:#556688;border:1px solid #556688;border-radius:4px;padding:4px;")

        tel_row.addWidget(self.disp_vib)
        tel_row.addWidget(self.disp_corr)
        tel_row.addWidget(self.disp_rpm)
        tel_row.addWidget(self.disp_carga)
        tel_row.addWidget(self.label_tel, stretch=2)
        layout.addLayout(tel_row)

        self.tabs = QTabWidget()
        self.tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        t_dsp = self._crear_tab_arquitectura_dsp()
        t_bode = self._crear_tab_bode_polos_dsp()
        t_mod = self._crear_tab_modulacion() 
        
        t1 = QWidget()
        l1 = QVBoxLayout()
        l1.setContentsMargins(0,0,0,0)
        self._crear_tab_procesado_pg(l1)
        t1.setLayout(l1)

        t2 = QWidget()
        l2 = QVBoxLayout()
        l2.setContentsMargins(0,0,0,0)
        self._crear_tab_crudo_pg(l2)
        t2.setLayout(l2)

        t3 = self._crear_tab_didactica()
        t4 = self._crear_tab_rul()
        t5 = self._crear_tab_historico()
        t6 = self._crear_tab_avanzado()
        t7 = self._crear_tab_sustentacion_envivo()

        self.tabs.addTab(t_dsp, "💻  Arquitectura dsPIC MAC")
        self.tabs.addTab(t_bode, "🎚️  Bode Polos-Ceros IIR")
        self.tabs.addTab(t_mod, "🎛️ Modulación AM y Aislamiento") 
        self.tabs.addTab(t1, "📈  Análisis Predictivo")
        self.tabs.addTab(t2, "📡  Telemetría Cruda")
        self.tabs.addTab(t3, "🔬  Laboratorio de Filtros")
        self.tabs.addTab(t4, "⏳  Vida Útil RUL")
        self.tabs.addTab(t5, "📊  Historial Tendencia")
        self.tabs.addTab(t6, "🧲  Diagnóstico Avanzado")
        self.tabs.addTab(t7, "🎓  Sustentación: Análisis en Vivo")

        layout.addWidget(self.tabs, stretch=1)
        central.setLayout(layout)
        self.setCentralWidget(central)

        self.btn_iniciar.clicked.connect(self._iniciar)
        self.btn_detener.clicked.connect(self._detener)

    # =========================================================================
    # NUEVA PESTAÑA: BODE Y POLOS-CEROS EN PLANO Z
    # =========================================================================
    def _crear_tab_bode_polos_dsp(self):
        w = QWidget()
        main = QVBoxLayout()
        main.setSpacing(6)
        main.setContentsMargins(8, 8, 8, 8)

        math_box = QLabel(
            "<b style='color:#FFD700;font-size:10pt;'>Análisis en el Dominio Z del Filtro IIR Biquad</b><br><br>"
            "<span style='color:#AADDFF;'>Todo filtro digital implementado en el dsPIC vive matemáticamente en el "
            "<b>plano Z</b>. La ubicación de sus <b>polos y ceros</b> determina por completo su comportamiento: "
            "estabilidad ancho de banda y atenuación.<br><br>"
            "<span style='color:#00FFCC;font-family:Consolas;'>H z = b0 + b1·z⁻¹ + b2·z⁻² / 1 + a1·z⁻¹ + a2·z⁻²</span><br><br>"
            "Este es exactamente el filtro biquad usado para suavizar la vibración cruda. Un filtro es "
            "<b>estable</b> únicamente si todos sus polos caen <i>dentro</i> del círculo unitario — condición que "
            "se verifica visualmente en el diagrama de la derecha.<br><br>"
            "<span style='color:#FF8C00;'>El panel inferior compara qué pasaría con la respuesta en frecuencia si el "
            "dsPIC usara un formato de punto fijo más pobre Q7 8 bits en vez del <b>Q15 de 15 bits acumulador "
            "de 40 bits</b> que realmente usa</span> — mostrando por qué la arquitectura interna del DSP ancho de "
            "acumuladores formato fraccionario no es un detalle menor sino lo que hace posible un filtro preciso "
            "en tiempo real.</span>"
        )
        math_box.setWordWrap(True)
        math_box.setStyleSheet(f"background:{C_PANEL};border:1px solid {C_BORDER};border-radius:6px;padding:10px;")
        main.addWidget(math_box)

        b0, b1, b2 = 0.020083, 0.040167, 0.020083
        a1, a2 = -1.561018, 0.641351
        b_coefs = [b0, b1, b2]
        a_coefs = [1.0, a1, a2]

        w_norm, h_resp = freqz(b_coefs, a_coefs, worN=1024)
        freq_normalizada = w_norm / np.pi  

        mag_db = 20 * np.log10(np.maximum(np.abs(h_resp), 1e-12))
        fase_deg = np.unwrap(np.angle(h_resp)) * 180 / np.pi

        def cuantizar(coefs, bits):
            escala = 2 ** (bits - 1)
            return [round(c * escala) / escala for c in coefs]

        b_q7 = cuantizar(b_coefs, 8)
        a_q7 = [1.0] + cuantizar([a1, a2], 8)
        _, h_q7 = freqz(b_q7, a_q7, worN=1024)
        mag_q7_db = 20 * np.log10(np.maximum(np.abs(h_q7), 1e-12))

        fig_bode = Figure(facecolor=C_BG)
        fig_bode.subplots_adjust(left=0.09, right=0.95, top=0.90, bottom=0.12, wspace=0.30, hspace=0.45)
        cv_bode = FigureCanvas(fig_bode)
        cv_bode.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        gs = gridspec.GridSpec(2, 2, figure=fig_bode, wspace=0.30, hspace=0.45)

        ax_mag = fig_bode.add_subplot(gs[0, 0])
        _estilizar_ax(ax_mag, "Magnitud — Q15 real vs Q7 degradado", "dB", C_VIB)
        ax_mag.set_xlabel("Frecuencia normalizada × Fs/2", color=C_TEXT_DIM, fontsize=7)
        ax_mag.plot(freq_normalizada, mag_db, color=C_VIB, lw=2.0, label='Q15 dsPIC real')
        ax_mag.plot(freq_normalizada, mag_q7_db, color='#FF6B6B', lw=1.5, ls='--', label='Q7 simulado 8 bits')
        ax_mag.axhline(-3, color='#FFD700', ls=':', lw=1.0, alpha=0.8, label='-3 dB corte')
        ax_mag.legend(fontsize=6, facecolor='#0A1020', edgecolor=C_BORDER, labelcolor='white')

        ax_fase = fig_bode.add_subplot(gs[0, 1])
        _estilizar_ax(ax_fase, "Respuesta de Fase", "Grados", '#AA88FF')
        ax_fase.set_xlabel("Frecuencia normalizada × Fs/2", color=C_TEXT_DIM, fontsize=7)
        ax_fase.plot(freq_normalizada, fase_deg, color='#AA88FF', lw=2.0)

        ax_pz = fig_bode.add_subplot(gs[1, 0])
        ax_pz.set_facecolor(C_AX_BG)
        ax_pz.set_title("Diagrama de Polos y Ceros Plano Z", color='#AACCFF', fontsize=9, fontweight='bold')
        ax_pz.tick_params(colors='#6688AA', labelsize=7)
        for s in ax_pz.spines.values():
            s.set_edgecolor(C_BORDER)
        ax_pz.grid(True, color=C_GRID, linestyle='--', linewidth=0.5, alpha=0.6)

        theta_circulo = np.linspace(0, 2 * np.pi, 200)
        ax_pz.plot(np.cos(theta_circulo), np.sin(theta_circulo), color='#556688', lw=1.0, ls='--')

        ceros = np.roots(b_coefs)
        polos = np.roots(a_coefs)
        ax_pz.scatter(ceros.real, ceros.imag, marker='o', s=90, facecolors='none',
                      edgecolors=C_VIB, linewidths=2, label='Ceros', zorder=3)
        ax_pz.scatter(polos.real, polos.imag, marker='x', s=90, color=C_ALARM,
                      linewidths=2, label='Polos', zorder=3)
        ax_pz.axhline(0, color='#334455', lw=0.6)
        ax_pz.axvline(0, color='#334455', lw=0.6)
        ax_pz.set_xlim(-1.5, 1.5)
        ax_pz.set_ylim(-1.5, 1.5)
        ax_pz.set_aspect('equal')
        ax_pz.legend(fontsize=7, facecolor='#0A1020', edgecolor=C_BORDER, labelcolor='white', loc='upper right')

        ax_txt = fig_bode.add_subplot(gs[1, 1])
        ax_txt.axis('off')

        radios_polos = np.abs(polos)
        estable = np.all(radios_polos < 1.0)
        color_estable = '#39FF14' if estable else '#FF3333'
        texto_estabilidad = "ESTABLE" if estable else "INESTABLE"

        resumen = (
            f"Polos: {polos[0]:.4f}, {polos[1]:.4f}\n"
            f"|Polo| máx: {np.max(radios_polos):.4f}\n"
            f"Ceros: {ceros[0]:.4f}, {ceros[1]:.4f}\n\n"
            f"Condición de estabilidad:\n|z_polo| < 1  →  {texto_estabilidad}\n\n"
            f"Coefs. cuantizados a Q7:\n"
            f"b = {b_q7}\n"
            f"a1,a2 = {a_q7[1]:.4f}, {a_q7[2]:.4f}"
        )
        ax_txt.text(0.02, 0.95, resumen, transform=ax_txt.transAxes, va='top', ha='left',
                    color=C_TEXT, fontsize=8.5, family='monospace',
                    bbox=dict(boxstyle='round', facecolor='#0D1525', edgecolor=color_estable, linewidth=1.5))

        main.addWidget(cv_bode)
        w.setLayout(main)
        return w

    # =========================================================================
    # PESTAÑA: MODULACIÓN AM Y AISLAMIENTO DE FRECUENCIA PROFESIONAL
    # =========================================================================
    def _crear_tab_modulacion(self):
        w = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        math_box = QLabel(
            "<b style='color:#00FFCC;font-size:11pt;'>Aislamiento Espectral y Modulación de Amplitud AM</b><br><br>"
            "<span style='color:#AADDFF;font-size:10pt;'>Esta sección demuestra la capacidad del sistema para aislar componentes frecuenciales específicos y aplicar técnicas de modulación de comunicaciones:<br>"
            "1. <b>Adquisición Cruda:</b> Captura de la señal de vibración transitoria con presencia de ruido mecánico y electromagnético.<br>"
            "2. <b>Aislamiento Fundamental:</b> Utilizando el índice de la frecuencia dominante Bin_Dom y su magnitud calculada por la FFT en el dsPIC "
            "se sintetiza la onda senoidal fundamental aislando la falla mecánica de las frecuencias parásitas.<br>"
            "3. <b>Modulación AM:</b> La componente fundamental aislada actúa como señal moduladora sobre una onda portadora de alta frecuencia "
            "demostrando la manipulación de espectros para potenciales sistemas de telemetría RF.</span>"
        )
        math_box.setWordWrap(True)
        math_box.setStyleSheet(f"background:{C_PANEL};border:1px solid #00FFCC;border-radius:6px;padding:10px;")
        layout.addWidget(math_box)

        self.glw_mod = pg.GraphicsLayoutWidget()
        layout.addWidget(self.glw_mod, stretch=1)

        self.p_mod_raw = self.glw_mod.addPlot(title="1. Señal de Vibración Transitoria Dominio Temporal con Ruido Blanco")
        self.p_mod_raw.setLabel('left', "Amplitud Cruda")
        self.p_mod_raw.showGrid(x=True, y=True, alpha=0.3)
        self.p_mod_raw.setXRange(0, 300)
        self.p_mod_raw.setYRange(-self.LIM_VIB, self.LIM_VIB)
        _bloquear_grafica_pg(self.p_mod_raw)
        self.line_mod_raw = self.p_mod_raw.plot(pen=pg.mkPen(C_RAW_DIM, width=1.5))
        
        self.glw_mod.nextRow()

        self.p_mod_pure = self.glw_mod.addPlot(title="2. Componente Fundamental Aislada Reconstrucción desde Bin FFT Dominante")
        self.p_mod_pure.setLabel('left', "Amplitud Extraída")
        self.p_mod_pure.showGrid(x=True, y=True, alpha=0.3)
        self.p_mod_pure.setXRange(0, 300)
        self.p_mod_pure.setYRange(-self.LIM_VIB, self.LIM_VIB)
        _bloquear_grafica_pg(self.p_mod_pure)
        self.line_mod_pure = self.p_mod_pure.plot(pen=pg.mkPen('#FFD700', width=2.5))
        
        self.glw_mod.nextRow()

        self.p_mod_am = self.glw_mod.addPlot(title="3. Señal Modulada en Amplitud AM con Portadora de Alta Frecuencia")
        self.p_mod_am.setLabel('left', "Amplitud Modulada")
        self.p_mod_am.setLabel('bottom', "Tiempo Desplazamiento de Fase Animado")
        self.p_mod_am.showGrid(x=True, y=True, alpha=0.3)
        self.p_mod_am.setXRange(0, 300)
        self.p_mod_am.setYRange(-self.LIM_VIB * 2, self.LIM_VIB * 2)
        _bloquear_grafica_pg(self.p_mod_am)
        self.line_mod_am = self.p_mod_am.plot(pen=pg.mkPen('#00FFCC', width=1))

        w.setLayout(layout)
        return w

    def _crear_tab_arquitectura_dsp(self):
        w = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(15)

        # ---------------------------------------------------------------------
        # COLUMNA IZQUIERDA: BÚFER CIRCULAR MODULO ADDRESSING
        # ---------------------------------------------------------------------
        col_izq = QVBoxLayout()
        
        lbl_titulo_circ = QLabel("Direccionamiento Circular por Hardware Anillo de 64 muestras")
        lbl_titulo_circ.setFont(QFont("Consolas", 11, QFont.Bold))
        lbl_titulo_circ.setStyleSheet("color:#00FFCC; background:#0D1525; padding:8px; border:1px solid #1A3060; border-radius:4px;")
        lbl_titulo_circ.setAlignment(Qt.AlignCenter)
        col_izq.addWidget(lbl_titulo_circ)

        # Usar pyqtgraph para dibujar el anillo
        self.glw_circ = pg.GraphicsLayoutWidget()
        self.p_circ = self.glw_circ.addPlot()
        self.p_circ.hideAxis('left')
        self.p_circ.hideAxis('bottom')
        self.p_circ.setAspectLocked(True)
        self.p_circ.setRange(xRange=[-1.2, 1.2], yRange=[-1.2, 1.2])
        _bloquear_grafica_pg(self.p_circ)

        # Precalcular coordenadas del anillo
        angulos = np.linspace(0, 2 * np.pi, self.n_puntos_anillo, endpoint=False)
        self.x_circ = np.cos(angulos)
        self.y_circ = np.sin(angulos)
        
        # Puntos base grises
        self.scatter_circ_base = pg.ScatterPlotItem(x=self.x_circ, y=self.y_circ, size=12, brush=pg.mkBrush(100, 100, 120, 80))
        self.p_circ.addItem(self.scatter_circ_base)
        
        # Puntos activos se actualizarán en _datos
        self.scatter_circ_active = pg.ScatterPlotItem(size=14)
        self.p_circ.addItem(self.scatter_circ_active)

        # Texto al centro del anillo
        self.text_ptr = pg.TextItem(text="Ptr: 0\nWait", color='#FFD700', anchor=(0.5, 0.5))
        self.text_ptr.setFont(QFont("Consolas", 14, QFont.Bold))
        self.text_ptr.setPos(0, 0)
        self.p_circ.addItem(self.text_ptr)

        col_izq.addWidget(self.glw_circ, stretch=1)
        
        txt_desc_circ = QLabel("El dsPIC actualiza el puntero de escritura en hardware Modulo Addressing.\nAl llegar a la posición 63 salta a la 0 en 0 ciclos de CPU\nliberando a la ALU para procesar el filtro IIR continuo.")
        txt_desc_circ.setFont(QFont("Consolas", 9))
        txt_desc_circ.setStyleSheet("color:#AADDFF; background:transparent;")
        txt_desc_circ.setAlignment(Qt.AlignCenter)
        col_izq.addWidget(txt_desc_circ)

        layout.addLayout(col_izq, stretch=1)

        # ---------------------------------------------------------------------
        # COLUMNA DERECHA: MOTOR DSP BUSES Y ACUMULADOR
        # ---------------------------------------------------------------------
        col_der = QVBoxLayout()
        
        lbl_titulo_dsp = QLabel("Monitor de Núcleo: Arquitectura Harvard MAC")
        lbl_titulo_dsp.setFont(QFont("Consolas", 11, QFont.Bold))
        lbl_titulo_dsp.setStyleSheet("color:#FFD700; background:#0D1525; padding:8px; border:1px solid #1A3060; border-radius:4px;")
        lbl_titulo_dsp.setAlignment(Qt.AlignCenter)
        col_der.addWidget(lbl_titulo_dsp)

        # Panel visual del DSP
        grp_dsp = QGroupBox()
        grp_dsp.setStyleSheet("QGroupBox{border:1px solid #2A5090;border-radius:6px; background:#0A0E1A;}")
        grid_dsp = QGridLayout()
        grid_dsp.setSpacing(15)

        # Formatos de display
        style_bus = "background:#0D1525; color:{}; border:2px solid {}; border-radius:5px; padding:10px; font-family:Consolas; font-size:16px; font-weight:bold;"
        style_mac = "background:#1A0000; color:#FF4444; border:2px solid #FF0000; border-radius:5px; padding:15px; font-family:Consolas; font-size:24px; font-weight:bold;"
        
        # Elementos
        self.lbl_dsp_adc = QLabel("ADC Raw Int: 0")
        self.lbl_dsp_adc.setStyleSheet(style_bus.format('#00FFCC', '#00AA88'))
        self.lbl_dsp_adc.setAlignment(Qt.AlignCenter)

        self.lbl_dsp_q15 = QLabel("Bus X Data Q15: 0.0000")
        self.lbl_dsp_q15.setStyleSheet(style_bus.format('#FFD700', '#AA8800'))
        self.lbl_dsp_q15.setAlignment(Qt.AlignCenter)

        self.lbl_dsp_coef = QLabel("Bus Y Coef Q15: +0.0201")
        self.lbl_dsp_coef.setStyleSheet(style_bus.format('#39FF14', '#00AA00'))
        self.lbl_dsp_coef.setAlignment(Qt.AlignCenter)

        self.lbl_dsp_acc = QLabel("Acc A 40-bit: 0x0000000000")
        self.lbl_dsp_acc.setStyleSheet(style_mac)
        self.lbl_dsp_acc.setAlignment(Qt.AlignCenter)

        # Diagrama de flujo
        grid_dsp.addWidget(self.lbl_dsp_adc, 0, 0, 1, 2)
        
        flecha1 = QLabel("▼ Conversión a Fraccionario"); flecha1.setAlignment(Qt.AlignCenter); flecha1.setStyleSheet("color:#556688; font-weight:bold;")
        grid_dsp.addWidget(flecha1, 1, 0, 1, 2)

        grid_dsp.addWidget(self.lbl_dsp_q15, 2, 0)
        grid_dsp.addWidget(self.lbl_dsp_coef, 2, 1)

        flecha2 = QLabel("↘ Dual Fetch Líneas Paralelas ↙"); flecha2.setAlignment(Qt.AlignCenter); flecha2.setStyleSheet("color:#556688; font-weight:bold;")
        grid_dsp.addWidget(flecha2, 3, 0, 1, 2)

        grid_dsp.addWidget(self.lbl_dsp_acc, 4, 0, 1, 2)

        grp_dsp.setLayout(grid_dsp)
        col_der.addWidget(grp_dsp)

        # Cuadro explicativo
        txt_desc_mac = QLabel(
            "<b style='color:#FFD700;font-size:10pt;'>Ejecución de Instrucción MAC en 1 Ciclo</b><br><br>"
            "<span style='color:#AADDFF;'>El dsPIC permite operaciones de Procesamiento Digital de Señales extremadamente veloces gracias a su arquitectura Harvard Modificada.<br><br>"
            "Durante el filtro IIR el microcontrolador hace <b>Fetch simultáneo</b>: lee la muestra de vibración Bus X y el coeficiente del filtro Bus Y en el mismo instante.<br><br>"
            "La operación <span style='color:#FF4444;font-family:Consolas;'>MAC W4*W5 A</span> multiplica ambas variables fraccionarias Q15 y suma el resultado al Acumulador A de 40 bits en <b>un solo ciclo de reloj</b> impidiendo el <i>overflow</i> matemático.</span>"
        )
        txt_desc_mac.setWordWrap(True)
        txt_desc_mac.setStyleSheet(f"background:{C_PANEL};border:1px solid {C_BORDER};border-radius:6px;padding:10px;")
        col_der.addWidget(txt_desc_mac)

        col_der.addStretch()
        layout.addLayout(col_der, stretch=1)
        w.setLayout(layout)
        return w

    def _limpiar_graficas(self):
        self.buffer_vib_largo.clear()
        self.matriz_fft.fill(0)
        self.hist_v_p1.fill(0)
        self.hist_c_p1.fill(0)
        self.hist_iir_p1.fill(0)
        self.hist_v_p2.fill(0)
        self.hist_c_p2.fill(0)
        self.hist_etime = [0.0] * self.MAX_DATOS
        self.hist_efreq = [0.0] * self.MAX_DATOS
        self.fft_bins = np.zeros(32)
        self._set_estado("🧹 BÚFERES LIMPIADOS", '#FFD700')

    def _toggle_motor_remoto(self, checked):
        if self.hilo_trabajo is not None:
            if checked:
                self.hilo_trabajo.cambiar_actuador(1) 
                self.btn_parada.setText("▶ ENCENDER MOTOR")
                self._set_estado("🛑  MOTOR APAGADO DESDE HMI", '#FF3333', bg='#330000', border='red')
            else:
                self.hilo_trabajo.cambiar_actuador(0)
                self.btn_parada.setText("🛑 APAGAR MOTOR")
                self._set_estado("▶  MOTOR REANUDADO", '#00FFCC')

    def _crear_tab_sustentacion_envivo(self):
        w = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        
        panel_imgs = QWidget()
        lay_imgs = QVBoxLayout()
        lay_imgs.setContentsMargins(0, 0, 0, 0)
        
        lbl_titulo_img = QLabel("🖼️ Galería del Artículo IEEE")
        lbl_titulo_img.setFont(QFont("Consolas", 10, QFont.Bold))
        lbl_titulo_img.setStyleSheet("color:#AACCFF; background:#0D1525; padding:4px; border:1px solid #1A3060;")
        lbl_titulo_img.setAlignment(Qt.AlignCenter)
        
        self.lista_imgs = QListWidget()
        self.lista_imgs.setFixedHeight(180)
        self.lista_imgs.setStyleSheet("""
            QListWidget { background:#0A1020; color:#8899BB; font-size:9pt; font-family:Consolas; border:1px solid #1A3060; }
            QListWidget::item { padding: 6px; border-bottom: 1px solid #152040; }
            QListWidget::item:selected { background:#1A3060; color:#FFD700; font-weight:bold; }
            QListWidget::item:hover { background:#142035; color:#00FFCC; }
        """)
        
        nombres_imagenes = [
            ("1. Cód. C Hanning FFT", "fft_c.png"),
            ("2. Espectro FFT Real", "fig1_espectro_fft.png"),
            ("3. Impulso Filtro IIR", "fig2_impulso_iir.png"),
            ("4. Bode IIR", "fig3_bode_discreto.png"),
            ("5. Leakage vs Hanning", "fig4_hanning_comparacion.png"),
            ("6. Filtrado de Señal", "fig5_senal_filtrada.png"),
            ("7. Cód. Py Cepstrum", "python_cepstrum.png"),
            ("8. Envolvente Hilbert", "hilbert_envolvente.png"),
            ("9. Teorema Parseval", "fig6_parseval.png")
        ]
        
        for nombre, archivo in nombres_imagenes:
            self.lista_imgs.addItem(nombre)
            
        self.lbl_visor_img = VisorImagen()
        
        lay_imgs.addWidget(lbl_titulo_img)
        lay_imgs.addWidget(self.lista_imgs)
        lay_imgs.addWidget(self.lbl_visor_img, stretch=1)
        panel_imgs.setLayout(lay_imgs)
        
        self.dic_imagenes = {n: a for n, a in nombres_imagenes}
        self.lista_imgs.currentTextChanged.connect(self._mostrar_imagen_seleccionada)

        panel_graf = QWidget()
        lay_graf = QVBoxLayout()
        lay_graf.setContentsMargins(0, 0, 0, 0)
        
        lbl_titulo_graf = QLabel("🔴 ANÁLISIS EN VIVO: ESPACIO DE ESTADOS Y TRANSFORMADAS")
        lbl_titulo_graf.setFont(QFont("Consolas", 12, QFont.Bold))
        lbl_titulo_graf.setStyleSheet("color:#39FF14; background:#0F1830; padding:8px; border-radius:5px; border:1px solid #39FF14;")
        lbl_titulo_graf.setAlignment(Qt.AlignCenter)
        lay_graf.addWidget(lbl_titulo_graf)

        self.glw_sust = pg.GraphicsLayoutWidget()
        lay_graf.addWidget(self.glw_sust, stretch=1)

        self.p_sust_fft = self.glw_sust.addPlot(title="1. Transformada Rápida de Fourier FFT en dsPIC")
        self.p_sust_fft.setLabel('left', "Amplitud")
        self.p_sust_fft.setLabel('bottom', "Bin de Frecuencia")
        self.p_sust_fft.setXRange(-0.5, 31.5)
        self.p_sust_fft.setYRange(0, self.LIM_VIB)
        _bloquear_grafica_pg(self.p_sust_fft)
        
        self.bar_sust_fft = pg.BarGraphItem(x=range(32), height=np.zeros(32), width=0.8, brush='#AA88FF')
        self.p_sust_fft.addItem(self.bar_sust_fft)

        self.p_sust_iir = self.glw_sust.addPlot(title="2. Filtro IIR Butterworth Atenuación Ruido")
        self.p_sust_iir.setLabel('left', "Vibración")
        self.p_sust_iir.setYRange(0, self.LIM_VIB * 2)
        _bloquear_grafica_pg(self.p_sust_iir)
        self.p_sust_iir.addLegend(offset=(10, 10))
        self.line_sust_raw = self.p_sust_iir.plot(pen=pg.mkPen(C_RAW_DIM, width=1, style=Qt.DashLine), name="Señal Cruda")
        self.line_sust_iir = self.p_sust_iir.plot(pen=pg.mkPen(C_IIR, width=2), name="Señal Filtrada")

        self.glw_sust.nextRow()

        self.p_sust_pars = self.glw_sust.addPlot(title="3. Teorema de Parseval Conservación Energía")
        self.p_sust_pars.setLabel('left', "Energía")
        _bloquear_grafica_pg(self.p_sust_pars)
        self.p_sust_pars.addLegend(offset=(10, 10))
        self.line_sust_etime = self.p_sust_pars.plot(pen=pg.mkPen(C_VIB, width=2.5), name="Tiempo x²")
        self.line_sust_efreq = self.p_sust_pars.plot(pen=pg.mkPen('#AA88FF', width=1.5, style=Qt.DashLine), name="Freq X²")

        # =====================================================================
        # EL WOW FACTOR: Retrato de Fase Phase Portrait
        # =====================================================================
        self.p_sust_orbit = self.glw_sust.addPlot(title="4. Espacio de Estados: Trayectoria de Fase")
        self.p_sust_orbit.setLabel('left', "Amplitud x[n]")
        self.p_sust_orbit.setLabel('bottom', "Amplitud x[n-1]")
        _bloquear_grafica_pg(self.p_sust_orbit)
        self.p_sust_orbit.addLegend(offset=(10, 10))
        
        self.line_orbit_raw = self.p_sust_orbit.plot(pen=None, symbol='o', symbolSize=3, symbolBrush=pg.mkBrush(QColor(255,107,107,150)), name="Ruido Crudo")
        self.line_orbit_iir = self.p_sust_orbit.plot(pen=pg.mkPen(C_IIR, width=2.5), name="Órbita Estable IIR dsPIC")

        panel_graf.setLayout(lay_graf)
        
        layout.addWidget(panel_imgs, stretch=1)
        layout.addWidget(panel_graf, stretch=3)
        w.setLayout(layout)
        return w

    def _mostrar_imagen_seleccionada(self, texto):
        archivo = self.dic_imagenes.get(texto)
        if archivo:
            self.lbl_visor_img.set_imagen(archivo)

    def _crear_tab_procesado_pg(self, layout):
        self.panel_datos_t1 = PanelDatosVivo([
            ('ml_status', 'ESTADO ML',  '',   '#00FFCC'),
            ('v_proc',   'VIB EMA',  'u.a', C_VIB),
            ('v_iir',    'VIB IIR',  'u.a', C_IIR),
            ('c_proc',   'CORR EMA', 'mA',  C_CORR),
            ('rpm',      'RPM EJE',    'rpm', '#AA88FF'),
            ('bin_dom',  'BIN DOM',    '',    '#FF88AA')
        ])
        layout.addWidget(self.panel_datos_t1)

        self.glw_p1 = pg.GraphicsLayoutWidget()
        layout.addWidget(self.glw_p1)

        self.p1_v = self.glw_p1.addPlot(title="Vibración Multi-Filtro EMA e IIR Biquad y Corriente")
        self.p1_v.setLabel('left', "Vibración u.a.", color=C_VIB)
        self.p1_v.showGrid(x=True, y=True, alpha=0.3)
        self.p1_v.setXRange(0, self.MAX_DATOS)
        self.p1_v.setYRange(0, self.LIM_VIB * 1.6)
        _bloquear_grafica_pg(self.p1_v)

        self.p1_c = pg.ViewBox()
        self.p1_v.showAxis('right')
        self.p1_v.scene().addItem(self.p1_c)
        self.p1_v.getAxis('right').linkToView(self.p1_c)
        self.p1_c.setXLink(self.p1_v)
        self.p1_v.getAxis('right').setLabel('Corriente mA', color=C_CORR)
        self.p1_c.setYRange(0, self.LIM_CORR * 1.6)
        _bloquear_grafica_pg(self.p1_c)

        def updateViews_p1():
            self.p1_c.setGeometry(self.p1_v.vb.sceneBoundingRect())
            self.p1_c.linkedViewChanged(self.p1_v.vb, self.p1_c.XAxis)
        
        updateViews_p1()
        QTimer.singleShot(0, updateViews_p1)
        self.p1_v.vb.sigResized.connect(updateViews_p1)

        c_vib_color = QColor(C_VIB)
        c_vib_color.setAlpha(50)
        c_iir_color = QColor(C_IIR)
        c_iir_color.setAlpha(30)
        c_corr_color = QColor(C_CORR)
        c_corr_color.setAlpha(30)

        self.line_v1 = self.p1_v.plot(pen=pg.mkPen(C_VIB, width=2), fillLevel=0, brush=pg.mkBrush(c_vib_color), name="Vibración EMA")
        self.line_v1_iir = self.p1_v.plot(pen=pg.mkPen(C_IIR, width=2, style=Qt.DashLine), fillLevel=0, brush=pg.mkBrush(c_iir_color), name="Vibración IIR")
        self.line_c1 = pg.PlotDataItem(pen=pg.mkPen(C_CORR, width=2), fillLevel=0, brush=pg.mkBrush(c_corr_color), name="Corriente")
        self.p1_c.addItem(self.line_c1)

        lim_v = pg.InfiniteLine(pos=self.LIM_VIB, angle=0, pen=pg.mkPen(C_ALARM, width=1, style=Qt.DashLine))
        self.p1_v.addItem(lim_v)
        lim_c = pg.InfiniteLine(pos=self.LIM_CORR, angle=0, pen=pg.mkPen(C_ALARM, width=1, style=Qt.DashLine))
        self.p1_c.addItem(lim_c)
        
        self.line_ref = pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen(C_REF, width=1.5, style=Qt.DotLine))
        self.p1_c.addItem(self.line_ref)

        self.glw_p1.nextRow()

        self.p1_fft = self.glw_p1.addPlot(title="Espectrograma en Cascada FFT")
        self.p1_fft.setLabel('left', "Bin Frecuencia", color='#8899BB')
        self.p1_fft.setLabel('bottom', "Muestras tiempo →", color=C_TEXT_DIM)
        _bloquear_grafica_pg(self.p1_fft)
        
        self.img_fft = pg.ImageItem()
        self.p1_fft.addItem(self.img_fft)
        
        colormap = pg.colormap.get('plasma')
        self.img_fft.setColorMap(colormap)
        self.img_fft.setRect(QRectF(0, 0, self.MAX_DATOS, 32))

    def _crear_tab_crudo_pg(self, layout):
        self.panel_datos_t2 = PanelDatosVivo([
            ('v_crud',  'VIB CRUDA',  'u.a', C_VIB_RAW),
            ('c_crud',  'CORR CRUDA', 'mA',  C_CORR_RAW),
            ('rpm',     'RPM EJE',    'rpm', '#AA88FF'),
            ('bin_dom', 'BIN DOM',    '',    '#FF88AA'),
            ('alertas', 'ALERTAS',    '',    C_ALARM),
        ])
        layout.addWidget(self.panel_datos_t2)

        self.glw_p2 = pg.GraphicsLayoutWidget()
        layout.addWidget(self.glw_p2)

        self.p2_v = self.glw_p2.addPlot(title="Picos Crudos en Tiempo Real sin filtro")
        self.p2_v.setLabel('left', "Vibración pico u.a.", color=C_VIB_RAW)
        self.p2_v.showGrid(x=True, y=True, alpha=0.3)
        self.p2_v.setXRange(0, self.MAX_DATOS)
        self.p2_v.setYRange(0, self.LIM_VIB * 1.8)
        _bloquear_grafica_pg(self.p2_v)

        self.p2_c = pg.ViewBox()
        self.p2_v.showAxis('right')
        self.p2_v.scene().addItem(self.p2_c)
        self.p2_v.getAxis('right').linkToView(self.p2_c)
        self.p2_c.setXLink(self.p2_v)
        self.p2_v.getAxis('right').setLabel('Corriente pico mA', color=C_CORR_RAW)
        self.p2_c.setYRange(0, self.LIM_CORR * 1.8)
        _bloquear_grafica_pg(self.p2_c)

        def updateViews_p2():
            self.p2_c.setGeometry(self.p2_v.vb.sceneBoundingRect())
            self.p2_c.linkedViewChanged(self.p2_v.vb, self.p2_c.XAxis)
        
        updateViews_p2()
        self.p2_v.vb.sigResized.connect(updateViews_p2)

        self.line_v2 = self.p2_v.plot(pen=pg.mkPen(C_VIB_RAW, width=1.5), symbol='o', symbolSize=4, symbolBrush=C_VIB_RAW)
        self.line_c2 = pg.PlotDataItem(pen=pg.mkPen(C_CORR_RAW, width=1.5), symbol='o', symbolSize=4, symbolBrush=C_CORR_RAW)
        self.p2_c.addItem(self.line_c2)

        lim_v2 = pg.InfiniteLine(pos=self.LIM_VIB*1.2, angle=0, pen=pg.mkPen(C_ALARM, width=1, style=Qt.DashLine))
        self.p2_v.addItem(lim_v2)

    def _crear_tab_didactica(self):
        contenedor = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        self.sub_tabs = QTabWidget()
        self.sub_tabs.setStyleSheet("QTabBar::tab{background:#0A1020;padding:5px 10px;color:#557799;font-size:8pt;} QTabBar::tab:selected{background:#0F1830;color:#FFD700;font-weight:bold;}")
        self.sub_tabs.addTab(self._tab_ema(),       "① EMA vs IIR Biquad")
        self.sub_tabs.addTab(self._tab_rms(),       "② True RMS")
        self.sub_tabs.addTab(self._tab_fft(),       "③ FFT — Espectro")
        self.sub_tabs.addTab(self._tab_pasabaja(),  "④ Pasa-Bajas RPM")
        self.sub_tabs.addTab(self._tab_parseval(),  "⑤ Teorema de Parseval")
        layout.addWidget(self.sub_tabs)
        contenedor.setLayout(layout)
        return contenedor

    def _crear_tab_avanzado(self):
        contenedor = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(8,8,8,8)

        self.glw_avanzado = pg.GraphicsLayoutWidget()
        layout.addWidget(self.glw_avanzado, stretch=3)

        self.p_env = self.glw_avanzado.addPlot(title="ANÁLISIS DE ENVOLVENTE Detección de Impactos")
        self.p_env.setLabel('left', "Amplitud u.a.", color='#FFD700')
        self.p_env.showGrid(x=True, y=True, alpha=0.3)
        self.p_env.setXRange(0, self.MAX_DATOS)
        self.p_env.setYRange(0, self.LIM_VIB * 2)
        _bloquear_grafica_pg(self.p_env)
        
        pen_env = pg.mkPen('#FFD700', width=2)
        brush_env = QColor('#FFD700')
        brush_env.setAlpha(30)
        self.line_envolvente = self.p_env.plot(pen=pen_env, fillLevel=0, brush=pg.mkBrush(brush_env))

        self.glw_avanzado.nextRow()

        self.p_ceps = self.glw_avanzado.addPlot(title="ANÁLISIS CEPSTRAL Periodicidad de Impactos Alta Res.")
        self.p_ceps.setLabel('left', "Amplitud", color='#00FFCC')
        self.p_ceps.setLabel('bottom', "Quefrency Tiempo", color=C_TEXT_DIM)
        self.p_ceps.showGrid(x=True, y=True, alpha=0.3)
        self.p_ceps.setXRange(0, self.MAX_DATOS_LARGO // 2)
        _bloquear_grafica_pg(self.p_ceps)
        
        self.line_cepstrum = self.p_ceps.plot(pen=pg.mkPen('#00FFCC', width=2), fillLevel=0, brush=pg.mkBrush(QColor(0,255,204,30)), stepMode=False)

        grp_analisis = QGroupBox("Síntesis del Análisis de Filtros en Tiempo Real")
        grp_analisis.setStyleSheet("QGroupBox{border:1px solid #2A5090;border-radius:6px;margin-top:10px;font-weight:bold;color:#7AAFFF;font-size:9pt;} QGroupBox::title{subcontrol-origin:margin;left:10px;padding:0 5px;}")
        lay_analisis = QVBoxLayout()
        
        self.txt_analisis = QTextEdit()
        self.txt_analisis.setReadOnly(True)
        self.txt_analisis.setStyleSheet(f"background:#050811; color:{C_TEXT}; font-family:Consolas; font-size:10pt; border:none; padding: 5px;")
        self.txt_analisis.setMaximumHeight(160)
        
        lay_analisis.addWidget(self.txt_analisis)
        grp_analisis.setLayout(lay_analisis)
        
        layout.addWidget(grp_analisis, stretch=1)

        contenedor.setLayout(layout)
        return contenedor

    def _tab_ema(self):
        w = QWidget()
        main = QVBoxLayout()
        main.setSpacing(6)
        self.panel_datos_ema = PanelDatosVivo([('v_crud', 'VIB CRUDA', 'u.a', C_RAW_DIM), ('v_proc', 'VIB EMA', 'u.a', C_EMA), ('v_iir', 'VIB IIR', 'u.a', C_IIR), ('rpm',    'RPM',       'rpm', '#AA88FF')])
        main.addWidget(self.panel_datos_ema)

        math_box = QLabel("<b style='color:#FFD700;font-size:10pt;'>Comparación de Filtros en Tiempo Real</b><br><br><span style='color:#00BFFF;font-family:Consolas;font-size:10pt;'>EMA Filtro de Promedio Móvil Exponencial:</span><span style='color:#AADDFF;'> Ideal para tendencias visuales sin consumir ciclos de CPU. Suaviza la señal temporal aplicando un coeficiente recursivo.</span><br><br><span style='color:#39FF14;font-family:Consolas;font-size:10pt;'>IIR Biquad Butterworth:</span><span style='color:#AADDFF;'> Pasa-Bajas agresivo de 2do orden en el Plano Z. Utiliza instrucciones MAC nativas del DSP para eliminar frecuencias parásitas ruido blanco electromagnético garantizando que la FFT procese espectros limpios.</span>")
        math_box.setWordWrap(True)
        math_box.setStyleSheet(f"background:{C_PANEL};border:1px solid {C_BORDER};border-radius:6px;padding:10px;")
        main.addWidget(math_box)

        alpha_row = QHBoxLayout()
        alpha_row.addWidget(QLabel("Simulador de α Solo afecta al EMA visual:"))
        self.slider_alpha = QSlider(Qt.Horizontal)
        self.slider_alpha.setRange(1, 99)
        self.slider_alpha.setValue(int(self.alpha_didact * 100))
        self.lbl_alpha = QLabel(f"{self.alpha_didact:.2f}  dsPIC >> {self._alpha_to_shift(self.alpha_didact)}")
        self.lbl_alpha.setStyleSheet("color:#FFD700;font-family:Consolas;min-width:200px;")
        self.slider_alpha.valueChanged.connect(self._on_alpha_change)
        alpha_row.addWidget(self.slider_alpha, stretch=1)
        alpha_row.addWidget(self.lbl_alpha)
        main.addLayout(alpha_row)

        self.fig_ema = Figure(facecolor=C_BG)
        self.fig_ema.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.15)
        self.cv_ema  = FigureCanvas(self.fig_ema)
        self.cv_ema.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.ax_ema = self.fig_ema.add_subplot(111)
        _estilizar_ax(self.ax_ema, "Ruido vs EMA vs IIR Biquad", "Amplitud", C_EMA)
        
        self.line_raw_ema, = self.ax_ema.plot([], [], color=C_RAW_DIM, lw=0.9, alpha=0.7, label='Señal Cruda')
        self.line_ema_out, = self.ax_ema.plot([], [], color=C_EMA, lw=2.0, label='Salida EMA Suave')
        self.line_iir_out, = self.ax_ema.plot([], [], color=C_IIR, lw=2.0, ls='--', label='Salida IIR Biquad Agresivo')
        
        self.ax_ema.set_xlim(0, self.MAX_DATOS - 1)
        self.ax_ema.set_ylim(0, self.LIM_VIB * 2)
        self.ax_ema.legend(loc='upper left', fontsize=7, facecolor='#0A1020', edgecolor=C_BORDER, labelcolor='white')
        main.addWidget(self.cv_ema)
        w.setLayout(main)
        return w

    def _alpha_to_shift(self, alpha):
        if alpha <= 0: return "∞"
        shift = round(-np.log2(alpha))
        return str(int(max(0, shift)))

    def _on_alpha_change(self, val):
        self.alpha_didact = val / 100.0
        self.ema_sim = 0.0
        shift = self._alpha_to_shift(self.alpha_didact)
        self.lbl_alpha.setText(f"{self.alpha_didact:.2f}  dsPIC >> {shift}")

    def _tab_rms(self):
        w = QWidget()
        main = QVBoxLayout()
        main.setSpacing(6)
        self.panel_datos_rms = PanelDatosVivo([('c_proc', 'TRUE RMS', 'mA', C_CORR), ('c_crud', 'PICO CRUDO', 'mA', C_CORR_RAW), ('rpm', 'RPM', 'rpm', '#AA88FF')])
        main.addWidget(self.panel_datos_rms)

        math_box = QLabel("<b style='color:#FFD700;font-size:10pt;'>True RMS — Valor Eficaz Real</b><br><br><span style='color:#00FFCC;font-family:Consolas;font-size:10pt;'>RMS = √ 1/N · Σ x²[i] </span><br><br><span style='color:#AADDFF;'>Implementación en el dsPIC:<br><span style='color:#FFD700;font-family:Consolas;'>1. Restar offset DC de hardware fijo: valor_ac = ADC[i] − 2048<br>2. Acumular cuadrados: suma += valor_ac²<br>3. Media: media = suma >> 6  ÷64 muestras<br>4. Raíz entera optimizada: RMS = integer_sqrt media<br>5. Convertir a mA: mA = RMS × factor_sensor ÷ 1000</span><br><br>El True RMS es indispensable para medir la energía térmica real generada en los devanados ya que la corriente de un motor DC no es perfectamente sinusoidal y un cálculo de valor pico generaría errores de medición superiores al 40%.</span>")
        math_box.setWordWrap(True)
        math_box.setStyleSheet(f"background:{C_PANEL};border:1px solid {C_BORDER};border-radius:6px;padding:10px;")
        main.addWidget(math_box)

        self.fig_rms = Figure(facecolor=C_BG)
        self.fig_rms.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.15, wspace=0.35)
        self.cv_rms  = FigureCanvas(self.fig_rms)
        self.cv_rms.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        gs_rms = gridspec.GridSpec(1, 2, figure=self.fig_rms, wspace=0.35)

        self.ax_rms_hist = self.fig_rms.add_subplot(gs_rms[0])
        _estilizar_ax(self.ax_rms_hist, "Historial True RMS Corriente", "mA RMS", C_CORR)
        self.line_rms_hist, = self.ax_rms_hist.plot([], [], color=C_CORR, lw=1.5, label='True RMS')
        self.line_rms_peak, = self.ax_rms_hist.plot([], [], color=C_CORR_RAW, lw=1.0, ls='--', alpha=0.7, label='Pico crudo')
        self.ax_rms_hist.set_xlim(0, self.MAX_DATOS - 1)
        self.ax_rms_hist.set_ylim(0, self.LIM_CORR * 1.5)
        self.ax_rms_hist.legend(fontsize=7, facecolor='#0A1020', edgecolor=C_BORDER, labelcolor='white')

        self.ax_rms_bar = self.fig_rms.add_subplot(gs_rms[1])
        self.ax_rms_bar.set_facecolor(C_AX_BG)
        self.ax_rms_bar.set_title("RMS vs Pico actual", color='#AACCFF', fontsize=9, fontweight='bold')
        self.ax_rms_bar.tick_params(colors='#6688AA', labelsize=8)
        for s in self.ax_rms_bar.spines.values(): s.set_edgecolor(C_BORDER)
        self.bars_rms = self.ax_rms_bar.bar(['True RMS', 'Pico Crudo'], [0, 0], color=[C_CORR, C_CORR_RAW], width=0.5, edgecolor=C_BORDER)
        self.ax_rms_bar.set_ylim(0, self.LIM_CORR * 1.5)
        self.ax_rms_bar.set_ylabel("mA", color='#8899BB', fontsize=8)
        self.txt_rms_factor = self.ax_rms_bar.text(0.5, 0.95, "Factor Pico/RMS: —", transform=self.ax_rms_bar.transAxes, ha='center', color='#FFD700', fontsize=8)

        main.addWidget(self.cv_rms)
        w.setLayout(main)
        return w

    def _tab_fft(self):
        w = QWidget()
        main = QVBoxLayout()
        main.setSpacing(6)
        self.panel_datos_fft = PanelDatosVivo([('v_proc', 'VIB EMA', 'u.a', C_VIB), ('bin_dom', 'BIN DOM', '', '#FF88AA'), ('rpm', 'RPM', 'rpm', '#AA88FF')])
        main.addWidget(self.panel_datos_fft)

        math_box = QLabel("<b style='color:#FFD700;font-size:10pt;'>FFT — Transformada Rápida de Fourier DSP Edge</b><br><br><span style='color:#AADDFF;'>El microcontrolador ejecuta una FFT de 64 puntos utilizando bibliotecas nativas de DSP procesando la señal en el dominio temporal y proyectándola al dominio de la frecuencia para aislar vibraciones mecánicas.<br><br><span style='color:#FFD700;font-family:Consolas;'>Resolución de frecuencia: Δf = Fs / N<br>Fs = RPM × 16 / 60  Muestreo Adaptativo</span><br><br>Se ha implementado una ventana de Hanning directamente en el código C para mitigar el efecto de fuga espectral Spectral Leakage concentrando la energía en el bin central de la falla mecánica detectada.</span>")
        math_box.setWordWrap(True)
        math_box.setStyleSheet(f"background:{C_PANEL};border:1px solid {C_BORDER};border-radius:6px;padding:10px;")
        main.addWidget(math_box)

        self.fig_fft_d = Figure(facecolor=C_BG)
        self.fig_fft_d.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.15, wspace=0.35)
        self.cv_fft_d  = FigureCanvas(self.fig_fft_d)
        self.cv_fft_d.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        gs_fft = gridspec.GridSpec(1, 2, figure=self.fig_fft_d, wspace=0.35)

        self.ax_fft_bars = self.fig_fft_d.add_subplot(gs_fft[0])
        self.ax_fft_bars.set_facecolor(C_AX_BG)
        self.ax_fft_bars.set_title("Espectro de Amplitud Transmitido por dsPIC", color='#AACCFF', fontsize=9, fontweight='bold')
        self.ax_fft_bars.tick_params(colors='#6688AA', labelsize=7)
        self.ax_fft_bars.set_xlabel("Bin de Frecuencia", color=C_TEXT_DIM, fontsize=7)
        self.ax_fft_bars.set_ylabel("Amplitud", color='#AA88FF', fontsize=8)
        for s in self.ax_fft_bars.spines.values(): s.set_edgecolor(C_BORDER)
        self.bar_fft = self.ax_fft_bars.bar(range(32), self.fft_bins, color='#AA88FF', edgecolor='none', width=0.8)
        self.ax_fft_bars.set_xlim(-0.5, 31.5)
        self.ax_fft_bars.set_ylim(0, self.LIM_VIB)

        self.ax_fft_cascade = self.fig_fft_d.add_subplot(gs_fft[1])
        self.ax_fft_cascade.set_facecolor('black')
        self.ax_fft_cascade.set_title("Cascada FFT Histórica", color='#AACCFF', fontsize=9, fontweight='bold')
        self.ax_fft_cascade.tick_params(colors='#6688AA', labelsize=7)
        for s in self.ax_fft_cascade.spines.values(): s.set_edgecolor(C_BORDER)
        self.img_fft_d = self.ax_fft_cascade.imshow(self.matriz_fft, aspect='auto', cmap='inferno', origin='lower', vmin=0, vmax=self.LIM_VIB/2, extent=[0, self.MAX_DATOS, 0, 31], interpolation='bilinear')

        main.addWidget(self.cv_fft_d)
        w.setLayout(main)
        return w

    def _tab_pasabaja(self):
        w = QWidget()
        main = QVBoxLayout()
        main.setSpacing(6)
        self.panel_datos_rpm = PanelDatosVivo([('rpm', 'RPM MEDIDA', 'rpm', '#AA88FF'), ('rpm_obj', 'RPM OBJETIVO', 'rpm', '#FFD700')])
        main.addWidget(self.panel_datos_rpm)

        math_box = QLabel("<b style='color:#FFD700;font-size:10pt;'>Filtro Pasa-Altas Digital — Detección Automática de RPM</b><br><br><span style='color:#AADDFF;'>El sistema extrae las RPM contando los picos de las escobillas generados en el colector del motor eliminando el componente de corriente continua.<br><br><span style='color:#FFD700;font-family:Consolas;'>delta = ADC[n] − ADC[n-1]<br>hp_filtrado = 31130 × hp_filtrado + delta >> 15<br>Si hp_filtrado > umbral_dinamico → cuenta un pico</span><br><br>El uso de aritmética de enteros y desplazamientos de bits en lugar de divisiones permite que el dsPIC procese este algoritmo dentro de una interrupción sin bloquear el sistema operativo calculando finalmente:<br><span style='color:#00FFCC;font-family:Consolas;'>RPM = picos × Fs × 60 / N_muestras × N_delgas</span></span>")
        math_box.setWordWrap(True)
        math_box.setStyleSheet(f"background:{C_PANEL};border:1px solid {C_BORDER};border-radius:6px;padding:10px;")
        main.addWidget(math_box)

        self.fig_rpm = Figure(facecolor=C_BG)
        self.fig_rpm.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.15)
        self.cv_rpm  = FigureCanvas(self.fig_rpm)
        self.cv_rpm.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.ax_rpm_hist = self.fig_rpm.add_subplot(111)
        _estilizar_ax(self.ax_rpm_hist, "RPM Medidas vs RPM Objetivo", "RPM", '#AA88FF')
        self.hist_rpm_med = [0.0] * self.MAX_DATOS
        self.hist_rpm_obj = [0.0] * self.MAX_DATOS
        self.line_rpm_med, = self.ax_rpm_hist.plot([], [], color='#AA88FF', lw=1.5, label='RPM medidas')
        self.line_rpm_obj, = self.ax_rpm_hist.plot([], [], color='#FFD700', lw=1.0, ls='--', alpha=0.8, label='RPM objetivo')
        self.ax_rpm_hist.set_xlim(0, self.MAX_DATOS - 1)
        self.ax_rpm_hist.set_ylim(0, 12000)
        self.ax_rpm_hist.legend(fontsize=7, facecolor='#0A1020', edgecolor=C_BORDER, labelcolor='white')
        main.addWidget(self.cv_rpm)
        w.setLayout(main)
        return w

    def _tab_parseval(self):
        w = QWidget()
        main = QVBoxLayout()
        main.setSpacing(6)
        self.panel_datos_pars = PanelDatosVivo([
            ('e_time', 'ENERGÍA TIEMPO', 'u²', C_VIB), 
            ('e_freq', 'ENERGÍA FFT', 'u²', '#AA88FF'), 
            ('err_pars', 'DESVIACIÓN', '%', C_ALARM)
        ])
        main.addWidget(self.panel_datos_pars)

        math_box = QLabel("<b style='color:#FFD700;font-size:10pt;'>Teorema de Parseval — Validación Espectral Continua</b><br><br><span style='color:#AADDFF;'>Este teorema fundamental de procesamiento de señales estipula que la energía total de una señal debe conservarse íntegra al transformar su dominio físico.<br><br><span style='color:#00FFCC;font-family:Consolas;font-size:10pt;'>Dominio Temporal: E_t = Σ |x[n]|²  ==  Dominio Frecuencial: E_f = 1/N Σ |X[k]|²</span><br><br>Esta pestaña valida experimentalmente la integridad algorítmica de la FFT generada por el dsPIC. Si ambas energías se superponen demostramos irrefutablemente que no existen fugas de energía matemática ni corrupción de datos en el búfer de telemetría enviado por Modbus RTU.</span>")
        math_box.setWordWrap(True)
        math_box.setStyleSheet(f"background:{C_PANEL};border:1px solid {C_BORDER};border-radius:6px;padding:10px;")
        main.addWidget(math_box)

        self.fig_pars = Figure(facecolor=C_BG)
        self.fig_pars.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.15)
        self.cv_pars  = FigureCanvas(self.fig_pars)
        self.cv_pars.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.ax_pars = self.fig_pars.add_subplot(111)
        _estilizar_ax(self.ax_pars, "Conservación de Energía Parseval: Tiempo vs Frecuencia", "Energía u²", '#39FF14')
        self.line_etime, = self.ax_pars.plot([], [], color=C_VIB, lw=2.5, alpha=0.9, label='Energía Tiempo x[n]²')
        self.line_efreq, = self.ax_pars.plot([], [], color='#AA88FF', lw=1.5, ls='--', label='Energía Frecuencia X[k]²')
        
        self.ax_pars.set_xlim(0, self.MAX_DATOS - 1)
        self.ax_pars.legend(fontsize=7, facecolor='#0A1020', edgecolor=C_BORDER, labelcolor='white')
        
        main.addWidget(self.cv_pars)

        self.tabla_parseval = QTableWidget(0, 4)
        self.tabla_parseval.setHorizontalHeaderLabels(["Tiempo HH:MM:SS", "Energía Tiempo Σx²", "Energía Frec 1/N ΣX²", "Desviación %"])
        self.tabla_parseval.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tabla_parseval.verticalHeader().setVisible(False)
        self.tabla_parseval.setFixedHeight(120)
        self.tabla_parseval.setStyleSheet("""
            QTableWidget { background-color: #0D1525; color: #C8D8F0; gridline-color: #1A3060; border: 1px solid #1A3060; font-family: Consolas; font-size: 9pt; }
            QHeaderView::section { background-color: #0F1830; color: #AACCFF; padding: 4px; border: 1px solid #1A3060; font-weight: bold; }
        """)
        main.addWidget(self.tabla_parseval)

        w.setLayout(main)
        return w

    def _crear_tab_rul(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f"QScrollArea{{border:none;background:{C_BG};}} QScrollBar:vertical{{background:#0D1525;width:8px;border-radius:4px;}} QScrollBar::handle:vertical{{background:#1A3060;border-radius:4px;}}")
        w = QWidget()
        main = QVBoxLayout()
        main.setSpacing(8)
        main.setContentsMargins(8,8,8,8)

        grp_placa = QGroupBox("Datos de Placa del Motor Base de Datos Integrada")
        grp_placa.setStyleSheet("QGroupBox{border:1px solid #2A5090;border-radius:6px;margin-top:10px;font-weight:bold;color:#7AAFFF;font-size:9pt;} QGroupBox::title{subcontrol-origin:margin;left:10px;padding:0 5px;}")
        grid_placa = QGridLayout()
        grid_placa.setSpacing(8)

        campos_placa = [
            ("RPM nominales:", "rpm_nom", "2200", "rpm", 0, 0), 
            ("Corriente en vacío:", "i_vacio", "350", "mA", 0, 3), 
            ("Corriente con carga:", "i_nom", "2400", "mA", 1, 0), 
            ("Corriente de atorado:", "i_bloq", "5600", "mA", 1, 3), 
            ("Vibración sana nominal:", "vib_nom", "20", "u.a.", 2, 0),     
            ("Vida útil fabricante:", "vida_nom", "17520", "h", 2, 3)      
        ]
        
        self._inputs_placa = {}
        for label_txt, key, default, unit, row, col in campos_placa:
            lbl = QLabel(label_txt)
            lbl.setStyleSheet(f"color:{C_TEXT};font-size:9pt;")
            inp = QLineEdit(default)
            inp.setFixedWidth(90)
            inp.setStyleSheet(f"background:#0D1525;border:1px solid {C_BORDER};border-radius:4px;padding:3px 6px;color:#FFD700;font-family:Consolas;")
            lbl_u = QLabel(unit)
            lbl_u.setStyleSheet(f"color:{C_TEXT_DIM};font-size:8pt;min-width:28px;")
            
            grid_placa.addWidget(lbl, row, col)
            grid_placa.addWidget(inp, row, col+1)
            grid_placa.addWidget(lbl_u, row, col+2)
            self._inputs_placa[key] = inp

        lbl_tag = QLabel("Tag/Serial:")
        lbl_tag.setStyleSheet(f"color:{C_TEXT};font-size:9pt;")
        self.input_tag = QLineEdit()
        self.input_tag.setPlaceholderText("Ej: MOT-001")
        self.input_tag.setFixedWidth(90)
        self.input_tag.setStyleSheet(f"background:#0D1525;border:1px solid {C_BORDER};border-radius:4px;padding:3px 6px;color:#FFD700;font-family:Consolas;")
        grid_placa.addWidget(lbl_tag, 3, 0)
        grid_placa.addWidget(self.input_tag, 3, 1, 1, 2)

        self.db_motores = {
            "Personalizado...": [0, 0, 0, 0, 0, 0, 0, 0],
            "Pololu 25D 131:1 Tu Motor": [76, 150, 650, 2400, 6, 131, 12, 5000],
            "Pololu 37D 70:1 Torque Medio": [150, 300, 2500, 5500, 6, 70, 15, 6000],
            "Johnson JS-540 Estándar 12V": [16000, 800, 4500, 20000, 5, 1, 8, 3000],
            "Mabuchi RS-775 Alta Potencia": [18000, 1200, 8000, 40000, 5, 1, 10, 2000],
            "Maxon RE-MAX 29 Precisión": [7200, 100, 1200, 8000, 9, 1, 3, 10000],
            "Bühler DC Motor Automotriz": [3000, 500, 3000, 12000, 5, 1, 6, 5000],
            "Motorreductor Industrial 24V 50:1": [60, 400, 1800, 7500, 6, 50, 10, 8000]
        }
        
        lay_db = QHBoxLayout()
        lay_db.addWidget(QLabel("Cargar perfil de motor:"))
        self.combo_motores = QComboBox()
        self.combo_motores.addItems(self.db_motores.keys())
        self.combo_motores.setStyleSheet("background:#0D1525; color:#00FFCC; border:1px solid #1A3060; padding:4px;")
        self.combo_motores.currentTextChanged.connect(self._autocompletar_motor)
        lay_db.addWidget(self.combo_motores, stretch=1)
        grid_placa.addLayout(lay_db, 4, 0, 1, 6)

        self.btn_confirmar_placa = QPushButton("✔  Confirmar datos de placa y activar RUL")
        self.btn_confirmar_placa.setStyleSheet("QPushButton{background:#002244;color:#00BFFF;border:1px solid #0055AA;border-radius:5px;font-weight:bold;padding:6px 18px;font-size:9pt;} QPushButton:hover{background:#003366;}")
        self.btn_confirmar_placa.clicked.connect(self._confirmar_placa)
        grid_placa.addWidget(self.btn_confirmar_placa, 5, 0, 1, 6)
        
        grp_placa.setLayout(grid_placa)
        main.addWidget(grp_placa)

        grp_rul = QGroupBox("Pronóstico Exponencial de Vida Útil RUL")
        grp_rul.setStyleSheet("QGroupBox{border:1px solid #2A5090;border-radius:6px;margin-top:10px;font-weight:bold;color:#7AAFFF;font-size:9pt;} QGroupBox::title{subcontrol-origin:margin;left:10px;padding:0 5px;}")
        rul_row = QHBoxLayout()
        rul_row.setSpacing(20)

        self.lbl_rul_valor = QLabel("—")
        self.lbl_rul_valor.setFont(QFont("Consolas", 38, QFont.Bold))
        self.lbl_rul_valor.setStyleSheet("color:#00FFCC;background:transparent;")
        self.lbl_rul_valor.setAlignment(Qt.AlignCenter)
        
        self.lbl_rul_unidad = QLabel("horas restantes")
        self.lbl_rul_unidad.setFont(QFont("Consolas", 11))
        self.lbl_rul_unidad.setStyleSheet(f"color:{C_TEXT_DIM};background:transparent;")
        self.lbl_rul_unidad.setAlignment(Qt.AlignCenter)
        
        self.lbl_rul_conf = QLabel("Intervalo de confianza: ±—")
        self.lbl_rul_conf.setFont(QFont("Consolas", 9))
        self.lbl_rul_conf.setStyleSheet("color:#FFD700;background:transparent;")
        self.lbl_rul_conf.setAlignment(Qt.AlignCenter)
        
        self.lbl_rul_di = QLabel("DI actual: —  |  Umbral paramétrico: 6.0σ")
        self.lbl_rul_di.setFont(QFont("Consolas", 9))
        self.lbl_rul_di.setStyleSheet(f"color:{C_TEXT_DIM};background:transparent;")
        self.lbl_rul_di.setAlignment(Qt.AlignCenter)
        
        self.lbl_rul_estado = QLabel("⏸  Esperando datos de placa")
        self.lbl_rul_estado.setFont(QFont("Consolas", 10, QFont.Bold))
        self.lbl_rul_estado.setStyleSheet(f"background:{C_PANEL};color:#556688;border:1px solid {C_BORDER};border-radius:4px;padding:4px;")
        self.lbl_rul_estado.setAlignment(Qt.AlignCenter)

        col_num = QVBoxLayout()
        col_num.addWidget(self.lbl_rul_valor)
        col_num.addWidget(self.lbl_rul_unidad)
        col_num.addWidget(self.lbl_rul_conf)
        col_num.addWidget(self.lbl_rul_di)
        col_num.addWidget(self.lbl_rul_estado)
        
        rul_row.addLayout(col_num, stretch=1)
        grp_rul.setLayout(rul_row)
        main.addWidget(grp_rul)

        self.fig_rul = Figure(facecolor=C_BG)
        self.fig_rul.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.15, wspace=0.35)
        self.cv_rul  = FigureCanvas(self.fig_rul)
        self.cv_rul.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        gs_rul = gridspec.GridSpec(1, 2, figure=self.fig_rul, wspace=0.35)

        self.ax_di = self.fig_rul.add_subplot(gs_rul[0])
        _estilizar_ax(self.ax_di, "Índice de Degradación DI — Historial", "DI σ", '#FF8C00')
        self.line_di,   = self.ax_di.plot([], [], color='#FF8C00', lw=1.5, label='DI actual')
        self.line_di_ma,= self.ax_di.plot([], [], color='#FFD700', lw=1.0, ls='--', alpha=0.8, label='DI media móvil 20')
        self.ax_di.axhline(self._di_umbral, color=C_ALARM, ls='--', lw=1.0, alpha=0.8, label=f'Umbral falla {self._di_umbral}σ')
        self.ax_di.set_xlim(0, 100)
        self.ax_di.set_ylim(0, self._di_umbral * 1.5)
        self.ax_di.legend(fontsize=7, facecolor='#0A1020', edgecolor=C_BORDER, labelcolor='white')

        self.ax_rul_proj = self.fig_rul.add_subplot(gs_rul[1])
        _estilizar_ax(self.ax_rul_proj, "Proyección RUL Degradación Exponencial", "DI σ", '#00FFCC')
        self.line_rul_hist, = self.ax_rul_proj.plot([], [], color='#00FFCC', lw=1.5, label='DI histórico')
        self.line_rul_pred, = self.ax_rul_proj.plot([], [], color='#FF4488', lw=1.5, ls='--', label='Proyección P-F Curve')
        self.fill_rul_ci    = None
        self.ax_rul_proj.axhline(self._di_umbral, color=C_ALARM, ls='--', lw=1.0, alpha=0.8)
        self.ax_rul_proj.set_xlim(0, 200)
        self.ax_rul_proj.set_ylim(0, self._di_umbral * 1.5)
        self.ax_rul_proj.set_xlabel("Muestras pasadas → futuras", color=C_TEXT_DIM, fontsize=7)
        self.ax_rul_proj.legend(fontsize=7, facecolor='#0A1020', edgecolor=C_BORDER, labelcolor='white')
        
        main.addWidget(self.cv_rul)
        w.setLayout(main)
        scroll.setWidget(w)
        return scroll

    def _crear_tab_historico(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f"QScrollArea{{border:none;background:{C_BG};}} QScrollBar:vertical{{background:#0D1525;width:8px;border-radius:4px;}} QScrollBar::handle:vertical{{background:#1A3060;border-radius:4px;}}")
        w = QWidget()
        main = QVBoxLayout()
        main.setSpacing(6)
        main.setContentsMargins(6,6,6,6)
        
        fila1 = QHBoxLayout()
        fila1.setSpacing(8)

        grp_firma = QGroupBox("Firma de Arranque")
        grp_firma.setStyleSheet("QGroupBox{border:1px solid #2A5090;border-radius:6px;margin-top:8px;font-weight:bold;color:#7AAFFF;font-size:9pt;} QGroupBox::title{subcontrol-origin:margin;left:10px;padding:0 5px;}")
        firma_lay = QVBoxLayout()
        
        self.lbl_firma_sim = QLabel("Similitud con motor sano: —")
        self.lbl_firma_sim.setFont(QFont("Consolas", 20, QFont.Bold))
        self.lbl_firma_sim.setStyleSheet("color:#00FFCC;background:transparent;")
        self.lbl_firma_sim.setAlignment(Qt.AlignCenter)
        firma_lay.addWidget(self.lbl_firma_sim)

        self.lbl_firma_detalle = QLabel("RPM pico: —  |  I pico: — mA  |  T estab.: — s")
        self.lbl_firma_detalle.setFont(QFont("Consolas", 9))
        self.lbl_firma_detalle.setStyleSheet(f"color:{C_TEXT_DIM};background:transparent;")
        self.lbl_firma_detalle.setAlignment(Qt.AlignCenter)
        firma_lay.addWidget(self.lbl_firma_detalle)

        self.fig_firma = Figure(facecolor=C_BG)
        self.fig_firma.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15, hspace=0.4)
        self.cv_firma  = FigureCanvas(self.fig_firma)
        self.cv_firma.setFixedHeight(130)
        gs_f = gridspec.GridSpec(2, 1, figure=self.fig_firma, hspace=0.15)
        
        self.ax_firma_c = self.fig_firma.add_subplot(gs_f[0])
        self.ax_firma_v = self.fig_firma.add_subplot(gs_f[1])
        
        for ax, col, lbl in [(self.ax_firma_c, C_CORR, 'I mA'), (self.ax_firma_v, C_VIB,  'Vib')]:
            ax.set_facecolor(C_AX_BG)
            ax.tick_params(colors='#6688AA', labelsize=6)
            ax.set_ylabel(lbl, color=col, fontsize=7)
            for s in ax.spines.values(): 
                s.set_edgecolor(C_BORDER)
            ax.grid(True, color=C_GRID, ls='--', lw=0.4, alpha=0.7)
            
        self.line_firma_c, = self.ax_firma_c.plot([], [], color=C_CORR, lw=1.5)
        self.line_firma_v, = self.ax_firma_v.plot([], [], color=C_VIB,  lw=1.5)
        self.line_firma_c_ref, = self.ax_firma_c.plot([], [], color='#FFD700', lw=1.0, ls='--', alpha=0.7, label='Ref sana')
        self.line_firma_v_ref, = self.ax_firma_v.plot([], [], color='#FFD700', lw=1.0, ls='--', alpha=0.7)
        
        firma_lay.addWidget(self.cv_firma)
        grp_firma.setLayout(firma_lay)
        fila1.addWidget(grp_firma, stretch=2)

        grp_tend = QGroupBox("Alerta Temprana — Tendencia entre Sesiones")
        grp_tend.setStyleSheet("QGroupBox{border:1px solid #2A5090;border-radius:6px;margin-top:8px;font-weight:bold;color:#7AAFFF;font-size:9pt;} QGroupBox::title{subcontrol-origin:margin;left:10px;padding:0 5px;}")
        tend_lay = QVBoxLayout()
        
        self.lbl_tendencia = QLabel("⏸  Sin historial suficiente mín. 5 sesiones")
        self.lbl_tendencia.setFont(QFont("Consolas", 9, QFont.Bold))
        self.lbl_tendencia.setWordWrap(True)
        self.lbl_tendencia.setStyleSheet(f"background:{C_PANEL};color:#556688;border:1px solid {C_BORDER};border-radius:4px;padding:6px;")
        
        self.lbl_tend_detalle = QLabel("Pendiente DI: —  |  R²: —  |  Similitud actual: —")
        self.lbl_tend_detalle.setFont(QFont("Consolas", 8))
        self.lbl_tend_detalle.setStyleSheet(f"color:{C_TEXT_DIM};background:transparent;")
        
        tend_lay.addWidget(self.lbl_tendencia)
        tend_lay.addWidget(self.lbl_tend_detalle)

        self.fig_tend = Figure(facecolor=C_BG)
        self.fig_tend.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)
        self.cv_tend  = FigureCanvas(self.fig_tend)
        self.cv_tend.setFixedHeight(130)
        self.ax_tend = self.fig_tend.add_subplot(111)
        _estilizar_ax(self.ax_tend, "DI medio por sesión", "DI σ", '#FF8C00')
        
        self.line_tend_di,  = self.ax_tend.plot([], [], color='#FF8C00', lw=1.5, marker='o', markersize=3, label='DI sesión')
        self.line_tend_reg, = self.ax_tend.plot([], [], color='#FF4488', lw=1.0, ls='--', alpha=0.8, label='Regresión')
        self.ax_tend.axhline(3.0, color=C_ALARM, ls='--', lw=0.8, alpha=0.7)
        self.ax_tend.set_xlim(0, 10)
        self.ax_tend.set_ylim(0, 4)
        self.ax_tend.legend(fontsize=6, facecolor='#0A1020', edgecolor=C_BORDER, labelcolor='white')
        
        tend_lay.addWidget(self.cv_tend)
        grp_tend.setLayout(tend_lay)
        fila1.addWidget(grp_tend, stretch=3)
        
        main.addLayout(fila1)

        grp_hist = QGroupBox("Sesiones Anteriores base de datos local")
        grp_hist.setStyleSheet("QGroupBox{border:1px solid #2A5090;border-radius:6px;margin-top:6px;font-weight:bold;color:#7AAFFF;font-size:9pt;} QGroupBox::title{subcontrol-origin:margin;left:10px;padding:0 5px;}")
        hist_lay = QVBoxLayout()
        enc_lay = QHBoxLayout()
        
        for txt, col_w in [("ID",40),("Inicio",150),("Horas",55),("DI med",60),("DI máx",60),("Sobrec.",55),("Desb.",55),("Similitud",75),("Sensor",120)]:
            lh = QLabel(txt)
            lh.setFixedWidth(col_w)
            lh.setFont(QFont("Consolas", 7, QFont.Bold))
            lh.setStyleSheet("color:#AACCFF;background:#0D1525;padding:2px;")
            lh.setAlignment(Qt.AlignCenter)
            enc_lay.addWidget(lh)
            
        enc_lay.addStretch()
        hist_lay.addLayout(enc_lay)

        scroll2 = QScrollArea()
        scroll2.setWidgetResizable(True)
        scroll2.setFixedHeight(160)
        scroll2.setStyleSheet(f"background:{C_AX_BG};border:1px solid {C_BORDER};border-radius:4px;")
        
        self._hist_container = QWidget()
        self._hist_container.setStyleSheet(f"background:{C_AX_BG};")
        self._hist_vlay = QVBoxLayout()
        self._hist_vlay.setSpacing(1)
        self._hist_vlay.setContentsMargins(0,0,0,0)
        self._hist_container.setLayout(self._hist_vlay)
        scroll2.setWidget(self._hist_container)
        hist_lay.addWidget(scroll2)

        btn_row = QHBoxLayout()
        self.btn_exportar = QPushButton("💾  Exportar a Excel")
        self.btn_exportar.setStyleSheet("QPushButton{background:#002244;color:#00BFFF;border:1px solid #0055AA;border-radius:5px;font-weight:bold;padding:5px 14px;} QPushButton:hover{background:#003366;}")
        self.btn_exportar.clicked.connect(self._exportar_excel)
        
        self.btn_refrescar = QPushButton("🔄  Refrescar tabla")
        self.btn_refrescar.setStyleSheet("QPushButton{background:#0F1830;color:#7AAFFF;border:1px solid #2A5090;border-radius:5px;font-weight:bold;padding:5px 14px;} QPushButton:hover{background:#142035;}")
        self.btn_refrescar.clicked.connect(self._refrescar_panel_historico)
        
        self.lbl_export_status = QLabel("")
        self.lbl_export_status.setFont(QFont("Consolas", 8))
        self.lbl_export_status.setStyleSheet(f"color:{C_TEXT_DIM};background:transparent;")
        
        btn_row.addWidget(self.btn_exportar)
        btn_row.addWidget(self.btn_refrescar)
        btn_row.addWidget(self.lbl_export_status)
        btn_row.addStretch()
        hist_lay.addLayout(btn_row)

        grp_hist.setLayout(hist_lay)
        main.addWidget(grp_hist)
        w.setLayout(main)
        scroll.setWidget(w)
        
        self._refrescar_panel_historico()
        return scroll

    def _autocompletar_motor(self, nombre_perfil):
        datos = self.db_motores.get(nombre_perfil)
        if datos and nombre_perfil != "Personalizado...":
            llaves = ['rpm_nom', 'i_vacio', 'i_nom', 'i_bloq', 'delgas', 'reductora', 'vib_nom', 'vida_nom']
            for i, key in enumerate(llaves):
                if key == 'delgas':
                    self.input_delgas.setText(str(datos[i]))
                elif key == 'reductora':
                    self.input_reductora.setText(str(datos[i]))
                else:
                    self._inputs_placa[key].setText(str(datos[i]))

    def _confirmar_placa(self):
        try:
            p = {}
            for key, inp in self._inputs_placa.items():
                val = float(inp.text().replace(',', '.'))
                if val <= 0: raise ValueError(f"{key} debe ser > 0")
                p[key] = val
            if p['i_vacio'] >= p['i_nom']: raise ValueError("Corriente vacío debe ser < corriente nominal")
            if p['i_nom'] >= p['i_bloq']: raise ValueError("Corriente nominal debe ser < corriente de bloqueo")
            
            tag_motor = self.input_tag.text().strip().upper()
            if not tag_motor: raise ValueError("El Tag/Serial es obligatorio")
            self._activo_id = tag_motor
            
            self._placa = p
            self._placa_cargada = True
            
            raw_string = f"{p['rpm_nom']}{p['i_nom']}{p['i_bloq']}{time.time()}"
            self._motor_id = hashlib.md5(raw_string.encode()).hexdigest()[:8].upper()
            
            self._calibrando_rul = True
            self._buffer_calibracion.clear()
            self._hist_di.clear()
            self._hist_di_t.clear()
            
            if hasattr(self, '_di_acc'):
                self._di_acc.clear()
            
            try:
                perfil_db = self._db.obtener_perfil_motor(self._activo_id)
                if perfil_db and 't0_vida' in perfil_db:
                    self._t0_vida_motor = perfil_db['t0_vida']
                else:
                    self._t0_vida_motor = time.time()
                    self.db_queue.put((self._db.guardar_perfil_motor, (self._activo_id, self._t0_vida_motor), {}))
            except AttributeError:
                self._t0_vida_motor = time.time()
            
            self._rul_activo = True
            
            self.lbl_rul_estado.setText(f"⏳ CALIBRANDO ACTIVO: {self._activo_id} Req. {self._MUESTRAS_CALIBRACION} iteraciones...")
            self.lbl_rul_estado.setStyleSheet("background:#1A1000;color:#FFD700;border:1px solid #886600;border-radius:4px;padding:4px;")
            self.btn_confirmar_placa.setText("✔  Reiniciar calibración de línea base")
        except ValueError as e:
            self.lbl_rul_estado.setText(f"⚠  Error: {e}")
            self.lbl_rul_estado.setStyleSheet("background:#2A0000;color:#FF4444;border:1px solid red;border-radius:4px;padding:4px;")

    def _motor_en_regimen_estable(self, c, v, r, imp):
        if not self._placa_cargada: return False
        p = self._placa
        return (r > p['rpm_nom'] * 0.85 and 
                c < p['i_nom'] * 1.5 and 
                v < p['vib_nom'] * 2.0 and 
                imp < p['vib_nom'] * 0.25)

    def _calcular_rul(self, c_proc, v_proc, rpm_val):
        if not self._placa_cargada: return
        
        t_actual = time.time()
        if self._t0_vida_motor is None:
            self._t0_vida_motor = t_actual
            
        t_horas = (t_actual - self._t0_vida_motor) / 3600.0 
        
        x_actual = np.array([c_proc, v_proc, rpm_val, self.impacto_actual])
        
        if self._calibrando_rul:
            if self._motor_en_regimen_estable(c_proc, v_proc, rpm_val, self.impacto_actual):
                self._buffer_calibracion.append(x_actual)
                progreso = len(self._buffer_calibracion)
                
                if progreso >= self._MUESTRAS_CALIBRACION:
                    matriz_calibracion = np.array(self._buffer_calibracion)
                    self._ref_sano = np.mean(matriz_calibracion, axis=0)
                    
                    self._cov_sano = np.cov(matriz_calibracion, rowvar=False)
                    epsilon = np.diag(self._cov_sano) * 0.01 + 1e-8
                    self._cov_sano += np.diag(epsilon)
                    self._inv_cov_sano = np.linalg.inv(self._cov_sano)
                    
                    self._calibrando_rul = False
                    self._di_umbral = 6.0
                    
                    self.lbl_rul_estado.setText("✅ LÍNEA BASE ESTABLECIDA - Monitoreo RUL Activo")
                    self.lbl_rul_estado.setStyleSheet("background:#001A00;color:#00FF66;border:1px solid #00AA33;border-radius:4px;padding:4px;")
            return

        if self._ref_sano is None or self._inv_cov_sano is None: return

        delta = x_actual - self._ref_sano
        di_raw = float(np.sqrt(np.dot(np.dot(delta.T, self._inv_cov_sano), delta)))
        
        if not hasattr(self, '_di_acc'):
            self._di_acc = deque(maxlen=7) 
            
        self._di_acc.append(di_raw)
        di = float(np.median(self._di_acc))
        di = min(di, self._di_umbral * 1.5)
        
        self._hist_di.append(di)
        self._hist_di_t.append(t_horas)
        
        if len(self._hist_di) > self._MAX_HIST_RUL: 
            self._hist_di.pop(0)
            self._hist_di_t.pop(0)
            
        n = len(self._hist_di)
        ventana_ma = min(20, n)
        di_ma = np.convolve(self._hist_di, np.ones(ventana_ma)/ventana_ma, mode='valid')
        
        if n >= 30:
            t_arr = np.array(self._hist_di_t[-len(di_ma):], dtype=float)
            di_arr = np.array(di_ma, dtype=float)
            
            log_di = np.log(di_arr + 1e-6)
            b, ln_a = np.polyfit(t_arr, log_di, 1)
            a = np.exp(ln_a)
            
            residuos = log_di - (b * t_arr + ln_a)
            std_res = float(np.std(residuos)) if len(residuos) > 2 else 0.1
            
            if b > 1e-6: 
                try:
                    t_falla_horas = np.log(self._di_umbral / a) / b
                    self._rul_horas = max(0.0, t_falla_horas - t_horas)
                    self._rul_std_horas = (std_res / b)
                    
                    horas_proj = min(self._rul_horas * 1.2 + 1.0, 5000.0) 
                    t_fut = np.linspace(t_horas, t_horas + horas_proj, 50)
                    di_fut = a * np.exp(b * t_fut)
                    
                    di_fut_u = np.exp(ln_a + std_res) * np.exp(b * t_fut)
                    di_fut_l = np.exp(ln_a - std_res) * np.exp(b * t_fut)
                    
                    t_hist_plot = np.array(self._hist_di_t, dtype=float)
                    t0 = t_hist_plot[0]
                    
                    self.line_rul_hist.set_data(t_hist_plot - t0, self._hist_di)
                    self.line_rul_pred.set_data(t_fut - t0, di_fut)
                    
                    if self.fill_rul_ci is not None: 
                        self.fill_rul_ci.remove()
                    self.fill_rul_ci = self.ax_rul_proj.fill_between(t_fut - t0, di_fut_l, di_fut_u, color='#FF4488', alpha=0.15, label='±1σ confianza')
                    
                    self.ax_rul_proj.set_xlim(0, max((t_fut[-1] - t0)*1.05, 1.0))
                    self.cv_rul.draw_idle()
                except Exception:
                    pass
            else:
                self._rul_horas = max(0.0, self._placa['vida_nom'] - t_horas)
                self._rul_std_horas = self._placa['vida_nom'] * 0.1
                
                t_hist_plot = np.array(self._hist_di_t, dtype=float)
                t0 = t_hist_plot[0]
                self.line_rul_hist.set_data(t_hist_plot - t0, self._hist_di)
                self.line_rul_pred.set_data([], [])
                if self.fill_rul_ci is not None: 
                    self.fill_rul_ci.remove()
                    self.fill_rul_ci = None
                self.ax_rul_proj.set_xlim(0, max((t_hist_plot[-1] - t0)*1.05, 1.0))
                self.cv_rul.draw_idle()
        else:
            self._rul_horas = self._placa['vida_nom']
            self._rul_std_horas = self._placa['vida_nom'] * 0.1
            
        t_arr_full = np.array(self._hist_di_t, dtype=float) - self._hist_di_t[0] if self._hist_di_t else np.array([])
        di_arr_full = np.array(self._hist_di, dtype=float)
        if len(t_arr_full) > 0:
            self.line_di.set_data(t_arr_full, di_arr_full)
            
            if len(di_ma) > 0:
                offset_ma = len(self._hist_di) - len(di_ma)
                t_ma = t_arr_full[offset_ma:]
                self.line_di_ma.set_data(t_ma, di_ma)
                
            self.ax_di.set_xlim(0, max(float(t_arr_full[-1]) * 1.1, 1.0))
            self.ax_di.set_ylim(0, max(self._di_umbral * 1.5, float(np.max(di_arr_full)) * 1.2))
            self.cv_rul.draw_idle()
        
        self.lbl_rul_di.setText(f"DI actual: {di:.3f}σ  |  Media móvil: {float(di_ma[-1]):.3f}σ  |  Umbral paramétrico: {self._di_umbral:.1f}σ" if len(di_ma) > 0 else f"DI actual: {di:.3f}σ  |  Umbral: {self._di_umbral:.1f}σ")
        
        if self._rul_horas is not None:
            h = self._rul_horas
            ci = self._rul_std_horas or 0.0
            dias = int(h // 24)
            horas = int(h % 24)
            vida_nom = self._placa['vida_nom']
            
            if n >= 30 and b <= 1e-6:
                color_rul = '#00FFCC'; estado = "✅  TENDENCIA ESTABLE — Sin degradación detectable"
                bg_estado = '#001A00'; border_estado = '#00AA33'
            elif h > vida_nom * 0.5: 
                color_rul = '#00FFCC'; estado = "✅  OPERACIÓN ESTABLE (Estimación Heurística)"
                bg_estado = '#001A00'; border_estado = '#00AA33'
            elif h > vida_nom * 0.2: 
                color_rul = '#FFD700'; estado = "⚠  DEGRADACIÓN DETECTADA — Sugiere inspección"
                bg_estado = '#1A1000'; border_estado = '#886600'
            elif h > 0: 
                color_rul = '#FF4444'; estado = "⛔  FALLA INMINENTE — Mantenimiento urgente"
                bg_estado = '#2A0000'; border_estado = 'red'
            else: 
                color_rul = '#FF0000'; estado = "🔴  LÍMITE ESTADÍSTICO SUPERADO"
                bg_estado = '#330000'; border_estado = 'darkred'
                
            self.lbl_rul_valor.setText(f"{h:,.1f}")
            self.lbl_rul_valor.setStyleSheet(f"color:{color_rul};background:transparent;")
            self.lbl_rul_conf.setText(f"Interv. confianza: ±{ci:,.1f} h  |  Aprox: {dias}d {horas}h restantes")
            self.lbl_rul_estado.setText(estado)
            self.lbl_rul_estado.setStyleSheet(f"background:{bg_estado};color:{color_rul};border:1px solid {border_estado};border-radius:4px;padding:4px;")

    def _actualizar_panel_firma(self, res: dict, sim: float | None):
        try:
            sim_txt = f"{sim:.1f} %" if sim is not None else "Sin ref."
            sim_col = ('#00FFCC' if sim and sim > 85 else '#FFD700' if sim and sim > 65 else '#FF4444')
            self.lbl_firma_sim.setText(f"Similitud: {sim_txt}")
            self.lbl_firma_sim.setStyleSheet(f"color:{sim_col};background:transparent;font-size:20pt;font-weight:bold;")
            
            rpm_p = res.get('rpm_pico') or 0
            c_p = res.get('c_pico') or 0
            t_e = res.get('t_estab_s') or 0
            self.lbl_firma_detalle.setText(f"RPM pico: {rpm_p:.0f}  |  I pico: {c_p:.1f} mA  |  T estab.: {t_e:.1f} s")
            
            muestras = res.get('muestras', [])
            if muestras:
                ts = [m['t'] for m in muestras]
                cs = [m['c_proc'] for m in muestras]
                vs = [m['v_proc'] for m in muestras]
                self.line_firma_c.set_data(ts, cs)
                self.line_firma_v.set_data(ts, vs)
                self.ax_firma_c.set_xlim(min(ts), max(ts)+1)
                self.ax_firma_c.set_ylim(0, max(cs)*1.3 + 1)
                self.ax_firma_v.set_xlim(min(ts), max(ts)+1)
                self.ax_firma_v.set_ylim(0, max(vs)*1.3 + 1)
                
            if self._firma_ref_vec:
                chunk = len(self._firma_ref_vec) // 3
                c_ref = np.array(self._firma_ref_vec[:chunk]) * 2400.0
                v_ref = np.array(self._firma_ref_vec[chunk:2*chunk]) * 50.0
                t_ref = list(range(len(c_ref)))
                self.line_firma_c_ref.set_data(t_ref[:len(muestras)], c_ref[:len(muestras)])
                self.line_firma_v_ref.set_data(t_ref[:len(muestras)], v_ref[:len(muestras)])
            self.cv_firma.draw_idle()
        except Exception as e:
            logging.warning(f"Error en _actualizar_panel_firma: {e}")

    def _refrescar_panel_historico(self):
        try:
            sesiones = self._db.obtener_sesiones(30)
            while self._hist_vlay.count():
                item = self._hist_vlay.takeAt(0)
                if item.widget(): 
                    item.widget().deleteLater()
                    
            for s in sesiones:
                row_w = QWidget()
                row_w.setStyleSheet(f"background:#0D1525;border-bottom:1px solid {C_BORDER};")
                row_lay = QHBoxLayout()
                row_lay.setContentsMargins(0,1,0,1)
                row_lay.setSpacing(0)
                
                sim = s['similitud_pct']
                dim = s['di_medio'] or 0
                dimx = s['di_max'] or 0
                sobr = s['sobrecargas'] or 0
                desb = s['desbalances'] or 0
                
                sim_col = ('#00FFCC' if sim and sim > 80 else '#FFD700' if sim and sim > 60 else '#FF4444' if sim else '#556688')
                di_col  = ('#00FFCC' if dim < 1.0 else '#FFD700' if dim < 2.0 else '#FF4444')
                
                vals_cols = [
                    (str(s['id']), 40, '#AACCFF'), 
                    (s['inicio'] or '—', 150, C_TEXT), 
                    (f"{s['horas_op'] or 0:.1f}h", 55, C_TEXT), 
                    (f"{dim:.2f}σ" if dim else '—', 60, di_col), 
                    (f"{dimx:.2f}σ" if dimx else '—',60,di_col), 
                    (str(sobr), 55, '#FF4444' if sobr else C_TEXT), 
                    (str(desb), 55, '#FFA500' if desb else C_TEXT), 
                    (f"{sim:.0f}%" if sim else '—', 75, sim_col), 
                    (s['sensor_tipo'] or '—', 120, C_TEXT_DIM)
                ]
                
                for txt, w, fg in vals_cols:
                    lbl = QLabel(txt)
                    lbl.setFixedWidth(w)
                    lbl.setFont(QFont("Consolas", 8))
                    lbl.setStyleSheet(f"color:{fg};background:transparent;padding:1px 3px;")
                    lbl.setAlignment(Qt.AlignCenter)
                    row_lay.addWidget(lbl)
                    
                row_lay.addStretch()
                row_w.setLayout(row_lay)
                self._hist_vlay.addWidget(row_w)
                
            tend = self._db.tendencia_di(30)
            if len(tend) >= 2:
                xs = list(range(len(tend)))
                dis = [t['di_medio'] or 0 for t in tend]
                self.line_tend_di.set_data(xs, dis)
                
                xs_arr = np.array(xs, dtype=float)
                di_arr = np.array(dis, dtype=float)
                A = np.vstack([xs_arr, np.ones(len(xs_arr))]).T
                (a, b), _, _, _ = np.linalg.lstsq(A, di_arr, rcond=None)
                
                self.line_tend_reg.set_data(xs_arr, a*xs_arr + b)
                self.ax_tend.set_xlim(0, max(len(tend)-1, 5))
                self.ax_tend.set_ylim(0, max(max(dis)*1.3, 1.0))
                self.cv_tend.draw_idle()
                
                res_tend = self._tendencia.analizar()
                nivel = res_tend['nivel']
                col_map = {'ok':'#00FFCC','alerta':'#FFD700','critico':'#FF4444','sin_datos':'#556688'}
                bg_map = {'ok':'#001A00','alerta':'#1A1000','critico':'#2A0000','sin_datos':C_PANEL}
                color = col_map.get(nivel, '#556688')
                bg = bg_map.get(nivel, C_PANEL)
                
                self.lbl_tendencia.setText(res_tend['mensaje'])
                self.lbl_tendencia.setStyleSheet(f"background:{bg};color:{color};border:1px solid {color};border-radius:4px;padding:6px;")
                
                pend = res_tend.get('pendiente_di')
                r2 = res_tend.get('r2')
                sim_a= res_tend.get('sim_actual')
                self.lbl_tend_detalle.setText(f"Pendiente DI: {pend:+.4f}σ/sesión  |  R²: {r2:.3f}  |  Similitud actual: {sim_a:.1f}%  |  Sesiones: {len(tend)}" if pend is not None and r2 is not None and sim_a is not None else f"Sesiones analizadas: {len(tend)}")
        except Exception as e:
            logging.warning(f"Error en _refrescar_panel_historico: {e}")

    def _cargar_tendencia_bd(self):
        try:
            tend = self._db.tendencia_di(30)
            for t in tend: 
                self._tendencia.agregar_sesion(t['id'], t['di_medio'], self._db.obtener_sesiones(1)[0].get('similitud_pct') if self._db.obtener_sesiones(1) else None)
        except Exception as e:
            logging.warning(f"Error en _cargar_tendencia_bd: {e}")

    def _exportar_excel(self):
        try:
            ruta = self._db.exportar_excel()
            self.lbl_export_status.setText(f"✅  Guardado: {ruta}")
            self.lbl_export_status.setStyleSheet("color:#00FFCC;background:transparent;")
        except ImportError:
            self.lbl_export_status.setText("⚠  Instala openpyxl:  pip install openpyxl")
            self.lbl_export_status.setStyleSheet("color:#FFD700;background:transparent;")
        except Exception as e:
            self.lbl_export_status.setText(f"⚠  Error: {e}")
            self.lbl_export_status.setStyleSheet("color:#FF4444;background:transparent;")

    def _tick_horas(self):
        self._horas_acum += 1.0

    def _registrar_evento(self, tipo, d):
        evento = {'tipo': tipo, 'hora_op': self._horas_acum, 'c_proc': d.get('c_proc', 0.0), 'c_crud': d.get('c_crud', 0.0), 'v_proc': d.get('v_proc', 0.0), 'v_crud': d.get('v_crud', 0.0), 'rpm': d.get('rpm', 0), 'bin_dom': d.get('bin_dom', 0), 'diagnostico': self._diagnosticar_faway(tipo, d)}
        self._historial_eventos.append(evento)
        if len(self._historial_eventos) > 50: 
            self._historial_eventos.pop(0)
        
        self.db_queue.put((self._db.guardar_evento, (tipo, self._horas_acum, dict(d), evento['diagnostico']), {}))

    def _diagnosticar_faway(self, tipo, d):
        c = d.get('c_proc', 0.0)
        cv = d.get('c_crud', 0.0)
        v = d.get('v_proc', 0.0)
        r = d.get('rpm', 0)
        b = d.get('bin_dom', 0)
        p = self._placa if self._placa_cargada else {}
        
        i_nom = p.get('i_nom', 300.0)
        i_bloq = p.get('i_bloq', 900.0)
        r_nom = p.get('rpm_nom', 1750.0)
        v_nom = p.get('vib_nom', 5.0)
        
        ratio_i = c / max(i_nom, 1.0)
        ratio_ip = cv / max(i_nom, 1.0)
        ratio_v = v / max(v_nom, 1.0)
        ratio_rpm = r / max(r_nom, 1.0)
        
        causa, norma, accion = "Desconocido", "N/A", "N/A"
        
        if tipo == 'SOBRECARGA':
            if ratio_ip > 2.5 and ratio_v < 1.5: 
                causa = "Sobrecarga mecánica súbita atascamiento"; norma = "NEMA MG1-2021 §12.48 / IEEE 1099 §6.3.2"; accion = "Verificar carga acoplada correas o rodamientos bloqueados"
            elif ratio_i > 1.5 and ratio_v > 2.5: 
                causa = "Desbalance de carga y vibración excesiva"; norma = "ISO 10816-3 Zona D / IEEE 1099 §6.4"; accion = "Balanceo dinámico urgente e inspección de rodamientos"
            elif ratio_i > 1.3 and ratio_rpm < 0.88: 
                causa = "Caída de tensión de alimentación o falla de fase"; norma = "NEMA MG1-2021 §12.45 / IEEE Std 1100"; accion = "Verificar tensión de red fusibles y borneras de conexión"
            elif ratio_i > 1.2 and ratio_v < 1.2 and ratio_rpm > 0.95: 
                causa = "Sobrecarga térmica por ciclo de trabajo excesivo"; norma = "IEC 60034-1 Clase de servicio S1→S3"; accion = "Reducir ciclo de trabajo o mejorar ventilación"
            elif ratio_ip > 1.8 and ratio_rpm < 0.7: 
                causa = "Arranque con carga plena corriente de bloqueo"; norma = "NEMA MG1-2021 §12.38 — arranque directo"; accion = "Implementar arrancador suave o variador de frecuencia"
            else: 
                causa = "Sobrecorriente transitoria — posible pico de red"; norma = "IEEE 1100-2005 §8.3 power quality"; accion = "Instalar filtro EMI o supresor de transitorios en alimentación"
        else:
            if ratio_v > 3.0 and b > 0: 
                causa = f"Desbalance mecánico severo frecuencia bin {b}"; norma = "ISO 10816-3 Zona D — vibración inaceptable"; accion = "Balanceo dinámico inmediato inspección de rotor y acoplamiento"
            elif ratio_v > 2.0 and ratio_i > 1.1: 
                causa = "Excentricidad de rotor o desgaste de rodamiento"; norma = "ISO 10816-3 Zona C / IEEE 1099 §7.2.1"; accion = "Análisis de firma de corriente MCSA y cambio de rodamientos"
            elif ratio_v > 1.5 and ratio_rpm > 1.05: 
                causa = "Resonancia mecánica — frecuencia natural excitada"; norma = "ISO 10816-1 §5.4 — análisis modal"; accion = "Verificar rigidez de montaje y frecuencias de resonancia"
            elif ratio_v > 1.0 and ratio_i < 1.05 and ratio_rpm > 0.98: 
                causa = "Desalineación de acoplamiento angular o paralela"; norma = "ISO 10816-3 Zona B→C / NEMA MG1 §7.06"; accion = "Realineación laser del acoplamiento motor-carga"
            else: 
                causa = "Vibración elevada sin patrón eléctrico claro"; norma = "ISO 10816-3 Zona B — monitoreo recomendado"; accion = "Incrementar frecuencia de monitoreo inspección visual"
        return {'causa': causa, 'norma': norma, 'accion': accion}

    def _entrenar_ml_async(self, datos):
        """Método de entrenamiento del modelo ML desplazado a hilo en background."""
        try:
            nuevo_modelo = IsolationForest(n_estimators=50, contamination='auto', random_state=42)
            nuevo_modelo.fit(datos)
            self._ml_modelo = nuevo_modelo  
            self._ml_entrenado = True
        except Exception as e:
            logging.warning(f"Error asíncrono en ajuste ML: {e}")
        finally:
            self._ml_entrenando = False

    def _iniciar(self):
        try:
            rpm = int(self.input_rpm.text())
            puerto_serie = self.input_puerto.text().strip()
            if not (10 <= rpm <= 10000): raise ValueError
            
            modo = self.combo_modo.currentText()
            
            if "Hardware" in modo:
                if not puerto_serie: raise ValueError("Puerto inválido")
                self.hilo_trabajo = TrabajoModbus(puerto_serie, rpm)
                self.hilo_trabajo.datos_listos.connect(self._datos)
                self.hilo_trabajo.start()
                self.hilo_trabajo.cambiar_actuador(1 if self.btn_actuador.isChecked() else 0)
                QTimer.singleShot(800, lambda: self.hilo_trabajo.escribir_registro(148, int(self.input_delgas.text().strip() or 3)))
                QTimer.singleShot(1200, lambda: self.hilo_trabajo.escribir_registro(149, int(self.input_reductora.text().strip() or 0)))
            else:
                self.hilo_trabajo = None 
                self.timer_pruebas.start(150) 

            self.btn_iniciar.setEnabled(False)
            self.btn_detener.setEnabled(True)
            self.input_rpm.setEnabled(False)
            self.input_puerto.setEnabled(False)
            self._set_estado("⟳  Iniciando monitoreo...", '#00FFCC')
            
            self._horas_acum = 0.0
            self._historial_eventos.clear()
            self._sobrecarga_count = 0
            self._desbalance_count = 0
            self._di_sesion.clear()
            self._muestra_sesion  = 0
            self._similitud_sesion = None
            
            self._ml_entrenado = False
            self._ml_entrenando = False
            self._ml_buffer.clear()
            self._muestras_desde_entrenamiento = 0
            
            self.disponibilidad_tecnica = 100.0
            self._timer_horas.start(3600000)

            placa_snap  = self._placa if self._placa_cargada else None
            self.db_queue.put((self._db.abrir_sesion, (rpm, "ACS758_FIXED", placa_snap), {}))
            
            self._firma = AnalizadorFirma(rpm, self._placa if self._placa_cargada else None)

            ref = self._db.obtener_firma_referencia()
            self._firma_ref_vec = ref['vector'] if ref else None
        except ValueError as e:
            self._set_estado(f"⚠  {e}", '#FF4444', bg='#2A0000', border='#FF2222')

    def _detener(self):
        if self.hilo_trabajo:
            self.hilo_trabajo.detener()
            self.hilo_trabajo.wait()
            self.hilo_trabajo = None
            
        if hasattr(self, 'timer_pruebas'):
            self.timer_pruebas.stop()
            
        self.btn_iniciar.setEnabled(True)
        self.btn_detener.setEnabled(False)
        self.input_rpm.setEnabled(True)
        self.input_puerto.setEnabled(True)
        self._timer_horas.stop()
        
        # RESETEO DE KALMAN AL DETENER
        self.kalman_corr.x = 0.0
        self.kalman_vib.x = 0.0
        self.kalman_rpm.x = 0.0
        
        di_arr = self._di_sesion
        di_medio = float(np.mean(di_arr)) if di_arr else None
        di_max   = float(np.max(di_arr))  if di_arr else None
        
        self.db_queue.put((
            self._db.cerrar_sesion, 
            (), 
            {
                'horas_op': self._horas_acum, 
                'sobrecargas': self._sobrecarga_count, 
                'desbalances': self._desbalance_count, 
                'di_medio': di_medio, 
                'di_max': di_max, 
                'similitud_pct': self._similitud_sesion
            }
        ))
        
        if di_medio is not None:
            ultimas = self._db.obtener_sesiones(1)
            sid = ultimas[0]['id'] if ultimas else 0
            self._tendencia.agregar_sesion(sid, di_medio, self._similitud_sesion)
            
        self._set_estado("■  MONITOREO DETENIDO", '#AAAAAA', border='#444444')
        try: 
            self._refrescar_panel_historico()
        except Exception as e: 
            logging.warning(f"Error al refrescar tabla histórica: {e}")

    def _set_estado(self, txt, color='#00FFCC', bg=C_BG, border=None):
        b = border or color
        self.label_tel.setText(txt)
        self.label_tel.setStyleSheet(f"background:{C_BG};color:{color};border:1px solid {b};border-radius:4px;padding:4px;")

    @pyqtSlot(dict)
    def _datos(self, d):
        try:
            if d.get('error', False):
                # RESETEO DE KALMAN AL HABER ERROR (Corte Modbus)
                self.kalman_corr.x = 0.0
                self.kalman_vib.x = 0.0
                self.kalman_rpm.x = 0.0
                self._set_estado("⚠ RUIDO EMI - ESPERANDO PAQUETE SANO...", '#FFD700', bg='#1A1000', border='#886600')
                return 

            e_tiempo = 0.0
            e_freq = 0.0
            error_pars = 0.0

            c_proc = float(d.get('c_proc', 0.0))
            v_proc = float(d.get('v_proc', 0.0))
            raw_rpm_in = float(d.get('rpm', 0.0))
            c_crud = float(d.get('c_crud', 0.0))
            v_crud = float(d.get('v_crud', 0.0))
            ref_sana = d.get('ref_sana', 0)
            
            try:
                reductora_val = float(self.input_reductora.text().replace(',', '.'))
            except:
                reductora_val = 0.0
            
            rpm_eje_raw = raw_rpm_in / max(reductora_val, 1.0)

            if not hasattr(self, '_rpm_buffer'):
                self._rpm_buffer = deque(maxlen=6)

            if rpm_eje_raw > 1.0: 
                self._rpm_buffer.append(rpm_eje_raw)

            if c_proc > 50.0:  
                if len(self._rpm_buffer) > 0:
                    rpm_eje = float(np.median(self._rpm_buffer))
                else:
                    try: rpm_eje = float(self.input_rpm.text().replace(',', '.'))
                    except: rpm_eje = 2200.0
            else:
                rpm_eje = 0.0
                self._rpm_buffer.clear()

            fft_cruda_dspic = np.array(d.get('fft_vib', [0]*32), dtype=float)
            
            if c_proc < 10.0 or rpm_eje < 10.0:
                fft_para_dibujar = np.zeros(32)
                bin_dom = 0
            else:
                fft_para_dibujar = fft_cruda_dspic
                bin_dom = int(d.get('bin_dom', 0))
                if bin_dom == 0:
                    bin_dom = int(np.argmax(fft_para_dibujar[1:]) + 1)
                    
            self.fft_bins = fft_para_dibujar

            self.hist_c_p1[:-1] = self.hist_c_p1[1:]; self.hist_c_p1[-1] = c_proc
            self.hist_v_p1[:-1] = self.hist_v_p1[1:]; self.hist_v_p1[-1] = v_proc
            self.hist_c_p2[:-1] = self.hist_c_p2[1:]; self.hist_c_p2[-1] = c_crud
            self.hist_v_p2[:-1] = self.hist_v_p2[1:]; self.hist_v_p2[-1] = v_crud

            self.buffer_vib_largo.append(v_crud)
            n_actual = len(self.buffer_vib_largo)

            self._iir_x.pop(0)
            self._iir_x.append(v_crud)
            
            b0, b1, b2 = 0.020083, 0.040167, 0.020083
            a1, a2 = -1.561018, 0.641351
            
            y_new = (b0 * self._iir_x[2] + b1 * self._iir_x[1] + b2 * self._iir_x[0] 
                     - a1 * self._iir_y[2] - a2 * self._iir_y[1])
                     
            self._iir_y.pop(0)
            self._iir_y.append(y_new)
            v_iir = max(0.0, y_new) 
            self.hist_iir_p1[:-1] = self.hist_iir_p1[1:]; self.hist_iir_p1[-1] = v_iir

            # =================================================================
            # ACTUALIZACIÓN DEL SIMULADOR DE ARQUITECTURA DSP
            # =================================================================
            if hasattr(self, 'ptr_circular'):
                self.ptr_circular = (self.ptr_circular + 1) % self.n_puntos_anillo
                
                opacidades = np.zeros(self.n_puntos_anillo)
                for i in range(self.n_puntos_anillo):
                    distancia = (self.ptr_circular - i) % self.n_puntos_anillo
                    if distancia == 0:
                        opacidades[i] = 255 
                    elif distancia < 10:
                        opacidades[i] = 200 - (distancia * 15) 
                    else:
                        opacidades[i] = 0

                brushes = [pg.mkBrush(0, 255, 204, int(op)) for op in opacidades]
                self.scatter_circ_active.setData(x=self.x_circ, y=self.y_circ, brush=brushes)
                self.text_ptr.setText(f"Ptr: {self.ptr_circular}\nAcc")

                adc_raw_sim = int(v_crud * 40.95) 
                self.lbl_dsp_adc.setText(f"ADC Raw Int: {adc_raw_sim}")
                
                q15_data = (adc_raw_sim / 2048.0) - 1.0 
                self.lbl_dsp_q15.setText(f"Bus X Data Q15: {q15_data:+.4f}")
                
                coef_y = b0 if self.ptr_circular % 2 == 0 else -a1
                self.lbl_dsp_coef.setText(f"Bus Y Coef Q15: {coef_y:+.4f}")

                acc_sim = int(v_iir * 153023)
                self.lbl_dsp_acc.setText(f"Acc A 40-bit: 0x{acc_sim:010X}")

            # =================================================================
            # ACTUALIZACIÓN DE LA PESTAÑA MODULACIÓN AM INDUSTRIAL
            # =================================================================
            if hasattr(self, 'fase_modulacion'):
                self.fase_modulacion += 0.25 
                t_visual = np.linspace(0, 4 * np.pi, 300)
                
                if len(self.buffer_vib_largo) > 0:
                    arr_largo = np.array(self.buffer_vib_largo)
                    if len(arr_largo) >= 300:
                        v_cruda_plot = arr_largo[-300:]
                    else:
                        v_cruda_plot = np.pad(arr_largo, (300 - len(arr_largo), 0), 'constant')
                else:
                    v_cruda_plot = np.zeros(300)
                
                v_cruda_plot = v_cruda_plot - np.mean(v_cruda_plot)
                self.line_mod_raw.setData(v_cruda_plot)

                frecuencia_relativa = max(1, bin_dom * 0.4) 
                amplitud_real = self.fft_bins[bin_dom] if bin_dom < len(self.fft_bins) else 0
                onda_pura = amplitud_real * np.sin(frecuencia_relativa * t_visual + self.fase_modulacion)
                self.line_mod_pure.setData(onda_pura)

                portadora = np.cos(40 * t_visual)
                onda_am = (amplitud_real + onda_pura) * portadora
                self.line_mod_am.setData(onda_am)

            if not d.get('_predicho', False):
                self.kalman_corr.update(c_proc)
                self.kalman_vib.update(v_proc)
                self.kalman_rpm.update(rpm_eje) 
            
            self._carga_pct, estado_carga = self._calcular_pct(c_proc, rpm_eje)

            alerta = d.get('alertas', 0)
            if alerta & 0b01:
                self._sobrecarga_count += 1; self._registrar_evento('SOBRECARGA', d)
                self._set_estado(f"⛔  SOBRECARGA  {c_crud:.1f} mA | CARGA: {self._carga_pct}%", '#FF2222', bg='#330000', border='red')
            elif alerta & 0b10:
                self._desbalance_count += 1; self._registrar_evento('DESBALANCE', d)
                self._set_estado(f"⚠  DESBALANCE  Bin: {bin_dom} | {self._rul_tiempo_real_str()}", '#FFA500', bg='#2A1800', border='orange')
            else:
                self._set_estado(f"✅  NORMAL  |  RPM: {rpm_eje:.0f}  |  CARGA: {self._carga_pct}%  |  {self._rul_tiempo_real_str()}", '#00FFCC')

            total_fallas = self._sobrecarga_count + self._desbalance_count
            if total_fallas == 0:
                self.disponibilidad_tecnica = 100.0
            else:
                mtbf = max(self._horas_acum, 0.1) / total_fallas
                self.disponibilidad_tecnica = (mtbf / (mtbf + self.MTTR_TEORICO)) * 100.0

            if self._placa_cargada and rpm_eje > 10:
                carga_n = self._carga_pct / 100.0
                eta_max = 0.85 
                if carga_n < 0.05: eta_calc = 0.0 
                else:
                    eta_calc = eta_max - 0.4 * (carga_n - 0.75)**2
                    eta_calc = max(0.1, min(eta_max, eta_calc)) 
                self._eta_pct = eta_calc * 100
            else:
                self._eta_pct = 0.0

            try: rpm_obj = float(self.input_rpm.text())
            except: rpm_obj = 0.0

            self.envolvente = np.abs(hilbert(self.hist_v_p1))
            self.line_envolvente.setData(self.envolvente)

            if n_actual >= 64: 
                arr_vib = np.array(self.buffer_vib_largo)
                arr_vib = arr_vib - np.mean(arr_vib)
                arr_vib = arr_vib * np.hanning(n_actual)
                espectro_largo = np.abs(np.fft.fft(arr_vib))
                cepstrum_largo = np.abs(np.fft.ifft(np.log(espectro_largo + 1e-6)))
                mitad = n_actual // 2
                idx_start = max(1, n_actual // 50) 
                idx_end = min(mitad, n_actual // 5)
                
                if idx_end > idx_start:
                    self.impacto_actual = float(np.max(cepstrum_largo[idx_start:idx_end]))
                
                self.line_cepstrum.setData(cepstrum_largo[:mitad])
                self.p_ceps.setXRange(0, mitad)

                # --- VALIDACIÓN DEL TEOREMA DE PARSEVAL ---
                arr_64 = np.array(list(self.buffer_vib_largo)[-64:])
                arr_64_sin_dc = arr_64 - np.mean(arr_64)
                arr_64_hanning = arr_64_sin_dc * np.hanning(64)
                
                e_tiempo = float(np.sum(np.square(arr_64_hanning)))
                fft_64 = np.abs(np.fft.fft(arr_64_hanning))
                e_freq = float(np.sum(np.square(fft_64)) / 64.0)
                error_pars = abs(e_tiempo - e_freq) / max(e_tiempo, 1e-6) * 100.0
                
                self.hist_etime.append(e_tiempo); self.hist_etime.pop(0)
                self.hist_efreq.append(e_freq); self.hist_efreq.pop(0)
                
                x_pars = list(range(self.MAX_DATOS))
                self.line_etime.set_data(x_pars, self.hist_etime)
                self.line_efreq.set_data(x_pars, self.hist_efreq)
                
                if hasattr(self, 'line_sust_etime'):
                    self.line_sust_etime.setData(x_pars, self.hist_etime)
                    self.line_sust_efreq.setData(x_pars, self.hist_efreq)
                
                max_val_pars = max(max(self.hist_etime), max(self.hist_efreq), 1.0)
                self.ax_pars.set_ylim(0, max_val_pars * 1.2)
                if hasattr(self, 'p_sust_pars'): self.p_sust_pars.setYRange(0, max_val_pars * 1.2)
                self.cv_pars.draw_idle()
                
                if hasattr(self, 'tabla_parseval'):
                    self.tabla_parseval.insertRow(0)
                    self.tabla_parseval.setItem(0, 0, QTableWidgetItem(time.strftime("%H:%M:%S")))
                    self.tabla_parseval.setItem(0, 1, QTableWidgetItem(f"{e_tiempo:.1f}"))
                    self.tabla_parseval.setItem(0, 2, QTableWidgetItem(f"{e_freq:.1f}"))
                    item_error = QTableWidgetItem(f"{error_pars:.4f} %")
                    item_error.setForeground(QColor('#FF3333') if error_pars > 1.0 else QColor('#39FF14'))
                    self.tabla_parseval.setItem(0, 3, item_error)
                    if self.tabla_parseval.rowCount() > 8: self.tabla_parseval.removeRow(8)

            if self._rul_activo: 
                self._calcular_rul(c_proc, v_proc, rpm_eje)

            # --- APRENDIZAJE AUTOMÁTICO ---
            ml_estado_txt = "MOTOR DETENIDO"
            pred = 1
            if not self._calibrando_rul and rpm_eje > 50:
                p_nom = self._placa if self._placa_cargada else {'rpm_nom': 1750.0, 'vib_nom': 5.0, 'i_nom': 300.0}
                
                v_norm = min(v_proc / max(p_nom['vib_nom'], 1.0), 10.0)
                c_norm = min(c_proc / max(p_nom['i_nom'], 1.0), 10.0)
                r_norm = min(rpm_eje / max(p_nom['rpm_nom'], 1.0), 3.0)
                impacto_norm = min(self.impacto_actual / max(p_nom['vib_nom'] * 0.25, 1.0), 10.0)
                
                vector_estado_ml = [v_norm, c_norm, r_norm, impacto_norm]

                if not d.get('_predicho', False):
                    es_muestra_sana = self._motor_en_regimen_estable(c_proc, v_proc, rpm_eje, self.impacto_actual)

                    if not self._ml_entrenado:
                        if es_muestra_sana:
                            self._ml_buffer.append(vector_estado_ml)
                            
                        faltan = self._ML_SAMPLES_TRAIN - len(self._ml_buffer)
                        if faltan > 0: 
                            ml_estado_txt = f"CALIBRANDO {faltan}"
                        elif not self._ml_entrenando:
                            self._ml_entrenando = True
                            ml_estado_txt = "ENTRENANDO..."
                            threading.Thread(target=self._entrenar_ml_async, args=(np.array(self._ml_buffer),), daemon=True).start()
                        else:
                            ml_estado_txt = "ENTRENANDO..."
                    else:
                        pred = self._ml_modelo.predict(np.array([vector_estado_ml]))[0]
                        ml_estado_txt = "ANOMALÍA DETECTADA" if pred == -1 else "ML: NORMAL"
                        
                        if es_muestra_sana:
                            self._ml_buffer.append(vector_estado_ml)
                            self._muestras_desde_entrenamiento += 1
                            
                            if self._muestras_desde_entrenamiento >= 200:
                                self._muestras_desde_entrenamiento = 0
                                if not self._ml_entrenando:
                                    self._ml_entrenando = True
                                    threading.Thread(target=self._entrenar_ml_async, args=(np.array(self._ml_buffer),), daemon=True).start()

            score_vib = min(30, v_proc * 0.6)
            score_iir = min(20, v_iir * 0.4)
            score_carga = max(0, self._carga_pct - 100) * 0.5
            score_ml = 15 if pred == -1 else 0
            self.health_score = max(0, 100 - score_vib - score_iir - score_carga - score_ml)

            diagnostico_txt = self.analizar_salud_motor(self._carga_pct, v_proc, self.impacto_actual)

            self.disp_vib.set_valor(v_proc)
            self.disp_corr.set_valor(c_proc)
            self.disp_rpm.set_valor(rpm_eje)
            self.disp_carga.set_valor(self._carga_pct)
            self.disp_carga.set_color_acento(estado_carga[1]) 

            self.dash_secundario.actualizar_dashboard(
                self._carga_pct, estado_carga[0], estado_carga[1], 
                self._rul_tiempo_real_str(), self._eta_pct, self.disponibilidad_tecnica, 
                self._activo_id if hasattr(self, '_activo_id') else self._motor_id
            )
            self.dash_secundario.actualizar_avanzado(
                self.health_score, self.confianza_sistema, self.drift_operacional, diagnostico_txt
            )
            self._actualizar_reporte_texto(
                self._carga_pct, v_proc, self.impacto_actual, 
                self.health_score, self.confianza_sistema, diagnostico_txt
            )

            self.line_v1.setData(self.hist_v_p1)
            self.line_v1_iir.setData(self.hist_iir_p1) 
            self.line_c1.setData(self.hist_c_p1)
            self.line_ref.setValue(ref_sana)

            max_fft_val = max(float(np.max(fft_para_dibujar)) * 1.2, 10.0) 
            
            for i, (bar, h) in enumerate(zip(self.bar_fft, fft_para_dibujar)):
                bar.set_height(h)
                bar.set_color('#FF4488' if i == bin_dom and h > 2.0 else '#AA88FF')
                
            self.ax_fft_bars.set_ylim(0, max_fft_val)
            
            if hasattr(self, 'bar_sust_fft'):
                self.bar_sust_fft.setOpts(height=fft_para_dibujar)
                self.p_sust_fft.setYRange(0, max_fft_val)
                
            self.matriz_fft = np.roll(self.matriz_fft, -1, axis=1)
            self.matriz_fft[:, -1] = fft_para_dibujar
            
            self.img_fft.setImage(self.matriz_fft.T[:, ::-1], autoLevels=False, levels=(0, max_fft_val))
            
            if hasattr(self, 'img_fft_d'):
                self.img_fft_d.set_array(self.matriz_fft)
                self.img_fft_d.set_clim(vmin=0, vmax=max_fft_val)
                self.cv_fft_d.draw_idle()

            self.line_v2.setData(self.hist_v_p2)
            self.line_c2.setData(self.hist_c_p2)

            x = list(range(self.MAX_DATOS))
            self.hist_raw_didact.append(v_crud); self.hist_raw_didact.pop(0)
            self.ema_sim += self.alpha_didact * (v_crud - self.ema_sim)
            self.hist_ema_didact.append(self.ema_sim); self.hist_ema_didact.pop(0)
            self.hist_iir_didact.append(v_iir); self.hist_iir_didact.pop(0)
            
            self.line_raw_ema.set_data(x, self.hist_raw_didact)
            self.line_ema_out.set_data(x, self.hist_ema_didact)
            self.line_iir_out.set_data(x, self.hist_iir_didact) 
            
            if hasattr(self, 'line_sust_raw'):
                self.line_sust_raw.setData(x, self.hist_raw_didact)
                self.line_sust_iir.setData(x, self.hist_iir_didact)

            if hasattr(self, 'line_orbit_raw'):
                v_raw_arr = np.array(self.hist_v_p2[-60:])
                v_iir_arr = np.array(self.hist_iir_p1[-60:])
                if len(v_raw_arr) > 1:
                    self.line_orbit_raw.setData(v_raw_arr[:-1], v_raw_arr[1:])
                    self.line_orbit_iir.setData(v_iir_arr[:-1], v_iir_arr[1:])
            
            for col in self.ax_ema.collections[:]: col.remove()
            self.ax_ema.fill_between(x, self.hist_ema_didact, color=C_EMA, alpha=0.12)
            self.cv_ema.draw_idle()

            self.hist_rms_didact.append(c_proc); self.hist_rms_didact.pop(0)
            self.line_rms_hist.set_data(x, self.hist_rms_didact); self.line_rms_peak.set_data(x, self.hist_c_p2)
            for bar, h in zip(self.bars_rms, [c_proc, c_crud]): bar.set_height(h)
            factor = (c_crud / c_proc) if c_proc > 0 else 0
            self.txt_rms_factor.set_text(f"Factor Pico/RMS: {factor:.2f}"); self.cv_rms.draw_idle()

            self.hist_rpm_med.append(float(rpm_eje)); self.hist_rpm_med.pop(0); self.hist_rpm_obj.append(rpm_obj); self.hist_rpm_obj.pop(0)
            self.line_rpm_med.set_data(x, self.hist_rpm_med); self.line_rpm_obj.set_data(x, self.hist_rpm_obj); self.cv_rpm.draw_idle()

            datos_vivo = {
                'ml_status': ml_estado_txt, 'v_proc': v_proc, 'v_iir': v_iir, 'c_proc': c_proc, 'v_crud': v_crud, 'c_crud': c_crud, 
                'rpm': rpm_eje, 'ref_sana': ref_sana, 'bin_dom': bin_dom, 'alertas': d.get('alertas', 0), 'rpm_obj': rpm_obj,
                'nema_load': self._carga_pct,
                'e_time': e_tiempo, 'e_freq': e_freq, 'err_pars': error_pars
            }

            self.panel_datos_t1.actualizar(datos_vivo)
            self.panel_datos_t2.actualizar(datos_vivo)
            self.panel_datos_ema.actualizar(datos_vivo)
            self.panel_datos_rms.actualizar(datos_vivo)
            self.panel_datos_fft.actualizar(datos_vivo)
            self.panel_datos_rpm.actualizar(datos_vivo)
            if hasattr(self, 'panel_datos_pars'): 
                self.panel_datos_pars.actualizar(datos_vivo)

            try:
                di_actual = self._hist_di[-1] if self._hist_di else None
                if di_actual is not None: self._di_sesion.append(di_actual)
                es_pred = bool(d.get('_predicho', False))
                self.db_queue.put((self._db.guardar_muestra, (self._muestra_sesion, dict(d)), {'di': di_actual, 'predicho': es_pred}))
                self._muestra_sesion += 1
            except Exception as e:
                logging.warning(f"Error al encolar guardar_muestra: {e}")

            try:
                if self._firma and not self._firma.completada:
                    completa = self._firma.agregar_muestra(self._muestra_sesion, d)
                    if completa:
                        sim = self._firma.calcular_similitud(self._firma_ref_vec)
                        self._similitud_sesion = sim
                        if len(self.hist_sim) > 5:
                            self.drift_operacional = abs(sim - np.mean(self.hist_sim))

                    sim_para_confianza = sim if sim else 100.0
                    self.confianza_sistema = np.clip(
                        (sim_para_confianza * 0.4) + ((100 - self.drift_operacional) * 0.3) + (self.health_score * 0.3),
                        0, 100
                    )

                    res = self._firma.resumen()
                    self.db_queue.put((
                        self._db.guardar_firma, 
                        (), 
                        {
                            'vector': res['vector_norm'], 
                            'rpm_pico': res['rpm_pico'], 
                            'c_pico': res['c_pico'], 
                            't_estab_s': res['t_estab_s'], 
                            'similitud_pct': sim
                        }
                    ))
                    try: self._actualizar_panel_firma(res, sim)
                    except Exception as e:
                        logging.warning(f"Error en actualizar panel de firma UI: {e}")
            except Exception as e:
                logging.warning(f"Error en procesamiento de firma: {e}")

        except Exception as e:
            logging.error(f"[_datos] Fallo crítico no capturado: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    v = PanelHMI()
    v.show()
    sys.exit(app.exec_())