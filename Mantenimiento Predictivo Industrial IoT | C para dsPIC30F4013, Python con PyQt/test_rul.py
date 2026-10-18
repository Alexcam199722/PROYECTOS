import sys
from PyQt5.QtWidgets import QApplication
app = QApplication(sys.argv)
import hilo_HMI as h

def debug_rul(self):
    import PyQt5.QtWidgets as qw
    print('rul: scroll...')
    scroll = qw.QScrollArea()
    print('rul: grp_placa...')
    grp_placa = qw.QGroupBox('Datos de Placa')
    print('rul: grp_placa OK')
    print('rul: figura...')
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FC
    fig = Figure(facecolor='black')
    cv = FC(fig)
    print('rul: figura OK')
    print('rul: grp_mant...')
    grp_mant = qw.QGroupBox('Mantenimiento')
    print('rul: grp_mant OK')
    import PyQt5.QtWidgets as qw2
    w_inner = qw2.QWidget()
    main = qw2.QVBoxLayout()
    w_inner.setLayout(main)
    scroll.setWidget(w_inner)
    return scroll

h.PanelHMI._crear_tab_rul = debug_rul
v = h.PanelHMI()
print('DONE')

