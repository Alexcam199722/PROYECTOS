# =============================================================================
# ARCHIVO: hilo_modbus.py
# DESCRIPCIÓN: Controlador de adquisición de datos Modbus RTU multihilo
# =============================================================================

from PyQt5.QtCore import QThread, pyqtSignal
import minimalmodbus
import time

class TrabajoModbus(QThread):
    datos_listos = pyqtSignal(dict)

    def __init__(self, puerto, rpm_deseadas, id_sensor=3, estado_ventilador=0):
        super().__init__()
        self.puerto = puerto
        self.rpm_deseadas = rpm_deseadas
        self.id_sensor = id_sensor
        self.estado_ventilador = estado_ventilador
        self.corriendo = True
        self.cmd_actuador = None 
        # Cola de peticiones asíncronas
        self.peticion_escritura = None  

    def cambiar_actuador(self, estado):
        """Inyecta la orden de conmutación desde la interfaz HMI"""
        self.cmd_actuador = estado

    def escribir_registro(self, registro, valor):
        """Asigna en cola la orden de reconfiguración de registros (Delgas/Reductora)"""
        self.peticion_escritura = {'registro': registro, 'valor': valor}

    def run(self):
        try:
            cliente = minimalmodbus.Instrument(self.puerto, 1)
            cliente.serial.baudrate = 115200
            cliente.serial.timeout = 0.2
            cliente.clear_buffers_before_each_transaction = True

            # Configuración de RPM inicial en el registro objetivo
            cliente.write_register(0, self.rpm_deseadas, 0, functioncode=6)
            
        except Exception as e:
            self.datos_listos.emit({'error': True, 'mensaje': f"Falla de inicialización Modbus: {e}"})
            return

        while self.corriendo:
            paquete = {'error': False}
            try:
                # 1. Escritura asíncrona de estado de actuador
                if self.cmd_actuador is not None:
                    cliente.write_register(141, self.cmd_actuador, 0, functioncode=6)
                    self.cmd_actuador = None 
                    time.sleep(0.04) # Tiempo de latencia de bus RS-485

                # 2. Escritura asíncrona de configuración de motor
                if self.peticion_escritura is not None:
                    reg = self.peticion_escritura['registro']
                    val = self.peticion_escritura['valor']
                    cliente.write_register(reg, val, 0, functioncode=6)
                    self.peticion_escritura = None
                    time.sleep(0.04)

                # 3. Lectura de tramas de telemetría del dsPIC
                regs = cliente.read_registers(0, 4)
                time.sleep(0.04)
                
                regs_analitica = cliente.read_registers(132, 7)
                time.sleep(0.04)
                
                regs_crudos = cliente.read_registers(142, 2)
                time.sleep(0.04)
                
                # Extracción del buffer espectral (32 Bins FFT)
                regs_fft_vib = cliente.read_registers(68, 32)

                # =========================================================
                # ETAPA DE ACONDICIONAMIENTO: ANTI-WINDUP Y HARD CLIPPING
                FACTOR_CORRECCION = 2.0
                TOPE_ABSOLUTO = 50.0  # Límite máximo visual en pantalla
                # =========================================================

                # 🟢 VARIABLES INTACTAS (Corriente y RPM)
                paquete['rpm'] = regs_analitica[6]      
                paquete['c_proc'] = regs[1]             
                paquete['c_crud'] = regs_crudos[0]      
                
                # Extracción cruda con el factor base
                v_crud_raw = regs_crudos[1] / FACTOR_CORRECCION
                v_proc_raw = regs[2] / FACTOR_CORRECCION
                v_iir_raw = regs_analitica[2] / FACTOR_CORRECCION

                # 🛠️ CONDICIONAL ANTI-WINDUP (Desatasco inteligente)
                # Si el dsPIC se atascó arriba (> 45) pero la vibración cruda ya está normal (< 25)
                if v_proc_raw > 45.0 and v_crud_raw < 25.0:
                    # Calculamos qué tan desfasado está el dsPIC respecto a la realidad
                    ratio_descarga = v_crud_raw / max(v_proc_raw, 1.0)
                    
                    # Forzamos los filtros procesados a bajar a la realidad cruda instantáneamente
                    v_proc_final = v_crud_raw
                    v_iir_final  = v_crud_raw
                    # Aplastamos la FFT multiplicándola por el ratio para que no se quede pegada
                    fft_raw = [ (val / FACTOR_CORRECCION) * ratio_descarga for val in regs_fft_vib ]
                else:
                    # Comportamiento nominal sin atasco
                    v_proc_final = v_proc_raw
                    v_iir_final  = v_iir_raw
                    fft_raw = [ val / FACTOR_CORRECCION for val in regs_fft_vib ]

                # 🛠️ SATURACIÓN MATEMÁTICA (Hard Clipping)
                # Si el valor (real o falso) supera 50, se corta limpiamente a 50 para no deformar gráficas
                paquete['v_crud']  = min(v_crud_raw, TOPE_ABSOLUTO)
                paquete['v_proc']  = min(v_proc_final, TOPE_ABSOLUTO)
                paquete['v_iir']   = min(v_iir_final, TOPE_ABSOLUTO)
                paquete['fft_vib'] = [min(val, TOPE_ABSOLUTO) for val in fft_raw]

                # 🟢 VARIABLES DE CONTROL Y REFERENCIAS (Intactas)
                paquete['alertas'] = regs[3]            
                paquete['bin_dom'] = regs_analitica[0]  
                paquete['ref_sana'] = regs_analitica[4] 

            except Exception as e:
                paquete['error'] = True
                paquete['mensaje'] = "Ruido EMI: Integridad de paquete RS-485 comprometida"
            
            self.datos_listos.emit(paquete)
            time.sleep(0.4)

        cliente.serial.close()
        
    def detener(self):
        self.corriendo = False