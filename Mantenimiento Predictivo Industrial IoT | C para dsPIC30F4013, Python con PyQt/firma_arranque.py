"""
firma_arranque.py — Captura y análisis de la firma de arranque del motor.

La firma de arranque son los primeros N_MUESTRAS ciclos desde que el motor
recibe tensión hasta que estabiliza RPM. En ese transitorio se revelan:

  • Estado de rodamientos  → pico de corriente y tiempo de subida
  • Estado del rotor       → oscilaciones de RPM en la rampa
  • Estado del aislamiento → relación i_pico / i_nominal
  • Desbalance incipiente  → vibración durante el transitorio de RPM

Índice de Similitud (IS):
  Compara el vector de firma actual con el vector de referencia (sesión sana)
  usando correlación cruzada normalizada (NCC). IS = 100% → motor idéntico.

Referencias:
  IEEE 1099-2022 §5.2  — Motor Current Signature Analysis (MCSA)
  IEC 60034-1 §8.8     — arranque directo y corriente de bloqueo
  NEMA MG1-2021 §12.38 — full-voltage starting
"""

import numpy as np
from collections import deque


# ─────────────────────────────────────────────────────────────────────────────
N_MUESTRAS_FIRMA = 12   # primeros 12 ciclos ≈ 6 s a 500 ms/ciclo
T_ESTAB_RPM_PCT  = 0.95  # considerar estabilizado cuando RPM ≥ 95% del objetivo
MIN_RPM_ARRANQUE = 50    # RPM mínimas para considerar que arrancó
# ─────────────────────────────────────────────────────────────────────────────


class AnalizadorFirma:
    """
    Captura la firma de arranque en tiempo real y calcula métricas.
    Se instancia cada vez que el usuario pulsa "Iniciar".
    """

    def __init__(self, rpm_objetivo: float, placa: dict | None = None):
        self.rpm_objetivo = max(rpm_objetivo, 1.0)
        self.placa        = placa or {}
        self.i_nom        = self.placa.get('i_nom',  300.0)
        self.i_bloq       = self.placa.get('i_bloq', 900.0)

        self._muestras: list[dict] = []
        self._capturando  = True    # False cuando la firma está completa
        self._t_arranque  = 0       # índice de muestra del primer dato
        self._estabilizado = False
        self._t_estab     = None    # índice de muestra cuando RPM se estabilizó

        # Resultados calculados
        self.rpm_pico    : float | None = None
        self.c_pico      : float | None = None
        self.t_estab_s   : float | None = None   # segundos hasta estabilización
        self.vector_norm : list  | None = None   # firma normalizada para comparar
        self.completada  : bool  = False

    # ─────────────────────────────────────────────────────────────────────────
    def agregar_muestra(self, t_idx: int, datos: dict) -> bool:
        """
        Agrega una muestra al buffer de firma.
        Retorna True cuando la firma queda completa.
        """
        if not self._capturando:
            return True

        rpm = datos.get('rpm', 0)
        c   = datos.get('c_proc', 0.0)
        v   = datos.get('v_proc', 0.0)

        # Esperar a que el motor empiece a girar
        if not self._muestras and rpm < MIN_RPM_ARRANQUE:
            return False

        self._muestras.append({
            't': t_idx,
            'c_proc': c,
            'v_proc': v,
            'rpm':    rpm,
        })

        # Detectar estabilización de RPM
        if not self._estabilizado and rpm >= self.rpm_objetivo * T_ESTAB_RPM_PCT:
            self._estabilizado = True
            self._t_estab      = len(self._muestras) - 1

        # Completar cuando tenemos N muestras desde el arranque
        if len(self._muestras) >= N_MUESTRAS_FIRMA:
            self._capturando = False
            self._calcular_metricas()
            return True

        return False

    # ─────────────────────────────────────────────────────────────────────────
    def _calcular_metricas(self):
        """Calcula las métricas de la firma y el vector normalizado."""
        if not self._muestras:
            return

        cs   = np.array([m['c_proc'] for m in self._muestras])
        vs   = np.array([m['v_proc'] for m in self._muestras])
        rpms = np.array([m['rpm']    for m in self._muestras])

        self.rpm_pico  = float(np.max(rpms))
        self.c_pico    = float(np.max(cs))

        if self._t_estab is not None:
            self.t_estab_s = self._t_estab * 0.5   # 500 ms por ciclo
        else:
            self.t_estab_s = N_MUESTRAS_FIRMA * 0.5  # nunca estabilizó

        # Vector de firma normalizado para NCC:
        # concatena [c_norm, v_norm, rpm_norm] como serie temporal plana
        c_norm   = cs   / max(self.i_bloq, 1.0)
        v_norm   = vs   / 50.0
        rpm_norm = rpms / max(self.rpm_objetivo, 1.0)
        self.vector_norm = np.concatenate([c_norm, v_norm, rpm_norm]).tolist()
        self.completada  = True

    # ─────────────────────────────────────────────────────────────────────────
    def calcular_similitud(self, vector_referencia: list | None) -> float:
        """
        Similitud entre esta firma y la de referencia.
        Usa correlación cruzada normalizada (NCC) → [0, 100] %.
        Si no hay referencia, devuelve None.
        """
        if vector_referencia is None or self.vector_norm is None:
            return None

        a = np.array(self.vector_norm,      dtype=float)
        b = np.array(vector_referencia,      dtype=float)

        # Igualar longitud
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]

        # NCC: correlación de Pearson entre los dos vectores
        if np.std(a) < 1e-9 or np.std(b) < 1e-9:
            return 100.0 if np.allclose(a, b, atol=1e-3) else 0.0

        ncc = float(np.corrcoef(a, b)[0, 1])
        # Convertir [-1, 1] → [0, 100] %
        similitud = max(0.0, (ncc + 1.0) / 2.0 * 100.0)
        return round(similitud, 1)

    # ─────────────────────────────────────────────────────────────────────────
    def resumen(self) -> dict:
        return {
            'completada' : self.completada,
            'rpm_pico'   : self.rpm_pico,
            'c_pico'     : self.c_pico,
            't_estab_s'  : self.t_estab_s,
            'muestras'   : self._muestras,
            'vector_norm': self.vector_norm,
        }


# ─────────────────────────────────────────────────────────────────────────────
# DETECTOR DE TENDENCIA (alertas tempranas por pendiente de DI entre sesiones)
# ─────────────────────────────────────────────────────────────────────────────

class DetectorTendencia:
    """
    Analiza el DI medio entre sesiones históricas y detecta si hay una
    tendencia de degradación antes de que se dispare cualquier umbral.

    Devuelve:
      - pendiente_di: cambio de DI por sesión (positivo = empeora)
      - sesiones_a_falla: estimado de sesiones hasta DI = umbral
      - nivel: 'ok' | 'alerta' | 'critico'
      - mensaje: texto para mostrar en la UI
    """

    UMBRAL_FALLA  = 3.0
    UMBRAL_ALERTA = 0.05   # pendiente DI/sesión que activa alerta temprana
    UMBRAL_CRITICO= 0.15   # pendiente que activa alerta crítica
    MIN_SESIONES  = 5      # mínimo de sesiones para calcular tendencia

    def __init__(self):
        self._historial: deque = deque(maxlen=30)  # (sesion_id, di_medio)

    def agregar_sesion(self, sesion_id: int, di_medio: float | None,
                       similitud_pct: float | None = None):
        if di_medio is not None:
            self._historial.append({
                'id': sesion_id,
                'di': di_medio,
                'sim': similitud_pct,
            })

    def analizar(self) -> dict:
        n = len(self._historial)
        if n < self.MIN_SESIONES:
            return {
                'nivel'            : 'sin_datos',
                'pendiente_di'     : None,
                'sesiones_a_falla' : None,
                'r2'               : None,
                'mensaje'          : f"Acumulando historial — {n}/{self.MIN_SESIONES} sesiones",
                'di_actual'        : self._historial[-1]['di'] if n > 0 else None,
                'sim_actual'       : self._historial[-1]['sim'] if n > 0 else None,
            }

        indices = np.arange(n, dtype=float)
        di_vals = np.array([h['di'] for h in self._historial], dtype=float)

        # Regresión lineal
        A = np.vstack([indices, np.ones(n)]).T
        (a, b), res, _, _ = np.linalg.lstsq(A, di_vals, rcond=None)

        ss_res = float(np.sum((di_vals - (a * indices + b)) ** 2))
        ss_tot = float(np.var(di_vals) * n)
        r2     = 1.0 - ss_res / ss_tot if ss_tot > 1e-9 else 0.0

        di_actual  = float(di_vals[-1])
        sim_actual = self._historial[-1]['sim']

        if a > 1e-6:
            sesiones_a_falla = max(0.0, (self.UMBRAL_FALLA - di_actual) / a)
        else:
            sesiones_a_falla = None  # pendiente plana o mejorando

        # Tendencia de similitud (debe bajar si el motor se degrada)
        sim_vals = [h['sim'] for h in self._historial if h['sim'] is not None]
        sim_pendiente = None
        if len(sim_vals) >= 3:
            sv = np.array(sim_vals[-min(len(sim_vals), n):])
            iv = np.arange(len(sv), dtype=float)
            (sa, _), _, _, _ = np.linalg.lstsq(
                np.vstack([iv, np.ones(len(iv))]).T, sv, rcond=None
            )
            sim_pendiente = float(sa)  # negativo = similitud cayendo

        # Clasificación
        if a >= self.UMBRAL_CRITICO and r2 > 0.5:
            nivel = 'critico'
            msg   = (f"⛔  DEGRADACIÓN ACELERADA — DI sube {a:.3f}σ/sesión  "
                     f"(R²={r2:.2f})  →  falla estimada en {sesiones_a_falla:.0f} sesiones"
                     if sesiones_a_falla else
                     f"⛔  DEGRADACIÓN ACELERADA — DI sube {a:.3f}σ/sesión (R²={r2:.2f})")
        elif a >= self.UMBRAL_ALERTA and r2 > 0.4:
            nivel = 'alerta'
            msg   = (f"⚠  Tendencia de degradación detectada — DI +{a:.3f}σ/sesión  "
                     f"→  planificar mantenimiento en ~{sesiones_a_falla:.0f} sesiones"
                     if sesiones_a_falla else
                     f"⚠  Tendencia leve de degradación ({a:.3f}σ/sesión, R²={r2:.2f})")
        elif sim_pendiente is not None and sim_pendiente < -1.0 and len(sim_vals) >= 5:
            nivel = 'alerta'
            msg   = f"⚠  Similitud con motor sano cayendo {abs(sim_pendiente):.1f}%/sesión — inspección recomendada"
        else:
            nivel = 'ok'
            msg   = (f"✅  Motor estable — DI {a:+.4f}σ/sesión  "
                     f"(R²={r2:.2f})  |  DI actual: {di_actual:.3f}σ")

        return {
            'nivel'            : nivel,
            'pendiente_di'     : round(a, 5),
            'sesiones_a_falla' : round(sesiones_a_falla, 1) if sesiones_a_falla else None,
            'r2'               : round(r2, 3),
            'mensaje'          : msg,
            'di_actual'        : round(di_actual, 3),
            'sim_actual'       : sim_actual,
            'sim_pendiente'    : round(sim_pendiente, 2) if sim_pendiente else None,
            'di_vals'          : di_vals.tolist(),
            'indices'          : indices.tolist(),
            'pendiente_linea'  : (a, b),
        }