import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import CONDICION_IMAGEN as k

plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'figure.dpi': 300
})

CARPETA_IMAGENES = r"C:\Users\favio\Documents\IMAGENES_PROYECTO_SA828"
CARPETA_SALIDA = r"C:\Proyectos\PROYECTO_SA828\Audios"

FS = 48000                  # Hz, frecuencia de muestreo del PCM1808 y del filtro de Hilbert
F_PORTADORA = 1800          # Hz
MUESTRAS_POR_PIXEL = 20     # 48000 / 20 = 2400 pixeles por segundo

## Estructura de una línea: SYNC + PORCHE TRASERO + VIDEO + PORCHE DELANTERO
## Con 256 pixeles: 400 + 240 + 5120 + 240 = 6000 muestras = 125 ms por línea
MUESTRAS_SYNC = 400         # portadora a amplitud máxima
MUESTRAS_PORCHE_TRAS = 240  # nivel negro
MUESTRAS_PORCHE_DEL = 240   # nivel negro

## Cabecera de inicio de cuadro: silencio y luego portadora pura
SILENCIO_INICIAL_S = 0.2
PORTADORA_PURA_S = 1.0
MUESTRAS_GUARDA_CABECERA = 480  # silencio entre la cabecera y la primera línea, para que la envolvente caiga
SILENCIO_FINAL_S = 0.2

NIVEL_PICO = 0.9            # fracción de la escala completa, deja margen antes del recorte del ADC


def crear_linea_APT(matriz_base, fs, nombre_archivo, muestras_por_pixel=MUESTRAS_POR_PIXEL):
    ## La funcion portadora es una señal f(t) = A(t) * sin(2*pi*f*t)
    ## A(t) es la señal normalizada donde los pixeles pasan de 0 a 255, a 0.0 a 1.0
    ## f es la frecuencia portadora, 1800 Hz
    ## t es el tiempo, representado por la frecuencia de muestreo de 48000 Hz
    try:
        f_portadora = F_PORTADORA
        filas = matriz_base.shape[0]
        columnas = matriz_base.shape[1]

        ## Zonas de cada línea, en niveles de 0 a 255
        sync = np.ones(MUESTRAS_SYNC) * 255
        porche_tras = np.zeros(MUESTRAS_PORCHE_TRAS)
        porche_del = np.zeros(MUESTRAS_PORCHE_DEL)

        ## Cada pixel se repite muestras_por_pixel veces, así la envolvente varía
        ## más lento que la portadora y el filtro de Hilbert puede recuperarla
        video = np.repeat(matriz_base.astype(np.float64), muestras_por_pixel, axis=1)

        matriz_sync = np.tile(sync, (filas, 1))                 ## forma (n_filas, sync)
        matriz_tras = np.tile(porche_tras, (filas, 1))
        matriz_del = np.tile(porche_del, (filas, 1))
        matriz_final = np.hstack((matriz_sync, matriz_tras, video, matriz_del))
        muestras_linea = matriz_final.shape[1]

        ## Se aplana la matriz a 1 buffer de 1 dimension y se agrega la cabecera de cuadro
        silencio_ini = np.zeros(int(SILENCIO_INICIAL_S * fs))
        cabecera = np.ones(int(PORTADORA_PURA_S * fs)) * 255
        guarda = np.zeros(MUESTRAS_GUARDA_CABECERA)
        silencio_fin = np.zeros(int(SILENCIO_FINAL_S * fs))
        buffer = np.concatenate((silencio_ini, cabecera, guarda, matriz_final.flatten(), silencio_fin))

        ## Se genera la señal modulada
        A = buffer / 255.0
        t = np.arange(len(buffer)) / fs
        senal_salida = NIVEL_PICO * A * np.sin(2 * np.pi * f_portadora * t)
        audio_16bits = np.int16(np.clip(np.round(senal_salida * 32767), -32768, 32767))

        os.makedirs(os.path.dirname(nombre_archivo), exist_ok=True)
        wavfile.write(nombre_archivo, fs, audio_16bits)
        print(f"Archivo {nombre_archivo} generado con éxito")
        print(f"{filas} x {columnas} pixeles, {muestras_linea} muestras por línea ({1000 * muestras_linea / fs:.0f} ms), "
              f"duración total {len(buffer) / fs:.1f} s")

        fig, ax = plt.subplots(figsize=(8, 4))
        inicio = len(silencio_ini) + len(cabecera) + len(guarda)    ## inicio de la primera línea
        muestras_zoom = 1200                          ## sync, porche trasero y primeros pixeles
        ax.plot(t[inicio:inicio + muestras_zoom], senal_salida[inicio:inicio + muestras_zoom])
        ax.set_title("Señal modulada, inicio de línea")
        ax.set_xlabel("Tiempo [s]")
        ax.set_ylabel("Amplitud")
        ax.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        #plt.savefig(os.path.join(CARPETA_SALIDA, f"{os.path.basename(nombre_archivo)[:-4]}.png"))
        plt.show()
        return nombre_archivo
    except Exception as e:
        print(f"[¡] Error: {e}")


if __name__ == '__main__':
    lista_direcciones = [f for f in os.listdir(CARPETA_IMAGENES) if f.lower().endswith(k.EXTENSIONES)]
    print("Ingrese el numero de la imagen que necesita enviar")
    for x in enumerate(lista_direcciones):
        print(f"{x}")

    try:
        op = int(input("Ingrese el número de imagen que se quiere enviar: \n"))
        print(lista_direcciones[op])
        matriz_base = k.cargar_normalizar(os.path.join(CARPETA_IMAGENES, lista_direcciones[op]))
        print(matriz_base)
        print(matriz_base.shape)

        nombre_archivo = os.path.join(CARPETA_SALIDA, f"{lista_direcciones[op].split('.')[0]}_modulada.wav")
        crear_linea_APT(matriz_base, FS, nombre_archivo)
    except (ValueError, IndexError, FileNotFoundError) as e:
        print(f"Error: {e}")