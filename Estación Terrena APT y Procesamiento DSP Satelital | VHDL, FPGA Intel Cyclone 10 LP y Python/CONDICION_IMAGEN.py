import os
import numpy as np
from PIL import Image

ANCHO_IMAGEN = 256
ALTO_IMAGEN = 256
EXTENSIONES = ('.tif', '.tiff', '.jpg', '.jpeg', '.png')


def cargar_normalizar(ruta_imagen):
    ruta_minuscula = ruta_imagen.lower()

    if ruta_minuscula.endswith(('.tif', '.tiff')):
        import rasterio
        with rasterio.open(ruta_imagen) as dataset:
            imagen = dataset.read(1).astype(np.float64)

        # Los GeoTIFF suelen traer NaN como dato vacío
        minimo = np.nanmin(imagen)
        maximo = np.nanmax(imagen)
        imagen = np.nan_to_num(imagen, nan=minimo)

        if maximo == minimo:
            imagen_final = np.zeros_like(imagen)   # imagen plana, evita dividir entre cero
        else:
            imagen_final = (imagen - minimo) / (maximo - minimo)

        imagen_final_casteada = (imagen_final * 255).astype(np.uint8)
        imagen_temp = Image.fromarray(imagen_final_casteada)

    elif ruta_minuscula.endswith(('.jpg', '.jpeg', '.png')):
        imagen_temp = Image.open(ruta_imagen).convert('L')

    else:
        raise ValueError("Ingrese una ruta con una imagen en formato válido")

    # La imagen se fuerza a 256 x 256, por lo que una imagen no cuadrada se deforma
    imagen_redimensionada = imagen_temp.resize((ANCHO_IMAGEN, ALTO_IMAGEN))
    return np.array(imagen_redimensionada)


if __name__ == '__main__':
    carpeta = r"C:\Users\favio\Documents\IMAGENES_PROYECTO_SA828"
    lista_direcciones = [f for f in os.listdir(carpeta) if f.lower().endswith(EXTENSIONES)]
    for n_l in enumerate(lista_direcciones):
        print(f"{n_l}")

    try:
        op = int(input("Ingrese el numero de imagen que quiere enviar: "))
        matriz_base = cargar_normalizar(os.path.join(carpeta, lista_direcciones[op]))
        print(matriz_base)
        print(matriz_base.shape)
    except (ValueError, IndexError, FileNotFoundError) as e:
        print(f"[¡] Error: {e}")