"""Exporta una prueba de YOLO11 a la aplicación web.

Copia el mejor peso de resultados/yolo11/testN/ a app/models/<modelo>_testN/ junto con
el orden de clases y modelo.json (tipo de modelo, tamaño de entrada y métricas de prueba).
La aplicación web lista en su selector todas las carpetas de app/models/ con ese formato.

Uso, desde la carpeta DeepLearningModelos/:
    python exportar_modelo.py test3
    python exportar_modelo.py resultados/yolo11/test3

Si la carpeta de destino ya existe, se reemplaza su contenido (por ejemplo, tras reevaluar la prueba).
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RESULTADOS_YOLO = BASE_DIR / "resultados" / "yolo11"
APP_MODELS_DIR = BASE_DIR / "app" / "models"


def leer_json(ruta):
    return json.loads(Path(ruta).read_text(encoding="utf-8"))


def exportar_prueba_yolo(test_dir, destino_base=APP_MODELS_DIR):
    """Exporta la prueba test_dir a destino_base/<modelo>_<prueba>/ y devuelve esa carpeta."""
    test_dir = Path(test_dir)
    peso = test_dir / "weights" / "best.pt"
    for requerido in (peso, test_dir / "clases.json", test_dir / "configuracion.json"):
        if not requerido.exists():
            raise FileNotFoundError(f"Falta {requerido}: la prueba no terminó de entrenar o la ruta no es correcta.")

    configuracion = leer_json(test_dir / "configuracion.json")
    ruta_metricas = test_dir / "metricas_test.json"
    if not ruta_metricas.exists():
        print(f"Aviso: {ruta_metricas.name} no existe (el notebook no llegó a la evaluación); se exporta sin métricas.")
    metricas = leer_json(ruta_metricas) if ruta_metricas.exists() else None

    arquitectura = Path(configuracion["modelo_base"]).stem  # yolo11m-cls.pt -> yolo11m-cls
    hp = configuracion["hiperparametros"]
    modelo = {
        "nombre": f"{arquitectura.replace('yolo', 'YOLO', 1)} · {test_dir.name}",
        "tipo": "ultralytics",
        "imgsz": hp["imgsz"],
        "metricas_test": metricas,
        "origen": {
            "notebook": "yolo11.ipynb",
            "prueba": test_dir.name,
            "entrenado": configuracion.get("inicio"),
            "epocas": f"{configuracion.get('epocas_ejecutadas', '?')}/{hp['epochs']}",
            "clases": configuracion["datos"]["clases"],
        },
    }

    # Se escribe recién aquí, cuando ya se leyó todo lo necesario
    destino = Path(destino_base) / f"{arquitectura}_{test_dir.name}"
    if destino.exists():
        shutil.rmtree(destino)  # solo contiene una exportación anterior de esta misma prueba
    destino.mkdir(parents=True)
    shutil.copy2(peso, destino / "model.pt")
    shutil.copy2(test_dir / "clases.json", destino / "classes.json")
    (destino / "modelo.json").write_text(json.dumps(modelo, indent=2, ensure_ascii=False), encoding="utf-8")
    return destino


def main():
    parser = argparse.ArgumentParser(description="Exporta una prueba de YOLO11 a la aplicación web (app/models/).")
    parser.add_argument("prueba", help='nombre de la prueba ("test3") o ruta a su carpeta')
    args = parser.parse_args()

    test_dir = Path(args.prueba)
    if not test_dir.is_dir():
        test_dir = RESULTADOS_YOLO / args.prueba
    try:
        destino = exportar_prueba_yolo(test_dir)
    except FileNotFoundError as exc:
        sys.exit(f"Error: {exc}")
    except (json.JSONDecodeError, KeyError) as exc:
        sys.exit(f"Error: configuracion.json o metricas_test.json de {test_dir} no tiene el formato esperado ({exc!r}).")
    print(f"Exportado a {destino}")
    print("Reiniciar la aplicación web (o recargar la página) para verlo en el selector de modelos.")


if __name__ == "__main__":
    main()
