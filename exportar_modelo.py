"""Exporta una prueba de entrenamiento (YOLO11, ResNet-50 o ConvNeXt-Tiny) a la aplicación web.

Copia el mejor peso de resultados/<modelo>/testN/ a app/models/<arquitectura>_testN/ junto con
el orden de clases y modelo.json (tipo de modelo, preprocesamiento y métricas de prueba).
La aplicación web lista en su selector todas las carpetas de app/models/ con ese formato.

- YOLO11 (yolo11.ipynb): el best.pt de Ultralytics se copia tal cual ("tipo": "ultralytics").
- ResNet-50 (resnet50.ipynb) y ConvNeXt-Tiny (convnextTiny.ipynb): el modelo completo se guarda en
  media precisión ("tipo": "torch"), para que model.pt pese la mitad (~47 y ~56 MB). La aplicación
  lo vuelve a float32 al cargarlo.

El tipo se decide por el "modelo_base" de configuracion.json: si termina en ".pt" es YOLO.

Uso, desde la carpeta DeepLearningModelos/ (el modelo es opcional; por defecto, yolo11):
    python exportar_modelo.py test3                       -> resultados/yolo11/test3
    python exportar_modelo.py resnet50 test1              -> resultados/resnet50/test1
    python exportar_modelo.py convnext_tiny test1         -> resultados/convnext_tiny/test1
    python exportar_modelo.py resultados/resnet50/test1   (una ruta también sirve)

Si la carpeta de destino ya existe, se reemplaza su contenido (por ejemplo, tras reevaluar la prueba).
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RESULTADOS_DIR = BASE_DIR / "resultados"
RESULTADOS_YOLO = RESULTADOS_DIR / "yolo11"
APP_MODELS_DIR = BASE_DIR / "app" / "models"
MODELO_POR_DEFECTO = "yolo11"


def leer_json(ruta):
    return json.loads(Path(ruta).read_text(encoding="utf-8"))


def leer_prueba(test_dir):
    """Verifica los archivos obligatorios de la prueba y devuelve (configuracion, metricas o None)."""
    test_dir = Path(test_dir)
    for requerido in (test_dir / "weights" / "best.pt", test_dir / "clases.json", test_dir / "configuracion.json"):
        if not requerido.exists():
            raise FileNotFoundError(f"Falta {requerido}: la prueba no terminó de entrenar o la ruta no es correcta.")

    configuracion = leer_json(test_dir / "configuracion.json")
    ruta_metricas = test_dir / "metricas_test.json"
    if not ruta_metricas.exists():
        print(f"Aviso: {ruta_metricas.name} no existe (el notebook no llegó a la evaluación); se exporta sin métricas.")
    metricas = leer_json(ruta_metricas) if ruta_metricas.exists() else None
    return configuracion, metricas


def preparar_destino(destino_base, nombre):
    """Crea destino_base/nombre vacía (borra la exportación anterior de esa misma prueba, si existía)."""
    destino = Path(destino_base) / nombre
    if destino.exists():
        shutil.rmtree(destino)  # solo contiene una exportación anterior de esta misma prueba
    destino.mkdir(parents=True)
    return destino


def escribir_modelo_json(destino, modelo):
    (destino / "modelo.json").write_text(json.dumps(modelo, indent=2, ensure_ascii=False), encoding="utf-8")


def exportar_prueba_yolo(test_dir, destino_base=APP_MODELS_DIR):
    """Exporta la prueba test_dir a destino_base/<modelo>_<prueba>/ y devuelve esa carpeta."""
    test_dir = Path(test_dir)
    configuracion, metricas = leer_prueba(test_dir)

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
    destino = preparar_destino(destino_base, f"{arquitectura}_{test_dir.name}")
    shutil.copy2(test_dir / "weights" / "best.pt", destino / "model.pt")
    shutil.copy2(test_dir / "clases.json", destino / "classes.json")
    escribir_modelo_json(destino, modelo)
    return destino


def exportar_prueba_torch(test_dir, destino_base=APP_MODELS_DIR):
    """Exporta una prueba de PyTorch (p. ej. ResNet-50) a destino_base/<modelo>_<prueba>/ y devuelve esa carpeta.

    best.pt es el modelo completo (torch.save(modelo)) en float32. Se guarda en media precisión,
    que reduce el archivo a la mitad (~47 MB en ResNet-50, bajo el límite de 100 MB de GitHub);
    la aplicación web lo vuelve a float32 al cargarlo.
    """
    import torch  # solo se necesita para este tipo de modelo

    test_dir = Path(test_dir)
    configuracion, metricas = leer_prueba(test_dir)

    arquitectura = configuracion["modelo_base"]  # p. ej. resnet50
    hp = configuracion["hiperparametros"]
    pre = configuracion["preprocesamiento"]
    modelo = {
        "nombre": f"{configuracion.get('nombre_modelo', arquitectura)} · {test_dir.name}",
        "tipo": "torch",
        "imgsz": pre["imgsz"],
        "redimension": pre["redimension"],
        "normalizacion": pre["normalizacion"],
        "salida": "logits",
        "metricas_test": metricas,
        "origen": {
            "notebook": configuracion.get("notebook", f"{arquitectura}.ipynb"),
            "prueba": test_dir.name,
            "entrenado": configuracion.get("inicio"),
            "epocas": f"{configuracion.get('epocas_ejecutadas', '?')}/{hp['epochs']}",
            "clases": configuracion["datos"]["clases"],
            "pesos_preentrenados": configuracion.get("pesos_preentrenados"),
        },
    }

    peso = test_dir / "weights" / "best.pt"
    try:
        red = torch.load(peso, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ValueError(f"No se pudo cargar {peso}: {exc}") from exc
    if not isinstance(red, torch.nn.Module):
        raise ValueError(f"{peso} no contiene un modelo completo (torch.save(modelo)), sino {type(red).__name__}.")
    red.half()  # media precisión: el archivo pesa la mitad

    # Se escribe recién aquí, cuando ya se leyó todo lo necesario
    destino = preparar_destino(destino_base, f"{arquitectura}_{test_dir.name}")
    torch.save(red, destino / "model.pt")
    shutil.copy2(test_dir / "clases.json", destino / "classes.json")
    escribir_modelo_json(destino, modelo)
    return destino


def exportar_prueba(test_dir, destino_base=APP_MODELS_DIR):
    """Exporta la prueba con la función que corresponde a su modelo_base (YOLO si termina en .pt)."""
    test_dir = Path(test_dir)
    if not (test_dir / "configuracion.json").exists():
        raise FileNotFoundError(
            f"Falta {test_dir / 'configuracion.json'}: la prueba no terminó de entrenar o la ruta no es correcta."
        )
    modelo_base = leer_json(test_dir / "configuracion.json")["modelo_base"]
    if str(modelo_base).endswith(".pt"):  # yolo11m-cls.pt
        return exportar_prueba_yolo(test_dir, destino_base)
    return exportar_prueba_torch(test_dir, destino_base)


def resolver_prueba(argumentos):
    """Convierte los argumentos de la línea de comandos ([modelo] prueba) en la carpeta de la prueba."""
    if len(argumentos) == 1:
        modelo, prueba = MODELO_POR_DEFECTO, argumentos[0]
    else:
        modelo, prueba = argumentos
    test_dir = Path(prueba)
    if not test_dir.is_dir() and len(test_dir.parts) == 1:  # un nombre ("test3"), no una ruta
        test_dir = RESULTADOS_DIR / modelo / prueba
    return test_dir


def main():
    parser = argparse.ArgumentParser(
        description="Exporta una prueba de YOLO11, ResNet-50 o ConvNeXt-Tiny a la aplicación web (app/models/)."
    )
    parser.add_argument(
        "modelo", nargs="?", help=f'carpeta de resultados/ ("yolo11", "resnet50" o "convnext_tiny"; por defecto, {MODELO_POR_DEFECTO})'
    )
    parser.add_argument("prueba", help='nombre de la prueba ("test3") o ruta a su carpeta')
    args = parser.parse_args()

    test_dir = resolver_prueba([a for a in (args.modelo, args.prueba) if a is not None])
    try:
        destino = exportar_prueba(test_dir)
    except FileNotFoundError as exc:
        sys.exit(f"Error: {exc}")
    except (json.JSONDecodeError, KeyError) as exc:
        sys.exit(f"Error: configuracion.json o metricas_test.json de {test_dir} no tiene el formato esperado ({exc!r}).")
    except ValueError as exc:  # peso de PyTorch ilegible o que no es un modelo completo
        sys.exit(f"Error: {exc}")
    print(f"Exportado a {destino}")
    print("Reiniciar la aplicación web (o recargar la página) para verlo en el selector de modelos.")


if __name__ == "__main__":
    main()
