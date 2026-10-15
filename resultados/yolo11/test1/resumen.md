# test1 — YOLO11m-cls

- **Fecha:** 2026-10-15T07:03:12  ·  **duración:** 20.1 min  ·  **GPU:** NVIDIA GeForce RTX 4060 Laptop GPU
- **Datos:** 25 clases, división {'train': 14128, 'val': 3028, 'test': 3028}
- **Hiperparámetros:** imgsz 224, épocas 47/50 (paciencia 10), lote -1, AdamW lr0 0.0005, dropout 0.2, cls_pw 0.0
- **Mejor época (val top-1):** 37

| Métrica (prueba) | Valor |
|---|---|
| imagenes_test | 3028 |
| accuracy | 0.9921 |
| accuracy_top5 | 0.9974 |
| precision_macro | 0.9933 |
| recall_macro | 0.9883 |
| f1_macro | 0.9907 |
| precision_ponderada | 0.9921 |
| recall_ponderado | 0.9921 |
| f1_ponderado | 0.9921 |

| Costo | Valor |
|---|---|
| parametros_millones | 10.39 |
| gflops | 4.85 |
| tamano_best_pt_mb | 21.0 |
| latencia_cpu_ms | 23.5 |
| latencia_gpu_ms | 2.8 |
| duracion_entrenamiento_min | 20.1 |
| gpu | NVIDIA GeForce RTX 4060 Laptop GPU |
