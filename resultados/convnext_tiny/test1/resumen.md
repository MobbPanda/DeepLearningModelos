# test1 — ConvNeXt-Tiny

- **Fecha:** 2026-10-15T21:16:21  ·  **duración:** 49.8 min  ·  **GPU:** NVIDIA GeForce RTX 4060 Laptop GPU
- **Pesos preentrenados:** IMAGENET1K_V1
- **Datos:** 25 clases, división {'train': 14128, 'val': 3028, 'test': 3028}
- **Hiperparámetros:** imgsz 224, épocas 50/50 (paciencia 10; 3 con el backbone congelado), lote 64, AdamW lr cabeza 0.001 / backbone 5e-05, dropout 0.2, label smoothing 0.1, stochastic depth 0.1, weight decay 0.05, peso_clases 0.0, aumentación randaugment (rotación 0°, escala 0.5, borrado 0.4)
- **Mejor época (val top-1):** 43

| Métrica (prueba) | Valor |
|---|---|
| imagenes_test | 3028 |
| accuracy | 0.9944 |
| accuracy_top5 | 0.9977 |
| precision_macro | 0.9941 |
| recall_macro | 0.9888 |
| f1_macro | 0.9913 |
| precision_ponderada | 0.9944 |
| recall_ponderado | 0.9944 |
| f1_ponderado | 0.9944 |

| Costo | Valor |
|---|---|
| parametros_millones | 27.84 |
| gflops | 8.91 |
| tamano_best_pt_mb | 111.5 |
| tamano_modelo_app_mb | 55.7 |
| latencia_cpu_ms | 32.0 |
| latencia_gpu_ms | 3.7 |
| duracion_entrenamiento_min | 49.8 |
| gpu | NVIDIA GeForce RTX 4060 Laptop GPU |
