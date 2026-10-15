# test2 — ResNet-50

- **Fecha:** 2026-10-15T16:27:34  ·  **duración:** 38.1 min  ·  **GPU:** NVIDIA GeForce RTX 4060 Laptop GPU
- **Pesos preentrenados:** IMAGENET1K_V2
- **Datos:** 25 clases, división {'train': 14128, 'val': 3028, 'test': 3028}
- **Hiperparámetros:** imgsz 224, épocas 50/50 (paciencia 10; 3 con el backbone congelado), lote 64, AdamW lr cabeza 0.001 / backbone 0.0001, dropout 0.2, label smoothing 0.1, peso_clases 0.0
- **Mejor época (val top-1):** 41

| Métrica (prueba) | Valor |
|---|---|
| imagenes_test | 3028 |
| accuracy | 0.9941 |
| accuracy_top5 | 0.9977 |
| precision_macro | 0.9945 |
| recall_macro | 0.9923 |
| f1_macro | 0.9933 |
| precision_ponderada | 0.9941 |
| recall_ponderado | 0.9941 |
| f1_ponderado | 0.9941 |

| Costo | Valor |
|---|---|
| parametros_millones | 23.56 |
| gflops | 8.17 |
| tamano_best_pt_mb | 94.6 |
| tamano_modelo_app_mb | 47.1 |
| latencia_cpu_ms | 40.7 |
| latencia_gpu_ms | 3.4 |
| duracion_entrenamiento_min | 38.1 |
| gpu | NVIDIA GeForce RTX 4060 Laptop GPU |
