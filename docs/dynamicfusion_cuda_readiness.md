# Verifica implementazione DynamicFusion da zero su CUDA

## Esito rapido
Stato attuale: **non ancora pronto** per una implementazione "da zero" completa e manutenibile di DynamicFusion.

## Cosa c'è già di buono
- Struttura modulare presente per componenti chiave (`TSDFVolume`, `WarpField`, `GaussNewtonSolver`).
- Kernel CUDA separati per TSDF e solver (`kernels/tsdf_kernels.cu`, `kernels/solver_kernels.cu`).
- Pipeline end-to-end funzionante in forma "monolitica" dentro `tests/main.cu` (`DynamicFusionPipeline`).

## Gap critici da chiudere
1. **Classe pubblica `DynamicFusion` dichiarata ma non implementata**
   - `include/dynamic_fusion.h` dichiara API e passi di pipeline, ma non esiste una controparte in `src/` con metodi definiti.
   - Oggi la logica reale vive soprattutto in `tests/main.cu` (classe `DynamicFusionPipeline`), quindi non è una libreria pulita riusabile.

2. **Dipendenze build fragili / non portabili**
   - `CMakeLists.txt` richiede Open3D con hint hardcoded locale (`/home/hydran00/...`).
   - Nell'ambiente corrente la configurazione fallisce per assenza `nvcc`.

3. **Mancano test automatici unitari/integrati reali**
   - C'è un eseguibile demo (`run_fusion`) ma non una suite di test riproducibile (tracking, convergenza solver, qualità ricostruzione).

4. **Separazione libreria/app incompleta**
   - Molta logica pipeline e visualizzazione è accoppiata nel file di test.
   - Per "da zero" serve spostare pipeline in `src/dynamic_fusion.cu` + mantenere `tests/main.cu` come solo runner.

## Priorità consigliata (ordine di lavoro)
1. Implementare `src/dynamic_fusion.cu` con i metodi di `include/dynamic_fusion.h`.
2. Estrarre da `tests/main.cu` la logica pipeline in classi libreria.
3. Rendere CMake portabile:
   - rimuovere path hardcoded Open3D,
   - aggiungere opzioni (`BUILD_VISUALIZER`, `WITH_OPEN3D`) e fallback headless.
4. Aggiungere test minimi:
   - integrazione TSDF sintetica,
   - corrispondenze ICP,
   - convergenza Gauss-Newton su scenario controllato.
5. Definire benchmark (tempo/frame, RMSE geometrico, drift pose).

## Check eseguito in questa verifica
- `cmake -S . -B build` → fallisce per toolkit CUDA non disponibile (`nvcc` non trovato) nell'ambiente corrente.

