# Knowledge Base: RecSys End-to-End

Guía didáctica para data scientists que quieren entender el proyecto de recomendación de películas de principio a fin. Cada sección explica un concepto desde cero, cómo está implementado en este repositorio, y por qué se tomaron las decisiones de diseño que se tomaron.

---

## Índice

1. [Semantic Release y Conventional Commits](#1-semantic-release-y-conventional-commits)
2. [Generación de eventos históricos (Simulador 1)](#2-generación-de-eventos-históricos-simulador-1)
3. [Feature Engineering offline](#3-feature-engineering-offline)
4. [Preparación del dataset: el input al modelo](#4-preparación-del-dataset-el-input-al-modelo)
5. [Arquitectura Two-Tower](#5-arquitectura-two-tower)
6. [ONNX: exportar y servir modelos en producción](#6-onnx-exportar-y-servir-modelos-en-producción)
7. [El truco de la torre de películas: embeddings precomputados](#7-el-truco-de-la-torre-de-películas-embeddings-precomputados)
8. [PyFlink: procesamiento de eventos en tiempo real](#8-pyflink-procesamiento-de-eventos-en-tiempo-real)
9. [Redis: el feature store en tiempo real](#9-redis-el-feature-store-en-tiempo-real)
10. [Flujo de datos online](#10-flujo-de-datos-online)
11. [Inferencia en producción: de una petición a una recomendación](#11-inferencia-en-producción-de-una-petición-a-una-recomendación)
12. [Simulador online (Simulador 2)](#12-simulador-online-simulador-2)
13. [Prometheus y Grafana: observabilidad](#13-prometheus-y-grafana-observabilidad)
14. [Loop de reentrenamiento](#14-loop-de-reentrenamiento)

---

## 1. Semantic Release y Conventional Commits

### ¿Qué problema resuelve?

En un proyecto con múltiples colaboradores, decidir cuándo publicar una nueva versión, qué número asignarle, y actualizar el changelog manualmente es tedioso y propenso a errores. **Semantic Release** automatiza todo esto: analiza los commits que hay desde el último release, decide si hay un bump de versión, lo hace, y publica el release sin intervención humana.

Para que esto funcione, los commits tienen que seguir un formato estructurado: **Conventional Commits**.

### ¿Qué es Conventional Commits?

Es una convención sencilla para los mensajes de commit. El formato es:

```
<tipo>[scope opcional]: <descripción>
```

Los tipos que usamos en este proyecto y su efecto en la versión:

| Tipo | Significado | Efecto en versión |
|---|---|---|
| `feat:` | Nueva funcionalidad | Bump **minor** (0.11.0 → 0.12.0) |
| `fix:` | Corrección de bug | Bump **patch** (0.11.0 → 0.11.1) |
| `chore:` | Tareas de mantenimiento | Sin bump |
| `docs:` | Documentación | Sin bump |
| `refactor:` | Refactorización | Sin bump |
| `test:` | Tests | Sin bump |

Ejemplos reales del historial del proyecto:
```
feat: model ranking
fix: mlflow
chore: cleaning
```

### Cómo está implementado

La configuración vive en `pyproject.toml`:

```toml
[tool.semantic_release]
version_toml = ["pyproject.toml:project.version"]  # dónde está la versión
tag_format = "v{version}"                           # formato de los tags git
commit_message = "chore(release): v{version} [skip ci]"  # el commit que crea el release

[tool.semantic_release.branches.main]
match = "(main|master)"
prerelease = false          # main → release estable (v0.12.0)

[tool.semantic_release.branches.dev]
match = ".*"
prerelease_token = "rc"
prerelease = true           # cualquier otra rama → prerelease (v0.12.0-rc.3)
```

El workflow de CI/CD que lo ejecuta es `.github/workflows/release.yml`. Se dispara en cada push a `main` y hace:

1. Checkout del repo con historial completo (`fetch-depth: 0`, necesario para leer todos los commits desde el último tag)
2. Ejecuta `python-semantic-release` que analiza los commits
3. Si hay cambios relevantes: actualiza la versión en `pyproject.toml`, actualiza `CHANGELOG.md`, crea el tag git y hace push

El pipeline de CI que valida cada PR vive en `.github/workflows/ci.yml` y ejecuta:
```bash
uv run ruff check .   # linting
uv run mypy .         # type checking
uv run pytest --tb=short  # tests
```

### Por qué funciona así y no de otra manera

La alternativa más común es hacer los releases manualmente: el desarrollador decide "ya es hora de la versión 0.12", edita el número en `pyproject.toml`, escribe el changelog a mano, hace el tag. Es error-prone y genera rozamiento. Semantic Release elimina toda esa fricción, pero a cambio impone la disciplina de los Conventional Commits: si un commit no sigue el formato, el sistema no puede decidir qué tipo de cambio es.

El prerelease automático en ramas no-main es especialmente útil para este proyecto: cuando se trabaja en una feature nueva (por ejemplo, en la rama `feature/two-tower-precompute`), cada push genera automáticamente una versión `rc` que se puede desplegar en staging sin tocar producción.

[↑ Volver al índice](#índice)

---

## 2. Generación de eventos históricos (Simulador 1)

### ¿Por qué necesitamos fabricar datos?

MovieLens 20M nos da ~20 millones de ratings explícitos (usuario X dio 4 estrellas a película Y). Pero un modelo de recomendación moderno no aprende de ratings: aprende de **comportamiento implícito**. Lo que nos interesa predecir es "¿hará click el usuario en esta película si se la mostramos?", no "¿le pondrá 4 estrellas?".

El problema es que MovieLens no tiene clicks, ni impresiones, ni sesiones. Solo tiene el resultado final (el rating). El Simulador 1 infiere hacia atrás qué secuencia de interacciones plausiblemente llevó a ese rating, y genera la tabla de eventos completa.

El resultado: ~150-170 millones de filas en `data/processed/events.parquet`.

### El funnel de interacción

La lógica central está en `src/data/generate_events.py`. Para cada rating real de MovieLens, se genera un "funnel positivo":

```
impression  @ t - (20 a 40 min)   ← el usuario vio la carátula en la home
view        @ t - (1 a 5 min antes de click)    ← abrió la ficha de la película
click       @ t - (5 a 15 min antes del rating) ← le dio play
rating      @ t                    ← timestamp real de MovieLens
```

Todos estos eventos tienen `label=1` porque el usuario sí interactuó con la película.

El esquema de cada evento (definido en `src/data/schemas.py`):

```python
{
    "event_id": uuid,              # identificador único
    "timestamp": int,              # unix epoch en segundos
    "user_id": int,                # del dataset de MovieLens
    "movie_id": int,               # del dataset de MovieLens
    "event_type": str,             # "impression" | "view" | "click" | "rating"
    "rating": float | None,        # solo para event_type=="rating"
    "session_id": uuid,            # agrupa eventos de una misma sesión
    "label": int,                  # 0 (negativo) o 1 (positivo)
    "recommendation_id": str | None  # se usa en la fase online, None aquí
}
```

### Sesiones

Los usuarios no ven películas de forma aislada: se conectan, navegan durante un rato, y se van. El código agrupa los ratings de cada usuario por proximidad temporal: si dos ratings consecutivos están a menos de 3600 segundos (1 hora), pertenecen a la misma sesión. Un gap mayor inicia una sesión nueva.

```python
SESSION_GAP_S = 3_600  # 60 minutos

# Si ts[i+1] - ts[i] > SESSION_GAP_S → nueva sesión (nuevo UUID)
```

### Negativos: lo más importante del dataset

Sin negativos, el modelo no aprende a discriminar. Si solo tiene ejemplos de películas que gustaron, aprende a predecir siempre "sí" para todo.

El simulador genera dos tipos de negativos:

**Tipo A — Impresiones sin view** (`label=0`): el usuario vio la carátula pero pasó de largo.
- **Cantidad**: 4-6 por cada película que sí puntuó en la sesión.
- **Composición** (parámetros `FRAC_POPULAR=0.40`, `FRAC_GENRE=0.40`, `FRAC_COLLAB=0.20`):
  - 40% películas populares ese mes (las que estarían en la portada)
  - 40% películas del mismo género (negativos "difíciles" — el modelo tiene que aprender a discriminar dentro del género)
  - 20% películas que usuarios similares vieron pero este usuario no (colaborativos)
- **Filtro crítico**: una película solo puede ser negativo si el usuario **nunca** la ha puntuado en ningún momento (se verifica contra `user_rated[user_id]`).

**Tipo B — Views sin click** (`label=0`): el usuario abrió la ficha pero no empezó a verla.
- **Cantidad**: 1 por cada 2-3 clicks en la sesión.
- Son negativos aún más difíciles que los Tipo A: el usuario mostró interés suficiente para abrir la ficha, pero no para ver la película.

### Estructuras de datos precalculadas

Antes de lanzar workers, el proceso principal precalcula tres estructuras grandes en memoria:

```python
genre_map = {movie_id: ["Action", "Comedy", ...]}        # géneros por película
popular_by_month = {(2015, 3): [movie_id, ...]}          # top-500 por mes
user_rated = {user_id: {movie_id, ...}}                   # películas vistas por usuario
movie_to_users = {movie_id: [user_id, ...]}              # índice invertido (para vecinos colaborativos)
```

Estas estructuras se heredan por los workers vía **copy-on-write** del fork de Unix. No se serializan, no se pasan por IPC: simplemente existen en la memoria del proceso padre y los hijos las leen sin copiarlas (mientras no las modifiquen).

### Multiprocessing

El procesamiento se paraleliza con `multiprocessing.Pool`:

```python
n_workers = min(cpu_count(), 4)   # capped a 4 (el COW amortiza las estructuras)
USERS_PER_BATCH = 500             # cada worker procesa 500 usuarios

# Cada worker escribe un chunk de Parquet directamente a disco
# y devuelve solo la ruta del archivo (no los ~700 eventos por usuario)
def _process_batch_task(args) -> str:
    batch_idx, user_batch = args
    all_events = []
    for user_id, ratings in user_batch:
        sessions = _build_sessions(ratings)
        events = _generate_user_events(sessions, ...)
        all_events.extend(events)
    pl.DataFrame(all_events).write_parquet(f"_batches/_batch_{batch_idx:04d}.parquet")
    return path
```

El diseño clave: los workers devuelven **solo una cadena de texto** (la ruta del archivo) por IPC, no los datos. Con 138K usuarios y ~700 eventos por usuario, serializar los datos sería ~67 GB de overhead. Al escribir directamente a disco y devolver solo rutas, el IPC es trivial.

Al final, se hace un merge de todos los chunks con Polars lazy:

```python
pl.scan_parquet(chunk_paths).sink_parquet(output_path)
```

### Por qué no usar el rating como label

El valor del rating (1-5 estrellas) se ignora como label. El modelo predice `P(click)`, no `P(rating alto)`. La razón: en producción, el modelo decide qué mostrar antes de que el usuario interactúe. La pregunta relevante es "¿hará click?", no "¿le gustará mucho?". Incorporar el rating como segundo objetivo (multitask learning) es una mejora posible pero no implementada aquí.

[↑ Volver al índice](#índice)

---

## 3. Feature Engineering offline

### ¿Qué es el feature engineering en este contexto?

Los eventos crudos (impresión, click, rating) no son directamente utilizables por el modelo. El modelo necesita vectores numéricos: "el usuario ha clickado mucho en películas de terror esta semana" o "esta película tiene alta popularidad en este momento". El feature engineering transforma los eventos en esos vectores.

El script `src/features/build_features.py` toma `events.parquet` y produce `train_dataset.parquet`, añadiendo todas las columnas de features que el modelo necesita.

### El principio más importante: point-in-time correctness

Cuando entrenamos el modelo, para cada evento de impresión tenemos que preguntarnos: **¿qué información habría tenido disponible en el momento exacto en que ocurrió esa impresión?**

Si usamos información del futuro (por ejemplo, cuántos clicks hizo el usuario en los 7 días *siguientes* a la impresión), estamos haciendo **data leakage**: el modelo aprende con información que en producción no tendría. Funciona bien en evaluación pero falla en producción.

Por eso todos los features de usuario se calculan con una ventana estrictamente **anterior** al timestamp del evento:

```python
# Feature: n_clicks_last_7d para el evento en timestamp t
clicks_in_window = clicks donde click.timestamp >= (t - 7 días) AND click.timestamp < t
n_clicks_last_7d = len(clicks_in_window)
```

### Split temporal (no aleatorio)

El dataset se divide en train/val/test por tiempo, nunca aleatoriamente:

```python
q80 = events_df["timestamp"].quantile(0.80)  # percentil 80 del tiempo
q90 = events_df["timestamp"].quantile(0.90)  # percentil 90 del tiempo

# train: timestamps < q80  (los primeros 80% del tiempo)
# val:   q80 <= timestamps < q90
# test:  timestamps >= q90 (el último 10% del tiempo)
```

¿Por qué es tan importante? Si hiciéramos split aleatorio, un evento del usuario X en enero podría estar en test, y eventos del mismo usuario en febrero en train. El modelo vería en entrenamiento información sobre ese usuario que temporalmente ocurrió *después* del evento de test que quiere predecir. La evaluación sería inflada y engañosa.

### Features de usuario (dinámicas)

Estas features capturan el comportamiento reciente del usuario. Se calculan por ventana temporal usando Polars:

**`genre_affinity_last_7d`** — lista de 19 floats (uno por género):
Proporción de clicks del usuario en cada género durante los últimos 7 días. Si el usuario ha clickado 10 películas, 4 de acción y 6 de comedia: `[0.4, 0.6, 0.0, ...]`.

**`n_clicks_last_7d`** — entero:
Número de clicks en los últimos 7 días.

**`favorite_genres`** — lista de strings (máximo 3):
Los 3 géneros con más clicks históricos del usuario. Calculado sobre todos los eventos, no solo los últimos 7 días.

**`avg_session_length`** — float:
Media de clicks por sesión del usuario. Un usuario con 5 sesiones y [3, 5, 2, 4, 6] clicks → 4.0.

**`days_since_last_activity`** — float o null:
Días transcurridos desde el evento anterior del mismo usuario. El primer evento de cada usuario tiene este campo como null (no hay evento anterior).

### Features de película (semi-estáticas)

Estas features son propiedades de la película. Se calculan una sola vez ancladas al percentil 80 del tiempo (`split_ts`):

**`genres_vector`** — lista de 19 floats (one-hot):
`[1.0, 0.0, 1.0, ...]` indica que la película pertenece a los géneros 1 y 3. Los 19 géneros canónicos se extraen del archivo `movies.csv`.

**`genome_top20`** — lista de 20 floats o null:
Los 20 tags del genome de MovieLens con mayor relevancia para esta película. Los genome scores son valores de relevancia temática (por ejemplo, "suspense": 0.89, "based on a book": 0.73). Si la película no tiene datos de genome, el campo es null.

**`year`** — entero o null:
Año extraído del título ("Inception (2010)" → 2010) con regex. Si el título no tiene año, null.

**`avg_rating`** — float o null:
Media de todos los ratings de MovieLens para esta película. Null si nunca fue valorada.

**`popularity_last_30d`** — entero:
Número de eventos (views o ratings) en una ventana de ±30 días alrededor del `split_ts`. Representa qué tan "de moda" estaba la película en el momento del split.

### Por qué solo se entrenan sobre impresiones

Al final del pipeline, el dataset se filtra a solo los eventos de tipo `impression`:

```python
dataset = dataset.filter(pl.col("event_type") == "impression")
```

El modelo aprende a responder "dado que mostramos esta película a este usuario, ¿hará click?". Esta es exactamente la pregunta que responde en producción: recibe una lista de candidatas y puntúa cada una. Si entrenáramos sobre clicks o views, estaríamos en una distribución diferente a la de serving.

[↑ Volver al índice](#índice)

---

## 4. Preparación del dataset: el input al modelo

### ¿Qué es RecSysDataModule?

`RecSysDataModule` (en `src/data/dataset.py`) es el puente entre los archivos Parquet y el modelo de PyTorch. Es una clase de **PyTorch Lightning** que encapsula toda la lógica de carga, vocabularios, normalización y creación de dataloaders.

Lightning usa un patrón estándar: defines `setup()` para preparar los datos y `train_dataloader()` / `val_dataloader()` para servirlos al trainer. Así el trainer no necesita saber nada sobre cómo están almacenados los datos.

### Vocabularios de IDs

Los IDs de usuario y película de MovieLens son enteros arbitrarios (ej: usuario 138493, película 89745). Las redes neuronales necesitan índices consecutivos que empiecen en 0 o 1 para hacer lookups en tablas de embeddings.

El DataModule construye dos diccionarios de vocabulario durante el `setup()`, **solo con los IDs del split de train**:

```python
train_user_ids = sorted(train_df["user_id"].unique().to_list())
self.user_vocab = {uid: idx + 1 for idx, uid in enumerate(train_user_ids)}
# Resultado: {138: 1, 245: 2, 891: 3, ...}
```

¿Por qué `idx + 1` y no `idx`? El índice 0 está reservado para usuarios/películas desconocidos (que aparecen en val o test pero no estaban en train). La capa de embedding tiene `padding_idx=0`, lo que significa que el índice 0 produce siempre un embedding de ceros.

### Normalización z-score

Los features numéricos escalares tienen escalas muy distintas: `n_clicks_last_7d` puede ir de 0 a 200, `avg_rating` de 0 a 5, `year` de 1900 a 2020. Si se pasan crudos al modelo, los features con valores grandes dominarán el gradiente.

La solución es normalización z-score: restar la media y dividir por la desviación estándar, de modo que todos los features tengan media ≈ 0 y desviación estándar ≈ 1.

```python
# Calculado SOLO sobre train split (para evitar data leakage)
for col in ("n_clicks_last_7d", "avg_session_length", "days_since_last_activity",
            "year", "popularity_last_30d", "avg_rating"):
    series = train_df[col].drop_nulls()
    mean = series.mean()
    std = max(series.std(), 1e-8)  # evitar división por cero
    self._norm_stats[col] = (mean, std)

# Aplicado a todos los splits con los stats de train
normalized = (value - mean) / std
```

Los valores null se rellenan con la media del train antes de normalizar, lo que equivale a asignarles el valor 0 después de la normalización. El modelo aprende que 0 significa "dato desconocido/promedio".

Estos stats de normalización se guardan en `vocab.json` junto con los vocabularios, para que el servidor de producción aplique exactamente la misma transformación.

### Los cinco tensores de salida

Cada batch que el DataModule entrega al modelo contiene cinco tensores:

```
user_ids      [batch_size]           int64  — índice de vocab del usuario
user_behavior [batch_size, 41]       float32 — features de comportamiento del usuario
movie_ids     [batch_size]           int64  — índice de vocab de la película
movie_meta    [batch_size, 42]       float32 — features de la película
labels        [batch_size]           float32 — 0.0 o 1.0
```

La dimensión 41 del comportamiento de usuario se desglosa así:

```
genre_affinity_last_7d  [19]   proporción de clicks por género (últimos 7 días)
favorite_genres_multihot [19]  one-hot de los géneros favoritos históricos
n_clicks_last_7d         [1]   z-normalizado
avg_session_length       [1]   z-normalizado
days_since_last_activity [1]   z-normalizado
─────────────────────────────
Total:                   41
```

La dimensión 42 de la película:

```
genres_vector    [19]   one-hot de géneros (estático)
genome_top20     [20]   relevancia de los 20 tags temáticos principales
year             [1]    z-normalizado
popularity_last_30d [1] z-normalizado
avg_rating       [1]    z-normalizado
─────────────────────────────
Total:           42
```

### Por qué estas dimensiones exactas

Los géneros de MovieLens son exactamente 19 (excluyendo el placeholder "(no genres listed)"). El genome top-20 es una decisión de diseño: hay 1128 tags en total, pero los 1128 más el embedding lookup harían el vector demasiado grande. Los 20 más relevantes capturan la esencia temática de la película con buen ratio señal/dimensión.

[↑ Volver al índice](#índice)

---

## 5. Arquitectura Two-Tower

### ¿Qué es un modelo two-tower?

Un modelo two-tower (también llamado dual encoder) es una arquitectura de red neuronal diseñada específicamente para sistemas de recomendación y recuperación de información. La idea fundamental es:

> En lugar de aprender una función de compatibilidad entre usuario y película directamente, aprende a mapear usuarios y películas a un espacio vectorial compartido donde los pares compatibles están cerca (producto punto alto) y los incompatibles lejos.

Esto tiene una ventaja enorme para producción: la "torre de película" solo depende de las features de la película, que no cambian entre requests. Se puede precomputar una sola vez. La "torre de usuario" se computa por request con las features actuales del usuario.

El código del modelo está en `src/models/two_tower.py`.

### Visión general de la arquitectura

```
                    TWO-TOWER MODEL
                    ═══════════════

 Usuario                              Película
 ───────                              ────────

 user_id ──→ [Embedding]              movie_id ──→ [Embedding]
             │  (dim=64)                           │  (dim=64)
             │                                     │
 behavior ───┘ cat [B, 64+41=105]    meta ────────┘ cat [B, 64+42=106]
             │                                     │
             ▼                                     ▼
          [MLP]                                 [MLP]
          Linear(105 → 256)                     Linear(106 → 256)
          LayerNorm → GELU → Dropout            LayerNorm → GELU → Dropout
          Linear(256 → 128)                     Linear(256 → 128)
          LayerNorm → GELU → Dropout            LayerNorm → GELU → Dropout
          Linear(128 → 64)                      Linear(128 → 64)
          Tanh                                  Tanh
             │                                     │
             ▼                                     ▼
     user_embedding [B, 64]           movie_embedding [B, 64]
             │                                     │
             └──────────── · (dot product) ────────┘
                                │
                             logit [B]
                                │
                           sigmoid(logit)
                                │
                           score [B] ∈ (0, 1)
                       P(click | user, movie)
```

### UserTower en detalle

```python
class UserTower(nn.Module):
    def __init__(self, n_users, embed_dim=64, behavior_dim=41,
                 hidden_dims=(256, 128), output_dim=64):
        self.embed = nn.Embedding(n_users + 1, embed_dim, padding_idx=0)
        # embed_dim + behavior_dim = 64 + 41 = 105
        self.mlp = _build_mlp(embed_dim + behavior_dim, hidden_dims, output_dim)

    def forward(self, user_ids, behavior):
        e = self.embed(user_ids)              # [B, 64]
        x = torch.cat([e, behavior], dim=-1)  # [B, 105]
        return self.mlp(x)                    # [B, 64]
```

El embedding de usuario captura la identidad "permanente" del usuario (sus preferencias globales de largo plazo). El vector `behavior` captura su estado actual (qué ha visto recientemente). Concatenarlos permite al modelo combinar ambas señales.

### MovieTower en detalle

Estructura idéntica a UserTower pero con `meta_dim=42`:

```python
class MovieTower(nn.Module):
    def __init__(self, n_movies, embed_dim=64, meta_dim=42,
                 hidden_dims=(256, 128), output_dim=64):
        self.embed = nn.Embedding(n_movies + 1, embed_dim, padding_idx=0)
        self.mlp = _build_mlp(embed_dim + meta_dim, hidden_dims, output_dim)
```

El embedding de película captura la identidad de la película en el espacio latente de preferencias. El vector `meta` añade features observables (género, popularidad, año).

### El bloque MLP: capa a capa

```python
def _build_mlp(in_dim, hidden_dims, out_dim) -> nn.Sequential:
    layers = []
    prev = in_dim
    for h in hidden_dims:          # (256, 128)
        layers += [
            nn.Linear(prev, h),
            nn.LayerNorm(h),       # normaliza activaciones → entrenamiento estable
            nn.GELU(),             # activación suave (sin "zona muerta" como ReLU)
            nn.Dropout(0.1),       # regularización: 10% de neuronas a 0 en train
        ]
        prev = h
    layers += [nn.Linear(prev, out_dim), nn.Tanh()]
    return nn.Sequential(*layers)
```

**LayerNorm**: normaliza las activaciones dentro de cada ejemplo (no por batch). Estabiliza el entrenamiento, especialmente cuando los valores de entrada tienen rangos muy distintos.

**GELU**: función de activación suave. A diferencia de ReLU que corta en 0 abruptamente, GELU tiene una transición continua. En la práctica converge más rápido en modelos de embeddings.

**Tanh en la capa de salida**: acota los embeddings a [-1, 1]. Esto garantiza que el producto punto esté acotado (no puede crecer indefinidamente) y que los embeddings tengan una magnitud razonable para el dot product.

### El score: por qué producto punto y no distancia euclídea

```python
def forward(self, user_ids, behavior, movie_ids, meta):
    u = self.encode_user(user_ids, behavior)    # [B, 64]
    m = self.encode_movie(movie_ids, meta)      # [B, 64]
    return torch.sigmoid((u * m).sum(dim=-1))   # [B]
```

`(u * m).sum(dim=-1)` es el producto punto entre los vectores de usuario y película. Si el usuario tiene un valor alto en la dimensión "terror" y la película también, el producto contribuye positivamente. Si son de signo opuesto, contribuye negativamente.

¿Por qué no distancia euclídea? El producto punto es computacionalmente equivalente (con vectores de norma fija), pero es más natural para modelos entrenados con BCE y más eficiente de vectorizar en numpy (`matrix @ vector`).

### Función de pérdida

El modelo se entrena con **Binary Cross-Entropy (BCE)**:

```
loss = -[y * log(pred) + (1 - y) * log(1 - pred)]
```

Donde `y ∈ {0, 1}` es el label y `pred ∈ (0, 1)` es el score del modelo. La métrica de validación es **AUC-ROC**: mide si el modelo rankea correctamente los clicks sobre los no-clicks, independientemente del umbral elegido.

[↑ Volver al índice](#índice)

---

## 6. ONNX: exportar y servir modelos en producción

### ¿Qué es ONNX?

ONNX (Open Neural Network Exchange) es un formato de representación de modelos de ML. Es como un "PDF para redes neuronales": un formato estándar que diferentes frameworks pueden leer y ejecutar, independientemente de si el modelo fue entrenado con PyTorch, TensorFlow, o cualquier otro.

¿Por qué no usar PyTorch directamente en producción? Principalmente por rendimiento y portabilidad:
- PyTorch está diseñado para entrenamiento (gradientes, autograd). En inferencia, ese overhead no se necesita.
- ONNX Runtime (el motor de ejecución de ONNX) está optimizado para inferencia: puede usar CUDA, CPU optimizations (AVX), o incluso hardware especializado sin cambiar el código.
- Los modelos ONNX son más fáciles de versionar y desplegar: son un solo archivo binario, sin dependencias de Python.

### Cómo se exporta en este proyecto

La función `export_onnx()` en `src/models/export_onnx.py` hace el export:

```python
torch.onnx.export(
    model,                               # el modelo PyTorch (solo UserTower)
    (dummy_user_ids, dummy_behavior),    # inputs de ejemplo para tracing
    output_path,                         # dónde guardar el .onnx
    opset_version=17,                    # versión del estándar ONNX
    input_names=["user_ids", "user_behavior"],
    output_names=["user_embedding"],
    dynamic_axes={
        "user_ids": {0: "batch"},        # la dimensión 0 (batch) es variable
        "user_behavior": {0: "batch"},
        "user_embedding": {0: "batch"},
    }
)
```

**Tracing vs Scripting**: PyTorch exporta ONNX por *tracing*. Pasa los inputs de ejemplo por el modelo y registra las operaciones que se ejecutan. El resultado es un grafo computacional estático (sin condicionales dinámicos). Si el modelo tiene `if` que dependen de los valores de los tensores, hay que usar TorchScript en su lugar.

**`opset_version=17`**: ONNX tiene versiones. La 17 es ampliamente soportada y cubre todas las operaciones que necesitamos (LayerNorm, GELU, Tanh).

**`dynamic_axes`**: sin esto, el modelo exportado esperaría exactamente el tamaño de batch de los inputs de ejemplo. Al marcar la dimensión 0 como dinámica, ONNX Runtime acepta cualquier tamaño de batch en inferencia.

### Verificación numérica

Después de exportar, el código verifica que el modelo ONNX produce los mismos resultados que PyTorch:

```python
sess = ort.InferenceSession(model_bytes)
ort_out = sess.run(["user_embedding"], ort_inputs)[0]

with torch.no_grad():
    pt_out = model(*dummy_inputs).numpy()

max_diff = float(np.abs(pt_out - ort_out).max())
if max_diff >= 5e-3:
    raise RuntimeError(f"ONNX vs PyTorch max diff {max_diff:.2e} exceeds 5e-3")
```

La tolerancia es `5e-3` (y no algo más ajustado como `1e-6`) porque las operaciones de LayerNorm y GELU tienen pequeñas diferencias numéricas entre el backend de PyTorch y el de ONNX Runtime (distintas optimizaciones de punto flotante). Para embeddings crudos esto es visible; para scores comprimidos con sigmoid (como hacía la versión anterior del modelo), el error se promediaba.

### Cómo se ejecuta en producción

ONNX Runtime (`onnxruntime` en Python) carga el archivo `.onnx` y lo ejecuta:

```python
sess = ort.InferenceSession(model_bytes)

outputs = sess.run(
    ["user_embedding"],                           # qué outputs queremos
    {
        "user_ids": np.array([user_idx]),         # int64 [1]
        "user_behavior": user_behavior[None, :],  # float32 [1, 41]
    }
)
user_embedding = outputs[0][0]  # extrae el vector [64]
```

ONNX Runtime es significativamente más rápido que PyTorch para inferencia single-batch porque no tiene overhead de autograd ni de Python. En este proyecto, la inferencia con ONNX tarda decenas de microsegundos, lo que permite responder una petición de recomendación en milisegundos.

[↑ Volver al índice](#índice)

---

## 7. El truco de la torre de películas: embeddings precomputados

### El problema de rendimiento en two-tower ingenuo

Imagina una implementación "naive" del two-tower: para cada petición de recomendación, tienes 200 candidatos. Para cada uno ejecutas el modelo completo (torre de usuario + torre de película). Eso son 200 llamadas al modelo ONNX.

Pero aquí hay algo importante: la torre de película solo depende de las features de la película. El género de "Inception" no cambia entre una petición y otra. La popularidad de ayer es la misma para el usuario A y el usuario B. El embedding de "Inception" es **idéntico** en todas las peticiones hasta el próximo reentrenamiento.

¿Por qué recalcularlo 200 veces por request, miles de veces por minuto?

### La solución: precomputar en el momento del export

En `src/models/export_onnx.py`, la función `precompute_movie_embeddings()` ejecuta la torre de película **una sola vez** sobre todo el catálogo:

```python
def precompute_movie_embeddings(model, movie_idxs, movie_metas) -> np.ndarray:
    movie_tower = model.movie_tower.cpu().eval()
    with torch.no_grad():
        embeddings = movie_tower(
            torch.tensor(movie_idxs, dtype=torch.long),   # [N_movies]
            torch.tensor(movie_metas, dtype=torch.float32) # [N_movies, 42]
        )
    return embeddings.numpy()  # [N_movies, 64]
```

El resultado (~27K vectores de dimensión 64) se guarda como columna `embedding` en `movie_features.parquet`. En `src/train.py`, la función `_export_serving_artifacts()` construye este parquet enriquecido al final del entrenamiento.

### El cambio en inferencia

En lugar de ejecutar la movie tower 200 veces:

```python
# Antes (una llamada ONNX por candidato):
for movie_id in candidate_ids:
    movie_embedding = onnx_session.run(movie_tower_inputs)
    score = sigmoid(dot(user_embedding, movie_embedding))
```

Ahora la scoring es:

```python
# Después (una llamada ONNX total + una multiplicación de matrices):
movie_embs = np.stack([movie_cache[mid].embedding for mid in movie_ids])  # [200, 64]
user_emb = onnx_session.run(user_tower_inputs)[0][0]                       # [64]
logits = movie_embs @ user_emb                                              # [200]
scores = 1.0 / (1.0 + np.exp(-logits))                                     # [200]
```

El coste pasa de O(n_candidatos) llamadas ONNX a O(1) llamada ONNX + una multiplicación matriz-vector en numpy (que es una operación BLAS altamente optimizada y tarda microsegundos).

### El contrato de artefactos

Esto implica un cambio en qué archivos genera el entrenamiento:

| Antes | Después |
|---|---|
| `model.onnx` (forward completo) | `user_tower.onnx` (solo torre de usuario) |
| `movie_features.parquet` (sin embeddings) | `movie_features.parquet` (con columna `embedding`) |
| `vocab.json` | `vocab.json` (sin cambios) |

La movie tower nunca se exporta a ONNX: sus outputs están todos en el parquet. No se necesita ejecutar de nuevo hasta el próximo reentrenamiento.

### Carga en el servidor

`OnnxScorer` (`src/serving/scorer.py`) carga el parquet al arrancar y construye un diccionario `movie_id → MovieRecord` donde cada `MovieRecord` tiene el embedding precomputado como numpy array:

```python
self.movie_cache: dict[int, MovieRecord] = self._load_movie_cache()
# movie_cache[550].embedding → np.array([0.23, -0.41, ...], dtype=float32)  # [64]
```

Este cache vive en memoria durante toda la vida del proceso. Con 27K películas y 64 dimensiones en float32, ocupa ~6 MB: completamente manejable.

[↑ Volver al índice](#índice)

---

## 8. PyFlink: procesamiento de eventos en tiempo real

### ¿Qué es Apache Flink?

Apache Flink es un framework de procesamiento de streams (flujos de datos en tiempo real). Mientras que herramientas como Spark procesan datos en batches (lotes), Flink procesa cada evento individualmente en cuanto llega, con latencias de milisegundos.

La abstracción clave de Flink es el **DataStream**: una secuencia potencialmente infinita de eventos que el sistema procesa de forma continua. Puedes aplicarle filtros, transformaciones, y funciones con estado (stateful operations) que recuerdan información de eventos anteriores.

**PyFlink** es la API de Python para Flink, que internamente ejecuta un JVM (Java Virtual Machine). Por eso el Dockerfile del streaming incluye Java 17.

### ¿Por qué Flink y no, por ejemplo, un consumidor Kafka simple?

Un consumidor Kafka simple podría leer eventos y actualizar Redis. El problema está en el estado: necesitamos mantener un historial de clicks de los últimos 7 días por usuario para calcular `genre_affinity_last_7d`. Con un consumidor simple, ese estado estaría en memoria del proceso y se perdería si el proceso reinicia. Flink tiene **checkpointing**: guarda el estado en disco cada 60 segundos, y si el proceso falla, se recupera exactamente desde donde lo dejó.

### El pipeline en este proyecto

El código está en `src/features/processor.py`. El pipeline completo:

```
RedPanda (topic: "events")
    → KafkaSource (lee eventos como strings JSON)
    → filter: solo event_type == "click"
    → keyBy: user_id (cada usuario va siempre al mismo worker)
    → UserFeatureProcessor (función con estado)
    → Redis HSET user:{user_id}
```

### La función con estado: UserFeatureProcessor

`UserFeatureProcessor` es una subclase de `KeyedProcessFunction`. Flink garantiza que todos los eventos del mismo `user_id` van al mismo hilo, en orden. Por eso no hay concurrencia que gestionar.

El estado que mantiene por usuario es un `ListState` de strings JSON:

```python
def open(self, ctx):
    self._state = ctx.get_list_state(
        ListStateDescriptor("click_history", Types.STRING())
    )
    self._redis = redis.Redis(host=REDIS_HOST, socket_timeout=1.0)
```

`ListState` es gestionado por Flink: se serializa en checkpoints, sobrevive a reinicios del proceso.

Para cada evento de click que llega:

```python
def process_element(self, event_json, ctx):
    event = json.loads(event_json)
    now_ts = int(time.time())

    # 1. Crear una entrada compacta con solo lo que necesitamos
    new_entry = json.dumps({
        "ts": event.get("timestamp", now_ts),
        "genres_vector": event.get("genres_vector") or [],
    })

    # 2. Actualizar el ListState: añadir la nueva entrada, podar las antiguas
    current = list(self._state.get() or [])
    current.append(new_entry)
    cutoff = now_ts - 7 * 24 * 3600  # hace 7 días
    current = [e for e in current if json.loads(e)["ts"] >= cutoff]
    self._state.update(current)

    # 3. Calcular features y escribir en Redis
    entries = [json.loads(e) for e in current]
    features = _compute_features(entries, now_ts)
    self._redis.hset(f"user:{user_id}", mapping={...})
    self._redis.expire(f"user:{user_id}", 7 * 24 * 3600)
```

### _compute_features: el cálculo de features

Esta función es intencionalmente pura (sin efectos secundarios, sin Flink, sin Redis). Calcula las features a partir de una lista de entradas:

```python
def _compute_features(entries, now_ts):
    entries_1h = [e for e in entries if e["ts"] >= now_ts - 3600]

    def avg_genre(subset):
        vecs = [e["genres_vector"] for e in subset if e.get("genres_vector")]
        if not vecs:
            return [0.0] * n_genres
        return [sum(v[i] for v in vecs) / len(vecs) for i in range(n_genres)]

    return {
        "genre_affinity_last_7d": avg_genre(entries),   # media de géneros en 7d
        "n_clicks_last_7d": len(entries),
        "genre_affinity_last_1h": avg_genre(entries_1h), # media de géneros en 1h
        "n_clicks_last_1h": len(entries_1h),
        "days_since_last_activity": (now_ts - max(e["ts"] for e in entries)) / 86400
                                    if entries else 7.0,
    }
```

El `genres_vector` que incluye cada evento (adjuntado por el servidor FastAPI al publicar en RedPanda) es el vector de géneros de la película clickada. El promedio de estos vectores a lo largo de los clicks recientes es la afinidad de género del usuario.

El formato exacto de la escritura en Redis, el TTL, y el patrón fire-and-forget se explican en la siguiente sección, [Redis: el feature store en tiempo real](#9-redis-el-feature-store-en-tiempo-real).

[↑ Volver al índice](#índice)

---

## 9. Redis: el feature store en tiempo real

### ¿Qué es Redis?

Redis (Remote Dictionary Server) es una base de datos en memoria de clave-valor. A diferencia de una base de datos relacional (donde los datos están en disco y las queries tardan milisegundos o más), Redis mantiene todos los datos en RAM y responde en microsegundos.

Es la herramienta estándar de la industria para lo que se llama un **feature store en tiempo real**: el lugar donde se almacenan los features de los usuarios (calculados por el sistema de streaming) para que el servidor de inferencia pueda leerlos con latencia mínima al atender cada petición.

### ¿Por qué Redis y no una base de datos convencional?

La latencia de serving impone restricciones estrictas. Una petición a `/recommendations` tiene que terminar en < 100ms para que el usuario no lo note. Dentro de ese presupuesto, hay que:

- Leer los features del usuario de algún almacén
- Correr el modelo ONNX
- Hacer el dot product con 200 candidatos
- Devolver la respuesta

Con PostgreSQL o cualquier base de datos en disco, la lectura de features costaría 5-20ms de latencia de red + I/O. Con Redis, cuesta < 1ms. La diferencia es estructural: Redis no va a disco.

La contrapartida es que Redis es **volátil**: si el proceso se reinicia sin persistencia configurada, los datos se pierden. Por eso en este proyecto Redis es el feature store online (volátil, rápido) y GCS es el almacén persistente (durable, lento). Son dos capas complementarias.

### Estructura de datos: HSET

Redis tiene múltiples estructuras de datos (strings, listas, sets, sorted sets...). Para los features de usuario usamos **HSET** (Hash), que almacena múltiples campos bajo una misma clave:

```
CLAVE: "user:12345"
CAMPOS:
    genre_affinity_last_7d   → "[0.3, 0.0, 0.5, 0.0, ...]"  ← JSON list
    n_clicks_last_7d         → "7"
    genre_affinity_last_1h   → "[0.0, 0.0, 0.8, 0.0, ...]"
    n_clicks_last_1h         → "2"
    days_since_last_activity → "0.08"
```

La alternativa habría sido guardar un JSON completo en una clave string (`SET user:12345 '{"genre_affinity_last_7d": [...], ...}'`). El Hash tiene una ventaja: permite leer o actualizar un campo individual sin deserializar y reserializar el JSON completo. En la práctica aquí se leen y escriben todos los campos a la vez, así que la diferencia es menor, pero el Hash es semánticamente más limpio.

Los valores de tipo lista se serializan a JSON (`json.dumps(v)`) porque Redis solo almacena strings. Al leer, hay que hacer el `json.loads()` inverso.

### Cómo escribe PyFlink

El procesador de PyFlink escribe con `HSET` y establece un TTL de 7 días:

```python
self._redis.hset(
    f"user:{user_id}",
    mapping={
        k: json.dumps(v) if isinstance(v, list) else str(v)
        for k, v in features.items()
    },
)
self._redis.expire(f"user:{user_id}", 7 * 24 * 3600)  # 604800 segundos
```

El TTL (Time To Live) hace que Redis elimine automáticamente la clave si no recibe ningún click del usuario en 7 días. Sin él, Redis acumularía datos de todos los usuarios que alguna vez visitaron la plataforma, creciendo indefinidamente.

El patrón **fire-and-forget** significa que si la escritura falla (timeout de red, Redis caído momentáneamente), el procesador de Flink registra el error pero no detiene el stream. El evento de click se procesa pero el feature store queda momentáneamente desactualizado. En producción real se añadiría un dead-letter sink para reintentar escrituras fallidas.

### Cómo lee FastAPI

En `src/serving/app.py`, al recibir una petición de recomendaciones:

```python
user_data = redis_client.hgetall(f"user:{user_id}")
# Resultado: {'genre_affinity_last_7d': '[0.3, 0.0, ...]', 'n_clicks_last_7d': '7', ...}
# Si el usuario no existe: {}  (dict vacío, se maneja como cold start)
```

`hgetall` devuelve todos los campos del hash en un solo round-trip a Redis. La librería `redis-py` lo convierte directamente a un dict Python.

### Redis Warmup: precalentar el feature store

Al arrancar el stack por primera vez, Redis está vacío. Los primeros usuarios que lleguen serían todos "cold start" y recibirían recomendaciones no personalizadas.

Para evitar esto, el servicio `redis-warmup` (`src/features/load_warm_users.py`) lee el dataset de training y preescribe en Redis las features históricas de todos los usuarios antes de que el sistema empiece a recibir tráfico:

```python
# Para cada usuario del dataset de training:
redis_client.hset(f"user:{user_id}", mapping={
    "genre_affinity_last_7d": json.dumps(user_features["genre_affinity_last_7d"]),
    "n_clicks_last_7d": str(user_features["n_clicks_last_7d"]),
    # ...
})
```

Esto es el equivalente de un "warm start": el sistema arranca con el conocimiento histórico de los usuarios, y PyFlink solo necesita actualizar esas features a medida que llegan nuevos eventos.

### Redis en el contexto del stack

```
                    ESCRIBE                      LEE
PyFlink   ──────────────────────→  Redis  ←──────────────  FastAPI
(cada click del usuario)          (RAM)         (cada request de recomendaciones)

redis-warmup ───────────────────→  Redis
(al arrancar, features históricas)
```

Redis es el punto de encuentro entre el sistema de streaming y el sistema de serving. Sin él, habría que hacer una query a una base de datos transaccional en cada petición, o calcular los features on-the-fly en el momento del serving (lo que requeriría acceso al historial completo de eventos del usuario).

[↑ Volver al índice](#índice)

---

## 10. Flujo de datos online

### Visión general

Una vez el sistema está desplegado, los datos fluyen continuamente a través de varios componentes:

```
Simulador/Usuario
     │
     ▼
 FastAPI /events ──publish──→ RedPanda
     │                topic: "events"      topic: "model-predictions"
     │                     │                         │
     │              PyFlink consume                  │
     │                     │                         │
     │               Redis HSET                      │
     │                                               │
     ▼                                          Events Sink
 FastAPI /recommendations                           │
     │                                              ▼
     └──publish──→ RedPanda                    GCS (Parquet)
              topic: "model-predictions"     dt=YYYY-MM-DD/
```

### El topic "events"

Cuando el simulador (o un usuario real) hace click, el cliente POST al endpoint `/events` de FastAPI. FastAPI publica el evento en el topic `events` de RedPanda (un broker de mensajería compatible con el protocolo Kafka).

Pero antes de publicar, FastAPI enriquece el evento añadiendo `genres_vector` de la película desde su cache en memoria (`scorer.movie_cache`). Este enriquecimiento es crucial: permite que PyFlink calcule la afinidad de género sin tener que hacer su propio lookup de películas.

```python
# En src/serving/app.py, endpoint POST /events
movie = _scorer.movie_cache.get(event.movie_id)
if movie:
    event_dict["genres_vector"] = movie.genres_vector
kafka_producer.send("events", value=json.dumps(event_dict).encode())
```

### El topic "model-predictions"

Para cada petición de recomendaciones, FastAPI publica en `model-predictions` un registro por cada película recomendada:

```json
{
    "recommendation_id": "uuid-correlador",
    "user_id": 12345,
    "movie_id": 550,
    "score": 0.83,
    "position": 1,
    "timestamp": 1718000000,
    "model_version": "v0.12.0",
    "user_features": "{\"genre_affinity_last_7d\": [0.3, ...], ...}"
}
```

El campo `user_features` es el snapshot de las features del usuario **en el momento de la predicción**. Esto es esencial para el reentrenamiento: cuando más adelante queramos saber si la recomendación fue buena, necesitamos recrear exactamente el estado del usuario en ese momento, no el estado actual.

El `recommendation_id` es el UUID que conecta este registro con los eventos futuros del usuario. Si el usuario hace click en `movie_id=550`, el evento de click también llevará `recommendation_id="uuid-correlador"`, permitiendo hacer el join para construir el dataset de reentrenamiento.

### Events Sink: la persistencia en GCS

El servicio `events_sink` (en `src/data/events_sink.py`) consume ambos topics y escribe Parquet a Google Cloud Storage con particionado por fecha:

```
gs://bucket/events/dt=2026-06-18/part-0001.parquet
gs://bucket/events/dt=2026-06-18/part-0002.parquet
gs://bucket/inference-logs/dt=2026-06-18/part-0001.parquet
```

El particionado por `dt=` es un patrón estándar de data lakes (Hive partitioning). Permite leer solo los datos de una fecha específica sin escanear todo el histórico:

```python
pl.scan_parquet("gs://bucket/events/dt=*/**.parquet")
    .filter(pl.col("timestamp") >= since_ts)
    .collect()
```

Los archivos se escriben en batches de 500 eventos o cada 60 segundos (lo que ocurra primero), para no crear millones de archivos pequeños.

### Diagrama del stack de servicios

Todos los servicios se orquestan con Docker Compose en `compose/streaming.yml`. Las dependencias son:

```
redpanda → redpanda-init → streaming-processor
                        → recsys-serving → prometheus → grafana
redis → redis-warmup → recsys-serving
                     → streaming-processor
```

`redis-warmup` (`src/features/load_warm_users.py`) carga el parquet de training al arrancar y preescribe en Redis los features de todos los usuarios históricos. Sin esto, todos los usuarios serían "cold start" al arrancar el sistema y tendrían recomendaciones genéricas hasta que el simulador generase suficientes clicks.

[↑ Volver al índice](#índice)

---

## 11. Inferencia en producción: de una petición a una recomendación

### El flujo completo de una petición

Cuando el simulador (o un usuario real) pide recomendaciones, esto es lo que ocurre en `src/serving/app.py`:

```
GET /recommendations/12345?n=5
        │
        ▼
1. Fetch features de Redis
   redis.hgetall("user:12345")
   → {genre_affinity_last_7d, n_clicks_last_7d, ...}
        │
        ▼
2. Construir vector de comportamiento [41]
   scorer.build_user_behavior(user_data)
   → np.array([0.3, 0.0, ..., 7, 4.0, 0.08])
        │
        ▼
3. Obtener índice de vocab del usuario
   scorer.user_idx(12345) → 4821
        │
        ▼
4. Generar candidatos (~200 películas)
   generate_candidates(user_data, movie_cache)
   → [550, 9428, 1265, ...]  # filtrados por género + popularidad
        │
        ▼
5. Scoring con Two-Tower
   scorer.score(4821, behavior_vector, candidate_ids)
   → [0.83, 0.71, 0.12, ...]  # una probabilidad por candidata
        │
        ▼
6. Ranking: top-5 por score
   argsort descending → [550, 9428, ...] (top 5)
        │
        ▼
7. Respuesta + métricas + log de inferencia
```

### Generación de candidatos

El catálogo completo tiene ~27K películas. Puntuar todas sería viable computacionalmente (un matrix multiply de [27K, 64] @ [64] tarda ~1ms en numpy), pero es una buena práctica reducir el espacio de candidatos primero. La función `generate_candidates()` en `src/serving/candidates.py` hace un filtrado por relevancia:

1. Si el usuario tiene géneros favoritos en Redis, filtra películas que compartan al menos un género.
2. Si el filtro produce menos de `n/2` candidatos, usa el catálogo completo.
3. Ordena por `popularity_last_30d` descendente.
4. Devuelve las top-200.

Es una estrategia simple pero efectiva: personalización por género + popularidad como tie-breaker.

### La llamada al modelo

`OnnxScorer.score()` en `src/serving/scorer.py`:

```python
def score(self, user_idx, user_behavior, movie_ids):
    # Embeddings de películas: ya precomputados, solo lookup
    movie_embs = np.stack([self.movie_cache[mid].embedding for mid in movie_ids])
    # [200, 64]

    # Embedding de usuario: una sola llamada ONNX
    user_emb = self._encode_user(user_idx, user_behavior)
    # [64]

    # Dot product vectorizado: [200, 64] @ [64] = [200]
    logits = movie_embs @ user_emb

    # Sigmoid: logit → probabilidad
    return 1.0 / (1.0 + np.exp(-logits))
    # [200] en (0, 1)
```

El único punto donde se ejecuta el modelo ONNX es `_encode_user()`:

```python
def _encode_user(self, user_idx, user_behavior):
    outputs = self._session.run(
        ["user_embedding"],
        {
            "user_ids": np.array([user_idx], dtype=np.int64),
            "user_behavior": user_behavior[np.newaxis, :].astype(np.float32),
        }
    )
    return outputs[0][0]  # extrae [64] del output [1, 64]
```

### Cold start: usuarios sin features en Redis

Si un usuario nuevo hace su primera petición, `redis.hgetall("user:12345")` devuelve un dict vacío. `build_user_behavior()` maneja esto con defaults: todos los features escalares usan la media del training (→ 0 después de z-normalización), y los vectores de género son todos ceros. El embedding de vocab para un ID desconocido es el vector de ceros (padding_idx=0).

El resultado es que un usuario nuevo recibe recomendaciones basadas puramente en los embeddings globales de los ítems, sin personalización. A medida que hace clicks, PyFlink actualiza Redis y las siguientes peticiones ya tienen personalización.

### Registro de inferencia para reentrenamiento

Después de generar las recomendaciones, FastAPI las publica en el topic `model-predictions` de forma asíncrona (en background, sin bloquear la respuesta al cliente). El campo `user_features` contiene el snapshot de features del usuario en este momento exacto, que se usará para reconstruir el vector de comportamiento al reentrenar.

[↑ Volver al índice](#índice)

---

## 12. Simulador online (Simulador 2)

### ¿Por qué necesitamos un simulador?

En producción real, los usuarios son personas reales. Para un proyecto didáctico (o para probar el sistema antes de tener usuarios reales), necesitamos un generador de tráfico sintético que:

1. Pida recomendaciones a la API
2. Simule si el usuario haría click o no (basándose en los scores del modelo)
3. Genere los eventos correspondientes

Esto crea un **feedback loop**: el simulador genera clicks → PyFlink actualiza features → las próximas recomendaciones son diferentes → el simulador se comporta diferente.

El código está en `src/simulator/online_simulator.py`.

### asyncio: concurrencia sin threads

El simulador necesita simular múltiples usuarios simultáneamente. Con threads, habría que gestionar locks y la GIL de Python limitaría el rendimiento. Con `asyncio`, un solo hilo puede manejar cientos de "sesiones" concurrentes.

La clave de asyncio es que todo es **cooperativo**: una corrutina cede el control con `await`. Mientras espera la respuesta HTTP, el event loop puede procesar otra corrutina. No hay paralelismo real (el CPU solo ejecuta una cosa a la vez), pero para tareas I/O-bound como hacer peticiones HTTP es perfectamente eficiente.

```python
async def _worker(pool, config, client):
    while True:
        user_id = pool.pick()
        await _run_session(user_id, config, client)   # cede el control mientras espera HTTP
        gap = random.uniform(10, 60)
        await asyncio.sleep(gap)                       # cede el control durante el sleep

async def main(config):
    async with httpx.AsyncClient(timeout=10.0) as client:
        # Lanza K workers concurrentes
        for _ in range(config.max_concurrent):
            asyncio.create_task(_worker(pool, config, client))
        asyncio.create_task(_add_cold_users_to_pool(pool, ...))
        await asyncio.Event().wait()  # espera indefinidamente (hasta Ctrl+C)
```

Con `max_concurrent=10`, hay exactamente 10 sesiones activas en cualquier momento.

### Una sesión de usuario simulada

```python
async def _run_session(user_id, config, client):
    session_id = uuid.uuid4()

    for page in range(config.max_pages_per_session):  # máximo 5 "páginas"
        # 1. Pedir recomendaciones
        resp = await client.get(f"{api_url}/recommendations/{user_id}?n=5")
        recs = resp.json()["recommendations"]

        clicked_page = False
        for rec in recs:
            # 2. Generar impresión (siempre)
            await client.post(f"{api_url}/events", json=impression_event)

            # 3. ¿Hace click? (depende del score del modelo y la temperatura)
            prob = _click_prob(rec["score"], config.temperature)
            if random.random() < prob:
                await client.post(f"{api_url}/events", json=click_event)
                await asyncio.sleep(config.watch_time_seconds)  # "ve la película"
                clicked_page = True
                break  # solo un click por página → vuelve a la home

        if not clicked_page:
            break  # el usuario se fue sin hacer click → fin de sesión
```

El modelo de "como mucho un click por página y luego vuelve a la home" refleja un comportamiento realista: el usuario elige una película para ver, la ve, y luego puede o no volver a pedir más.

### La temperatura: controlando el realismo

La función `_click_prob()` mapea el score del modelo a una probabilidad de click:

```python
def _click_prob(score, temperature):
    score = max(1e-7, min(1 - 1e-7, score))
    logit = math.log(score / (1 - score))   # inverso del sigmoid
    noise = random.gauss(0, 0.5)            # ruido gaussiano
    return 1.0 / (1.0 + math.exp(-(logit * temperature + noise)))
```

Esta función hace tres cosas:
1. **Convierte el score a un logit**: `log(p/(1-p))`. Un score de 0.9 → logit ≈ 2.2.
2. **Añade ruido**: los usuarios reales no son perfectamente racionales. Un ruido gaussiano de desviación 0.5 refleja variabilidad en la decisión.
3. **Escala por temperatura**:
   - `temperature=1.0`: el comportamiento sigue aproximadamente los scores del modelo
   - `temperature=2.0`: mucho más determinista. Los ítems con score alto casi siempre se clickan, los de score bajo casi nunca.
   - `temperature=0.1`: casi aleatorio. Los clicks no dependen del modelo.

La temperatura permite estudiar cómo se comporta el sistema bajo distintos patrones de usuario, o simular el efecto de un modelo de peor o mejor calidad.

### Pool de usuarios y cold start

El pool de usuarios empieza con todos los IDs del dataset de training (warm users: usuarios que ya tienen features en Redis). Periódicamente se añaden usuarios nuevos (cold start, IDs > 200000) a una tasa configurable (`new_users_per_hour=6.0`).

El `UserPool` no necesita locks porque asyncio es single-threaded: no hay dos corrutinas ejecutando `pool.pick()` o `pool.add_cold_user()` simultáneamente.

[↑ Volver al índice](#índice)

---

## 13. Prometheus y Grafana: observabilidad

### ¿Qué es observabilidad?

Un sistema en producción que no se puede observar es un sistema que no se puede operar. Observabilidad significa tener suficiente información sobre el comportamiento interno del sistema para responder preguntas como: "¿el modelo está funcionando bien?", "¿hay algún componente lento?", "¿el CTR ha bajado en las últimas horas?".

En este stack, la observabilidad se implementa con dos herramientas:
- **Prometheus**: base de datos de series temporales para métricas numéricas.
- **Grafana**: herramienta de visualización que lee de Prometheus y genera dashboards.

### Cómo funcionan juntos

```
FastAPI /metrics ◄── scrape cada 15s ── Prometheus ── query ──► Grafana
    │                                       │
    │ expone métricas                       │ almacena series temporales
    │ en formato texto                      │ evalúa alertas
    ▼                                       ▼
recsys_clicks_total                    alert: CTR < 1%
recsys_impressions_total               alert: latencia P95 > 500ms
recsys_recommendation_score{...}
```

**Prometheus** no recibe datos: los va a buscar ("scraping") al endpoint `/metrics` de FastAPI cada 15 segundos. Esto significa que FastAPI no necesita saber que Prometheus existe; simplemente expone sus métricas y Prometheus las recoge.

La configuración del scraping está en `docker/prometheus/prometheus.yml`:

```yaml
scrape_configs:
  - job_name: "recsys-serving"
    static_configs:
      - targets: ["recsys-serving:8000"]
    scrape_interval: 15s
```

### Métricas definidas en FastAPI

En `src/serving/app.py`, las métricas se declaran al inicio del módulo con `prometheus_client`:

```python
IMPRESSIONS = Counter("recsys_impressions_total", "Cumulative impressions")
CLICKS = Counter("recsys_clicks_total", "Cumulative clicks")
RECOMMENDATIONS_SERVED = Counter(
    "recsys_recommendations_served_total",
    "Recommendation requests served",
    ["model_version"]  # label: permite ver por versión de modelo
)
RECOMMENDATION_SCORE = Histogram(
    "recsys_recommendation_score",
    "Distribution of recommendation scores",
    ["model_version"],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)
UNIQUE_MOVIES_24H = Gauge("recsys_unique_movies_recommended_24h", "Catalog coverage")
```

Los tipos de métricas:
- **Counter**: solo sube. Ideal para eventos acumulativos (clicks, requests).
- **Histogram**: distribuye los valores en buckets. Permite calcular percentiles (P50, P95).
- **Gauge**: puede subir y bajar. Para valores actuales (usuarios activos, películas únicas recomendadas).

### Alertas

Las alertas se configuran en `docker/prometheus/alert_rules.yml`. Prometheus evalúa estas reglas continuamente y dispara alertas cuando se cumplen:

```yaml
groups:
  - name: recsys
    rules:
      - alert: LowCTR
        expr: |
          rate(recsys_clicks_total[10m])
          / rate(recsys_impressions_total[10m]) < 0.01
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "CTR below 1% for 10 minutes"

      - alert: HighRecommendationLatency
        expr: |
          histogram_quantile(0.95,
            rate(http_request_duration_seconds_bucket{
              handler="/recommendations/{user_id}"
            }[5m])) > 0.5
        for: 5m
        labels:
          severity: critical
```

La alerta `LowCTR` detecta degradación del modelo: si el CTR cae por debajo del 1% durante 10 minutos seguidos, algo va mal (el modelo empeoró, el simulador cambió comportamiento, o hay un bug).

### Dashboards de Grafana

El dashboard preconfigurado en `docker/grafana/dashboards/recsys.json` tiene estos paneles:

| Panel | Lo que muestra | Por qué importa |
|---|---|---|
| CTR (1h rolling) | `rate(clicks) / rate(impressions)` | La métrica reina: ¿recomendamos bien? |
| Clicks & Impressions Rate | Ambas tasas por separado | Detectar si el tráfico cambió |
| Recommendations Served | Por versión de modelo | Detectar si se desplegó una nueva versión |
| Recommendation Latency | P50/P95/P99 del endpoint | ¿El serving es rápido? |
| Score Distribution | P50 y P90 de los scores | Detectar colapso del modelo (todos scores ≈ 0.9) |
| Catalog Coverage | Películas únicas recomendadas en 24h | ¿El modelo recomienda diversidad o siempre lo mismo? |

El **score distribution** es especialmente útil para detectar problemas del modelo. Si todos los scores son muy altos (≥ 0.9), el modelo probablemente colapsó en un modo degenerado donde siempre predice positivo. Si todos son muy bajos (≤ 0.1), lo contrario. Una distribución saludable tiene varianza.

[↑ Volver al índice](#índice)

---

## 14. Loop de reentrenamiento

### El ciclo de mejora continua

Un modelo de ML en producción no es estático. El comportamiento de los usuarios cambia, el catálogo evoluciona, y el propio modelo puede degradarse con el tiempo (distributional shift). El loop de reentrenamiento permite actualizar el modelo continuamente con datos reales de producción.

El ciclo completo:

```
Datos de producción (eventos + logs de inferencia)
    │
    ▼
1. Build retrain dataset
   Join: inference logs ⋈ click events
    │
    ▼
2. Entrenar modelo v(N+1)
   Mismo código de training, nuevo dataset
    │
    ▼
3. Evaluar offline
   AUC en split temporal
    │
    ▼
4. Promover si mejora
   promote.py: Staging → Production
    │
    ▼
5. Desplegar nueva imagen
   FastAPI carga el modelo desde MLflow Production
```

### Construcción del dataset de reentrenamiento

El script `src/data/build_retrain_dataset.py` hace el join entre los logs de inferencia y los eventos de click:

```python
def _assign_labels(inference_df, clicks_df):
    # El join exacto usa recommendation_id + movie_id como llave compuesta
    click_keys = clicks_df.select(["recommendation_id", "movie_id"]).with_columns(
        pl.lit(1).alias("label")
    )
    return inference_df.join(
        click_keys, on=["recommendation_id", "movie_id"], how="left"
    ).with_columns(pl.col("label").fill_null(0))
    # Si la película recomendada no aparece en los clicks → label=0
```

La belleza de `recommendation_id`: no hay ambigüedad. Si el usuario 12345 vio la película 550 como recomendación número 2 de la petición X, y 3 minutos después hizo click, el `recommendation_id` lo une inequívocamente. Sin él, habría que hacer joins por `(user_id, movie_id, ventana_temporal)`, mucho más propenso a falsos positivos.

El campo `user_features` del log de inferencia se expande de vuelta a columnas individuales:

```python
def _expand_user_features(df):
    return df.with_columns([
        pl.col("user_features").map_elements(
            lambda s: json.loads(s).get("genre_affinity_last_7d", [0.0]*19)
        ).alias("genre_affinity_last_7d"),
        # ... más columnas
    ])
```

Esto reproduce exactamente el estado de las features del usuario en el momento de la predicción, sin necesidad de recomputarlas desde el historial de eventos.

### MLflow y la transición Staging → Production

El ciclo de versiones de modelos se gestiona con MLflow:

- **Staging**: cada vez que termina un entrenamiento, el modelo se registra automáticamente en staging (`_register_mlflow_model()` en `src/train.py`).
- **Production**: `src/models/promote.py` compara el AUC del modelo en Staging con el de Production y solo promueve si hay mejora:

```python
def promote(tracking_uri, model_name="two-tower-recsys"):
    staging_auc = client.get_run(staging[0].run_id).data.metrics["val_auc"]
    prod_auc = ... if production else 0.0   # 0.0 si no hay Production todavía

    if staging_auc > prod_auc:
        client.transition_model_version_stage(model_name, staging[0].version, "Production")
        print(f"Promoted: {prod_auc:.4f} → {staging_auc:.4f}")
    else:
        print(f"No promotion: Staging {staging_auc:.4f} ≤ Production {prod_auc:.4f}")
```

Este guardián automático previene que un modelo peor llegue a producción, aunque el entrenamiento termine sin errores.

### Carga automática del nuevo modelo en FastAPI

`OnnxScorer` resuelve el modelo desde MLflow al arrancar:

```python
def _resolve_from_mlflow(self):
    versions = client.get_latest_versions("two-tower-recsys", stages=["Production"])
    artifact_uri = versions[0].source  # gs://bucket/mlflow/artifacts/...
    return artifact_uri, versions[0].version
```

Esto significa que para desplegar un nuevo modelo solo hay que:
1. Hacer `make model-promote` (promueve en MLflow)
2. Reiniciar el contenedor de FastAPI (o esperar al próximo despliegue)

No hay que reconstruir la imagen Docker ni cambiar variables de entorno.

### Automatización con Airflow

Para que este loop ocurra automáticamente sin intervención manual, el proyecto tiene DAGs de Airflow en `dags/`. El DAG `daily_retrain` ejecuta en secuencia:

```
build_retrain_dataset → train_model → promote_if_better → deploy_serving
```

Airflow se despliega en su propia VM (`airflow-vm`) para no competir recursos con el stack de streaming. El Makefile tiene comandos para gestionarlo (`make airflow-deploy`, `make retrain-manual`).

[↑ Volver al índice](#índice)

---

*Documento generado como referencia técnica didáctica del proyecto RecSys End-to-End. Los fragmentos de código son extractos simplificados; ver los archivos fuente para la implementación completa.*
