# Proyecto RecSys End-to-End: Resumen de Diseño

## Objetivo del proyecto

Construir un sistema de recomendación de películas end-to-end con fines didácticos, utilizando un stack de producción real: RedPanda, PyFlink, Redis, DNN exportada a ONNX, MLflow, FastAPI, Prometheus y Grafana.

---

## 1. Datos de partida: MovieLens 20M

MovieLens proporciona 6 archivos:

| Archivo | Contenido | Uso en el proyecto |
|---|---|---|
| `ratings.csv` | ~20M filas: userId, movieId, rating, timestamp | **Fuente principal**. De aquí se derivan todos los eventos sintéticos |
| `movies.csv` | ~27K películas: movieId, title, genres | Catálogo. Features de película (géneros, año extraído del título) |
| `genome-scores.csv` | movieId, tagId, relevance (float 0-1) | Features ricas de película: relevancia en ~1128 dimensiones temáticas |
| `genome-tags.csv` | tagId, tag | Nombres de los tags del genome |
| `tags.csv` | ~465K tags libres de usuarios | Opcional, útil para enriquecer |
| `links.csv` | movieId, imdbId, tmdbId | Para enriquecer catálogo vía TMDB si se desea (no necesario para arrancar) |

**Lo que MovieLens NO tiene:** perfiles de usuario, eventos implícitos (views, clicks), sesiones, ni información de contexto. Solo tiene ratings explícitos.

---

## 2. Datos que hay que fabricar: tabla de eventos

La única tabla que se necesita generar es `events`. Representa toda la actividad de los usuarios, no solo los ratings. Se fabrica a partir de los ratings de MovieLens desenrollando el funnel de interacción.

### Esquema de la tabla de eventos

| Campo | Tipo | Descripción |
|---|---|---|
| event_id | uuid | Identificador único del evento |
| timestamp | datetime | Cuándo ocurrió |
| user_id | int | Quién (viene de MovieLens) |
| movie_id | int | Sobre qué película |
| event_type | string | `impression`, `view`, `click`, `rating` |
| rating | float, nullable | Solo cuando event_type = "rating" |
| session_id | uuid | Agrupa eventos de una misma sesión |

### Proceso de generación

#### Paso 1: Construir sesiones a partir de los ratings

Se agrupan los ratings de cada usuario por proximidad temporal. Si dos ratings consecutivos están a menos de 60 minutos, pertenecen a la misma sesión. Si hay un gap mayor, es sesión nueva.

```
Ratings del user 123 ordenados por tiempo:

rating movie_305  @ 14:02
rating movie_22   @ 14:18
rating movie_891  @ 14:45
--- gap de 6 horas ---
rating movie_100  @ 20:30
rating movie_455  @ 20:52

→ session_A: [movie_305, movie_22, movie_891]
→ session_B: [movie_100, movie_455]
```

#### Paso 2: Generar el funnel positivo (por cada rating real)

Para cada película que el usuario sí puntuó, se genera la cadena de eventos hacia atrás. El timestamp del rating es el ancla:

```
impression  @ t - 30min   ← vio la carátula en la home/carrusel
view        @ t - 20min   ← abrió la ficha de la película
click       @ t - 10min   ← le dio play
rating      @ t           ← dato REAL de MovieLens
```

Deltas razonables:
- `impression` → 20-40 min antes del rating
- `view` → 1-5 min después de la impression
- `click` → 5-15 min después del view
- `rating` → timestamp real de MovieLens

#### Paso 3: Generar negativos (lo más importante)

##### Tipo A: Impresiones sin view ("vio la carátula y pasó de largo")

- **Cantidad:** 4-6 impresiones negativas por cada película puntuada en la sesión.
- **Selección de películas negativas (criterios de realismo):**
  - ~40% películas populares en esa ventana temporal (muchos ratings en los 30 días circundantes). Simulan lo que "estaría en la home".
  - ~40% películas del mismo género que las positivas. Enseñan al modelo a discriminar dentro de un género.
  - ~20% películas que usuarios similares sí consumieron pero el usuario actual no. Generan negativos "difíciles".
- **Filtro:** nunca usar como negativo una película que el usuario haya puntuado en algún momento.
- **Ubicación temporal:** al principio de la sesión, intercaladas con las impresiones positivas.

##### Tipo B: Views sin click ("abrió la ficha pero no le convenció")

- **Cantidad:** 1 view sin click por cada 2-3 películas clickadas en la sesión.
- **Selección:** películas muy similares a las consumidas pero con alguna diferencia (mismo género y época). Enseñan matices de preferencia.

#### Paso 4: Señal del rating

Para el modelo inicial, tratamiento binario simple: todo click es label=1, toda impression/view sin click es label=0. El modelo predice "¿interactuará o no?". El valor del rating se ignora como label (se puede incorporar después como peso o segundo objetivo).

### Volumen resultante estimado

Para un usuario con 100 ratings:

| Evento | Cantidad aprox. |
|---|---|
| Impression positiva (luego click) | ~100 |
| Impression negativa (sin view) | ~500 |
| View positiva (luego click) | ~100 |
| View negativa (sin click) | ~35 |
| Click | ~100 |
| Rating (dato real) | 100 |
| **Total** | **~835 eventos** |

Con 138K usuarios y ~20M ratings → **~150-170M filas de eventos**.

---

## 3. Qué predice el modelo

### Predicción concreta

**`P(click | user, movie, contexto)`** — la probabilidad de que un usuario concreto haga click en una película concreta, dado el estado actual de ambos.

Es un modelo de **scoring pairwise**: recibe un par (usuario, película) y devuelve un float entre 0 y 1. No genera una lista directamente.

### Cómo se generan las recomendaciones top-5

1. **Candidate generation:** se filtra a 200-500 candidatas (por popularidad, género afín, collaborative filtering simple).
2. **Scoring:** se puntúa cada candidata con el modelo ONNX.
3. **Ranking:** se ordenan por score descendente y se devuelven las top-5.

### Cuándo se predice

Cada vez que el usuario "vuelve a la home" (no solo al iniciar sesión). Esto es lo que justifica el stack de streaming: las features del usuario cambian **dentro de la sesión** con cada interacción, y las recomendaciones se recalculan en tiempo real.

```
Usuario entra → API devuelve top-5 con features actuales
    → usuario clicka peli de terror → evento a RedPanda
    → PyFlink actualiza features en Redis
    → usuario vuelve a la home → API devuelve top-5 DIFERENTE
      (porque genre_affinity_last_1h cambió)
```

### Arquitectura del modelo

**Two-tower (dual encoder):**
- **Torre de usuario:** embedding de user_id + features de comportamiento reciente (calculadas por PyFlink)
- **Torre de película:** embedding de movie_id + género + año + popularidad reciente
- **Score:** dot product entre ambos vectores

Entrenamiento con binary cross-entropy.

---

## 4. Los dos simuladores

### Simulador 1: Generador de datos históricos (offline)

- Se ejecuta **una sola vez**.
- Toma los 20M de ratings de MovieLens y genera la tabla de eventos descrita arriba.
- Es un script batch de procesamiento. No interactúa con la infraestructura.
- **Propósito:** crear los datos para entrenar el modelo v1.

### Simulador 2: Generador de eventos en tiempo real

- Se ejecuta **continuamente** cuando toda la infraestructura está desplegada.
- Emite eventos a RedPanda simulando usuarios navegando la plataforma.
- **Interactúa con el modelo en vivo:** pide recomendaciones a la API, "decide" si el usuario hace click, genera eventos consecuentes.
- Modela comportamiento completo: inicio de sesión, duración, ciclos de vuelta a la home, fin de sesión.
- Crea un **feedback loop** donde el modelo influye en los datos que recibe después.

### Orden de implementación

```
1. Descargar MovieLens 20M
2. Simulador 1: generar tabla de eventos históricos
3. Feature engineering sobre esos eventos
4. Entrenar DNN two-tower → ONNX → MLflow (modelo v1)
5. Montar RedPanda + PyFlink + Redis
6. Montar FastAPI con modelo ONNX
7. Montar Prometheus + Grafana
8. Simulador 2: eventos en tiempo real contra la infraestructura
```

---

## 5. Validación del modelo

### Offline (antes de desplegar)

Split temporal (nunca aleatorio): entrenar con eventos hasta día X, testear con día X+1 en adelante.

| Métrica | Qué mide |
|---|---|
| AUC-ROC | ¿Distingue clicks de no-clicks? Métrica principal de calidad |
| NDCG@5 | ¿Las películas clickadas están arriba del ranking? |
| Precision@5 | De las 5 recomendadas, ¿cuántas reciben click? |
| Recall@5 | De las que habría clickado, ¿cuántas están en el top-5? |

### Online (con el sistema corriendo)

**Métricas de negocio** (expuestas en Prometheus, visualizadas en Grafana):

| Métrica | Qué mide |
|---|---|
| CTR | clicks / impresiones servidas (rolling 1h, 24h). Métrica reina |
| CTR@1 | Frecuencia de click en la primera recomendación |
| Películas únicas recomendadas | Diversidad del catálogo servido |
| Rating medio post-click | ¿Las recomendaciones satisfacen o son clickbait? |

**Métricas técnicas / de drift:**

| Métrica | Qué mide |
|---|---|
| Distribución de scores | Detectar colapso (todos en 0.9 o todos en 0.01) |
| Feature drift | Comparar distribución actual vs. entrenamiento |
| Latencia de inferencia | P50, P95, P99 del endpoint |
| Feature freshness | ¿Redis tiene datos actualizados o PyFlink se atrasó? |

---

## 6. Loop de reentrenamiento

### Datos que se almacenan para reentrenar

| Tabla | Dónde | Para qué |
|---|---|---|
| **Eventos crudos** | GCS/BigQuery (persistente) | Fuente de verdad. Labels reales para reentrenar. PyFlink, además de escribir en Redis, persiste cada evento |
| **Log de inferencia** | GCS/BigQuery (persistente) | Qué recomendó el modelo, con qué score, en qué posición, con qué versión |
| **Features actuales** | Redis (volátil) | Serving en tiempo real |
| **Snapshots de features** | GCS/BigQuery (periódico) | Reconstruir estado de features en el momento de la predicción sin recalcular |
| **Métricas** | Prometheus → Grafana | Monitoreo online |

### Cómo se construye el dataset de reentrenamiento

Se cruzan el log de inferencia con los eventos crudos:

```
Log: "al user 123 le recomendé movie 450 en posición 2, score 0.82, modelo v3"
Eventos: "user 123 hizo click en movie 450"
→ label = 1, el modelo acertó

Log: "al user 123 le recomendé movie 800 en posición 1, score 0.91, modelo v3"
Eventos: no hay click de user 123 en movie 800
→ label = 0, el modelo falló
```

### Ciclo

```
1. Join log de inferencia + eventos → dataset con labels reales
2. Calcular features (desde snapshots o recalcular desde eventos)
3. Entrenar modelo v(N+1)
4. Evaluar offline: ¿mejora AUC/NDCG vs modelo v(N)?
5. Si sí → registrar en MLflow, promover a producción
6. FastAPI carga el nuevo modelo ONNX
7. Monitorear métricas online: ¿sube el CTR?
8. Repetir
```

---

## 7. Arquitectura de datos completa

```
OFFLINE (una vez):
MovieLens 20M → Simulador 1 → Tabla de eventos históricos → Features → Modelo v1

PRODUCCIÓN (continuo):
Simulador 2 → RedPanda → PyFlink ──→ Redis (features online)
                │                        │
                │                   FastAPI + ONNX → Recomendaciones
                │                        │
                └──→ GCS/BQ              └──→ Log de inferencia (GCS/BQ)
              (eventos persistidos)            │
                     │                         │
                     └──── Join ───────────────┘
                              │
                     Dataset de reentrenamiento
                              │
                     Entrena modelo v(N+1) → MLflow → FastAPI actualiza modelo
```
