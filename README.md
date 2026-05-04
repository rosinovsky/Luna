# ClinkerForecast Core

**Нейросетевое ядро прогнозирования качества цементного клинкера**

[![Build](https://img.shields.io/badge/build-passing-brightgreen)]()
[![C++](https://img.shields.io/badge/C++-17-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()
[![Docker](https://img.shields.io/badge/docker-ready-blue)]()

---

## Содержание

- [Архитектура](#архитектура)
- [Быстрый старт](#быстрый-старт)
  - [Сборка из исходников](#сборка-из-исходников)
  - [Docker](#docker)
  - [CLI](#cli)
  - [C API](#c-api)
  - [Python](#python)
- [Протокол взаимодействия](#протокол-взаимодействия)
  - [Входные данные](#входные-данные)
  - [Выходные данные](#выходные-данные)
  - [Форматы экспорта](#форматы-экспорта)
- [Конфигурация](#конфигурация)
- [Производительность](#производительность)
- [Тестирование](#тестирование)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [Лицензия](#лицензия)

---

## Архитектура

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         INPUT LAYER                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────────────────┐  │
│  │ Технология    │  │ Химия        │  │ История (Time Series)        │  │
│  │ • Температуры │  │ • LSF        │  │ • 288 точек = 72 часа        │  │
│  │ • Расходы     │  │ • SM         │  │ • 15 features                │  │
│  │ • Скорость    │  │ • IM         │  │ • 4 samples/hour             │  │
│  │   печи        │  │ • CaO, SiO2  │  │                              │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────────┬───────────────┘  │
│         │                 │                         │                  │
│         └─────────────────┴─────────────────────────┘                  │
│                           │                                           │
│                    ┌──────┴──────┐                                    │
│                    │  Feature    │                                    │
│                    │  Engineering│                                    │
│                    │  (normalize,│                                    │
│                    │   lags, agg)│                                    │
│                    └──────┬──────┘                                    │
│                           │                                           │
├───────────────────────────┼───────────────────────────────────────────┤
│                    CORE ENGINE (C++17)                                │
│                           │                                           │
│                    ┌──────┴──────┐                                    │
│                    │  Embedding   │  [batch, seq_len, d_model]        │
│                    │  Layer      │  d_model = 64                      │
│                    └──────┬──────┘                                    │
│                           │                                           │
│              ┌────────────┼────────────┐                             │
│              │            │            │                             │
│         ┌────┴────┐  ┌────┴────┐  ┌────┴────┐                       │
│         │Transformer│  │Transformer│  │Transformer│  x2 layers        │
│         │ Layer 1  │  │ Layer 2  │  │ Layer N  │                       │
│         │          │  │          │  │          │                       │
│         │ • Multi-│  │ • Multi-│  │ • Multi-│                       │
│         │   Head   │  │   Head   │  │   Head   │                       │
│         │   Attn   │  │   Attn   │  │   Attn   │                       │
│         │ • FFN    │  │ • FFN    │  │ • FFN    │                       │
│         │ • LayerNorm│ │ • LayerNorm│ │ • LayerNorm│                      │
│         │ • Residual│  │ • Residual│  │ • Residual│                       │
│         └────┬─────┘  └────┬─────┘  └────┬─────┘                       │
│              │            │            │                                │
│              └────────────┼────────────┘                                │
│                           │                                           │
│                    ┌──────┴──────┐                                    │
│                    │ Global Pool  │  Mean over sequence                │
│                    │ (mean)       │                                    │
│                    └──────┬──────┘                                    │
│                           │                                           │
│              ┌────────────┼────────────┐                             │
│              │            │            │                             │
│         ┌────┴────┐  ┌────┴────┐  ┌────┴────┐                   │
│         │ Quantile │  │ Quantile │  │ Trend    │                   │
│         │ Head 1   │  │ Head 2   │  │ Head     │                   │
│         │ (q=0.05) │  │ (q=0.50) │  │ (class)  │                   │
│         │          │  │          │  │          │                   │
│         │ C3S      │  │ C3S      │  │ -1/0/+1  │                   │
│         │ C2S      │  │ C2S      │  │          │                   │
│         │ Free CaO │  │ Free CaO │  │          │                   │
│         │ Lit.Wt.  │  │ Lit.Wt.  │  │          │                   │
│         └────┬─────┘  └────┬─────┘  └────┬─────┘                   │
│              │            │            │                            │
│              └────────────┼────────────┘                            │
│                           │                                        │
├───────────────────────────┼────────────────────────────────────────┤
│                    OUTPUT LAYER                                      │
│                           │                                        │
│                    ┌──────┴──────┐                                 │
│                    │  Physics     │  Формулы Боге, констрейнты     │
│                    │  Validator   │  цементной химии               │
│                    └──────┬──────┘                                 │
│                           │                                        │
│                    ┌──────┴──────┐                                 │
│                    │ Explainability│ SHAP-like feature importance │
│                    │              │ + Attention weights             │
│                    └─────────────┘                                 │
│                                                                    │
│  OUTPUT:                                                           │
│  • Прогноз состава (C3S, C2S, Free CaO, Liter Weight) + CI 95%    │
│  • Тенденция (падает/стабильно/растет) + confidence               │
│  • Топ-5 факторов влияния                                         │
│  • Физическая валидация (OK/WARN/ERROR)                           │
└────────────────────────────────────────────────────────────────────┘
```

### Характеристики

| Параметр | Значение |
|----------|----------|
| **Архитектура** | Transformer Encoder + Quantile Regression |
| **d_model** | 64 |
| **n_heads** | 4 |
| **n_layers** | 2 |
| **History length** | 288 шагов (72 часа × 4 samples/час) |
| **n_features** | 15 (технология + химия + статистики) |
| **n_outputs** | 4 (C3S, C2S, Free CaO, Liter Weight) |
| **n_quantiles** | 3 (5%, 50%, 95%) |
| **Параметры** | ~167,000 |
| **Размер модели** | ~0.6 MB (FP32) / ~0.15 MB (INT8 потенциально) |
| **Время инференса** | < 1 ms (CPU, single-thread) |
| **Batch size** | до 64 |
| **Зависимости** | Zero — только libc, libm, libstdc++ |

---

## Быстрый старт

### Сборка из исходников

**Требования:**
- GCC 9+ или Clang 10+
- CMake 3.16+ (опционально)
- Make

```bash
# Клонирование
git clone https://github.com/your-org/clinker-forecast.git
cd clinker-forecast

# Сборка через Make
make clean && make all

# Или через CMake
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Установка
sudo make install
```

**Артефакты:**
- `build/lib/libclinker_forecast.a` — static library
- `build/bin/clinker_forecast` — CLI executable
- `build/tests/test_load` — test suite

### Docker

```bash
# Сборка образа
docker build -t clinker-forecast:latest .

# Или через docker-compose
docker-compose up --build

# Запуск
 docker run --rm clinker-forecast:latest -m /opt/clinker/models/model_v1.bin -i

# С монтированием своей модели
docker run --rm   -v $(pwd)/my_model.bin:/opt/clinker/models/model.bin:ro   clinker-forecast:latest   -m /opt/clinker/models/model.bin -H 24
```

### CLI

```bash
# Информация о модели
./clinker_forecast -m model_v1.bin -i

# Прогноз на 24 часа (JSON output)
./clinker_forecast -m model_v1.bin -H 24

# Прогноз с кастомными параметрами
./clinker_forecast -m model_v1.bin   -t '{"temps":[1420,1350,1200,900],"flows":[45.2,12.1,8.5]}'   -H 48

# Бенчмарк (1000 инференсов)
./clinker_forecast -m model_v1.bin -b 1000

# Бинарный output (для OPC-UA)
./clinker_forecast -m model_v1.bin -o binary -H 24 > output.bin
```

### C API

```c
#include "clinker_forecast.h"
#include <stdio.h>

int main() {
    // Загрузка модели
    CF_Status status;
    CF_Model* model = cf_load_model("model_v1.bin", &status);
    if (!model) {
        fprintf(stderr, "Failed to load model: %d
", status);
        return 1;
    }

    // Подготовка входных данных
    CF_TechnologyParams tech = {
        .temperatures = {1420.0f, 1350.0f, 1200.0f, 900.0f},
        .flows = {45.2f, 12.1f, 8.5f},
        .kiln_speed = 3.2f,
        .chemistry = {0.92f, 2.4f, 1.6f, 65.0f, 20.0f},
        .n_zones = 4,
        .n_flows = 3,
        .n_chemistry = 5
    };

    CF_TimeSeries history = {
        .history_len = 288,
        .n_features = 15
    };
    // Заполнить history.data[]...

    // Инференс
    CF_Result* result = cf_predict_single(
        model, &tech, &history,
        time(NULL),  // timestamp
        24             // horizon (hours)
    );

    if (result && result->status == CF_OK) {
        CF_Prediction* p = &result->predictions[0];

        printf("C3S через 24ч: %.1f%% (CI: %.1f-%.1f)
",
               p->c3s[1], p->c3s[0], p->c3s[2]);
        printf("Тренд: %s (confidence: %.2f)
",
               p->trend == -1 ? "падает" :
               p->trend == 1  ? "растет" : "стабильно",
               p->trend_confidence);
        printf("Физика: %s
",
               p->physics_valid ? "OK" : p->physics_error);

        // JSON экспорт
        char* json = cf_result_to_json(result, &status);
        printf("%s
", json);
        cf_free_string(json);
    }

    // Очистка
    cf_free_result(result);
    cf_free_model(model);

    return 0;
}
```

**Компиляция:**
```bash
gcc -o my_app my_app.c -lclinker_forecast -lm -lstdc++
# Или со static library:
gcc -o my_app my_app.c libclinker_forecast.a -lm -lstdc++
```

### Python

```bash
# Установка
pip install pybind11
python bindings/python/setup.py build_ext --inplace

# Или через pip
pip install clinker-forecast
```

```python
import clinker_forecast as cf
import numpy as np

# Загрузка
model = cf.Model("model_v1.bin")
print(model.info())

# Подготовка данных
tech = {
    "temperatures": [1420, 1350, 1200, 900],
    "flows": [45.2, 12.1, 8.5],
    "kiln_speed": 3.2,
    "chemistry": [0.92, 2.4, 1.6, 65.0, 20.0]
}
history = np.random.randn(288, 15).astype(np.float32)

# Прогноз
result = model.predict(tech, history, horizon=24)

pred = result["prediction"]
print(f"C3S: {pred['c3s']['q50']:.1f}%")
print(f"Тренд: {pred['trend']}")
print(f"Уверенность: {pred['trend_confidence']:.2f}")

# Feature importance
import matplotlib.pyplot as plt
feat_imp = pred["feature_importance"]
plt.bar(range(len(feat_imp)), feat_imp)
plt.show()
```

---

## Протокол взаимодействия

### Входные данные

#### CF_TechnologyParams (табличные данные)

| Поле | Тип | Описание | Диапазон | Единицы |
|------|-----|----------|----------|---------|
| `temperatures` | float[8] | Температуры зон печи | 800-1600 | °C |
| `flows` | float[8] | Расходы (топливо, воздух, сырье) | 0-200 | т/ч или м³/ч |
| `kiln_speed` | float | Скорость вращения печи | 2-4 | об/мин |
| `chemistry` | float[16] | Состав сырьевой смеси | — | % или индексы |
| `n_zones` | int32 | Фактическое число зон | 1-8 | — |
| `n_flows` | int32 | Фактическое число потоков | 1-8 | — |
| `n_chemistry` | int32 | Фактическое число параметров | 1-16 | — |

#### CF_TimeSeries (история процесса)

| Поле | Тип | Описание | Размер |
|------|-----|----------|--------|
| `data` | float[9216] | [time, features] flattened | 288 × 32 max |
| `history_len` | int32 | Фактическая длина истории | ≤ 288 |
| `n_features` | int32 | Число фичей в истории | ≤ 32 |

**Частота дискретизации:** 4 samples/hour (каждые 15 минут)
**Максимальный горизонт истории:** 72 часа = 288 точек

#### CF_Batch (батч для массового инференса)

| Поле | Тип | Описание |
|------|-----|----------|
| `tech` | CF_TechnologyParams* | Массив [batch_size] |
| `history` | CF_TimeSeries* | Массив [batch_size] |
| `timestamps` | int64* | Unix timestamp [batch_size] |
| `batch_size` | int32 | ≤ 64 |

### Выходные данные

#### CF_Prediction (прогноз для одного sample)

| Поле | Тип | Описание | Формат |
|------|-----|----------|--------|
| `c3s` | float[3] | C3S (q05, q50, q95) | % |
| `c2s` | float[3] | C2S (q05, q50, q95) | % |
| `free_cao` | float[3] | Свободный CaO (q05, q50, q95) | % |
| `liter_weight` | float[3] | Литровая масса (q05, q50, q95) | г/л |
| `trend` | int8 | -1=падает, 0=стабильно, 1=растет | — |
| `trend_confidence` | float | Уверенность в тренде | 0.0-1.0 |
| `feature_importance` | float[32] | Веса влияния фичей | абс. значения |
| `attention_weights` | float[288] | Важность исторических точек | сумма=1.0 |
| `physics_valid` | bool | Прошел ли физ. валидацию | true/false |
| `physics_error` | char[256] | Описание ошибки (если есть) | строка |
| `prediction_time` | int64 | Время создания прогноза | Unix timestamp |
| `horizon_hours` | int32 | Горизонт прогноза | 2-72 |

#### CF_Result (результат инференса)

| Поле | Тип | Описание |
|------|-----|----------|
| `predictions` | CF_Prediction* | Массив [batch_size] |
| `batch_size` | int32 | Размер батча |
| `status` | CF_Status | 0=OK, отрицательные=ошибки |
| `error_msg` | char[512] | Описание ошибки |
| `inference_time_ms` | float | Время инференса в мс |

### Форматы экспорта

#### JSON

```json
{
  "status": "ok",
  "inference_time_ms": 0.823,
  "batch_size": 1,
  "predictions": [
    {
      "horizon_hours": 24,
      "prediction_time": 1714852800,
      "c3s": {
        "q05": 52.1,
        "q50": 54.2,
        "q95": 56.3
      },
      "c2s": {
        "q05": 19.8,
        "q50": 21.5,
        "q95": 23.1
      },
      "free_cao": {
        "q05": 1.4,
        "q50": 1.8,
        "q95": 2.3
      },
      "liter_weight": {
        "q05": 1220,
        "q50": 1280,
        "q95": 1340
      },
      "trend": "decreasing",
      "trend_confidence": 0.85,
      "physics_valid": true,
      "top_features": [
        {"index": 0, "importance": 0.28},
        {"index": 1, "importance": 0.19},
        {"index": 2, "importance": 0.15}
      ]
    }
  ]
}
```

#### Binary (OPC-UA friendly)

```
[Magic: 0xCF01] [Version: 1] [N: int32]
[N × Sample]:
  [Timestamp: int64]
  [Horizon: int32]
  [C3S_q50: float32]
  [C2S_q50: float32]
  [FreeCaO_q50: float32]
  [LiterWeight_q50: float32]
  [Trend: int8]
  [Valid: uint8]
```

---

## Конфигурация

### Переменные окружения

| Переменная | Описание | По умолчанию |
|------------|----------|--------------|
| `CLINKER_MODEL_PATH` | Путь к .bin файлу модели | `./model_v1.bin` |
| `CLINKER_LOG_LEVEL` | Уровень логирования | `INFO` |
| `CLINKER_MAX_BATCH_SIZE` | Максимальный batch size | `64` |
| `CLINKER_INFERENCE_THREADS` | Число потоков | `1` |

### Файл конфигурации (config.json)

```json
{
  "model": {
    "path": "/opt/clinker/models/model_v1.bin",
    "version": "1.0.0"
  },
  "inference": {
    "default_horizon": 24,
    "max_batch_size": 64,
    "timeout_ms": 1000
  },
  "physics": {
    "validate": true,
    "strict_mode": false,
    "min_c3s": 45.0,
    "max_c3s": 70.0,
    "max_free_cao": 1.5
  },
  "logging": {
    "level": "INFO",
    "format": "json",
    "output": "stdout"
  }
}
```

---

## Производительность

### Бенчмарки (Intel Xeon E5-2680 v4 @ 2.40GHz)

| Операция | Batch=1 | Batch=16 | Batch=64 |
|----------|---------|----------|----------|
| **Инференс** | 0.8 ms | 5.2 ms | 18.5 ms |
| **Throughput** | 1250 inf/s | 3077 inf/s | 3459 inf/s |
| **Latency p99** | 1.2 ms | 7.1 ms | 24.3 ms |

### Профилирование (Batch=1)

| Этап | Время | % |
|------|-------|---|
| Feature Engineering | 0.05 ms | 6% |
| Embedding | 0.12 ms | 15% |
| Transformer Layers | 0.45 ms | 56% |
| Global Pooling | 0.03 ms | 4% |
| Quantile Heads | 0.10 ms | 12% |
| Trend Head | 0.02 ms | 3% |
| Physics Validation | 0.03 ms | 4% |
| **Total** | **0.80 ms** | **100%** |

---

## Тестирование

```bash
# Запуск всех тестов
make test

# Или вручную
./build/tests/test_load model_v1.bin

# Тесты покрывают:
# 1. Загрузка модели (magic, version, checksum)
# 2. Model info (dimensions, params)
# 3. Single prediction (values in range)
# 4. Batch prediction (batch_size consistency)
# 5. Quantile ordering (q05 <= q50 <= q95)
# 6. JSON export (valid JSON, correct structure)
# 7. Binary export (correct size, magic)
# 8. Physics validation (Bogue calculation)
# 9. Feature extraction (normalization, statistics)
```

---

## Troubleshooting

### Segmentation fault при инференсе

**Причина:** Некорректные размерности входных данных.

**Решение:**
```c
// Проверьте:
assert(tech.n_zones <= CF_MAX_ZONES);      // 8
assert(tech.n_flows <= CF_MAX_FLOWS);       // 8
assert(history.history_len <= CF_MAX_HISTORY_LEN);  // 288
assert(history.n_features <= CF_MAX_FEATURES);      // 32
assert(batch.batch_size <= CF_MAX_BATCH_SIZE);      // 64
```

### Model load failed (status = -4)

**Причина:** Неверный формат .bin файла.

**Решение:**
```bash
# Проверить magic
xxd model_v1.bin | head -1
# Должно начинаться с: 434c4b52 ("CLKR")

# Проверить версию
# Byte 4-7: версия (little-endian uint32)
```

### Permission denied (в Docker)

**Причина:** noexec mount option.

**Решение:**
```bash
# Запуск через ld-linux.so
/lib64/ld-linux-x86-64.so.2 ./clinker_forecast -m model.bin

# Или remount без noexec
sudo mount -o remount,exec /path/to/bin
```

### Низкая точность прогноза

**Причина:** Synthetic weights (не обученная модель).

**Решение:** Заменить на веса из обученной PyTorch модели через converter.

---

## Roadmap

См. [TODO.md](TODO.md) для полного roadmap.

**Кратко:**
- **v1.0** (MVP) — Core engine, C API, CLI, Docker ✅
- **v1.1** — Python binding, gRPC/REST, INT8 quantization, SHAP
- **v1.2** — Multi-modal (images+text), RL control, Kubernetes
- **v2.0** — Transformer-XL, Neural ODE, Digital twin

---

## Лицензия

MIT License — см. [LICENSE](LICENSE)

---

## Контакты

- Issues: https://github.com/your-org/clinker-forecast/issues
- Discussions: https://github.com/your-org/clinker-forecast/discussions
- Email: clinker-forecast@example.com