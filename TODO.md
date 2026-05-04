# ClinkerForecast Core — Roadmap & TODO

## 🔴 Critical (v1.0 — Production Ready)

### Модель
- [ ] Заменить synthetic weights на веса из обученной PyTorch модели
  - [ ] Создать converter: PyTorch state_dict -> custom .bin format
  - [ ] Валидация весов после конвертации (checksum, dimension check)
  - [ ] Тест на реальных данных (3-6 месяцев истории)
- [ ] Реализовать полноценный LSTM encoder для временных рядов
  - [ ] Bidirectional LSTM с skip connections
  - [ ] LayerNorm между LSTM слоями
- [ ] Добавить causal masking в Attention (для real-time streaming)
- [ ] Реализовать positional encoding (sinusoidal / learned)

### Инфраструктура
- [ ] CI/CD pipeline (GitHub Actions / GitLab CI)
  - [ ] Сборка под x86_64, ARM64, ARMv7
  - [ ] Автоматические тесты на каждый PR
  - [ ] Статический анализ (clang-tidy, cppcheck)
  - [ ] Fuzz testing входных данных
- [ ] Docker image с multi-stage build ✅
  - [ ] Оптимизировать размер образа (< 50 MB)
  - [ ] Distroless вариант для production
- [ ] Пакеты: .deb, .rpm, Conan, vcpkg

### API & Интеграция
- [ ] gRPC сервис (protobuf API)
  - [ ] Streaming inference (Server-Sent Events)
  - [ ] Batch inference endpoint
- [ ] REST API (OpenAPI 3.0)
  - [ ] /predict (single & batch)
  - [ ] /health (liveness & readiness)
  - [ ] /metrics (Prometheus format)
- [ ] OPC-UA интеграция (open62541)
  - [ ] Чтение данных из SCADA в real-time
  - [ ] Запись прогнозов обратно в SCADA
- [ ] MQTT брокер для edge-устройств

### Производительность
- [ ] INT8 квантизация весов
  - [ ] Post-training quantization (PTQ)
  - [ ] Quantization-aware training (QAT)
- [ ] SIMD оптимизации
  - [ ] AVX2 / AVX-512 для x86_64
  - [ ] NEON для ARM64
  - [ ] SVE для ARMv9
- [ ] OpenMP / TBB параллелизм
  - [ ] Parallel batch inference
  - [ ] Thread pool для feature engineering
- [ ] GPU backend (optional)
  - [ ] CUDA kernels для Attention
  - [ ] TensorRT optimization

## 🟡 High Priority (v1.1 — Enhanced)

### Explainability
- [ ] SHAP values (exact, not approximation)
  - [ ] KernelSHAP для табличных данных
  - [ ] DeepSHAP для нейросетевых слоев
- [ ] Attention visualization
  - [ ] Heatmap по временным шагам
  - [ ] Cross-attention между модальностями
- [ ] Counterfactual explanations
  - [ ] "Что если увеличить температуру на 10°?"

### Мониторинг
- [ ] Model drift detection
  - [ ] Data drift (PSI, KS-test)
  - [ ] Concept drift (accuracy degradation)
  - [ ] Feature drift (individual features)
- [ ] Автоматическое переобучение
  - [ ] Триггер по drift + human-in-the-loop
  - [ ] A/B тестирование новых версий
- [ ] Логирование предсказаний
  - [ ] Structured logging (JSON)
  - [ ] Интеграция с ELK / Grafana Loki

### Безопасность
- [ ] Model encryption at rest
  - [ ] AES-256 для .bin файлов
  - [ ] Hardware-backed keys (TPM/HSM)
- [ ] Input validation hardening
  - [ ] Санитизация всех входных данных
  - [ ] Rate limiting
- [ ] Аудит доступа
  - [ ] Логирование всех API вызовов
  - [ ] RBAC (Role-Based Access Control)

## 🟢 Medium Priority (v1.2 — Advanced)

### Мультимодальность
- [ ] Обработка изображений (термограммы печи)
  - [ ] CNN backbone (ResNet18/EfficientNet)
  - [ ] ROI extraction (зоны печи)
- [ ] Обработка текстов (журнал оператора)
  - [ ] NER для извлечения параметров
  - [ ] Sentiment analysis (аварийные ситуации)
- [ ] Геоданные (Graph Neural Networks)
  - [ ] Моделирование зон печи как графа

### Оптимизация процесса
- [ ] Reinforcement Learning для control
  - [ ] MPC (Model Predictive Control) интеграция
  - [ ] Автоматическая коррекция параметров
- [ ] Multi-objective optimization
  - [ ] Баланс качества vs энергопотребление
  - [ ] Pareto frontier для оператора

### Масштабирование
- [ ] Kubernetes deployment
  - [ ] Helm chart
  - [ ] HPA (Horizontal Pod Autoscaler)
  - [ ] GPU node selector
- [ ] Edge deployment
  - [ ] NVIDIA Jetson (ARM64 + CUDA)
  - [ ] Raspberry Pi 4 (ARMv7)
  - [ ] PLC интеграция (CODESYS)

## 🔵 Low Priority (v2.0 — Research)

### Архитектура
- [ ] Transformer-XL (длинные последовательности)
- [ ] Performer / Linformer (sub-quadratic attention)
- [ ] Neural ODE для физически-информированного моделирования
- [ ] Bayesian Neural Networks для uncertainty

### Новые задачи
- [ ] Прогнозирование отказов оборудования
- [ ] Оптимизация расхода энергии
- [ ] Управление выбросами CO2
- [ ] Цифровой двойник печи обжига

## 📋 Чек-лист релиза

### v1.0.0 (MVP)
- [x] Core engine (C++17)
- [x] C API
- [x] CLI утилита
- [x] Synthetic weights
- [x] Physics validation
- [x] Basic explainability
- [x] Docker image
- [ ] Реальные веса из PyTorch
- [ ] CI/CD
- [ ] Документация API

### v1.1.0 (Enhanced)
- [ ] Python binding
- [ ] gRPC/REST API
- [ ] INT8 quantization
- [ ] SHAP explainability
- [ ] Drift detection

### v1.2.0 (Advanced)
- [ ] Multi-modal (images + text)
- [ ] RL-based control
- [ ] Kubernetes deployment
- [ ] Edge inference

### v2.0.0 (Research)
- [ ] Transformer-XL
- [ ] Neural ODE
- [ ] Digital twin
- [ ] CO2 optimization