# =============================================================================
# ClinkerForecast Core — Multi-stage Docker Build
# =============================================================================
# Stage 1: Builder
FROM gcc:13-bookworm AS builder

WORKDIR /build

# Копируем весь проект
COPY . .

# Сборка
RUN make clean && make all -j$(nproc)

# Stage 2: Runtime (минимальный образ)
FROM debian:bookworm-slim AS runtime

LABEL maintainer="ClinkerForecast Team"
LABEL description="Neural forecasting engine for cement clinker quality"
LABEL version="1.0.0"

# Установка runtime зависимостей
RUN apt-get update && apt-get install -y --no-install-recommends     libstdc++6     && rm -rf /var/lib/apt/lists/*

# Создаем непривилегированного пользователя
RUN groupadd -r clinker && useradd -r -g clinker clinker

# Копируем артефакты из builder
COPY --from=builder /build/build/bin/clinker_forecast /usr/local/bin/
COPY --from=builder /build/build/lib/libclinker_forecast.a /usr/local/lib/
COPY --from=builder /build/core/include/clinker_forecast.h /usr/local/include/
COPY --from=builder /build/core/include/ops.h /usr/local/include/
COPY --from=builder /build/core/weights/model_v1.bin /opt/clinker/models/

# Права
RUN chown -R clinker:clinker /opt/clinker

USER clinker
WORKDIR /opt/clinker

# Healthcheck
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3     CMD /usr/local/bin/clinker_forecast -m /opt/clinker/models/model_v1.bin -i || exit 1

# Entrypoint
ENTRYPOINT ["/usr/local/bin/clinker_forecast"]
CMD ["-m", "/opt/clinker/models/model_v1.bin", "-H", "24"]