
#include "clinker_forecast.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    printf("Debug test starting...
");

    const char* model_path = argv[1];
    printf("Loading model: %s
", model_path);

    CF_Status status;
    CF_Model* model = cf_load_model(model_path, &status);
    if (!model) {
        printf("Failed to load model: %d
", status);
        return 1;
    }
    printf("Model loaded OK
");

    CF_ModelInfo info = cf_get_model_info(model);
    printf("Model: d_model=%d, params=%ld
", info.d_model, info.total_params);

    CF_TechnologyParams tech;
    memset(&tech, 0, sizeof(tech));
    tech.n_zones = 4;
    tech.n_flows = 3;
    tech.n_chemistry = 5;
    tech.temperatures[0] = 1420.0f;
    tech.temperatures[1] = 1350.0f;
    tech.temperatures[2] = 1200.0f;
    tech.temperatures[3] = 900.0f;
    tech.flows[0] = 45.2f;
    tech.flows[1] = 12.1f;
    tech.flows[2] = 8.5f;
    tech.kiln_speed = 3.2f;
    tech.chemistry[0] = 0.92f;
    tech.chemistry[1] = 2.4f;
    tech.chemistry[2] = 1.6f;
    tech.chemistry[3] = 65.0f;
    tech.chemistry[4] = 20.0f;

    CF_TimeSeries history;
    memset(&history, 0, sizeof(history));
    history.history_len = 288;
    history.n_features = 15;
    for (int i = 0; i < history.history_len * history.n_features; i++) {
        history.data[i] = 0.5f;
    }

    printf("Running predict_single...
");
    CF_Result* result = cf_predict_single(model, &tech, &history, 1234567890, 24);

    if (!result) {
        printf("predict_single returned NULL
");
        cf_free_model(model);
        return 1;
    }

    printf("Result status: %d
", result->status);
    printf("Batch size: %d
", result->batch_size);
    printf("Inference time: %.3f ms
", result->inference_time_ms);

    if (result->batch_size > 0) {
        CF_Prediction* p = &result->predictions[0];
        printf("C3S: q05=%.2f q50=%.2f q95=%.2f
", p->c3s[0], p->c3s[1], p->c3s[2]);
        printf("C2S: q05=%.2f q50=%.2f q95=%.2f
", p->c2s[0], p->c2s[1], p->c2s[2]);
        printf("Free CaO: q05=%.2f q50=%.2f q95=%.2f
", p->free_cao[0], p->free_cao[1], p->free_cao[2]);
        printf("Liter Weight: q05=%.0f q50=%.0f q95=%.0f
", p->liter_weight[0], p->liter_weight[1], p->liter_weight[2]);
        printf("Trend: %d (confidence: %.2f)
", p->trend, p->trend_confidence);
        printf("Physics valid: %s
", p->physics_valid ? "YES" : "NO");
    }

    cf_free_result(result);
    cf_free_model(model);

    printf("Debug test completed.
");
    return 0;
}
