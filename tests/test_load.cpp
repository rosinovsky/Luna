#include "clinker_forecast.h"
#include <cstdio>
#include <cstring>
#include <cmath>

static bool float_eq(float a, float b, float eps = 0.01f) {
    return std::fabs(a - b) < eps;
}

int main(int argc, char** argv) {
    printf("ClinkerForecast Test Suite v1.0.0\n");
    printf("==================================\n\n");

    // Test 1: Version
    printf("[TEST 1] Library version... ");
    const char* ver = cf_version();
    if (ver && strlen(ver) > 0) {
        printf("PASS (%s)\n", ver);
    } else {
        printf("FAIL\n");
        return 1;
    }

    // Test 2: Load model
    printf("[TEST 2] Load model... ");
    const char* model_path = (argc > 1) ? argv[1] : "core/weights/model_v1.bin";
    CF_Status status;
    CF_Model* model = cf_load_model(model_path, &status);
    if (!model) {
        printf("FAIL (status=%d, path=%s)\n", status, model_path);
        return 1;
    }
    printf("PASS\n");

    // Test 3: Model info
    printf("[TEST 3] Model info... ");
    CF_ModelInfo info = cf_get_model_info(model);
    if (info.d_model > 0 && info.total_params > 0) {
        printf("PASS (params=%ld, size=%.2f MB)\n", 
               info.total_params, info.model_size_bytes / (1024.0f*1024.0f));
    } else {
        printf("FAIL\n");
        cf_free_model(model);
        return 1;
    }

    // Test 4: Single prediction
    printf("[TEST 4] Single prediction... ");
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
        history.data[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    CF_Result* result = cf_predict_single(model, &tech, &history, 1234567890, 24);
    if (!result || result->status != CF_OK) {
        printf("FAIL (status=%d)\n", result ? result->status : -1);
        cf_free_model(model);
        return 1;
    }

    // Check prediction values
    CF_Prediction& pred = result->predictions[0];
    bool valid = true;
    valid &= (pred.c3s[1] > 0.0f && pred.c3s[1] < 100.0f);
    valid &= (pred.c2s[1] > 0.0f && pred.c2s[1] < 100.0f);
    valid &= (pred.free_cao[1] >= 0.0f && pred.free_cao[1] < 10.0f);
    valid &= (pred.liter_weight[1] > 1000.0f && pred.liter_weight[1] < 2000.0f);
    valid &= (pred.horizon_hours == 24);

    if (valid) {
        printf("PASS (C3S=%.1f%%, C2S=%.1f%%, CaO=%.2f%%, Lit=%.0f, time=%.2fms)\n",
               pred.c3s[1], pred.c2s[1], pred.free_cao[1], 
               pred.liter_weight[1], result->inference_time_ms);
    } else {
        printf("FAIL (invalid values)\n");
        cf_free_result(result);
        cf_free_model(model);
        return 1;
    }

    // Test 5: Quantile ordering
    printf("[TEST 5] Quantile ordering (q05 <= q50 <= q95)... ");
    bool order_ok = true;
    for (int i = 0; i < result->batch_size; i++) {
        if (pred.c3s[0] > pred.c3s[1] || pred.c3s[1] > pred.c3s[2]) order_ok = false;
        if (pred.c2s[0] > pred.c2s[1] || pred.c2s[1] > pred.c2s[2]) order_ok = false;
    }
    printf("%s\n", order_ok ? "PASS" : "FAIL");

    // Test 6: JSON export
    printf("[TEST 6] JSON export... ");
    char* json = cf_result_to_json(result, &status);
    if (json && strlen(json) > 100) {
        printf("PASS (%zu bytes)\n", strlen(json));
        cf_free_string(json);
    } else {
        printf("FAIL\n");
    }

    // Test 7: Binary export
    printf("[TEST 7] Binary export... ");
    size_t bin_len;
    uint8_t* bin = cf_result_to_binary(result, &bin_len, &status);
    if (bin && bin_len > 0) {
        printf("PASS (%zu bytes)\n", bin_len);
        cf_free_buffer(bin);
    } else {
        printf("FAIL\n");
    }

    // Test 8: Physics validation
    printf("[TEST 8] Physics validation... ");
    float dummy_c3a[3] = {5.0f, 7.0f, 9.0f};
    float dummy_c4af[3] = {8.0f, 10.0f, 12.0f};
    char phys_err[256];
    CF_Status phys = cf_validate_physics(pred.c3s, pred.c2s, dummy_c3a, dummy_c4af,
                                          pred.free_cao, phys_err, sizeof(phys_err));
    printf("%s (%s)\n", 
           phys == CF_OK ? "PASS" : "WARN",
           phys == CF_OK ? "valid" : phys_err);

    // Test 9: Feature extraction
    printf("[TEST 9] Feature extraction... ");
    float features[CF_MAX_FEATURES];
    int32_t n_features;
    CF_Status feat = cf_extract_features(&tech, &history, features, &n_features);
    if (feat == CF_OK && n_features > 0) {
        printf("PASS (%d features)\n", n_features);
    } else {
        printf("FAIL\n");
    }

    // Test 10: Batch prediction
    printf("[TEST 10] Batch prediction (n=4)... ");
    CF_TechnologyParams tech_batch[4];
    CF_TimeSeries hist_batch[4];
    int64_t timestamps[4] = {1, 2, 3, 4};

    for (int i = 0; i < 4; i++) {
        memcpy(&tech_batch[i], &tech, sizeof(tech));
        memcpy(&hist_batch[i], &history, sizeof(history));
    }

    CF_Batch batch;
    batch.tech = tech_batch;
    batch.history = hist_batch;
    batch.timestamps = timestamps;
    batch.batch_size = 4;

    CF_Result* batch_result = cf_predict(model, &batch, 24);
    if (batch_result && batch_result->batch_size == 4) {
        printf("PASS (time=%.2fms)\n", batch_result->inference_time_ms);
        cf_free_result(batch_result);
    } else {
        printf("FAIL\n");
    }

    // Cleanup
    cf_free_result(result);
    cf_free_model(model);

    printf("\n==================================\n");
    printf("All tests completed.\n");

    return 0;
}