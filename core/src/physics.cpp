#include "clinker_forecast.h"
#include "internal.h"
#include <cmath>
#include <cstring>
#include <cstdio>

// ============ КОНСТАНТЫ ЦЕМЕНТНОЙ ХИМИИ ============

static const float M_CaO = 56.077f;
static const float M_SiO2 = 60.084f;
static const float M_Al2O3 = 101.961f;
static const float M_Fe2O3 = 159.687f;
static const float M_SO3 = 80.063f;
static const float M_MgO = 40.304f;

static const float MIN_C3S = 45.0f, MAX_C3S = 70.0f;
static const float MIN_C2S = 15.0f, MAX_C2S = 35.0f;
static const float MAX_FREE_CAO = 1.5f;
static const float MAX_LITER_WEIGHT = 1500.0f;
static const float MIN_LITER_WEIGHT = 1100.0f;
static const float MAX_SO3 = 3.5f;
static const float MAX_MGO = 5.0f;

// ============ РАСЧЕТ МИНЕРАЛОГИИ ПО БОГЕ ============

static void bogue_calculation(const float* chemistry, int n_elements,
                                float* c3s, float* c2s, float* c3a, float* c4af) {
    float cao = (n_elements > 0) ? chemistry[0] : 0.0f;
    float sio2 = (n_elements > 1) ? chemistry[1] : 0.0f;
    float al2o3 = (n_elements > 2) ? chemistry[2] : 0.0f;
    float fe2o3 = (n_elements > 3) ? chemistry[3] : 0.0f;

    *c3s = 4.071f * cao - 7.600f * sio2 - 6.718f * al2o3 - 1.430f * fe2o3;
    *c2s = 2.867f * sio2 - 0.7544f * (*c3s);
    *c3a = 2.650f * al2o3 - 1.692f * fe2o3;
    *c4af = 3.043f * fe2o3;

    if (*c3s < 0.0f) {
        *c2s += *c3s;
        *c3s = 0.0f;
    }
    if (*c2s < 0.0f) *c2s = 0.0f;
    if (*c3a < 0.0f) {
        *c4af += *c3a;
        *c3a = 0.0f;
    }
    if (*c4af < 0.0f) *c4af = 0.0f;
}

// ============ ВАЛИДАЦИЯ ============

CF_Status cf_validate_physics(const float* c3s, const float* c2s,
                               const float* c3a, const float* c4af,
                               const float* free_cao, char* error_msg, size_t msg_len) {
    if (!c3s || !c2s || !c3a || !c4af || !free_cao || !error_msg) {
        return CF_ERROR_INVALID_INPUT;
    }

    float c3s_val = c3s[1];
    float c2s_val = c2s[1];
    float c3a_val = c3a[1];
    float c4af_val = c4af[1];
    float cao_val = free_cao[1];

    float total = c3s_val + c2s_val + c3a_val + c4af_val;
    if (std::abs(total - 100.0f) > 2.0f) {
        snprintf(error_msg, msg_len,
                 "PHYSICS_ERR: Sum of minerals = %.2f%% (expected 100+-2%%). "
                 "C3S=%.1f C2S=%.1f C3A=%.1f C4AF=%.1f",
                 total, c3s_val, c2s_val, c3a_val, c4af_val);
        return CF_ERROR_PHYSICS_VIOLATION;
    }

    if (c3s_val < MIN_C3S || c3s_val > MAX_C3S) {
        snprintf(error_msg, msg_len,
                 "PHYSICS_ERR: C3S = %.1f%% (expected %.1f-%.1f%%)",
                 c3s_val, MIN_C3S, MAX_C3S);
        return CF_ERROR_PHYSICS_VIOLATION;
    }

    if (c2s_val < MIN_C2S || c2s_val > MAX_C2S) {
        snprintf(error_msg, msg_len,
                 "PHYSICS_ERR: C2S = %.1f%% (expected %.1f-%.1f%%)",
                 c2s_val, MIN_C2S, MAX_C2S);
        return CF_ERROR_PHYSICS_VIOLATION;
    }

    if (cao_val > MAX_FREE_CAO) {
        snprintf(error_msg, msg_len,
                 "PHYSICS_ERR: Free CaO = %.2f%% (expected <= %.1f%%)",
                 cao_val, MAX_FREE_CAO);
        return CF_ERROR_PHYSICS_VIOLATION;
    }

    float ratio = c3s_val / (c2s_val + 1e-6f);
    if (ratio < 0.5f || ratio > 5.0f) {
        snprintf(error_msg, msg_len,
                 "PHYSICS_WARN: C3S/C2S ratio = %.2f (expected 0.5-5.0)",
                 ratio);
    }

    return CF_OK;
}

// ============ FEATURE ENGINEERING ============

CF_Status cf_extract_features(const CF_TechnologyParams* tech,
                               const CF_TimeSeries* history,
                               float* out_features, int32_t* out_n_features) {
    if (!tech || !out_features || !out_n_features) {
        return CF_ERROR_INVALID_INPUT;
    }

    int idx = 0;
    const int MAX_F = CF_MAX_FEATURES;

    for (int i = 0; i < tech->n_zones && idx < MAX_F; i++) {
        out_features[idx++] = (tech->temperatures[i] - 800.0f) / 800.0f;
    }

    for (int i = 0; i < tech->n_flows && idx < MAX_F; i++) {
        out_features[idx++] = tech->flows[i] / 100.0f;
    }

    if (idx < MAX_F) {
        out_features[idx++] = (tech->kiln_speed - 2.0f) / 2.0f;
    }

    for (int i = 0; i < tech->n_chemistry && idx < MAX_F; i++) {
        out_features[idx++] = tech->chemistry[i];
    }

    if (history && history->history_len > 0) {
        int n = history->history_len * history->n_features;

        if (idx < MAX_F) {
            float mean = 0.0f;
            for (int i = 0; i < n; i++) mean += history->data[i];
            out_features[idx++] = mean / n;
        }

        if (idx < MAX_F && history->history_len >= 24) {
            int recent = 24 * history->n_features;
            float mean = 0.0f;
            for (int i = n - recent; i < n; i++) mean += history->data[i];
            mean /= recent;
            float var = 0.0f;
            for (int i = n - recent; i < n; i++) {
                float diff = history->data[i] - mean;
                var += diff * diff;
            }
            out_features[idx++] = std::sqrt(var / recent);
        }

        if (idx < MAX_F && history->history_len >= 48) {
            int step = history->n_features;
            float early = 0.0f, late = 0.0f;
            for (int i = 0; i < step; i++) {
                early += history->data[i];
                late += history->data[n - step + i];
            }
            out_features[idx++] = (late - early) / (history->history_len / 4.0f);
        }

        if (idx < MAX_F && history->history_len >= 2) {
            float max_grad = 0.0f;
            for (int t = 1; t < history->history_len; t++) {
                for (int f = 0; f < history->n_features; f++) {
                    float grad = std::abs(
                        history->data[t * history->n_features + f] -
                        history->data[(t-1) * history->n_features + f]
                    );
                    if (grad > max_grad) max_grad = grad;
                }
            }
            out_features[idx++] = max_grad;
        }
    }

    if (tech->n_chemistry >= 3 && idx < MAX_F) {
        out_features[idx++] = tech->chemistry[0];
    }

    if (tech->n_chemistry >= 3 && idx < MAX_F) {
        out_features[idx++] = tech->chemistry[1];
    }

    for (int i = idx; i < MAX_F; i++) out_features[i] = 0.0f;

    *out_n_features = idx;
    return CF_OK;
}