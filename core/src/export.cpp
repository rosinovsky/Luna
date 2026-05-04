#include "clinker_forecast.h"
#include "internal.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>

// ============ JSON ЭКСПОРТ ============

char* cf_result_to_json(const CF_Result* result, CF_Status* status) {
    if (!result || !result->predictions) {
        if (status) *status = CF_ERROR_INVALID_INPUT;
        return nullptr;
    }

    std::string json = "{\n";
    json += "  \"status\": \"" + std::string(result->status == CF_OK ? "ok" : "error") + "\",\n";
    json += "  \"inference_time_ms\": " + std::to_string(result->inference_time_ms) + ",\n";
    json += "  \"batch_size\": " + std::to_string(result->batch_size) + ",\n";
    json += "  \"predictions\": [\n";

    for (int b = 0; b < result->batch_size; b++) {
        const CF_Prediction& p = result->predictions[b];

        json += "    {\n";
        json += "      \"horizon_hours\": " + std::to_string(p.horizon_hours) + ",\n";
        json += "      \"prediction_time\": " + std::to_string(p.prediction_time) + ",\n";

        // Quantiles
        json += "      \"c3s\": {\n";
        json += "        \"q05\": " + std::to_string(p.c3s[0]) + ",\n";
        json += "        \"q50\": " + std::to_string(p.c3s[1]) + ",\n";
        json += "        \"q95\": " + std::to_string(p.c3s[2]) + "\n";
        json += "      },\n";

        json += "      \"c2s\": {\n";
        json += "        \"q05\": " + std::to_string(p.c2s[0]) + ",\n";
        json += "        \"q50\": " + std::to_string(p.c2s[1]) + ",\n";
        json += "        \"q95\": " + std::to_string(p.c2s[2]) + "\n";
        json += "      },\n";

        json += "      \"free_cao\": {\n";
        json += "        \"q05\": " + std::to_string(p.free_cao[0]) + ",\n";
        json += "        \"q50\": " + std::to_string(p.free_cao[1]) + ",\n";
        json += "        \"q95\": " + std::to_string(p.free_cao[2]) + "\n";
        json += "      },\n";

        json += "      \"liter_weight\": {\n";
        json += "        \"q05\": " + std::to_string(p.liter_weight[0]) + ",\n";
        json += "        \"q50\": " + std::to_string(p.liter_weight[1]) + ",\n";
        json += "        \"q95\": " + std::to_string(p.liter_weight[2]) + "\n";
        json += "      },\n";

        // Trend
        const char* trend_str = (p.trend == -1) ? "decreasing" : 
                                (p.trend == 1) ? "increasing" : "stable";
        json += "      \"trend\": \"" + std::string(trend_str) + "\",\n";
        json += "      \"trend_confidence\": " + std::to_string(p.trend_confidence) + ",\n";

        // Physics
        json += "      \"physics_valid\": " + std::string(p.physics_valid ? "true" : "false") + ",\n";
        if (!p.physics_valid) {
            json += "      \"physics_error\": \"" + std::string(p.physics_error) + "\",\n";
        }

        // Feature importance (top 5)
        json += "      \"top_features\": [\n";
        for (int f = 0; f < 5 && f < CF_MAX_FEATURES; f++) {
            json += "        {\"index\": " + std::to_string(f) + ", \"importance\": " + 
                    std::to_string(p.feature_importance[f]) + "}";
            if (f < 4) json += ",";
            json += "\n";
        }
        json += "      ]\n";

        json += "    }";
        if (b < result->batch_size - 1) json += ",";
        json += "\n";
    }

    json += "  ]\n";
    json += "}";

    char* out = new char[json.length() + 1];
    strcpy(out, json.c_str());

    if (status) *status = CF_OK;
    return out;
}

// ============ БИНАРНЫЙ ЭКСПОРТ (OPC-UA friendly) ============

uint8_t* cf_result_to_binary(const CF_Result* result, size_t* out_len, CF_Status* status) {
    if (!result || !result->predictions) {
        if (status) *status = CF_ERROR_INVALID_INPUT;
        return nullptr;
    }

    int n = result->batch_size;
    size_t sample_size = 8 + 4 + 4*4 + 1 + 1;
    size_t total = 4 + 4 + 4 + n * sample_size;

    uint8_t* buf = new uint8_t[total];
    size_t pos = 0;

    // Magic
    buf[pos++] = 0xCF;
    buf[pos++] = 0x01;
    buf[pos++] = 0x00;
    buf[pos++] = 0x00;

    // Version
    buf[pos++] = 1;
    buf[pos++] = 0;
    buf[pos++] = 0;
    buf[pos++] = 0;

    // N samples
    memcpy(&buf[pos], &n, 4);
    pos += 4;

    for (int b = 0; b < n; b++) {
        const CF_Prediction& p = result->predictions[b];

        memcpy(&buf[pos], &p.prediction_time, 8);
        pos += 8;

        memcpy(&buf[pos], &p.horizon_hours, 4);
        pos += 4;

        memcpy(&buf[pos], &p.c3s[1], 4);
        pos += 4;

        memcpy(&buf[pos], &p.c2s[1], 4);
        pos += 4;

        memcpy(&buf[pos], &p.free_cao[1], 4);
        pos += 4;

        memcpy(&buf[pos], &p.liter_weight[1], 4);
        pos += 4;

        buf[pos++] = static_cast<uint8_t>(p.trend);

        buf[pos++] = p.physics_valid ? 1 : 0;
    }

    *out_len = pos;
    if (status) *status = CF_OK;
    return buf;
}

void cf_free_string(char* str) {
    delete[] str;
}

void cf_free_buffer(uint8_t* buf) {
    delete[] buf;
}