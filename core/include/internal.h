#ifndef CLINKER_INTERNAL_H
#define CLINKER_INTERNAL_H

#include "clinker_forecast.h"
#include <vector>
#include <string>

struct Tensor {
    std::vector<int> shape;
    std::vector<float> data;
    std::string name;
    float* ptr() { return data.data(); }
    const float* ptr() const { return data.data(); }
};

struct ModelConfig {
    int d_model = 64;
    int n_heads = 4;
    int n_layers = 2;
    int history_len = 288;
    int n_features = 15;
    int n_zones = 4;
    int n_flows = 3;
    int n_chemistry = 5;
    int n_outputs = 4;
    int n_quantiles = 3;
    float quantiles[CF_MAX_QUANTILES] = {0.05f, 0.50f, 0.95f};
    char version[32] = "1.0.0-synthetic";
    char created_at[32] = "2026-05-04";
};

struct CF_Model {
    ModelConfig config;
    std::vector<Tensor> weights;
    std::vector<std::string> weight_names;
    size_t total_params = 0;
    std::vector<float> workspace;
    size_t workspace_size = 0;
};

#endif
