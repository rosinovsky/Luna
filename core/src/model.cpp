#include "clinker_forecast.h"
#include "internal.h"\n#include "internal.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <chrono>

// ============ ВНУТРЕННИЕ СТРУКТУРЫ ============


// ============ LOAD WEIGHTS ============

static CF_Status load_weights_from_file(CF_Model* model, const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return CF_ERROR_MODEL_NOT_LOADED;

    // Magic
    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || strncmp(magic, "CLKR", 4) != 0) {
        fclose(f);
        return CF_ERROR_INVALID_WEIGHTS;
    }

    // Version
    uint32_t version;
    fread(&version, 4, 1, f);

    // Config JSON
    uint32_t config_len;
    fread(&config_len, 4, 1, f);
    std::vector<char> config_buf(config_len + 1, 0);
    fread(config_buf.data(), 1, config_len, f);
    // TODO: parse JSON config

    // Number of tensors
    uint32_t n_tensors;
    fread(&n_tensors, 4, 1, f);

    model->weights.reserve(n_tensors);
    model->weight_names.reserve(n_tensors);

    for (uint32_t i = 0; i < n_tensors; i++) {
        // Name
        uint32_t name_len;
        fread(&name_len, 4, 1, f);
        std::string name(name_len, '\0');
        fread(&name[0], 1, name_len, f);

        // Shape
        uint32_t ndims;
        fread(&ndims, 4, 1, f);
        Tensor tensor;
        tensor.name = name;
        tensor.shape.resize(ndims);
        size_t total_elements = 1;
        for (uint32_t d = 0; d < ndims; d++) {
            uint32_t dim;
            fread(&dim, 4, 1, f);
            tensor.shape[d] = dim;
            total_elements *= dim;
        }

        // Data
        uint32_t data_len;
        fread(&data_len, 4, 1, f);
        tensor.data.resize(total_elements);
        fread(tensor.data.data(), 4, total_elements, f);

        model->weights.push_back(std::move(tensor));
        model->weight_names.push_back(name);
        model->total_params += total_elements;
    }

    fclose(f);
    return CF_OK;
}

CF_Model* cf_load_model(const char* path, CF_Status* status) {
    CF_Model* model = new CF_Model();

    CF_Status st = load_weights_from_file(model, path);
    if (st != CF_OK) {
        delete model;
        if (status) *status = st;
        return nullptr;
    }

    // Pre-allocate workspace
    size_t ws = model->config.d_model * model->config.history_len * CF_MAX_BATCH_SIZE * 4;
    model->workspace.resize(ws);
    model->workspace_size = ws;

    if (status) *status = CF_OK;
    return model;
}

CF_Model* cf_load_model_from_buffer(const uint8_t* data, size_t len, CF_Status* status) {
    (void)data;
    (void)len;
    if (status) *status = CF_ERROR_INVALID_WEIGHTS;
    return nullptr;
}

void cf_free_model(CF_Model* model) {
    delete model;
}

CF_ModelInfo cf_get_model_info(const CF_Model* model) {
    CF_ModelInfo info = {};
    if (!model) return info;

    strncpy(info.version, model->config.version, sizeof(info.version) - 1);
    strncpy(info.created_at, model->config.created_at, sizeof(info.created_at) - 1);
    info.d_model = model->config.d_model;
    info.n_heads = model->config.n_heads;
    info.n_layers = model->config.n_layers;
    info.history_len = model->config.history_len;
    info.n_features = model->config.n_features;
    info.n_outputs = model->config.n_outputs;
    info.n_quantiles = model->config.n_quantiles;
    memcpy(info.quantiles, model->config.quantiles, sizeof(info.quantiles));
    info.total_params = model->total_params;
    info.model_size_bytes = model->total_params * sizeof(float);

    return info;
}

const char* cf_version(void) {
    return "ClinkerForecast Core v1.0.0-synthetic";
}