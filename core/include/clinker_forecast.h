#ifndef CLINKER_FORECAST_H
#define CLINKER_FORECAST_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>  /* для size_t */

/* ==================== КОНФИГУРАЦИЯ ==================== */

#define CF_MAX_BATCH_SIZE 64
#define CF_MAX_HISTORY_LEN 288
#define CF_MAX_FEATURES 32
#define CF_MAX_ZONES 8
#define CF_MAX_FLOWS 8
#define CF_MAX_CHEMISTRY 16
#define CF_MAX_OUTPUTS 8
#define CF_MAX_QUANTILES 5
#define CF_MAX_NAME_LEN 64

/* ==================== ТИПЫ ДАННЫХ ==================== */

struct CF_Model;
struct CF_Result;

typedef struct CF_Model CF_Model;
typedef struct CF_Result CF_Result;

/* Статус операции */
typedef enum {
    CF_OK = 0,
    CF_ERROR_INVALID_INPUT = -1,
    CF_ERROR_MODEL_NOT_LOADED = -2,
    CF_ERROR_OUT_OF_MEMORY = -3,
    CF_ERROR_INVALID_WEIGHTS = -4,
    CF_ERROR_DIMENSION_MISMATCH = -5,
    CF_ERROR_PHYSICS_VIOLATION = -6
} CF_Status;

/* Технологические параметры (табличные) */
typedef struct {
    float temperatures[CF_MAX_ZONES];
    float flows[CF_MAX_FLOWS];
    float kiln_speed;
    float chemistry[CF_MAX_CHEMISTRY];
    int32_t n_zones;
    int32_t n_flows;
    int32_t n_chemistry;
} CF_TechnologyParams;

/* Временной ряд (история) */
typedef struct {
    float data[CF_MAX_HISTORY_LEN * CF_MAX_FEATURES];
    int32_t history_len;
    int32_t n_features;
} CF_TimeSeries;

/* Входной батч */
typedef struct {
    CF_TechnologyParams* tech;
    CF_TimeSeries* history;
    int64_t* timestamps;
    int32_t batch_size;
} CF_Batch;

/* Предсказание для одного sample */
typedef struct {
    float c3s[CF_MAX_QUANTILES];
    float c2s[CF_MAX_QUANTILES];
    float free_cao[CF_MAX_QUANTILES];
    float liter_weight[CF_MAX_QUANTILES];
    float so3[CF_MAX_QUANTILES];
    int8_t trend;
    float trend_confidence;
    float feature_importance[CF_MAX_FEATURES];
    float attention_weights[CF_MAX_HISTORY_LEN];
    bool physics_valid;
    char physics_error[256];
    int64_t prediction_time;
    int32_t horizon_hours;
} CF_Prediction;

/* Результат инференса */
struct CF_Result {
    CF_Prediction* predictions;
    int32_t batch_size;
    CF_Status status;
    char error_msg[512];
    float inference_time_ms;
};

/* Информация о модели */
typedef struct {
    char version[32];
    char created_at[32];
    int32_t d_model;
    int32_t n_heads;
    int32_t n_layers;
    int32_t history_len;
    int32_t n_features;
    int32_t n_outputs;
    int32_t n_quantiles;
    float quantiles[CF_MAX_QUANTILES];
    int64_t total_params;
    size_t model_size_bytes;
} CF_ModelInfo;

/* ==================== API ФУНКЦИИ ==================== */

const char* cf_version(void);
CF_Model* cf_load_model(const char* path, CF_Status* status);
CF_Model* cf_load_model_from_buffer(const uint8_t* data, size_t len, CF_Status* status);
CF_Result* cf_predict(CF_Model* model, const CF_Batch* batch, int32_t horizon_hours);
CF_Result* cf_predict_single(CF_Model* model, const CF_TechnologyParams* tech, const CF_TimeSeries* history, int64_t timestamp, int32_t horizon_hours);
CF_ModelInfo cf_get_model_info(const CF_Model* model);
char* cf_result_to_json(const CF_Result* result, CF_Status* status);
uint8_t* cf_result_to_binary(const CF_Result* result, size_t* out_len, CF_Status* status);
void cf_free_model(CF_Model* model);
void cf_free_result(CF_Result* result);
void cf_free_string(char* str);
void cf_free_buffer(uint8_t* buf);
CF_Status cf_validate_physics(const float* c3s, const float* c2s, const float* c3a, const float* c4af, const float* free_cao, char* error_msg, size_t msg_len);
CF_Status cf_extract_features(const CF_TechnologyParams* tech, const CF_TimeSeries* history, float* out_features, int32_t* out_n_features);

#ifdef __cplusplus
}
#endif

#endif /* CLINKER_FORECAST_H */