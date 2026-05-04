#include "clinker_forecast.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

static void print_usage(const char* prog) {
    printf("Usage: %s [options]\n", prog);
    printf("Options:\n");
    printf("  -m, --model <path>       Path to model .bin file (required)\n");
    printf("  -t, --tech <json>        Technology params as JSON\n");
    printf("  -H, --horizon <hours>    Forecast horizon (2-72, default: 24)\n");
    printf("  -o, --output <format>    Output format: json, binary (default: json)\n");
    printf("  -i, --info               Show model info and exit\n");
    printf("  -b, --benchmark <n>      Run n inferences and show stats\n");
    printf("  -h, --help               Show this help\n");
    printf("\n");
    printf("Example:\n");
    printf("  %s -m model_v1.bin -t '{\"temps\":[1420,1350,1200,900],\"flows\":[45.2,12.1,8.5]}' -H 24\n", prog);
}

static bool parse_tech_json(const char* json, CF_TechnologyParams* tech) {
    memset(tech, 0, sizeof(*tech));
    tech->n_zones = 4;
    tech->n_flows = 3;
    tech->n_chemistry = 5;

    tech->temperatures[0] = 1420.0f;
    tech->temperatures[1] = 1350.0f;
    tech->temperatures[2] = 1200.0f;
    tech->temperatures[3] = 900.0f;
    tech->flows[0] = 45.2f;
    tech->flows[1] = 12.1f;
    tech->flows[2] = 8.5f;
    tech->kiln_speed = 3.2f;
    tech->chemistry[0] = 0.92f;
    tech->chemistry[1] = 2.4f;
    tech->chemistry[2] = 1.6f;
    tech->chemistry[3] = 65.0f;
    tech->chemistry[4] = 20.0f;

    const char* ptr = json;
    while (*ptr) {
        if (strncmp(ptr, "\"temps\"", 7) == 0) {
            ptr += 7;
            while (*ptr && *ptr != '[') ptr++;
            if (*ptr == '[') {
                ptr++;
                for (int i = 0; i < 4 && *ptr; i++) {
                    tech->temperatures[i] = strtof(ptr, nullptr);
                    while (*ptr && *ptr != ',' && *ptr != ']') ptr++;
                    if (*ptr == ',') ptr++;
                }
            }
        }
        if (strncmp(ptr, "\"flows\"", 7) == 0) {
            ptr += 7;
            while (*ptr && *ptr != '[') ptr++;
            if (*ptr == '[') {
                ptr++;
                for (int i = 0; i < 3 && *ptr; i++) {
                    tech->flows[i] = strtof(ptr, nullptr);
                    while (*ptr && *ptr != ',' && *ptr != ']') ptr++;
                    if (*ptr == ',') ptr++;
                }
            }
        }
        ptr++;
    }
    return true;
}

int main(int argc, char** argv) {
    const char* model_path = nullptr;
    const char* tech_json = nullptr;
    int horizon = 24;
    const char* output_format = "json";
    bool show_info = false;
    int benchmark = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--model") == 0) {
            if (i + 1 < argc) model_path = argv[++i];
        } else if (strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--tech") == 0) {
            if (i + 1 < argc) tech_json = argv[++i];
        } else if (strcmp(argv[i], "-H") == 0 || strcmp(argv[i], "--horizon") == 0) {
            if (i + 1 < argc) horizon = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0) {
            if (i + 1 < argc) output_format = argv[++i];
        } else if (strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--info") == 0) {
            show_info = true;
        } else if (strcmp(argv[i], "-b") == 0 || strcmp(argv[i], "--benchmark") == 0) {
            if (i + 1 < argc) benchmark = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }

    if (!model_path) {
        fprintf(stderr, "Error: Model path required\n");
        print_usage(argv[0]);
        return 1;
    }

    printf("ClinkerForecast CLI v1.0.0\n");
    printf("Loading model: %s\n", model_path);

    CF_Status status;
    CF_Model* model = cf_load_model(model_path, &status);
    if (!model) {
        fprintf(stderr, "Error loading model: %d\n", status);
        return 1;
    }

    if (show_info) {
        CF_ModelInfo info = cf_get_model_info(model);
        printf("\nModel Info:\n");
        printf("  Version:       %s\n", info.version);
        printf("  Created:       %s\n", info.created_at);
        printf("  Architecture:  d_model=%d, heads=%d, layers=%d\n",
               info.d_model, info.n_heads, info.n_layers);
        printf("  History:       %d steps\n", info.history_len);
        printf("  Outputs:       %d (with %d quantiles)\n",
               info.n_outputs, info.n_quantiles);
        printf("  Parameters:    %ld\n", info.total_params);
        printf("  Model size:    %.2f MB\n", info.model_size_bytes / (1024.0f * 1024.0f));
        cf_free_model(model);
        return 0;
    }

    CF_TechnologyParams tech;
    if (tech_json) {
        parse_tech_json(tech_json, &tech);
    } else {
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
    }

    CF_TimeSeries history;
    memset(&history, 0, sizeof(history));
    history.history_len = 288;
    history.n_features = 15;
    for (int i = 0; i < history.history_len * history.n_features; i++) {
        history.data[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
    }

    if (benchmark > 0) {
        printf("\nRunning benchmark: %d inferences...\n", benchmark);
        double total_ms = 0.0;
        double min_ms = 1e9, max_ms = 0.0;

        for (int i = 0; i < benchmark; i++) {
            CF_Result* result = cf_predict_single(model, &tech, &history,
                                                   time(nullptr), horizon);
            if (result) {
                total_ms += result->inference_time_ms;
                if (result->inference_time_ms < min_ms) min_ms = result->inference_time_ms;
                if (result->inference_time_ms > max_ms) max_ms = result->inference_time_ms;
                cf_free_result(result);
            }
        }

        printf("Results:\n");
        printf("  Total:  %.2f ms\n", total_ms);
        printf("  Mean:   %.3f ms\n", total_ms / benchmark);
        printf("  Min:    %.3f ms\n", min_ms);
        printf("  Max:    %.3f ms\n", max_ms);
        printf("  Throughput: %.1f inf/sec\n", 1000.0 * benchmark / total_ms);

        cf_free_model(model);
        return 0;
    }

    printf("\nRunning inference (horizon=%dh)...\n", horizon);

    CF_Result* result = cf_predict_single(model, &tech, &history,
                                           time(nullptr), horizon);
    if (!result) {
        fprintf(stderr, "Inference failed\n");
        cf_free_model(model);
        return 1;
    }

    printf("Inference time: %.3f ms\n\n", result->inference_time_ms);

    if (strcmp(output_format, "json") == 0) {
        char* json = cf_result_to_json(result, &status);
        if (json) {
            printf("%s\n", json);
            cf_free_string(json);
        }
    } else if (strcmp(output_format, "binary") == 0) {
        size_t len;
        uint8_t* bin = cf_result_to_binary(result, &len, &status);
        if (bin) {
            printf("Binary output: %zu bytes\n", len);
            fwrite(bin, 1, len, stdout);
            cf_free_buffer(bin);
        }
    }

    cf_free_result(result);
    cf_free_model(model);

    return 0;
}