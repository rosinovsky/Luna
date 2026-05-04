#include "clinker_forecast.h"
#include "ops.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector>

// ============ АКТИВАЦИИ ============

float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

float gelu(float x) {
    float c = 0.044715f;
    float sqrt_2_over_pi = 0.7978845608f;
    float x3 = x * x * x;
    float t = sqrt_2_over_pi * (x + c * x3);
    return 0.5f * x * (1.0f + std::tanh(t));
}

float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

// ============ МАТРИЧНЫЕ ОПЕРАЦИИ ============

void matmul(const float* A, const float* B, float* C, 
            int M, int K, int N, bool transpose_A) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                float a = transpose_A ? A[k * M + i] : A[i * K + k];
                sum += a * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

void matmul_bt(const float* A, const float* B, float* C,
               int M, int K, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[j * K + k];
            }
            C[i * N + j] = sum;
        }
    }
}

void linear(const float* X, const float* W, const float* b,
            float* Y, int batch, int in_features, int out_features) {
    for (int i = 0; i < batch; i++) {
        for (int j = 0; j < out_features; j++) {
            float sum = b ? b[j] : 0.0f;
            for (int k = 0; k < in_features; k++) {
                sum += X[i * in_features + k] * W[k * out_features + j];
            }
            Y[i * out_features + j] = sum;
        }
    }
}

// ============ LAYER NORM ============

void layer_norm(float* x, const float* gamma, const float* beta,
                int batch, int features, float eps) {
    for (int i = 0; i < batch; i++) {
        float* row = x + i * features;

        float mean = 0.0f;
        for (int j = 0; j < features; j++) mean += row[j];
        mean /= features;

        float var = 0.0f;
        for (int j = 0; j < features; j++) {
            float diff = row[j] - mean;
            var += diff * diff;
        }
        var /= features;

        float inv_std = 1.0f / std::sqrt(var + eps);
        for (int j = 0; j < features; j++) {
            row[j] = (row[j] - mean) * inv_std * gamma[j] + beta[j];
        }
    }
}

// ============ SOFTMAX ============

void softmax(float* x, int batch, int features) {
    for (int i = 0; i < batch; i++) {
        float* row = x + i * features;
        float max_val = row[0];
        for (int j = 1; j < features; j++) {
            if (row[j] > max_val) max_val = row[j];
        }
        float sum = 0.0f;
        for (int j = 0; j < features; j++) {
            row[j] = std::exp(row[j] - max_val);
            sum += row[j];
        }
        for (int j = 0; j < features; j++) {
            row[j] /= sum;
        }
    }
}

// ============ LSTM CELL ============

void lstm_step(const float* x, const float* h_prev, const float* c_prev,
               const float* W_ih, const float* W_hh,
               const float* b_ih, const float* b_hh,
               float* h_out, float* c_out,
               int input_size, int hidden_size) {
    int gates_size = 4 * hidden_size;
    std::vector<float> gates(gates_size);

    for (int g = 0; g < gates_size; g++) {
        gates[g] = b_ih[g];
        for (int i = 0; i < input_size; i++) {
            gates[g] += W_ih[g * input_size + i] * x[i];
        }
    }

    for (int g = 0; g < gates_size; g++) {
        gates[g] += b_hh[g];
        for (int h = 0; h < hidden_size; h++) {
            gates[g] += W_hh[g * hidden_size + h] * h_prev[h];
        }
    }

    for (int j = 0; j < hidden_size; j++) {
        float i_gate = sigmoid(gates[j]);
        float f_gate = sigmoid(gates[hidden_size + j]);
        float g_gate = std::tanh(gates[2 * hidden_size + j]);
        float o_gate = sigmoid(gates[3 * hidden_size + j]);

        c_out[j] = f_gate * c_prev[j] + i_gate * g_gate;
        h_out[j] = o_gate * std::tanh(c_out[j]);
    }
}

// ============ MULTI-HEAD ATTENTION ============

void multi_head_attention(const float* X, int batch, int seq_len, int d_model,
                          const float* W_q, const float* W_k, 
                          const float* W_v, const float* W_o,
                          const float* b_q, const float* b_k,
                          const float* b_v, const float* b_o,
                          int n_heads, float* output,
                          float* attn_weights_out) {
    int d_k = d_model / n_heads;

    std::vector<float> Q(batch * seq_len * d_model);
    std::vector<float> K(batch * seq_len * d_model);
    std::vector<float> V(batch * seq_len * d_model);

    linear(X, W_q, b_q, Q.data(), batch * seq_len, d_model, d_model);
    linear(X, W_k, b_k, K.data(), batch * seq_len, d_model, d_model);
    linear(X, W_v, b_v, V.data(), batch * seq_len, d_model, d_model);

    std::vector<float> attn_out(batch * seq_len * d_model);

    for (int b = 0; b < batch; b++) {
        for (int h = 0; h < n_heads; h++) {
            std::vector<float> q_head(seq_len * d_k);
            std::vector<float> k_head(seq_len * d_k);
            std::vector<float> v_head(seq_len * d_k);

            for (int s = 0; s < seq_len; s++) {
                for (int d = 0; d < d_k; d++) {
                    int src_idx = b * seq_len * d_model + s * d_model + h * d_k + d;
                    q_head[s * d_k + d] = Q[src_idx];
                    k_head[s * d_k + d] = K[src_idx];
                    v_head[s * d_k + d] = V[src_idx];
                }
            }

            std::vector<float> scores(seq_len * seq_len);
            matmul_bt(q_head.data(), k_head.data(), scores.data(), seq_len, d_k, seq_len);

            float scale = 1.0f / std::sqrt(static_cast<float>(d_k));
            for (size_t i = 0; i < scores.size(); i++) scores[i] *= scale;

            softmax(scores.data(), seq_len, seq_len);

            if (attn_weights_out && h == 0) {
                for (int s = 0; s < seq_len; s++) {
                    attn_weights_out[b * seq_len + s] = 0.0f;
                    for (int t = 0; t < seq_len; t++) {
                        attn_weights_out[b * seq_len + s] += scores[s * seq_len + t];
                    }
                    attn_weights_out[b * seq_len + s] /= seq_len;
                }
            }

            std::vector<float> head_out(seq_len * d_k);
            matmul(scores.data(), v_head.data(), head_out.data(), seq_len, seq_len, d_k, false);

            for (int s = 0; s < seq_len; s++) {
                for (int d = 0; d < d_k; d++) {
                    int dst_idx = b * seq_len * d_model + s * d_model + h * d_k + d;
                    attn_out[dst_idx] = head_out[s * d_k + d];
                }
            }
        }
    }

    linear(attn_out.data(), W_o, b_o, output, batch * seq_len, d_model, d_model);
}

// ============ FFN ============

void ffn(const float* X, const float* W1, const float* b1,
         const float* W2, const float* b2,
         float* output, int batch_seq, int d_model) {
    int hidden = d_model * 4;
    std::vector<float> hidden_vec(batch_seq * hidden);

    linear(X, W1, b1, hidden_vec.data(), batch_seq, d_model, hidden);
    for (size_t i = 0; i < hidden_vec.size(); i++) {
        hidden_vec[i] = gelu(hidden_vec[i]);
    }

    linear(hidden_vec.data(), W2, b2, output, batch_seq, hidden, d_model);
}