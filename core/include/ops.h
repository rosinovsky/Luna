#ifndef CLINKER_OPS_H
#define CLINKER_OPS_H

#ifdef __cplusplus
extern "C" {
#endif

void matmul(const float* A, const float* B, float* C, int M, int K, int N, bool transpose_A);
void matmul_bt(const float* A, const float* B, float* C, int M, int K, int N);
void linear(const float* X, const float* W, const float* b, float* Y, int batch, int in_features, int out_features);
void layer_norm(float* x, const float* gamma, const float* beta, int batch, int features, float eps);
void softmax(float* x, int batch, int features);
void lstm_step(const float* x, const float* h_prev, const float* c_prev,
               const float* W_ih, const float* W_hh,
               const float* b_ih, const float* b_hh,
               float* h_out, float* c_out, int input_size, int hidden_size);
void multi_head_attention(const float* X, int batch, int seq_len, int d_model,
                          const float* W_q, const float* W_k, const float* W_v, const float* W_o,
                          const float* b_q, const float* b_k, const float* b_v, const float* b_o,
                          int n_heads, float* output, float* attn_weights_out);
void ffn(const float* X, const float* W1, const float* b1,
         const float* W2, const float* b2,
         float* output, int batch_seq, int d_model);

// Activation functions
float relu(float x);
float gelu(float x);
float sigmoid(float x);

#ifdef __cplusplus
}
#endif

#endif