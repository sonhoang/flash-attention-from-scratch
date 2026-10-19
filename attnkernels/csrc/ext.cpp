#include <torch/extension.h>

torch::Tensor naive_attention(torch::Tensor q, torch::Tensor k, torch::Tensor v, bool causal,
                              double scale);
torch::Tensor flash_simt(torch::Tensor q, torch::Tensor k, torch::Tensor v, bool causal,
                         double scale);
torch::Tensor flash_wmma(torch::Tensor q, torch::Tensor k, torch::Tensor v, bool causal,
                         double scale, int64_t block_n);
torch::Tensor paged_decode(torch::Tensor q, torch::Tensor kc, torch::Tensor vc, torch::Tensor bt,
                           torch::Tensor ctx, int64_t max_ctx, double scale, int64_t partition);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Attention kernels written from scratch: naive, FlashAttention-style, paged decode";
    m.def("naive_attention", &naive_attention, "S = QK^T, softmax, PV as three kernels");
    m.def("flash_simt", &flash_simt, "tiled attention with online softmax, CUDA cores");
    m.def("flash_wmma", &flash_wmma, "tiled attention with online softmax, WMMA tensor cores");
    m.def("paged_decode", &paged_decode, "decode attention over a paged KV cache, split-K");
}
