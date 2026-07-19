#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>

#include <cuda/ptx>
#include <tvm/ffi/container/tensor.h>

#include <algorithm>
#include <array>
#include <cfloat>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace sglang {

SGL_DEVICE uint32_t smem_ptr_u32(const void* ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

SGL_DEVICE bool elect_one_sync(unsigned mask = 0xffffffffu) {
  int pred;
  asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      "elect.sync _|p, %1;\n\t"
      "selp.b32 %0, 1, 0, p;\n\t"
      "}\n"
      : "=r"(pred)
      : "r"(mask));
  return pred;
}

SGL_DEVICE void mbarrier_init(uint64_t* mbar, uint32_t expected_arrivers) {
  ::cuda::ptx::mbarrier_init(mbar, expected_arrivers);
}

// Required between mbarrier.init (generic proxy) and the first cp.async.bulk
// complete_tx on the barrier (async proxy); __syncthreads alone only orders
// the generic proxy.
SGL_DEVICE void fence_mbarrier_init() {
  asm volatile("fence.mbarrier_init.release.cluster;");
}

SGL_DEVICE void mbarrier_wait_parity(uint64_t* mbar, uint32_t parity) {
  while (!::cuda::ptx::mbarrier_try_wait_parity(mbar, parity))
    ;
}

SGL_DEVICE void mbarrier_arrive(uint64_t* mbar) {
  ::cuda::ptx::mbarrier_arrive(mbar);
}

SGL_DEVICE void mbarrier_arrive_expect_tx(uint64_t* mbar, uint32_t tx_bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_ptr_u32(mbar)), "r"(tx_bytes)
               : "memory");
}

SGL_DEVICE void bar_sync(uint32_t id, uint32_t num_threads) {
  asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(num_threads) : "memory");
}

SGL_DEVICE void tmem_alloc(uint32_t* smem_dst, uint32_t num_cols) {
  asm volatile(
      "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(smem_ptr_u32(smem_dst)), "r"(num_cols));
}

SGL_DEVICE void tmem_dealloc(uint32_t tmem_addr, uint32_t num_cols) {
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(tmem_addr), "r"(num_cols));
}

SGL_DEVICE void tmem_relinquish_alloc_permit() {
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
}

SGL_DEVICE void tmem_store_wait() {
  asm volatile("tcgen05.wait::st.sync.aligned;");
}

SGL_DEVICE void cp_async_bulk(void* smem_dst, const void* gmem_src, int bytes, uint64_t& mbar) {
  asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" ::"r"(
                   smem_ptr_u32(smem_dst)),
               "l"(gmem_src),
               "r"(bytes),
               "r"(smem_ptr_u32(&mbar))
               : "memory");
}

template <int N, typename T>
SGL_DEVICE void tmem_ld_32dp32bNx(uint32_t const& src_addr, T* dst_ptr_) {
  static_assert(N == 8, "attn_res production TMEM helpers only instantiate x8");
  uint32_t* dst_ptr = reinterpret_cast<uint32_t*>(dst_ptr_);
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x8.b32"
      "{%0, %1, %2, %3, %4, %5, %6, %7},"
      "[%8];\n"
      : "=r"(dst_ptr[0]),
        "=r"(dst_ptr[1]),
        "=r"(dst_ptr[2]),
        "=r"(dst_ptr[3]),
        "=r"(dst_ptr[4]),
        "=r"(dst_ptr[5]),
        "=r"(dst_ptr[6]),
        "=r"(dst_ptr[7])
      : "r"(src_addr));
}

template <int N, typename T>
SGL_DEVICE void tmem_st_32dp32bNx(uint32_t const& dst_addr, T* src_ptr_) {
  static_assert(N == 8, "attn_res production TMEM helpers only instantiate x8");
  uint32_t* src_ptr = reinterpret_cast<uint32_t*>(src_ptr_);
  asm volatile(
      "tcgen05.st.sync.aligned.32x32b.x8.b32"
      "[%8], {%0, %1, %2, %3, %4, %5, %6, %7};\n"
      :
      : "r"(src_ptr[0]),
        "r"(src_ptr[1]),
        "r"(src_ptr[2]),
        "r"(src_ptr[3]),
        "r"(src_ptr[4]),
        "r"(src_ptr[5]),
        "r"(src_ptr[6]),
        "r"(src_ptr[7]),
        "r"(dst_addr));
}

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------

constexpr int K_TILE = 1024;
constexpr int N_MAX = 12;
constexpr int CHUNK_DEPTH = 2;
constexpr int BLK = 288;                    // 1 producer warp + 8 consumer warps
constexpr int CONSUMER_THREADS = BLK - 32;  // 256
constexpr int CONSUMER_WARPS = CONSUMER_THREADS / 32;
constexpr int CONSUMER_GROUPS = 2;  // two 128-thread consumer groups
constexpr int CONSUMER_THREADS_PER_GROUP = CONSUMER_THREADS / CONSUMER_GROUPS;
constexpr int CONSUMER_BAR_ID = 1;                       // named barrier for the consumer rendezvous
constexpr int TMEM_CW_COLS = 32;                         // per-group columns for the cw slices
constexpr int TMEM_Q_COLS_PER_GROUP = 2 * TMEM_CW_COLS;  // cw + ow slices
constexpr int TMEM_Q_COLS_TOTAL = 2 * TMEM_Q_COLS_PER_GROUP;

template <int NC>
struct FwdSmemPlan {
  alignas(16) uint64_t bar_ready[CHUNK_DEPTH];
  alignas(16) uint64_t bar_consumed[CHUNK_DEPTH];
  alignas(16) float ws_sq[CONSUMER_WARPS][NC];
  alignas(16) float ws_dot[CONSUMER_WARPS][NC];
  // Out-norm partial sums; separate from ws_sq so a fast warp entering the
  // next token's pass A cannot clobber a slot a slow warp is still reading.
  alignas(16) float out_ssq[CONSUMER_WARPS];
  uint32_t tmem_base;
};

// N (total rows = nvb + 1) is a compile-time parameter: NUM_CHUNKS is a
// constexpr trip count, so the unrolled chunk loop constant-folds the
// per-chunk active-row count `an` and every row loop bound (the NV original
// took N at runtime and dispatched a switch(an) per chunk).
template <int H, int NC, int N, bool kUsePDL>
__global__ void __launch_bounds__(BLK, 1) attn_res_fwd_online_v2_kernel(
    const bf16_t* __restrict__ bank,        // [T, NB, H] rows 0..N-2
    const bf16_t* __restrict__ prefix_sum,  // [T, H] row N-1
    const bf16_t* __restrict__ cw,          // [H] score norm ⊙ proj weight
    const bf16_t* __restrict__ ow,          // [H] out norm weight
    bf16_t* __restrict__ output,            // [T, H] normalized: rmsnorm(mix) ⊙ ow
    int64_t stride_bm,                      // bank stride along T (in elements)
    int T,
    float rms_eps) {  // shared by the score norm and the out norm
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  constexpr float LOG2_E = 1.4426950408889634f;
  constexpr int N_CHUNK = NC;
  constexpr int NUM_BUFS = CHUNK_DEPTH * NC;
  constexpr int NUM_CHUNKS = (N + NC - 1) / NC;
  constexpr int NHT = H / K_TILE;
  constexpr int SLICES_PER_GROUP = (NHT + CONSUMER_GROUPS - 1) / CONSUMER_GROUPS;
  constexpr int VEC = 8;
  constexpr int ACC_PER_THREAD = SLICES_PER_GROUP * VEC;
  static_assert(H >= 4096 && H <= 8192);
  static_assert(H % K_TILE == 0);
  static_assert(N >= 2 && N <= N_MAX);
  static_assert(ACC_PER_THREAD <= TMEM_CW_COLS, "cw/ow slices must fit their TMEM columns");

  const int tid = threadIdx.x;
  const int wid = tid >> 5;
  const int lane = tid & 31;
  const int TB = T;
  const int num_ctas = gridDim.x;

  const int comp_wid = wid - 1;
  const int comp_tid = tid - 32;
  const int group = (comp_wid >= 4) ? 1 : 0;
  const int ct_in_group = (comp_tid >= 0) ? (comp_tid & (CONSUMER_THREADS_PER_GROUP - 1)) : -1;
  const int k_local = ct_in_group * VEC;

  extern __shared__ char smem_raw[];
  bf16_t* v_bufs = reinterpret_cast<bf16_t*>(smem_raw);  // [NUM_BUFS][H]
  constexpr size_t V_BYTES = (size_t)NUM_BUFS * H * sizeof(bf16_t);
  FwdSmemPlan<NC>& plan = *reinterpret_cast<FwdSmemPlan<NC>*>(smem_raw + V_BYTES);

  auto slot_of = [](long long gci, int n) { return (int)(gci % CHUNK_DEPTH) * N_CHUNK + n; };
  auto phase_of = [](long long gci) { return (int)((gci / CHUNK_DEPTH) & 1); };
  auto buf_ptr = [&](int slot) -> bf16_t* { return v_bufs + slot * H; };
  // Row n of token t: bank rows 0..N-2, prefix_sum as the last row.
  auto v_addr = [&](int n, int t) -> const bf16_t* {
    if (n < N - 1) return bank + (long long)t * stride_bm + (long long)n * H;
    return prefix_sum + (long long)t * H;
  };

  if (wid == 0 && lane < CHUNK_DEPTH) {
    mbarrier_init(&plan.bar_ready[lane], 1);
    mbarrier_init(&plan.bar_consumed[lane], CONSUMER_THREADS);
    fence_mbarrier_init();
  } else if (wid == 1) {
    tmem_alloc(&plan.tmem_base, TMEM_Q_COLS_TOTAL);
    tmem_relinquish_alloc_permit();
  }
  __syncthreads();

  const uint32_t my_tmem = (comp_tid >= 0) ? (plan.tmem_base + ((comp_wid >= 4) ? TMEM_Q_COLS_PER_GROUP : 0)) : 0;

  if (comp_tid >= 0) {
    // Stage cw at columns [0, TMEM_CW_COLS) and ow right after it.
    float q32[ACC_PER_THREAD];
#pragma unroll
    for (int si = 0; si < SLICES_PER_GROUP; si++) {
      int dt = si * CONSUMER_GROUPS + group;
      if (dt >= NHT) continue;
      int h_base = dt * K_TILE + k_local;
#pragma unroll
      for (int j = 0; j < VEC; j++) {
        int h = h_base + j;
        q32[si * VEC + j] = __bfloat162float(cw[h]);
      }
    }
#pragma unroll
    for (int si = 0; si < SLICES_PER_GROUP; si++) {
      tmem_st_32dp32bNx<VEC>(my_tmem + si * VEC, &q32[si * VEC]);
    }
#pragma unroll
    for (int si = 0; si < SLICES_PER_GROUP; si++) {
      int dt = si * CONSUMER_GROUPS + group;
      if (dt >= NHT) continue;
      int h_base = dt * K_TILE + k_local;
#pragma unroll
      for (int j = 0; j < VEC; j++) {
        int h = h_base + j;
        q32[si * VEC + j] = __bfloat162float(ow[h]);
      }
    }
#pragma unroll
    for (int si = 0; si < SLICES_PER_GROUP; si++) {
      tmem_st_32dp32bNx<VEC>(my_tmem + TMEM_CW_COLS + si * VEC, &q32[si * VEC]);
    }
    tmem_store_wait();
  }
  __syncthreads();

  if (wid == 0) {
    if (elect_one_sync()) {
      long long gci = 0;
      for (int tb = blockIdx.x; tb < TB; tb += num_ctas) {
        const int t = tb;
#pragma unroll
        for (int ci = 0; ci < NUM_CHUNKS; ci++, gci++) {
          const int ns = ci * N_CHUNK;
          const int an = min(N_CHUNK, N - ns);  // folds per unrolled iteration
          int chunk_slot = (int)(gci % CHUNK_DEPTH);
          int pc = phase_of(gci);
          mbarrier_wait_parity(&plan.bar_consumed[chunk_slot], pc ^ 1);
          mbarrier_arrive_expect_tx(&plan.bar_ready[chunk_slot], an * H * (int)sizeof(bf16_t));
#pragma unroll
          for (int n = 0; n < an; n++) {
            int slot = slot_of(gci, n);
            // Only prefix_sum is written by the immediately-preceding kernel;
            // bank rows are older. One wait before the first token's prefix
            // load covers everything after it.
            if (tb == blockIdx.x && ns + n == N - 1) {
              device::PDLWaitPrimary<kUsePDL>();
            }
            const bf16_t* src = v_addr(ns + n, t);
            cp_async_bulk(buf_ptr(slot), src, H * sizeof(bf16_t), plan.bar_ready[chunk_slot]);
          }
        }
      }
    }
  } else {
    float acc32[ACC_PER_THREAD] = {};

    long long gci = 0;
    for (int tb = blockIdx.x; tb < TB; tb += num_ctas) {
      float m_running = -FLT_MAX;
      float s_running = 0.f;
#pragma unroll
      for (int i = 0; i < ACC_PER_THREAD; i++) {
        acc32[i] = 0.f;
      }

#pragma unroll
      for (int ci = 0; ci < NUM_CHUNKS; ci++, gci++) {
        const int ns = ci * N_CHUNK;
        const int an = min(N_CHUNK, N - ns);  // folds per unrolled iteration
        int chunk_slot = (int)(gci % CHUNK_DEPTH);
        int pr = phase_of(gci);
        mbarrier_wait_parity(&plan.bar_ready[chunk_slot], pr);

        float sq_local[N_CHUNK] = {};
        float dot_local[N_CHUNK] = {};
        int4 v_cache[SLICES_PER_GROUP][N_CHUNK];

#pragma unroll
        for (int si = 0; si < SLICES_PER_GROUP; si++) {
          int dt = si * CONSUMER_GROUPS + group;
          if (dt >= NHT) continue;
          float qv[VEC];
          tmem_ld_32dp32bNx<VEC>(my_tmem + si * VEC, qv);

#pragma unroll
          for (int n = 0; n < an; n++) {
            int slot = slot_of(gci, n);
            int4 vp = *reinterpret_cast<const int4*>(buf_ptr(slot) + dt * K_TILE + k_local);
            v_cache[si][n] = vp;
            __nv_bfloat162* v2 = reinterpret_cast<__nv_bfloat162*>(&vp);
            float2 f0 = __bfloat1622float2(v2[0]);
            float2 f1 = __bfloat1622float2(v2[1]);
            float2 f2 = __bfloat1622float2(v2[2]);
            float2 f3 = __bfloat1622float2(v2[3]);
            sq_local[n] += f0.x * f0.x + f0.y * f0.y + f1.x * f1.x + f1.y * f1.y + f2.x * f2.x + f2.y * f2.y +
                           f3.x * f3.x + f3.y * f3.y;
            dot_local[n] += f0.x * qv[0] + f0.y * qv[1] + f1.x * qv[2] + f1.y * qv[3] + f2.x * qv[4] + f2.y * qv[5] +
                            f3.x * qv[6] + f3.y * qv[7];
          }
        }
        mbarrier_arrive(&plan.bar_consumed[chunk_slot]);

        float sq[N_CHUNK], dot[N_CHUNK];
#pragma unroll
        for (int n = 0; n < an; n++) {
          sq[n] = sq_local[n];
          dot[n] = dot_local[n];
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
          for (int n = 0; n < an; n++) {
            sq[n] += __shfl_xor_sync(0xffffffff, sq[n], offset);
            dot[n] += __shfl_xor_sync(0xffffffff, dot[n], offset);
          }
        }
        if (lane == 0) {
#pragma unroll
          for (int n = 0; n < an; n++) {
            plan.ws_sq[comp_wid][n] = sq[n];
            plan.ws_dot[comp_wid][n] = dot[n];
          }
        }
        bar_sync(CONSUMER_BAR_ID, CONSUMER_THREADS);

        float local_rsig = 0.f;
        float local_logit = 0.f;
        if (lane < an) {
          int n = lane;
          float sqs = 0.f;
          float dotss = 0.f;
#pragma unroll
          for (int w = 0; w < CONSUMER_WARPS; w++) {
            sqs += plan.ws_sq[w][n];
            dotss += plan.ws_dot[w][n];
          }
          local_rsig = rsqrtf(sqs / H + rms_eps);
          local_logit = dotss * local_rsig;
        }
        float logit_n[N_CHUNK];
#pragma unroll
        for (int n = 0; n < an; n++) {
          logit_n[n] = __shfl_sync(0xffffffff, local_logit, n);
        }

        float m_chunk = -FLT_MAX;
#pragma unroll
        for (int n = 0; n < an; n++) {
          m_chunk = fmaxf(m_chunk, logit_n[n]);
        }
        float m_new = fmaxf(m_running, m_chunk);
        float corr = exp2f((m_running - m_new) * LOG2_E);
        float w_n[N_CHUNK];
        float w_sum = 0.f;
#pragma unroll
        for (int n = 0; n < an; n++) {
          w_n[n] = exp2f((logit_n[n] - m_new) * LOG2_E);
          w_sum += w_n[n];
        }

#pragma unroll
        for (int si = 0; si < SLICES_PER_GROUP; si++) {
          int dt = si * CONSUMER_GROUPS + group;
          if (dt >= NHT) continue;
          float a[VEC];
#pragma unroll
          for (int j = 0; j < VEC; j++) {
            a[j] = acc32[si * VEC + j] * corr;
          }
#pragma unroll
          for (int n = 0; n < an; n++) {
            int4 vp = v_cache[si][n];
            __nv_bfloat162* v2 = reinterpret_cast<__nv_bfloat162*>(&vp);
            float2 f0 = __bfloat1622float2(v2[0]);
            float2 f1 = __bfloat1622float2(v2[1]);
            float2 f2 = __bfloat1622float2(v2[2]);
            float2 f3 = __bfloat1622float2(v2[3]);
            float wn = w_n[n];
            a[0] += wn * f0.x;
            a[1] += wn * f0.y;
            a[2] += wn * f1.x;
            a[3] += wn * f1.y;
            a[4] += wn * f2.x;
            a[5] += wn * f2.y;
            a[6] += wn * f3.x;
            a[7] += wn * f3.y;
          }
#pragma unroll
          for (int j = 0; j < VEC; j++) {
            acc32[si * VEC + j] = a[j];
          }
        }

        s_running = s_running * corr + w_sum;
        m_running = m_new;
      }

      // Fused out norm: mixed = acc32 * inv_s, out = rmsnorm(mixed) ⊙ ow.
      // sum(mixed^2) = inv_s^2 * sum(acc32^2), reduced across the 8 consumer
      // warps (skipped slices stay zero and contribute nothing).
      float inv_s = 1.f / s_running;
      float ssq = 0.f;
#pragma unroll
      for (int i = 0; i < ACC_PER_THREAD; i++) {
        ssq += acc32[i] * acc32[i];
      }
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        ssq += __shfl_xor_sync(0xffffffff, ssq, offset);
      }
      if (lane == 0) {
        plan.out_ssq[comp_wid] = ssq;
      }
      bar_sync(CONSUMER_BAR_ID, CONSUMER_THREADS);
      float total_sq = plan.out_ssq[0];
#pragma unroll
      for (int w = 1; w < CONSUMER_WARPS; w++) {
        total_sq += plan.out_ssq[w];
      }
      const float scale = inv_s * rsqrtf(total_sq * inv_s * inv_s / H + rms_eps);

      bf16_t* out_ptr = output + (long long)tb * H;
#pragma unroll
      for (int si = 0; si < SLICES_PER_GROUP; si++) {
        int dt = si * CONSUMER_GROUPS + group;
        if (dt >= NHT) continue;
        int h_base = dt * K_TILE + k_local;
        float wv[VEC];
        tmem_ld_32dp32bNx<VEC>(my_tmem + TMEM_CW_COLS + si * VEC, wv);
        bf16_t ov[VEC];
#pragma unroll
        for (int j = 0; j < VEC; j++) {
          ov[j] = __float2bfloat16(acc32[si * VEC + j] * scale * wv[j]);
        }
        *reinterpret_cast<int4*>(out_ptr + h_base) = *reinterpret_cast<int4*>(ov);
      }
    }
  }

  device::PDLTriggerSecondary<kUsePDL>();
  __syncthreads();
  if (wid == 1) {
    tmem_dealloc(plan.tmem_base, TMEM_Q_COLS_TOTAL);
  }
#else
  if (threadIdx.x == 0 && blockIdx.x == 0) printf("attn_res_fwd_online_v2_kernel requires sm_100a\n");
#endif
}

}  // namespace sglang

using namespace sglang;

template <int64_t kDim, uint32_t kMaxBankRows, bool kUsePDL>
struct AttnResFusedTmaKernel {
  using KernelFn = void (*)(const bf16_t*, const bf16_t*, const bf16_t*, const bf16_t*, bf16_t*, int64_t, int, float);

  struct Config {
    KernelFn fn;
    size_t smem_bytes;
    int grid_mul;  // CTAs per SM
  };

  template <int NC, int kGridMul, int kNumRows>
  static constexpr Config make_config() {
    constexpr size_t kSmem = (size_t)CHUNK_DEPTH * NC * kDim * sizeof(bf16_t) + sizeof(FwdSmemPlan<NC>);
    constexpr size_t kSmemAligned = (kSmem + 15) & ~15;
    return Config{
        attn_res_fwd_online_v2_kernel<static_cast<int>(kDim), NC, kNumRows, kUsePDL>,
        kSmemAligned,
        kGridMul,
    };
  }

  // Per-row-count config, aligned with the NV host dispatch: small row counts
  // run half-size chunks so two CTAs share an SM (their H=4096/8192 N<=4 /
  // N<=2 configs); larger counts keep the NV H=7168 production config, where
  // the NC=4 register footprint (288 threads x ~128 regs) caps one CTA/SM.
  template <uint32_t kNvb>
  static constexpr Config config_for() {
    constexpr int kNumRows = kNvb + 1;
    if constexpr (kNumRows <= 4) {
      return make_config<2, 2, kNumRows>();
    } else {
      return make_config<4, 1, kNumRows>();
    }
  }

  template <std::size_t... I>
  static constexpr auto make_table(std::index_sequence<I...>) {
    return std::array<Config, kMaxBankRows + 1>{Config{}, config_for<I + 1>()...};
  }
  static constexpr auto kTable = make_table(std::make_index_sequence<kMaxBankRows>{});
  static_assert(kMaxBankRows + 1 <= N_MAX, "row count exceeds the NV kernel contract");

  static void
  run(const tvm::ffi::TensorView prefix_sum,
      const tvm::ffi::TensorView bank,
      const tvm::ffi::TensorView cw,
      const tvm::ffi::TensorView ow,
      const tvm::ffi::TensorView out,
      int64_t nvb,
      double eps) {
    using namespace host;

    auto T_ = SymbolicSize{"num_tokens"};
    auto H_ = SymbolicSize{"hidden_size"};
    auto NB_ = SymbolicSize{"num_bank_slots"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({T_, H_}).with_dtype<bf16_t>().with_device(device).verify(prefix_sum).verify(out);
    TensorMatcher({T_, NB_, H_}).with_dtype<bf16_t>().with_device(device).verify(bank);
    TensorMatcher({H_}).with_dtype<bf16_t>().with_device(device).verify(cw).verify(ow);

    const auto num_tokens = static_cast<int64_t>(T_.unwrap());
    const auto H = static_cast<int64_t>(H_.unwrap());
    const auto NB = static_cast<int64_t>(NB_.unwrap());

    RuntimeCheck(H == kDim, "attn_res_fused_tma: H must be ", kDim, ", got ", H);
    RuntimeCheck(1 <= nvb && nvb <= kMaxBankRows && nvb <= NB);

    if (num_tokens == 0) return;

    [[maybe_unused]] static const bool _ = [] {
      for (uint32_t i = 1; i <= kMaxBankRows; ++i) {
        if (kTable[i].smem_bytes > 48 * 1024) {
          RuntimeDeviceCheck(
              cudaFuncSetAttribute(kTable[i].fn, cudaFuncAttributeMaxDynamicSharedMemorySize, kTable[i].smem_bytes));
        }
      }
      return true;
    }();

    const auto& config = kTable[nvb];
    const auto num_sm = runtime::get_sm_count(device.unwrap().device_id);
    const auto grid = std::min<int64_t>((int64_t)num_sm * config.grid_mul, num_tokens);
    LaunchKernel(grid, BLK, device.unwrap(), config.smem_bytes)
        .enable_pdl(kUsePDL)(
            config.fn,
            static_cast<const bf16_t*>(bank.data_ptr()),
            static_cast<const bf16_t*>(prefix_sum.data_ptr()),
            static_cast<const bf16_t*>(cw.data_ptr()),
            static_cast<const bf16_t*>(ow.data_ptr()),
            static_cast<bf16_t*>(out.data_ptr()),
            NB * H,
            static_cast<int>(num_tokens),
            static_cast<float>(eps));
  }
};
