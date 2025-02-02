// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/welford/thread/thread_welford.hpp"
#include "ck_tile/ops/welford/warp/warp_welford.hpp"

namespace ck_tile {

// TODO: Extract some type to wrapper class
template <typename Problem_>
struct Layernorm2dFwd
{
    using Problem = ck_tile::remove_cvref_t<Problem_>;

    using XDataType       = ck_tile::remove_cvref_t<typename Problem::XDataType>;
    using GammaDataType   = ck_tile::remove_cvref_t<typename Problem::GammaDataType>;
    using BetaDataType    = ck_tile::remove_cvref_t<typename Problem::BetaDataType>;
    using ComputeDataType = ck_tile::remove_cvref_t<typename Problem::ComputeDataType>;
    using YDataType       = ck_tile::remove_cvref_t<typename Problem::YDataType>;
    using MeanDataType    = ck_tile::remove_cvref_t<typename Problem::MeanDataType>;
    using InvStdDataType  = ck_tile::remove_cvref_t<typename Problem::InvStdDataType>;

    static constexpr bool kHasGamma   = !std::is_same_v<GammaDataType, ck_tile::null_type>;
    static constexpr bool kHasBeta    = !std::is_same_v<BetaDataType, ck_tile::null_type>;
    static constexpr bool kSaveMean   = !std::is_same_v<MeanDataType, ck_tile::null_type>;
    static constexpr bool kSaveInvStd = !std::is_same_v<InvStdDataType, ck_tile::null_type>;

    static constexpr ck_tile::index_t kMPerBlock = Problem::BlockShape::kMPerBlock;
    static constexpr ck_tile::index_t kNPerBlock = Problem::BlockShape::kNPerBlock;
    static constexpr bool kPadM                  = Problem::kPadM;
    static constexpr bool kPadN                  = Problem::kPadN;

    static constexpr ck_tile::index_t kNThreadPerWarp = Problem::BlockShape::kNThreadPerWarp;
    static constexpr ck_tile::index_t kNPerThread     = Problem::BlockShape::kNPerThread;

    static constexpr auto I0 = number<0>{};
    static constexpr auto I1 = number<1>{};

    struct Kargs
    {
        const void* p_x;
        const void* p_gamma;
        const void* p_beta;

        void* p_y;
        void* p_mean;
        void* p_invStd;

        float epsilon;

        ck_tile::index_t M;
        ck_tile::index_t N;
    };

    CK_TILE_HOST static constexpr Kargs MakeKargs(const void* p_x,
                                                  const void* p_gamma,
                                                  const void* p_beta,
                                                  void* p_y,
                                                  void* p_mean,
                                                  void* p_invStd,
                                                  float epsilon,
                                                  ck_tile::index_t M,
                                                  ck_tile::index_t N)
    {
        return Kargs{p_x, p_gamma, p_beta, p_y, p_mean, p_invStd, epsilon, M, N};
    }

    CK_TILE_HOST static constexpr auto GridSize(ck_tile::index_t M) { return M / kMPerBlock; }

    CK_TILE_HOST static constexpr auto BlockSize() { return Problem::BlockShape::kBlockSize; }

    CK_TILE_DEVICE static constexpr auto MakeXBlockTileDistribution()
    {
        using S = typename Problem::BlockShape;

        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<>,
                tuple<sequence<S::kMWarpPerBlock, S::kMThreadPerWarp, S::kMPerThread>,
                      sequence<S::kNWarpPerBlock, S::kNThreadPerWarp, S::kNPerThread>>,
                tuple<sequence<1, 2>, sequence<1, 2>>,
                tuple<sequence<0, 0>, sequence<1, 1>>,
                sequence<1, 2>,
                sequence<2, 2>>{});
    }

    CK_TILE_DEVICE static constexpr auto MakeGammaBetaBlockTileDistribution()
    {
        using S = typename Problem::BlockShape;

        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<S::kMWarpPerBlock, S::kMThreadPerWarp>,
                tuple<sequence<S::kNWarpPerBlock, S::kNThreadPerWarp, S::kNPerThread>>,
                tuple<sequence<0, 1>, sequence<0, 1>>,
                tuple<sequence<0, 0>, sequence<1, 1>>,
                sequence<1>,
                sequence<2>>{});
    }

    CK_TILE_DEVICE static int GetWelfordMaxCount(int N)
    {
        constexpr ck_tile::index_t kNThreadPerBlock = kNPerBlock / kNPerThread;

        int thread_id_n = get_thread_id() % kNThreadPerBlock;
        int max_count =
            __builtin_amdgcn_readfirstlane(N < kNPerBlock ? 0 : kNPerThread * (N / kNPerBlock));
        int n_per_block_tail_loop =
            __builtin_amdgcn_readfirstlane(N - max_count * kNThreadPerBlock);

        if(n_per_block_tail_loop > 0)
        {
            int thread_max_n = (thread_id_n + 1) * kNPerThread;
            int delta        = thread_max_n - n_per_block_tail_loop;
            delta            = clamp(thread_max_n - n_per_block_tail_loop, 0, kNPerThread);
            max_count += kNPerThread - delta;
        }

        return max_count;
    }

    template <typename DistributedTensor>
    CK_TILE_DEVICE static auto InvSqrt(const DistributedTensor& in_dstr_tensor,
                                       const ComputeDataType epsilon)
    {
        // TODO: Investigate fast inverse square root algorithm with epsilon
        constexpr auto spans = DistributedTensor::get_distributed_spans();

        DistributedTensor out_dstr_tensor;

        sweep_tile_span(spans[number<0>{}], [&](auto idx0) {
            constexpr auto i_idx   = make_tuple(idx0);
            out_dstr_tensor(i_idx) = type_convert<ComputeDataType>(1.0f) /
                                     ck_tile::sqrt(in_dstr_tensor[i_idx] + epsilon);
        });

        return out_dstr_tensor;
    }

    template <typename XBlockWindow,
              typename GammaBlockWindow,
              typename BetaBlockWindow,
              typename YBlockWindow,
              typename MeanBlockWindow,
              typename InvStdBlockWindow,
              bool Cond = (kHasGamma && kHasBeta)>
    CK_TILE_DEVICE std::enable_if_t<Cond>
    TwoPassLayernorm2dFwd(XBlockWindow& x_block_window,
                          GammaBlockWindow& gamma_block_window,
                          BetaBlockWindow& beta_block_window,
                          YBlockWindow& y_block_window,
                          MeanBlockWindow& mean_block_window,
                          InvStdBlockWindow& inv_std_block_window,
                          ComputeDataType epsilon,
                          ck_tile::index_t N) const
    {
        // TODO - Optimize tail loop to reduce move_tile_window()
        index_t num_n_tile_iteration =
            __builtin_amdgcn_readfirstlane(integer_divide_ceil(N, kNPerBlock));

        int welford_max_count = GetWelfordMaxCount(N);
        ThreadWelford<ComputeDataType, XDataType> thread_welford{welford_max_count};

        using XTensorType = decltype(load_tile(x_block_window));
        auto mean_compute_block_tensor =
            thread_welford.template MakeInitialMeanVarDistributedTensor<XTensorType>();
        auto var_compute_block_tensor =
            thread_welford.template MakeInitialMeanVarDistributedTensor<XTensorType>();

        clear_tile(mean_compute_block_tensor);
        clear_tile(var_compute_block_tensor);

        for(int iN = __builtin_amdgcn_readfirstlane(0); iN < num_n_tile_iteration; ++iN)
        {
            const auto x_block_tensor = load_tile(x_block_window);

            thread_welford(x_block_tensor, mean_compute_block_tensor, var_compute_block_tensor);
            move_tile_window(x_block_window, {0, kNPerBlock});
        }

        // TODO: support cross warp Welford
        WarpMergeWelford<ComputeDataType, true>{}(
            mean_compute_block_tensor, var_compute_block_tensor, thread_welford.cur_count_);

        auto inv_std_compute_block_tensor = InvSqrt(var_compute_block_tensor, epsilon);

        if constexpr(kSaveMean)
            store_tile(mean_block_window, cast_tile<MeanDataType>(mean_compute_block_tensor));
        if constexpr(kSaveInvStd)
            store_tile(inv_std_block_window,
                       cast_tile<InvStdDataType>(inv_std_compute_block_tensor));

        // reverse read x to reuse cache
        ck_tile::index_t stride_to_right_most_window =
            N % kNPerBlock == 0 ? N - kNPerBlock : N - N % kNPerBlock;

        move_tile_window(x_block_window, {0, -kNPerBlock});
        move_tile_window(gamma_block_window, {stride_to_right_most_window});
        move_tile_window(beta_block_window, {stride_to_right_most_window});
        move_tile_window(y_block_window, {0, stride_to_right_most_window});

        // Normalization
        for(int iN = __builtin_amdgcn_readfirstlane(0); iN < num_n_tile_iteration; ++iN)
        {
            const auto x_block_tensor     = load_tile(x_block_window);
            const auto gamma_block_tensor = load_tile(gamma_block_window);
            const auto beta_block_tensor  = load_tile(beta_block_window);

            constexpr auto x_spans = decltype(x_block_tensor)::get_distributed_spans();

            auto y_block_tensor =
                make_static_distributed_tensor<YDataType>(x_block_tensor.get_tile_distribution());

            sweep_tile_span(x_spans[I1], [&](auto idx1) {
                constexpr auto j_idx = make_tuple(idx1);
                const auto gamma     = type_convert<ComputeDataType>(gamma_block_tensor[j_idx]);
                const auto beta      = type_convert<ComputeDataType>(beta_block_tensor[j_idx]);

                sweep_tile_span(x_spans[I0], [&](auto idx0) {
                    constexpr auto i_idx   = make_tuple(idx0);
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);

                    const auto mean    = mean_compute_block_tensor[i_idx];
                    const auto inv_std = inv_std_compute_block_tensor[i_idx];

                    const auto x = type_convert<ComputeDataType>(x_block_tensor[i_j_idx]);
                    auto y       = (x - mean) * inv_std * gamma + beta;

                    y_block_tensor(i_j_idx) = type_convert<YDataType>(y);
                });
            });

            store_tile(y_block_window, y_block_tensor);

            move_tile_window(x_block_window, {0, -kNPerBlock});
            move_tile_window(gamma_block_window, {-kNPerBlock});
            move_tile_window(beta_block_window, {-kNPerBlock});
            move_tile_window(y_block_window, {0, -kNPerBlock});
        }
    }

    template <typename XBlockWindow,
              typename GammaBlockWindow,
              typename BetaBlockWindow,
              typename YBlockWindow,
              typename MeanBlockWindow,
              typename InvStdBlockWindow,
              bool Cond = (kHasGamma && kHasBeta)>
    CK_TILE_DEVICE std::enable_if_t<Cond>
    OnePassLayernorm2dFwd(XBlockWindow& x_block_window,
                          GammaBlockWindow& gamma_block_window,
                          BetaBlockWindow& beta_block_window,
                          YBlockWindow& y_block_window,
                          MeanBlockWindow& mean_block_window,
                          InvStdBlockWindow& inv_std_block_window,
                          ComputeDataType epsilon,
                          ck_tile::index_t N) const
    {
        int welford_max_count = GetWelfordMaxCount(N);
        ThreadWelford<ComputeDataType, XDataType> thread_welford{welford_max_count};

        using XTensorType = decltype(load_tile(x_block_window));
        auto mean_compute_block_tensor =
            thread_welford.template MakeInitialMeanVarDistributedTensor<XTensorType>();
        auto var_compute_block_tensor =
            thread_welford.template MakeInitialMeanVarDistributedTensor<XTensorType>();

        clear_tile(mean_compute_block_tensor);
        clear_tile(var_compute_block_tensor);

        const auto x_block_tensor = load_tile(x_block_window);
        thread_welford(x_block_tensor, mean_compute_block_tensor, var_compute_block_tensor);
        // TODO: support cross warp Welford
        WarpMergeWelford<ComputeDataType, true>{}(
            mean_compute_block_tensor, var_compute_block_tensor, thread_welford.cur_count_);

        auto inv_std_compute_block_tensor = InvSqrt(var_compute_block_tensor, epsilon);

        if constexpr(kSaveMean)
            store_tile(mean_block_window, cast_tile<MeanDataType>(mean_compute_block_tensor));
        if constexpr(kSaveInvStd)
            store_tile(inv_std_block_window,
                       cast_tile<InvStdDataType>(inv_std_compute_block_tensor));

        // normalize
        const auto gamma_block_tensor = load_tile(gamma_block_window);
        const auto beta_block_tensor  = load_tile(beta_block_window);

        constexpr auto x_spans = decltype(x_block_tensor)::get_distributed_spans();

        auto y_block_tensor =
            make_static_distributed_tensor<YDataType>(x_block_tensor.get_tile_distribution());

        sweep_tile_span(x_spans[I1], [&](auto idx1) {
            constexpr auto j_idx = make_tuple(idx1);
            const auto gamma     = type_convert<ComputeDataType>(gamma_block_tensor[j_idx]);
            const auto beta      = type_convert<ComputeDataType>(beta_block_tensor[j_idx]);

            sweep_tile_span(x_spans[I0], [&](auto idx0) {
                constexpr auto i_idx   = make_tuple(idx0);
                constexpr auto i_j_idx = make_tuple(idx0, idx1);

                const auto mean    = mean_compute_block_tensor[i_idx];
                const auto inv_std = inv_std_compute_block_tensor[i_idx];

                const auto x = type_convert<ComputeDataType>(x_block_tensor[i_j_idx]);
                auto y       = (x - mean) * inv_std * gamma + beta;

                y_block_tensor(i_j_idx) = type_convert<YDataType>(y);
            });
        });

        store_tile(y_block_window, y_block_tensor);
    }

    CK_TILE_DEVICE void operator()(Kargs kargs) const
    {
        const auto x_m_n = [&]() {
            const auto x_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                static_cast<const XDataType*>(kargs.p_x),
                make_tuple(kargs.M, kargs.N),
                make_tuple(kargs.N, 1),
                number<kNPerThread>{},
                number<1>{});

            return pad_tensor_view(x_dram_naive,
                                   make_tuple(number<kMPerBlock>{}, number<kNPerBlock>{}),
                                   sequence<kPadM, kPadN>{});
        }();

        const auto gamma_n = [&]() {
            const auto gamma_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                static_cast<const GammaDataType*>(kargs.p_gamma),
                make_tuple(kargs.N),
                make_tuple(1),
                number<kNPerThread>{},
                number<1>{});

            return pad_tensor_view(
                gamma_dram_naive, make_tuple(number<kNPerBlock>{}), sequence<kPadN>{});
        }();

        const auto beta_n = [&]() {
            const auto gamma_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                static_cast<const BetaDataType*>(kargs.p_beta),
                make_tuple(kargs.N),
                make_tuple(1),
                number<kNPerThread>{},
                number<1>{});

            return pad_tensor_view(
                gamma_dram_naive, make_tuple(number<kNPerBlock>{}), sequence<kPadN>{});
        }();

        const auto iM = get_block_id() * kMPerBlock;

        constexpr auto xDstr = MakeXBlockTileDistribution();

        auto x_block_window = make_tile_window(
            x_m_n, make_tuple(number<kMPerBlock>{}, number<kNPerBlock>{}), {iM, 0}, xDstr);

        const auto y_m_n = [&]() {
            const auto y_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                static_cast<YDataType*>(kargs.p_y),
                make_tuple(kargs.M, kargs.N),
                make_tuple(kargs.N, 1),
                number<kNPerThread>{},
                number<1>{});

            return pad_tensor_view(y_dram_naive,
                                   make_tuple(number<kMPerBlock>{}, number<kNPerBlock>{}),
                                   sequence<kPadM, kPadN>{});
        }();

        auto y_block_window = make_tile_window(
            y_m_n, make_tuple(number<kMPerBlock>{}, number<kNPerBlock>{}), {iM, 0});

        constexpr auto gammaDstr = MakeGammaBetaBlockTileDistribution();
        constexpr auto betaDstr  = gammaDstr;

        auto gamma_block_window =
            make_tile_window(gamma_n, make_tuple(number<kNPerBlock>{}), {0}, gammaDstr);

        auto beta_block_window = make_tile_window(
            beta_n, make_tuple(number<kMPerBlock>{}, number<kNPerBlock>{}), {0}, betaDstr);

        auto mean_block_window = [&]() {
            if constexpr(kSaveMean)
            {
                const auto mean_m = [&]() {
                    const auto mean_dram_naive =
                        make_naive_tensor_view_packed<address_space_enum::global>(
                            static_cast<MeanDataType*>(kargs.p_mean),
                            make_tuple(kargs.M),
                            number<1>{});

                    return pad_tensor_view(
                        mean_dram_naive, make_tuple(number<kMPerBlock>{}), sequence<kPadM>{});
                }();

                return make_tile_window(mean_m, make_tuple(number<kMPerBlock>{}), {iM});
            }
            else
                return make_null_tile_window(make_tuple(number<kMPerBlock>{}));
        }();

        auto inv_std_block_window = [&]() {
            if constexpr(kSaveInvStd)
            {
                const auto inv_std_m = [&]() {
                    const auto inv_std_dram_naive =
                        make_naive_tensor_view_packed<address_space_enum::global>(
                            static_cast<InvStdDataType*>(kargs.p_invStd),
                            make_tuple(kargs.M),
                            number<1>{});

                    return pad_tensor_view(
                        inv_std_dram_naive, make_tuple(number<kMPerBlock>{}), sequence<kPadM>{});
                }();

                return make_tile_window(inv_std_m, make_tuple(number<kMPerBlock>{}), {iM});
            }
            else
                return make_null_tile_window(make_tuple(number<kMPerBlock>{}));
        }();

        if(kargs.N <= kNPerBlock)
            OnePassLayernorm2dFwd(x_block_window,
                                  gamma_block_window,
                                  beta_block_window,
                                  y_block_window,
                                  mean_block_window,
                                  inv_std_block_window,
                                  static_cast<const ComputeDataType>(kargs.epsilon),
                                  kargs.N);
        else
            TwoPassLayernorm2dFwd(x_block_window,
                                  gamma_block_window,
                                  beta_block_window,
                                  y_block_window,
                                  mean_block_window,
                                  inv_std_block_window,
                                  static_cast<const ComputeDataType>(kargs.epsilon),
                                  kargs.N);
    }
};

} // namespace ck_tile
