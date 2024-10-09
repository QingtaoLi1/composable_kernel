#include "common.hpp"

#include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle.hpp"


ck::tensor_operation::device::DeviceGemmXdl<...>;


template <typename ADataType,
          typename BDataType,
          typename CDataType,
          typename AccDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          GemmSpecialization GemmSpec,
          ck::index_t BlockSize,
          ck::index_t MPerBlock,
          ck::index_t NPerBlock,
          ck::index_t K0PerBlock,
          ck::index_t K1,
          ck::index_t MPerXDL,
          ck::index_t NPerXDL,
          ck::index_t MXdlPerWave,
          ck::index_t NXdlPerWave,
          typename ABlockTransferThreadClusterLengths_K0_M_K1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          ck::index_t ABlockTransferSrcVectorDim,
          ck::index_t ABlockTransferSrcScalarPerVector,
          ck::index_t ABlockTransferDstScalarPerVector_K1,
          bool ABlockLdsAddExtraM,
          typename BBlockTransferThreadClusterLengths_K0_N_K1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          ck::index_t BBlockTransferSrcVectorDim,
          ck::index_t BBlockTransferSrcScalarPerVector,
          ck::index_t BBlockTransferDstScalarPerVector_K1,
          bool BBlockLdsAddExtraN,
          ck::index_t CThreadTransferSrcDstVectorDim,
          ck::index_t CThreadTransferDstScalarPerVector,
          ck::index_t NumPrefetch         = 1,
          ck::LoopScheduler LoopSched     = make_default_loop_scheduler(),
          ck::PipelineVersion PipelineVer = ck::PipelineVersion::v1>
struct DeviceGemmXdl : public DeviceGemm<ALayout, // DeviceGemm not important
                                         BLayout,
                                         CLayout,
                                         ADataType,
                                         BDataType,
                                         CDataType,
                                         AElementwiseOperation,
                                         BElementwiseOperation,
                                         CElementwiseOperation>
{
    // No constructor
};


template <index_t BlockSize,
          typename FloatAB,
          typename FloatAcc,
          typename FloatC,
          InMemoryDataOperationEnum CGlobalMemoryDataOperation,
          typename AGridDesc_K0_M_K1,
          typename BGridDesc_K0_N_K1,
          typename CGridDesc_M_N,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t K0PerBlock,
          index_t MPerXDL,
          index_t NPerXDL,
          index_t K1Value,
          index_t MXdlPerWave,
          index_t NXdlPerWave,
          typename ABlockTransferThreadClusterLengths_K0_M_K1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          index_t ABlockTransferSrcVectorDim,
          index_t ABlockTransferSrcScalarPerVector,
          index_t ABlockTransferDstScalarPerVector_K1,
          bool AThreadTransferSrcResetCoordinateAfterRun,
          bool ABlockLdsExtraM,
          typename BBlockTransferThreadClusterLengths_K0_N_K1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          index_t BBlockTransferSrcVectorDim,
          index_t BBlockTransferSrcScalarPerVector,
          index_t BBlockTransferDstScalarPerVector_K1,
          bool BThreadTransferSrcResetCoordinateAfterRun,
          bool BBlockLdsExtraN,
          typename CThreadTransferSrcDstAccessOrder,
          index_t CThreadTransferSrcDstVectorDim,
          index_t CThreadTransferDstScalarPerVector,
          index_t NumGemmKPrefetchStage = 1,
          LoopScheduler LoopSched       = make_default_loop_scheduler(),
          PipelineVersion PipelineVer   = PipelineVersion::v1>
struct GridwiseGemm_k0mk1_k0nk1_mn_xdlops_v2r3
{
    // No constructor

    template <bool HasMainKBlockLoop, typename Block2CTileMap = DefaultBlock2CTileMap>
    __device__ static void
    Run(const FloatAB* __restrict__ p_a_grid,
        const FloatAB* __restrict__ p_b_grid,
        FloatC* __restrict__ p_c_grid,
        void* __restrict__ p_shared,
        const AGridDesc_K0_M_K1& a_grid_desc_k0_m_k1,
        const BGridDesc_K0_N_K1& b_grid_desc_k0_n_k1,
        const CGridDesc_M0_N0_M1_N1_M2_M3_M4_N2& c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2,
        const AElementwiseOperation& a_element_op,
        const BElementwiseOperation& b_element_op,
        const CElementwiseOperation& c_element_op,
        const Block2CTileMap& block_2_ctile_map)
    {};
};

{
    // G2S
    template <typename ThreadGroup,
            typename SrcElementwiseOperation,
            typename DstElementwiseOperation,
            InMemoryDataOperationEnum DstInMemOp,
            typename BlockSliceLengths,
            typename ThreadClusterLengths,
            typename ThreadClusterArrangeOrder,
            typename SrcData,
            typename DstData,
            typename SrcDesc,
            typename DstDesc,
            typename SrcDimAccessOrder,
            typename DstDimAccessOrder,
            index_t SrcVectorDim,
            index_t DstVectorDim,
            index_t SrcScalarPerVector,
            index_t DstScalarPerVector,
            index_t SrcScalarStrideInVector,
            index_t DstScalarStrideInVector,
            bool ThreadTransferSrcResetCoordinateAfterRun,
            bool ThreadTransferDstResetCoordinateAfterRun,
            index_t NumThreadScratch = 1>
    struct ThreadGroupTensorSliceTransfer_v4r1
    {
        __device__ constexpr ThreadGroupTensorSliceTransfer_v4r1(
            const SrcDesc& src_desc,
            const Index& src_block_slice_origin,
            const SrcElementwiseOperation& src_element_op,
            const DstDesc& dst_desc,
            const Index& dst_block_slice_origin,
            const DstElementwiseOperation& dst_element_op){};
        
        template <typename SrcBuffer, index_t ThreadScratchId = 0>
        __device__ void RunRead(const SrcDesc& src_desc,
                                const SrcBuffer& src_buf,
                                Number<ThreadScratchId> thread_scratch_id = Number<ThreadScratchId>{})
        {}

        template <typename DstBuffer, index_t ThreadScratchId = 0>
        __device__ void RunWrite(const DstDesc& dst_desc,
                                DstBuffer& dst_buf,
                                Number<ThreadScratchId> thread_scratch_id = Number<ThreadScratchId>{})
        {}

        __device__ void MoveSrcSliceWindow(const SrcDesc& src_desc, const Index& step){}
        __device__ void MoveSrcSliceWindow(const DstDesc& dst_desc, const Index& step){}
        

    };

    // S-level gemm
    template <index_t BlockSize,
            typename FloatAB,
            typename FloatAcc,
            typename AK0MK1BlockDesc,
            typename BK0NK1BlockDesc,
            index_t MPerXDL,
            index_t NPerXDL,
            index_t MRepeat,
            index_t NRepeat,
            index_t KPack>
    struct BlockwiseGemmXdlops_k0mk1_k0nk1_m0n0m1n1m2m3m4n2_v1
    {
        // Only sanity check in constructor

        template <typename ABlockBuffer, typename BBlockBuffer, typename CThreadBuffer>
        __device__ void Run(const ABlockBuffer& a_block_buf,
                            const BBlockBuffer& b_block_buf,
                            CThreadBuffer& c_thread_buf) const
        {}
    };

    {
        // S2R
        template <typename SrcData,
                typename DstData,
                typename SrcDesc,
                typename DstDesc,
                typename SliceLengths,
                typename DimAccessOrder,
                index_t SrcVectorDim,
                index_t SrcScalarPerVector,
                index_t SrcScalarStrideInVector,
                typename enable_if<SrcDesc::IsKnownAtCompileTime() && DstDesc::IsKnownAtCompileTime(),
                                    bool>::type = false>
        struct ThreadwiseTensorSliceTransfer_v4
        {
            // Only sanity check in constructor
            
            template <typename SrcRefToOriginDisplacement,
                    typename DstOriginIdx,
                    typename SrcBuffer,
                    typename DstBuffer>
            __device__ void Run(const SrcDesc&,
                                const SrcRefToOriginDisplacement&,
                                const SrcBuffer& src_buf,
                                const DstDesc&,
                                const DstOriginIdx&,
                                DstBuffer& dst_buf) const
            {}

            template <typename SrcSliceMoveStepIdx>
            __device__ void MoveSrcSliceWindow(const SrcDesc&,
                                            const SrcSliceMoveStepIdx& src_slice_move_step_idx)
            {}
        };

        // R-level gemm
        template <typename base_type,   // mfma input type
                index_t MPerXdlops,
                index_t NPerXdlops,
                index_t KPack,
                bool TransposeC = false>
        struct XdlopsGemm
        {
            // Only sanity check in constructor

            template <class FloatA, class FloatB, class FloatC>
            __device__ void Run(const FloatA& p_a_wave, const FloatB& p_b_wave, FloatC& p_c_thread) const
            {}
        };
    }

    // R2G
    template <typename SrcData,
            typename DstData,
            typename SrcDesc,
            typename DstDesc,
            typename ElementwiseOperation,
            typename SliceLengths,
            typename DimAccessOrder,
            index_t DstVectorDim,
            index_t DstScalarPerVector,
            InMemoryDataOperationEnum DstInMemOp,
            index_t DstScalarStrideInVector,
            bool DstResetCoordinateAfterRun,
            typename enable_if<SrcDesc::IsKnownAtCompileTime(), bool>::type = false>
    struct ThreadwiseTensorSliceTransfer_v1r3
    {
        __device__ constexpr ThreadwiseTensorSliceTransfer_v1r3(const DstDesc& dst_desc,
                                                                const Index& dst_slice_origin_idx,
                                                                const ElementwiseOperation& element_op)
        {}

        template <typename SrcSliceOriginIdx, typename SrcBuffer, typename DstBuffer>
        __device__ void Run(const SrcDesc&,
                            const SrcSliceOriginIdx&,
                            const SrcBuffer& src_buf,
                            const DstDesc& dst_desc,
                            DstBuffer& dst_buf)
        {}

    };

}