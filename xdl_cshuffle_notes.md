## Transforms

grid_desc_Xraw:
- transform = [embed(M, K)] { MK, NK }
- low_idss  = [0]
- up_idss   = [1, 2]
- visible   = [1, 2]
- elem_space_size = MRaw * KRaw

(Padding)

grid_desc_ak0_m_ak1 / bk0_n_bk1:
- transforms    = [unmerge(AK0, AK1), pass_through(M)]
- old_low_idss  = [(1), (0)]
- new_up_idss   = [(0, 2), (1)]

block_desc_ak0_m_ak1 / bk0_n_bk1:
- transform = [embed(AK0PerBlock, MPerBlock, AK1)] {MK, NK}
- low_idss  = [0]
- up_idss   = [1, 2, 3]
- visible   = [1, 2, 3]
- elem_space_size = MPerBlock * KPerBlock

### StaticTensorTupleOfVectorBuffer
src_thread_scratch_desc_:
src_access_lengths_and_vector_length = src_access_lengths + <SrcScalarPerVector>
desc0:
- transform = [unmerge(src_access_lengths_and_vector_length)]
- low_idss  = [0]
- up_idss   = [1, 2, 3, 4]
- visible   = [1, 2, 3, 4]
- elem_space_size = src_access_lengths * SrcScalarPerVector
final:
- transforms    = [i == 2 ? merge(desc0[i], desc0[3]) : pass_through(desc0[i])]
- old_low_idss  = [i == 2 ? [i, 3] : [i]]
- new_up_idss   = [i]



## RunRead / RunWrite
0. Preliminaries
src_scalar_per_access   : for_all_dims{i == SrcVectorDim ? SrcScalarPerVector> : 1} = <1, 1, 8>
SliceIndices            : <AK0PerBlock, MPerBlock, AK1> / ThreadClusterLengths      = <1, 4, 8>
src_access_lengths      : SliceLengths / src_scalar_per_access                      = <1, 4, 1>
ordered_src_access_lengths                                                          = <4, 1, 1>

src_forward_steps / src_backward_steps: de-one-hot(src_scalar_per_access)

1. RunRead
Use llvm.amdgcn.raw.buffer.load.* series to move from (global) src_buf into src_vector_container; and then (via some cast or as_type?) to (VGPR) scratch.
src_thread_scratch_tuple_: StaticallyIndexedArray<StaticTensorTupleOfVectorBuffer, NumThreadScratch=1>,
- AddressSpaceEnum::Vgpr

2. RunWrite
(Via some cast or as_type?) from (VGPR) scratch to dst_vector_container; and then (via some cast or as_type?) to (LDS) dst_buf.


### Compute
BlockwiseGemmXdlops_k0mk1_k0nk1_m0n0m1n1m2m3m4n2_v1

#### Thread Copy
ThreadwiseTensorSliceTransfer_v4

a_block_desc_m0_m1_m2_k: (start from ak0_m_ak1)
- transforms    = [merge(AK0, AK1), unmerge(MRepeat, MWaves, MPerXDL)]
- old_low_idss  = [(0, 2), (1)]
- new_up_idss   = [(3), (0, 1, 2)]
SrcRefToOriginDisplacement: thread to block index mapping
- <m0, I0, I0, k * AMmaKStride>, where AMmaKStride = KPack * K0PerXdlops, KPack = max(lcm(AK1, BK1), mfma.k_per_blk); m0 is loop index of MRepeat, k is loop index inside KPerThread.

ThreadCopy::Run()
- Loop:
- - access_lengths                                                        = <1, 1, 1, KPack>
- - src_scalar_per_access                                                 = <1, 1, 1, AK1>
- - ordered_access_lengths    : access_lengths / src_scalar_per_access    = <1, 1, 1, KPack/AK1>
- - static_ford(ordered_access_lengths) {
    data_to_origin_disp_idx = ordered_access_idx * src_scalar_per_access;
    src_ref_to_data_disp_idx += data_to_origin_disp_idx;
}
- Copy data: block_buf -> src_tmp_vector -> dst_tmp_vector -> thread_buf


#### Others
thread_buf -> thread_vec
mfma compute (for order: m (load A) n (load B) k (compute))


### C shuffle and write out
c_thread_desc_m0_n0_m1_n1_m2_m3_m4_n2:
- pre-init  : <M0, M1, M2, N>   = (mfma.num_groups_per_blk, 1, mfma.group_size, 1)
- init      : <MRepeat, NRepeat, 1, 1, M0, M1, M2, N> (transform=unmerge(all), low=[0], up=...)

c_grid_desc_mblock_mperblock_nblock_nperblock_:
- init: <M, N>
- transforms    = [unmerge(MBlock, MPerBlock), unmerge(NBlock, NPerBlock)]
- old_low_idss  = [(0), (1)]
- new_up_idss   = [(0, 1), (2, 3)]

c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp: (only used to get lengths)
- init      : <MRepeat, NRepeat, MWaves, NWaves, MPerXdl, NPerXdl> (transform=unmerge(all), low=[0], up=...)
- transforms    = [pass_through(<MRepeat, NRepeat, MWaves, NWaves>),
               unmerge(mfma.num_groups_per_blk, mfma.num_input_blks, mfma.group_size),
               pass_through(mfma.num_threads_per_blk)]
- old_low_idss  = [(0), (1), (2), (3), (4), (5)]
- new_up_idss   = [(0), (1), (2), (3), (4, 5, 6), (7)]

c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2:
- init: <1, CShuffleMXdlPerWavePerShuffle * MWave * MPerXdl, 1, CShuffleNXdlPerWavePerShuffle * NWave * NPerXdl> (transform=unmerge(all), low=[0], up=...)
- transforms    = [freeze(0), unmerge(CShuffleMXdlPerWavePerShuffle, M1, M2, M3, M4),
                   freeze(0), unmerge(CShuffleNXdlPerWavePerShuffle, N1, N2)]
- old_low_idss  = [(0), (1), (2), (3)]
- new_up_idss   = [(), (0, 2, 4, 5, 6), (), (1, 3, 7)]


c_thread_mtx_on_block: (c_thread_m, c_thread_n)
* (waveId_m, waveId_n, _): thread_id -> merge(MWaves, NWaves, WaveSize)
* blk_idx               : (blk_i * mfma_instr.n_per_blk + blk_td,
                           xdlops_i * mfma_instr.m_per_blk + blk_id * mfma_instr.group_size)
* c_thread_m            : () -> unmerge(m0, waveId_m, blk_idx[I0])
* c_thread_n            : () -> unmerge(n0, waveId_n, blk_idx[I1])

m_thread_data_on_block_idx  : c_thread_m -> (M0, M1, ..., M4)
n_thread_data_on_block_idx  : c_thread_n -> (N0, N1, N2)


#### C data copy
num_access(src) = MXdlPerWave * NXdlPerWave * M2 * M4 / ScalarPerVector
                = MPerBlock * NPerBlock / ScalarPerVector
- static_for<access_id: num_access(src)>:
    num_access(dst) = CShuffleMXdlPerWavePerShuffle * CShuffleNXdlPerWavePerShuffle * M2 * M4 / ScalarPerVector
    c_thread_copy_vgpr_to_lds::Run():
    - - static_for<idx_1d: num_access(dst)>
        - - - static_for<i: DstScalarPerVector>:
            src_slice_origin_idx        = (0, 0, M1, N1, M2, M3, M4, N2)
            dst_scalar_step_in_vector   = (0, 0, 0, 0, 0, 0, 0, 1)
            idx_md                      : (SnakeCurved or not) idx_id -> delinearized indexes of each dim
            src_offset  = src_slice_origin_idx + idx_md + i * dst_scalar_step_in_vector
        dst_coord   : (0, 0, M1, N1, M2, M3, M4, N2)

    c_shuffle_block_copy_lds_to_global::Run():
    num_access(dst) = (CShuffleMXdlPerWavePerShuffle * MWave * MPerXdl)
                    * (CShuffleNXdlPerWavePerShuffle * NWave * NPerXdl)
                    /  CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
                    /  ScalarPerVector
    - - static_for<idx_1d: num_access(dst)>
        - - - static_for<i: DstScalarPerVector>:
        src_coord(origin)   : c_shuffle_block_desc_mblock_mperblock_nblock_nperblock (all-zeros)
        dst_coord(origin)   : c_grid_desc_mblock_mperblock_nblock_nperblock (all-zeros)




### Values

MainKBlockLoop = ceil(K / KPerBlock)
MWave = MPerBlock / (MXdlPerWave * MPerXdl)
NWave = NPerBlock / (NXdlPerWave * NPerXdl)

DefaultBlockToCTileMap:
Delinearize [M, N] into [M00, N0, M01=8].
grid_size = ceil((M00 * N0) / (MPerBlock * NPerBlock))
BottomIndex: (block_1d_id) -> (m, n) where
    block_1d_id is calculated along [M00, N0, M01], and
    (m, n) is the index inside the [M01, N0] area.
    Ref: the image in include\ck\tensor_operation\gpu\grid\block_to_ctile_map.hpp

#### MFMA
num_regs_per_blk    = group_size * num_groups_per_blk
num_input_blks      = wave_size / num_threads_per_blk