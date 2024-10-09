# // ######| ALayout| BLayout| CLayout|     AData|     BData|     CData|     AccData|         CShuffle|           A|           B|           C|           GEMM| NumGemmK| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|
# // ######|        |        |        |      Type|      Type|      Type|        Type|         DataType| Elementwise| Elementwise| Elementwise| Spacialization| Prefetch|  Size| Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave|         _MBlock_MWaveMPerXdl| ScalarPerVector|
# // ######|        |        |        |          |          |          |            |                 |   Operation|   Operation|   Operation|               |    Stage|      |      |      |      |    |    |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|         _NBlock_NWaveNPerXdl|   _NWaveNPerXdl|
# // ######|        |        |        |          |          |          |            |                 |            |            |            |               |         |      |      |      |      |    |    |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                             |                |
#          < ALayout, BLayout, CLayout, ADataType, BDataType, CDataType, AccDataType, CShuffleDataType,  AElementOp,  BElementOp,  CElementOp,    GemmDefault,        1,   256,   256,   128,    32,   8,   8,   32,   32,    4,    2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,               S<1, 32, 1, 8>,               8>;


import os
from typing import Dict


origin_str = "         <"
# fixed
origin_str += "ALayout, BLayout, CLayout, ADataType, BDataType, CDataType, AccDataType, CShuffleDataType,  AElementOp,  BElementOp,  CElementOp,    GemmDefault, 1,"
# BlockSize, MPerBlock, NPerBlock, KPerBlock
origin_str += "256,   256,   128,    32,"
# AK1, BK1, MPerXDL, NPerXDL, MXdlPerWave, NXdlPerWave
origin_str += "8,   8,   32,   32,    4,    2,"
# ABlockThreadCluster_K0_M_K1, A..._ArangeOrder, A..._SrcAccessOrder, A..._SrcVectorDim, A..._SrcScalarPerVector, A..._DstScalarPerVector_K1, A..._AddExtraM
origin_str += "S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,"
# B
origin_str += "S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,"
# CShuffleMXdlPerWavePerShuffle, CShuffleNXdlPerWavePerShuffle, CBlockThreadCluster_MBlock_MWaveMPerXdl_NBlock_NWaveNPerXdl, C..._ScalarPerVector_NWaveNPerXdl
origin_str += "1,           1,               S<1, 32, 1, 8>,               8>;"


BlockSize = 256
MPerBlock = 256
NPerBlock = 128
KPerBlock = 32
AK1 = 8
BK1 = 8
MPerXDL = 32
NPerXDL = 32
MXdlPerWave = 4
NXdlPerWave = 2

ABlockThreadCluster_K0_M_K1 = "S<4, 64, 1>"
ABlockThreadCluster_ArangeOrder = "S<1, 0, 2>"
ABlockThreadCluster_SrcAccessOrder = "S<1, 0, 2>"
ABlockThreadCluster_SrcVectorDim = 2
ABlockThreadCluster_SrcScalarPerVector = 8
ABlockThreadCluster_DstScalarPerVector_K1 = 8
ABlockThreadCluster_AddExtraM = 1

BBlockThreadCluster_K0_N_K1 = "S<4, 64, 1>"
BBlockThreadCluster_ArangeOrder = "S<1, 0, 2>"
BBlockThreadCluster_SrcAccessOrder = "S<1, 0, 2>"
BBlockThreadCluster_SrcVectorDim = 2
BBlockThreadCluster_SrcScalarPerVector = 8
BBlockThreadCluster_DstScalarPerVector_K1 = 8
BBlockThreadCluster_AddExtraN = 1

CShuffleMXdlPerWavePerShuffle = 1
CShuffleNXdlPerWavePerShuffle = 1
CBlockThreadCluster_MBlock_MWaveMPerXdl_NBlock_NWaveNPerXdl = "S<1, 32, 1, 8>"
CBlockThreadCluster_ScalarPerVector_NWaveNPerXdl = 8


# Other params
WaveSize = 64


def get_str():
    return "         <" + \
           "ALayout, BLayout, CLayout, ADataType, BDataType, CDataType, AccDataType, CShuffleDataType,  AElementOp,  BElementOp,  CElementOp,    GemmDefault, 1," + \
            str(BlockSize) + ",   " + str(MPerBlock) + ",   " + str(NPerBlock) + ",    " + str(KPerBlock) + "," + \
            str(AK1) + ",   " + str(BK1) + ",   " + str(MPerXDL) + ",   " + str(NPerXDL) + ",    " + str(MXdlPerWave) + ",    " + str(NXdlPerWave) + "," + \
            ABlockThreadCluster_K0_M_K1 + ",     " + ABlockThreadCluster_ArangeOrder + ",     " + ABlockThreadCluster_SrcAccessOrder + ",             " + str(ABlockThreadCluster_SrcVectorDim) + ",              " + str(ABlockThreadCluster_SrcScalarPerVector) + ",              " + str(ABlockThreadCluster_DstScalarPerVector_K1) + ",         " + str(ABlockThreadCluster_AddExtraM) + "," + \
            BBlockThreadCluster_K0_N_K1 + ",     " + BBlockThreadCluster_ArangeOrder + ",     " + BBlockThreadCluster_SrcAccessOrder + ",             " + str(BBlockThreadCluster_SrcVectorDim) + ",              " + str(BBlockThreadCluster_SrcScalarPerVector) + ",              " + str(BBlockThreadCluster_DstScalarPerVector_K1) + ",         " + str(BBlockThreadCluster_AddExtraN) + "," + \
            str(CShuffleMXdlPerWavePerShuffle) + ",           " + str(CShuffleNXdlPerWavePerShuffle) + ",               " + CBlockThreadCluster_MBlock_MWaveMPerXdl_NBlock_NWaveNPerXdl + ",               " + str(CBlockThreadCluster_ScalarPerVector_NWaveNPerXdl) + ">;\n"

def run(line_mod: Dict[int, str]):
        filename = "/home/aiscuser/qingtaoli/composable_kernel/example/01_gemm/gemm_xdl_fp16.cpp"
        with open(f"{filename}.std", "r") as f:
            lines = f.readlines()
        for line_num, line_str in line_mod.items():
            lines[line_num] = line_str
        with open(filename, "w") as f:
            f.writelines(lines)

        os.system("cd /home/aiscuser/qingtaoli/composable_kernel/build/; make clean; make -j example_gemm_xdl_fp16")
        os.system(f"echo {line_mod}")
        M = 2048
        K = 8192
        for N in [1024, 2048, 4096, 8192, 16384, 32768]:
            command = f"/home/aiscuser/qingtaoli/composable_kernel/build/bin/example_gemm_xdl_fp16 0 2 5 {M} {N} {K} {K} {K} {N}"
            os.system(command)
        
        N = 4096
        K = 16384
        for M in [1024, 2048]:
            command = f"/home/aiscuser/qingtaoli/composable_kernel/build/bin/example_gemm_xdl_fp16 0 2 5 {M} {N} {K} {K} {K} {N}"
            os.system(command)
          

def tune_BlockMNK_v1():
    MNKs = [
        (256, 128, 32),
        (128, 256, 32),
        (128, 128, 64),
        (256, 128, 64),
        (128, 256, 64)
    ]

    for MNK in MNKs:
        global MPerBlock, NPerBlock, KPerBlock

        MPerBlock = MNK[0]
        NPerBlock = MNK[1]
        KPerBlock = MNK[2]
        config_str = get_str()

        run({41: config_str})

def tune_direction():
    """Deprecated. Cannot modify."""
    Layouts = ["Row", "Col"]
    for ALayout in Layouts:
        for BLayout in Layouts:
            for CLayout in Layouts:
                line_mod = {
                    16: f"using ALayout = {ALayout};\n",
                    17: f"using BLayout = {BLayout};\n",
                    18: f"using CLayout = {CLayout};\n"
                }
                run(line_mod)


def tune_BlockMNK_v2():
    waves = BlockSize // WaveSize

    XdlPerWaves = [2, 4]
    global MXdlPerWave, NXdlPerWave, MPerBlock, NPerBlock, MWave, NWave, CBlockThreadCluster_MBlock_MWaveMPerXdl_NBlock_NWaveNPerXdl
    # CBlockThreadCluster_MBlock_MWaveMPerXdl_NBlock_NWaveNPerXdl = "S<1, 64, 1, 4>"
    for _MXdlPerWave in XdlPerWaves:
        _NXdlPerWave = 8 // _MXdlPerWave
        MXdlPerWave = _MXdlPerWave
        NXdlPerWave = _NXdlPerWave

        MWave = 2
        while MWave <= waves:
            NWave = waves // MWave
            MPerBlock = MWave * MPerXDL * MXdlPerWave
            NPerBlock = NWave * NPerXDL * NXdlPerWave

            try:
                # assert BlockSize == WaveSize * (MPerBlock // (MPerXDL * MXdlPerWave)) * (NPerBlock // (NPerXDL * NXdlPerWave))
                # assert (CShuffleNXdlPerWavePerShuffle * NWave * NPerXDL) % 64 == 0
                config_str = get_str()
                run({41: config_str})
            except AssertionError as ae:
                print (ae)
            finally:
                MWave *= 2


def tune_all():
    waves = BlockSize // WaveSize
    MWave = 2
    NWave = waves // MWave

    global MPerXDL, NPerXDL, MPerBlock, NPerBlock
    xdl_pairs = [
        (16, 16),
        (32, 32),
        (8, 64),
        (16, 64),
        (32, 64),
        (64, 64),
    ]
    
    for _MPerXDL, _NPerXDL in xdl_pairs:
        MPerXDL = _MPerXDL
        NPerXDL = _NPerXDL
        MPerBlock = MWave * MPerXDL * MXdlPerWave
        NPerBlock = NWave * NPerXDL * NXdlPerWave
        config_str = get_str()
        run({41: config_str})


def origin():
    run({41: get_str()})


if __name__ == "__main__":
    # tune_all()
    origin()
