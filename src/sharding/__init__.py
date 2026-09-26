"""src/sharding/__init__.py

Multi-GPU Sharding, Tensor Parallelism, and Pipeline Parallelism for Dual Tesla T4 GPUs.
"""

from src.sharding.comm import (
    AsyncCommHandle,
    DualGPUCommManager,
    all_reduce_dual,
    all_reduce_dual_inplace,
    all_reduce_pair,
    all_reduce_sum,
    can_device_access_peer,
    can_p2p,
    check_dual_gpu,
    enable_peer_access,
    get_comm_manager,
    get_device_count,
    init_p2p,
    is_cuda_available,
    is_p2p_available,
    measure_all_reduce_latency,
    measure_pcie_bandwidth,
    p2p_copy,
    p2p_transfer,
    reference_all_reduce_sum,
    require_dual_cuda,
    validate_comm_tensors,
)
from src.sharding.tp import (
    TPColumnParallelLinear,
    TPParallelAttention,
    TPParallelMLP,
    TPRowParallelLinear,
    dequantize_sym_int4,
    quantize_weight_sym_int4,
    slice_column_parallel_int4,
    slice_column_parallel_weight,
    slice_row_parallel_int4,
    slice_row_parallel_weight,
)
from src.sharding.pp import (
    PipelineStage,
    PipelineParallelQwen2,
)
from src.sharding.scope import (
    CPMultiGPUInferenceScope,
)
from src.sharding import comm, tp, pp, scope

__all__ = [
    # comm
    "AsyncCommHandle",
    "DualGPUCommManager",
    "all_reduce_dual",
    "all_reduce_dual_inplace",
    "all_reduce_pair",
    "all_reduce_sum",
    "can_device_access_peer",
    "can_p2p",
    "check_dual_gpu",
    "enable_peer_access",
    "get_comm_manager",
    "get_device_count",
    "init_p2p",
    "is_cuda_available",
    "is_p2p_available",
    "measure_all_reduce_latency",
    "measure_pcie_bandwidth",
    "p2p_copy",
    "p2p_transfer",
    "reference_all_reduce_sum",
    "require_dual_cuda",
    "validate_comm_tensors",
    "comm",
    # tp
    "TPColumnParallelLinear",
    "TPParallelAttention",
    "TPParallelMLP",
    "TPRowParallelLinear",
    "dequantize_sym_int4",
    "quantize_weight_sym_int4",
    "slice_column_parallel_int4",
    "slice_column_parallel_weight",
    "slice_row_parallel_int4",
    "slice_row_parallel_weight",
    "tp",
    # pp
    "PipelineStage",
    "PipelineParallelQwen2",
    "pp",
    # scope
    "CPMultiGPUInferenceScope",
    "scope",
]
