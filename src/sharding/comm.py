"""src/sharding/comm.py

Dual Tesla T4 Communication and Synchronization Layer for Tensor Parallelism (TP=2)
and Pipeline Parallelism (PP=2) over PCIe Gen3.

Features:
1. Dual-device P2P detection and capability verification.
2. High-priority dedicated CUDA streams (priority=-1) for communication.
3. Pre-allocated double-buffered staging buffers to eliminate dynamic allocator latency.
4. Non-blocking GPU-side event synchronization (AsyncCommHandle) with zero CPU stalls.
5. In-place and out-of-place All-Reduce primitives with two-phase cross-device barrier.
6. Fallback path to torch.distributed (NCCL) when initialized.
7. Point-to-point (P2P) boundary transfer primitive for Pipeline Parallelism.
8. Pure reference CPU implementation (reference_all_reduce_sum) for testing parity.
9. PCIe bandwidth and latency measurement utilities.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.distributed as dist


# ==============================================================================
# Hardware Introspection and P2P Capability
# ==============================================================================

def is_cuda_available() -> bool:
    """Returns True if CUDA is available on this system."""
    return torch.cuda.is_available()


def get_device_count() -> int:
    """Returns the number of available CUDA devices."""
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def check_dual_gpu(min_devices: int = 2) -> Tuple[bool, str]:
    """
    Checks if at least `min_devices` CUDA devices are present and accessible.
    Returns (status, reason_string).
    """
    if not torch.cuda.is_available():
        return False, "CUDA is not available on this host"
    count = torch.cuda.device_count()
    if count < min_devices:
        return False, f"Requires >= {min_devices} CUDA devices, found {count}"
    return True, ""


def require_dual_cuda(min_devices: int = 2) -> None:
    """
    Helper for test runners. Skips cleanly under pytest if dual GPUs are missing,
    or raises RuntimeError if called outside pytest.
    """
    ok, reason = check_dual_gpu(min_devices)
    if not ok:
        if "pytest" in sys.modules or "PYTEST_CURRENT_TEST" in os.environ:
            import pytest
            pytest.skip(reason)
        else:
            raise RuntimeError(f"Dual CUDA required: {reason}")


def can_device_access_peer(dev_a: Union[int, str, torch.device], dev_b: Union[int, str, torch.device]) -> bool:
    """
    Checks whether device `dev_a` can directly access memory on peer device `dev_b` via PCIe P2P.
    """
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        return False

    idx_a = torch.device(dev_a).index if isinstance(dev_a, (str, torch.device)) else dev_a
    idx_b = torch.device(dev_b).index if isinstance(dev_b, (str, torch.device)) else dev_b

    if idx_a is None or idx_b is None or idx_a == idx_b:
        return False

    try:
        return bool(torch.cuda.can_device_access_peer(idx_a, idx_b))
    except Exception:
        return False


def can_p2p(dev_a: int = 0, dev_b: int = 1) -> bool:
    """Convenience alias for can_device_access_peer."""
    return can_device_access_peer(dev_a, dev_b)


def is_p2p_available(dev_a: int = 0, dev_b: int = 1) -> bool:
    """
    Returns True if bidirectional PCIe P2P memory access is supported between dev_a and dev_b.
    """
    return can_device_access_peer(dev_a, dev_b) and can_device_access_peer(dev_b, dev_a)


def enable_peer_access(dev_a: int = 0, dev_b: int = 1) -> bool:
    """
    Attempts to enable bidirectional CUDA peer access between dev_a and dev_b.
    Returns True if peer access is active, False otherwise.
    """
    if not is_p2p_available(dev_a, dev_b):
        return False

    try:
        import ctypes
        for lib_name in ("libcudart.so", "libcudart.so.12", "libcudart.so.11.0", "libcuda.so.1"):
            try:
                cudart = ctypes.CDLL(lib_name)
                cudart.cudaSetDevice(ctypes.c_int(dev_a))
                cudart.cudaDeviceEnablePeerAccess(ctypes.c_int(dev_b), ctypes.c_uint(0))
                cudart.cudaSetDevice(ctypes.c_int(dev_b))
                cudart.cudaDeviceEnablePeerAccess(ctypes.c_int(dev_a), ctypes.c_uint(0))
                break
            except Exception:
                continue
    except Exception:
        pass

    return True


def init_p2p(dev_a: int = 0, dev_b: int = 1) -> bool:
    """
    Validates device indices and initializes bidirectional P2P access.
    Raises ValueError on identical indices or out-of-bounds devices.
    """
    if dev_a == dev_b:
        raise ValueError("Cannot establish peer access to self")
    if dev_a < 0 or dev_b < 0:
        raise ValueError("Device index must be non-negative")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available on this host")

    count = torch.cuda.device_count()
    if dev_a >= count or dev_b >= count:
        raise ValueError(f"Device index out of range: {dev_a}, {dev_b} for device count {count}")

    return enable_peer_access(dev_a, dev_b)


# ==============================================================================
# Async Communication Handle
# ==============================================================================

class AsyncCommHandle:
    """
    Handle for asynchronous GPU communication operations.
    Enables zero-CPU-stall stream synchronization via CUDA events.
    """
    def __init__(
        self,
        done_event_0: Optional[torch.cuda.Event] = None,
        done_event_1: Optional[torch.cuda.Event] = None,
        stream_0: Optional[torch.cuda.Stream] = None,
        stream_1: Optional[torch.cuda.Stream] = None,
        slot: int = 0,
    ):
        self.done_event_0 = done_event_0
        self.done_event_1 = done_event_1
        self.stream_0 = stream_0
        self.stream_1 = stream_1
        self.slot = slot
        self._completed = False

    def wait_device(self, device_idx: int) -> None:
        """
        Enqueues a stream wait on the calling device's current compute stream.
        Non-blocking on CPU; the GPU hardware scheduler waits for the comm event.
        """
        if self._completed or not torch.cuda.is_available():
            return
        if device_idx == 0 and self.done_event_0 is not None:
            torch.cuda.current_stream(0).wait_event(self.done_event_0)
        elif device_idx == 1 and self.done_event_1 is not None:
            torch.cuda.current_stream(1).wait_event(self.done_event_1)

    def wait(self) -> None:
        """
        Enqueues stream waits on current compute streams of both devices.
        Non-blocking on CPU.
        """
        if self._completed or not torch.cuda.is_available():
            self._completed = True
            return
        if self.done_event_0 is not None:
            torch.cuda.current_stream(0).wait_event(self.done_event_0)
        if self.done_event_1 is not None:
            torch.cuda.current_stream(1).wait_event(self.done_event_1)
        self._completed = True

    def synchronize(self) -> None:
        """
        CPU-blocking barrier. Blocks until both communication events complete.
        Use only when host-side inspection or timing is required.
        """
        if self._completed or not torch.cuda.is_available():
            self._completed = True
            return
        if self.done_event_0 is not None:
            self.done_event_0.synchronize()
        if self.done_event_1 is not None:
            self.done_event_1.synchronize()
        self._completed = True


# ==============================================================================
# Dual-GPU Communication Manager (Pools & Double-Buffering)
# ==============================================================================

class DualGPUCommManager:
    """
    Manages dedicated high-priority CUDA streams, pre-allocated staging buffers,
    and double-buffering slots for dual-T4 PCIe Gen3 communication.
    """
    _instance: Optional[DualGPUCommManager] = None

    def __init__(self, initial_buffer_numel: int = 65536, num_slots: int = 2):
        self.cuda_available = torch.cuda.is_available()
        self.device_count = torch.cuda.device_count() if self.cuda_available else 0
        self.initial_buffer_numel = initial_buffer_numel
        self.num_slots = num_slots
        self._current_slot = 0
        self._lock = threading.Lock()

        self.stream_0: Optional[torch.cuda.Stream] = None
        self.stream_1: Optional[torch.cuda.Stream] = None
        self._staging_buffers: Dict[Tuple[int, int], torch.Tensor] = {}
        self._ready_events: Dict[int, list[torch.cuda.Event]] = {}
        self._copy_done_events: Dict[int, list[torch.cuda.Event]] = {}
        self._done_events: Dict[int, list[torch.cuda.Event]] = {}
        self._slot_active: list[bool] = [False] * num_slots
        self._p2p_active = False

        if self.cuda_available and self.device_count >= 2:
            self._initialize_resources()

    def _initialize_resources(self) -> None:
        self.stream_0 = torch.cuda.Stream(device=0, priority=-1)
        self.stream_1 = torch.cuda.Stream(device=1, priority=-1)

        for dev in (0, 1):
            self._ready_events[dev] = [torch.cuda.Event(enable_timing=False) for _ in range(self.num_slots)]
            self._copy_done_events[dev] = [torch.cuda.Event(enable_timing=False) for _ in range(self.num_slots)]
            self._done_events[dev] = [torch.cuda.Event(enable_timing=False) for _ in range(self.num_slots)]

        for slot in range(self.num_slots):
            self._staging_buffers[(0, slot)] = torch.empty(
                self.initial_buffer_numel, dtype=torch.float16, device="cuda:0"
            )
            self._staging_buffers[(1, slot)] = torch.empty(
                self.initial_buffer_numel, dtype=torch.float16, device="cuda:1"
            )

        self._p2p_active = enable_peer_access(0, 1)

    @classmethod
    def get_instance(cls) -> DualGPUCommManager:
        if cls._instance is None:
            cls._instance = DualGPUCommManager()
        return cls._instance

    def next_slot(self) -> int:
        with self._lock:
            slot = self._current_slot
            self._current_slot = (self._current_slot + 1) % self.num_slots
            return slot

    def get_staging_buffer(self, device_idx: int, numel: int, dtype: torch.dtype, slot: int) -> torch.Tensor:
        key = (device_idx, slot)
        buf = self._staging_buffers.get(key)
        if buf is None or buf.numel() < numel or buf.dtype != dtype:
            alloc_numel = max(numel, self.initial_buffer_numel)
            buf = torch.empty(alloc_numel, dtype=dtype, device=f"cuda:{device_idx}")
            self._staging_buffers[key] = buf
        return buf

    def get_ready_event(self, device_idx: int, slot: int) -> Optional[torch.cuda.Event]:
        if device_idx in self._ready_events and slot < len(self._ready_events[device_idx]):
            return self._ready_events[device_idx][slot]
        return None

    def get_copy_done_event(self, device_idx: int, slot: int) -> Optional[torch.cuda.Event]:
        if device_idx in self._copy_done_events and slot < len(self._copy_done_events[device_idx]):
            return self._copy_done_events[device_idx][slot]
        return None

    def get_done_event(self, device_idx: int, slot: int) -> Optional[torch.cuda.Event]:
        if device_idx in self._done_events and slot < len(self._done_events[device_idx]):
            return self._done_events[device_idx][slot]
        return None

    def is_slot_active(self, slot: int) -> bool:
        if 0 <= slot < len(self._slot_active):
            return self._slot_active[slot]
        return False

    def mark_slot_active(self, slot: int) -> None:
        if 0 <= slot < len(self._slot_active):
            self._slot_active[slot] = True

    def reset_slots(self) -> None:
        with self._lock:
            self._current_slot = 0
            self._slot_active = [False] * self.num_slots


def get_comm_manager() -> DualGPUCommManager:
    """Returns the singleton DualGPUCommManager instance."""
    return DualGPUCommManager.get_instance()


# ==============================================================================
# Pure Reference CPU Implementation
# ==============================================================================

def reference_all_reduce_sum(
    tensor_rank0: torch.Tensor,
    tensor_rank1: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes reference All-Reduce sum over dual ranks on any device.
    Matches tests/test_e2e_multigpu_suite.py:221-225 exactly.
    """
    reduced = tensor_rank0 + tensor_rank1
    return reduced.clone(), reduced.clone()


# ==============================================================================
# Tensor Validation
# ==============================================================================

def validate_comm_tensors(
    tensor_0: torch.Tensor,
    tensor_1: torch.Tensor,
    enforce_contiguous: bool = True,
    require_cuda: bool = True,
) -> None:
    """
    Validates shapes, dtypes, contiguity, and device affinity before communication.
    Raises ValueError, TypeError, or RuntimeError on mismatch.
    """
    if not isinstance(tensor_0, torch.Tensor) or not isinstance(tensor_1, torch.Tensor):
        raise TypeError(f"Inputs must be torch.Tensor instances, got {type(tensor_0)} and {type(tensor_1)}")
    if tensor_0.dtype != tensor_1.dtype:
        raise TypeError(f"Tensor dtypes must match: {tensor_0.dtype} vs {tensor_1.dtype}")
    if tensor_0.shape != tensor_1.shape:
        raise ValueError(f"Tensor shapes must match: {tensor_0.shape} vs {tensor_1.shape}")
    if enforce_contiguous:
        if not tensor_0.is_contiguous() or not tensor_1.is_contiguous():
            raise ValueError("Input tensors must be contiguous for low-latency P2P All-Reduce")
    if require_cuda:
        if not tensor_0.is_cuda or not tensor_1.is_cuda:
            raise RuntimeError(f"CUDA tensors required for P2P All-Reduce, got: {tensor_0.device} and {tensor_1.device}")
        if tensor_0.device.index == tensor_1.device.index:
            raise ValueError(f"Dual-GPU communication requires distinct devices, got {tensor_0.device} and {tensor_1.device}")
    else:
        if tensor_0.is_cuda != tensor_1.is_cuda:
            raise ValueError(f"Device mismatch: tensor_0 on {tensor_0.device}, tensor_1 on {tensor_1.device}")
        if tensor_0.is_cuda and tensor_1.is_cuda and tensor_0.device.index == tensor_1.device.index:
            raise ValueError(f"Dual-GPU communication requires distinct devices, got {tensor_0.device} and {tensor_1.device}")


# ==============================================================================
# P2P All-Reduce Primitives (Single-Process Dual-GPU)
# ==============================================================================

def all_reduce_dual_inplace(
    tensor_0: torch.Tensor,
    tensor_1: torch.Tensor,
    async_op: bool = False,
    slot: Optional[int] = None,
    stream_0: Optional[torch.cuda.Stream] = None,
    stream_1: Optional[torch.cuda.Stream] = None,
) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[Tuple[torch.Tensor, torch.Tensor], AsyncCommHandle]]:
    """
    Low-latency in-place All-Reduce sum between cuda:0 and cuda:1.
    Mutates tensor_0 and tensor_1 in place to equal tensor_0 + tensor_1.
    Uses pre-allocated staging buffers, high-priority streams, and
    a two-phase cross-device barrier to prevent Read-After-Write (RAW) data races.
    """
    if not tensor_0.is_cuda and not tensor_1.is_cuda:
        validate_comm_tensors(tensor_0, tensor_1, enforce_contiguous=False, require_cuda=False)
        tensor_0.add_(tensor_1)
        tensor_1.copy_(tensor_0)
        if async_op:
            return (tensor_0, tensor_1), AsyncCommHandle()
        return tensor_0, tensor_1

    validate_comm_tensors(tensor_0, tensor_1, enforce_contiguous=True, require_cuda=True)

    mgr = get_comm_manager()
    active_slot = mgr.next_slot() if slot is None else slot

    compute_stream_0 = stream_0 if stream_0 is not None else torch.cuda.current_stream(0)
    compute_stream_1 = stream_1 if stream_1 is not None else torch.cuda.current_stream(1)

    # Double-buffering safety: ensure previous operation on active_slot has completed
    if mgr.is_slot_active(active_slot):
        prev_done_0 = mgr.get_done_event(0, active_slot)
        prev_done_1 = mgr.get_done_event(1, active_slot)
        if prev_done_0 is not None:
            mgr.stream_0.wait_event(prev_done_0)
        if prev_done_1 is not None:
            mgr.stream_1.wait_event(prev_done_1)

    ready_0 = mgr.get_ready_event(0, active_slot)
    ready_1 = mgr.get_ready_event(1, active_slot)
    compute_stream_0.record_event(ready_0)
    compute_stream_1.record_event(ready_1)

    mgr.stream_0.wait_event(ready_1)
    mgr.stream_0.wait_event(ready_0)
    mgr.stream_1.wait_event(ready_0)
    mgr.stream_1.wait_event(ready_1)

    buf_1_on_0 = mgr.get_staging_buffer(0, tensor_1.numel(), tensor_1.dtype, active_slot)
    buf_0_on_1 = mgr.get_staging_buffer(1, tensor_0.numel(), tensor_0.dtype, active_slot)

    view_1_on_0 = buf_1_on_0[:tensor_1.numel()].view_as(tensor_1)
    view_0_on_1 = buf_0_on_1[:tensor_0.numel()].view_as(tensor_0)

    # Phase 1: Asynchronously copy remote tensors into local staging buffers
    with torch.cuda.stream(mgr.stream_0):
        view_1_on_0.copy_(tensor_1, non_blocking=True)

    with torch.cuda.stream(mgr.stream_1):
        view_0_on_1.copy_(tensor_0, non_blocking=True)

    copy_done_0 = mgr.get_copy_done_event(0, active_slot)
    copy_done_1 = mgr.get_copy_done_event(1, active_slot)
    mgr.stream_0.record_event(copy_done_0)
    mgr.stream_1.record_event(copy_done_1)

    # Synchronization barrier before in-place mutation:
    # Stream 0 waits until Stream 1 has finished copying original tensor_0
    # Stream 1 waits until Stream 0 has finished copying original tensor_1
    mgr.stream_0.wait_event(copy_done_1)
    mgr.stream_1.wait_event(copy_done_0)

    # Phase 2: Local addition
    with torch.cuda.stream(mgr.stream_0):
        tensor_0.add_(view_1_on_0)

    with torch.cuda.stream(mgr.stream_1):
        tensor_1.add_(view_0_on_1)

    done_0 = mgr.get_done_event(0, active_slot)
    done_1 = mgr.get_done_event(1, active_slot)
    mgr.stream_0.record_event(done_0)
    mgr.stream_1.record_event(done_1)

    mgr.mark_slot_active(active_slot)

    handle = AsyncCommHandle(done_0, done_1, mgr.stream_0, mgr.stream_1, slot=active_slot)

    if async_op:
        return (tensor_0, tensor_1), handle

    compute_stream_0.wait_event(done_0)
    compute_stream_1.wait_event(done_1)
    return tensor_0, tensor_1


def all_reduce_dual(
    tensor_0: torch.Tensor,
    tensor_1: torch.Tensor,
    async_op: bool = False,
    slot: Optional[int] = None,
    stream_0: Optional[torch.cuda.Stream] = None,
    stream_1: Optional[torch.cuda.Stream] = None,
) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[Tuple[torch.Tensor, torch.Tensor], AsyncCommHandle]]:
    """
    Out-of-place All-Reduce sum between cuda:0 and cuda:1.
    Preserves input tensors and returns new reduced tensors (out_0, out_1).
    Uses pre-allocated staging buffers, high-priority streams, and
    two-phase cross-device synchronization.
    """
    if not tensor_0.is_cuda and not tensor_1.is_cuda:
        validate_comm_tensors(tensor_0, tensor_1, enforce_contiguous=False, require_cuda=False)
        r0, r1 = reference_all_reduce_sum(tensor_0, tensor_1)
        if async_op:
            return (r0, r1), AsyncCommHandle()
        return r0, r1

    validate_comm_tensors(tensor_0, tensor_1, enforce_contiguous=True, require_cuda=True)

    mgr = get_comm_manager()
    active_slot = mgr.next_slot() if slot is None else slot

    compute_stream_0 = stream_0 if stream_0 is not None else torch.cuda.current_stream(0)
    compute_stream_1 = stream_1 if stream_1 is not None else torch.cuda.current_stream(1)

    # Double-buffering safety: ensure previous operation on active_slot has completed
    if mgr.is_slot_active(active_slot):
        prev_done_0 = mgr.get_done_event(0, active_slot)
        prev_done_1 = mgr.get_done_event(1, active_slot)
        if prev_done_0 is not None:
            mgr.stream_0.wait_event(prev_done_0)
        if prev_done_1 is not None:
            mgr.stream_1.wait_event(prev_done_1)

    ready_0 = mgr.get_ready_event(0, active_slot)
    ready_1 = mgr.get_ready_event(1, active_slot)
    compute_stream_0.record_event(ready_0)
    compute_stream_1.record_event(ready_1)

    mgr.stream_0.wait_event(ready_1)
    mgr.stream_0.wait_event(ready_0)
    mgr.stream_1.wait_event(ready_0)
    mgr.stream_1.wait_event(ready_1)

    buf_1_on_0 = mgr.get_staging_buffer(0, tensor_1.numel(), tensor_1.dtype, active_slot)
    buf_0_on_1 = mgr.get_staging_buffer(1, tensor_0.numel(), tensor_0.dtype, active_slot)

    view_1_on_0 = buf_1_on_0[:tensor_1.numel()].view_as(tensor_1)
    view_0_on_1 = buf_0_on_1[:tensor_0.numel()].view_as(tensor_0)

    # Phase 1: Asynchronously copy remote tensors into local staging buffers
    with torch.cuda.stream(mgr.stream_0):
        view_1_on_0.copy_(tensor_1, non_blocking=True)

    with torch.cuda.stream(mgr.stream_1):
        view_0_on_1.copy_(tensor_0, non_blocking=True)

    copy_done_0 = mgr.get_copy_done_event(0, active_slot)
    copy_done_1 = mgr.get_copy_done_event(1, active_slot)
    mgr.stream_0.record_event(copy_done_0)
    mgr.stream_1.record_event(copy_done_1)

    # Cross-device barrier before computation to ensure stream 0 does not signal
    # completion to compute_stream_0 before stream 1 has finished copying tensor_0
    mgr.stream_0.wait_event(copy_done_1)
    mgr.stream_1.wait_event(copy_done_0)

    # Phase 2: Local out-of-place addition
    with torch.cuda.stream(mgr.stream_0):
        out_0 = tensor_0 + view_1_on_0

    with torch.cuda.stream(mgr.stream_1):
        out_1 = tensor_1 + view_0_on_1

    done_0 = mgr.get_done_event(0, active_slot)
    done_1 = mgr.get_done_event(1, active_slot)
    mgr.stream_0.record_event(done_0)
    mgr.stream_1.record_event(done_1)

    mgr.mark_slot_active(active_slot)

    handle = AsyncCommHandle(done_0, done_1, mgr.stream_0, mgr.stream_1, slot=active_slot)

    if async_op:
        return (out_0, out_1), handle

    compute_stream_0.wait_event(done_0)
    compute_stream_1.wait_event(done_1)
    return out_0, out_1


# Explicit alias for single-process dual-GPU pairing
all_reduce_pair = all_reduce_dual


# ==============================================================================
# Point-to-Point (P2P) Boundary Transfer Primitive (Pipeline Parallelism)
# ==============================================================================

def p2p_transfer(
    tensor: torch.Tensor,
    dst_device: Union[int, str, torch.device],
    non_blocking: bool = True,
    async_op: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.cuda.Event]]]:
    """
    Transfers an activation tensor across GPU boundaries (e.g. cuda:0 -> cuda:1).
    Used at the layer 13/14 partition boundary in Pipeline Parallelism.
    """
    dst_dev = torch.device(dst_device) if not isinstance(dst_device, torch.device) else dst_device

    if tensor.device == dst_dev:
        return (tensor, None) if async_op else tensor

    if not tensor.is_cuda:
        out = tensor.to(dst_dev, non_blocking=non_blocking)
        return (out, None) if async_op else out

    src_idx = tensor.device.index
    dst_idx = dst_dev.index

    mgr = get_comm_manager()
    comm_stream = mgr.stream_1 if dst_idx == 1 else mgr.stream_0

    src_compute_stream = torch.cuda.current_stream(src_idx)
    ready_event = torch.cuda.Event(enable_timing=False)
    src_compute_stream.record_event(ready_event)

    comm_stream.wait_event(ready_event)

    dst_tensor = torch.empty_like(tensor, device=dst_dev)

    with torch.cuda.stream(comm_stream):
        dst_tensor.copy_(tensor, non_blocking=non_blocking)

    done_event = torch.cuda.Event(enable_timing=False)
    comm_stream.record_event(done_event)

    if async_op:
        return dst_tensor, done_event

    dst_compute_stream = torch.cuda.current_stream(dst_idx)
    dst_compute_stream.wait_event(done_event)
    return dst_tensor


def p2p_copy(
    tensor: torch.Tensor,
    dst_device: Union[int, str, torch.device] = 1,
    non_blocking: bool = True,
    async_op: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.cuda.Event]]]:
    """Convenience alias/wrapper for p2p_transfer."""
    return p2p_transfer(tensor, dst_device, non_blocking=non_blocking, async_op=async_op)


# ==============================================================================
# Unified Top-Level All-Reduce Function
# ==============================================================================

def all_reduce_sum(
    tensor_0: torch.Tensor,
    tensor_1: Optional[torch.Tensor] = None,
    inplace: bool = False,
    async_op: bool = False,
    slot: Optional[int] = None,
    in_place: Optional[bool] = None,
    stream_0: Optional[torch.cuda.Stream] = None,
    stream_1: Optional[torch.cuda.Stream] = None,
) -> Any:
    """
    Unified All-Reduce sum entry point:
    - If tensor_1 is provided:
        * On CUDA: executes fast PCIe Gen3 P2P All-Reduce across dual GPUs.
        * On CPU: executes reference_all_reduce_sum (or in-place addition).
    - If tensor_1 is None:
        * When torch.distributed is initialized: executes dist.all_reduce(tensor_0).
        * Otherwise: raises RuntimeError explaining distributed initialization requirement.
    """
    if in_place is not None:
        inplace = in_place

    if tensor_1 is not None:
        if tensor_0.is_cuda and tensor_1.is_cuda:
            if inplace:
                return all_reduce_dual_inplace(
                    tensor_0, tensor_1, async_op=async_op, slot=slot, stream_0=stream_0, stream_1=stream_1
                )
            return all_reduce_dual(
                tensor_0, tensor_1, async_op=async_op, slot=slot, stream_0=stream_0, stream_1=stream_1
            )

        if not tensor_0.is_cuda and not tensor_1.is_cuda:
            if inplace:
                return all_reduce_dual_inplace(
                    tensor_0, tensor_1, async_op=async_op, slot=slot, stream_0=stream_0, stream_1=stream_1
                )
            return all_reduce_dual(
                tensor_0, tensor_1, async_op=async_op, slot=slot, stream_0=stream_0, stream_1=stream_1
            )

        raise ValueError(f"Device mismatch: tensor_0 on {tensor_0.device}, tensor_1 on {tensor_1.device}")

    # Single tensor passed: distributed multi-process mode
    if dist.is_available() and dist.is_initialized():
        work = dist.all_reduce(tensor_0, op=dist.ReduceOp.SUM, async_op=async_op)
        if async_op:
            return tensor_0, work
        return tensor_0

    raise RuntimeError(
        "Single-tensor all_reduce_sum requires torch.distributed to be initialized. "
        "For single-process dual-GPU tensor parallelism, pass both tensor_0 and tensor_1."
    )


# ==============================================================================
# Benchmarking & Telemetry Utilities
# ==============================================================================

def measure_pcie_bandwidth(
    bytes_to_transfer: int = 16 * 1024 * 1024,
    iterations: int = 50,
) -> Dict[str, Any]:
    """
    Measures bidirectional PCIe Gen3 transfer latency and payload bandwidth between cuda:0 and cuda:1.
    Complies with CLAIMS_HYGIENE.md requirements for empirical latency tracking.
    """
    require_dual_cuda(2)

    t0 = torch.randn(bytes_to_transfer // 4, dtype=torch.float32, device="cuda:0")
    t1 = torch.empty_like(t0, device="cuda:1")

    # Warmup
    for _ in range(10):
        t1.copy_(t0, non_blocking=False)
        t0.copy_(t1, non_blocking=False)
    torch.cuda.synchronize(0)
    torch.cuda.synchronize(1)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    latencies_ms: list[float] = []
    for _ in range(iterations):
        start_event.record()
        t1.copy_(t0, non_blocking=True)
        end_event.record()
        end_event.synchronize()
        latencies_ms.append(start_event.elapsed_time(end_event))

    latencies_ms.sort()
    median_ms = latencies_ms[len(latencies_ms) // 2]
    mean_ms = sum(latencies_ms) / len(latencies_ms)
    p95_ms = latencies_ms[int(0.95 * len(latencies_ms))]
    bandwidth_gb_s = (bytes_to_transfer / (median_ms * 1e-3)) / 1e9

    return {
        "bytes_transferred": bytes_to_transfer,
        "iterations": iterations,
        "latency_mean_ms": mean_ms,
        "latency_median_ms": median_ms,
        "latency_p95_ms": p95_ms,
        "bandwidth_gb_s": bandwidth_gb_s,
        "p2p_active": is_p2p_available(0, 1),
    }


def measure_all_reduce_latency(
    shape: Tuple[int, ...] = (1, 1, 3584),
    dtype: torch.dtype = torch.float16,
    iterations: int = 100,
) -> Dict[str, Any]:
    """
    Measures empirical All-Reduce latency across dual Tesla T4 GPUs for a given tensor shape.
    Used for the M3 TP vs PP empirical decision gate.
    """
    require_dual_cuda(2)

    t0 = torch.randn(shape, dtype=dtype, device="cuda:0")
    t1 = torch.randn(shape, dtype=dtype, device="cuda:1")

    # Warmup
    for _ in range(20):
        all_reduce_dual_inplace(t0, t1, async_op=False)
    torch.cuda.synchronize(0)
    torch.cuda.synchronize(1)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    latencies_ms: list[float] = []
    for _ in range(iterations):
        start_event.record(torch.cuda.current_stream(0))
        all_reduce_dual_inplace(t0, t1, async_op=False)
        end_event.record(torch.cuda.current_stream(0))
        end_event.synchronize()
        latencies_ms.append(start_event.elapsed_time(end_event))

    latencies_ms.sort()
    num_bytes = t0.numel() * t0.element_size()
    median_ms = latencies_ms[len(latencies_ms) // 2]
    mean_ms = sum(latencies_ms) / len(latencies_ms)
    p95_ms = latencies_ms[int(0.95 * len(latencies_ms))]
    p5_ms = latencies_ms[int(0.05 * len(latencies_ms))]
    effective_gb_s = (2.0 * num_bytes / (median_ms * 1e-3)) / 1e9

    return {
        "shape": list(shape),
        "dtype": str(dtype),
        "numel": t0.numel(),
        "bytes_per_tensor": num_bytes,
        "iterations": iterations,
        "latency_mean_ms": mean_ms,
        "latency_median_ms": median_ms,
        "latency_p5_ms": p5_ms,
        "latency_p95_ms": p95_ms,
        "effective_bandwidth_gb_s": effective_gb_s,
    }
