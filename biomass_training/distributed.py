import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedContext:
    """分布式训练的运行时上下文信息。"""

    enabled: bool
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


def setup_distributed(requested: bool, backend: str = "nccl") -> DistributedContext:
    """按需从 torchrun 环境变量初始化 DDP。"""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    should_enable = requested or world_size > 1
    if not should_enable:
        return DistributedContext(enabled=False)

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ["WORLD_SIZE"])

    if torch.cuda.is_available():
        # 每个进程绑定自己的本地 GPU，避免多个进程抢同一张卡。
        torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        dist.init_process_group(backend=backend)

    return DistributedContext(
        enabled=True,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
    )


def cleanup_distributed(ctx: DistributedContext) -> None:
    """如果已初始化分布式进程组，则在退出时销毁。"""
    if ctx.enabled and dist.is_initialized():
        dist.destroy_process_group()


def barrier(ctx: DistributedContext) -> None:
    """开启 DDP 时同步所有进程。"""
    if ctx.enabled and dist.is_initialized():
        dist.barrier()


def reduce_mean(value: float, device: torch.device, ctx: DistributedContext) -> float:
    """对所有进程上的标量求平均。"""
    if not ctx.enabled:
        return float(value)
    tensor = torch.tensor(float(value), device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= ctx.world_size
    return float(tensor.item())


def gather_objects(obj, ctx: DistributedContext):
    """从所有 rank 收集任意 Python 对象。"""
    if not ctx.enabled:
        return [obj]
    gathered = [None for _ in range(ctx.world_size)]
    dist.all_gather_object(gathered, obj)
    return gathered
