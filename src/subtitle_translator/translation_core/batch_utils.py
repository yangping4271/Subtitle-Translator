"""
批次大小计算工具
移植自 youtube-subtitle 项目
"""


def calculate_batch_sizes(
    total_count: int,
    target_batch_size: int = 20,
    min_batch_size: int = 15,
    max_batch_size: int = 25
) -> list[int]:
    """
    计算灵活的批次大小分配
    目标：每批15-25个句子，避免最后一批太少

    Args:
        total_count: 总数量
        target_batch_size: 目标批次大小
        min_batch_size: 最小批次大小
        max_batch_size: 最大批次大小

    Returns:
        list[int]: 每个批次的大小列表
    """
    if total_count <= 0:
        return []
    if total_count <= target_batch_size:
        return [total_count]

    batches = []
    remaining = total_count

    while remaining > 0:
        if remaining <= max_batch_size:
            batches.append(remaining)
            break
        elif remaining <= max_batch_size + min_batch_size:
            # 剩余数量在 max 和 max+min 之间，平均分成两批
            batch1 = (remaining + 1) // 2  # 向上取整
            batch2 = remaining - batch1
            batches.extend([batch1, batch2])
            break
        else:
            # 剩余数量较多，取 target_batch_size
            batches.append(target_batch_size)
            remaining -= target_batch_size

    return batches
