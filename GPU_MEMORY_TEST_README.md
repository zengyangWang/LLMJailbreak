# GPU 显存清理检测工具使用说明

## 📋 功能说明

这个工具会自动测试所有攻击方法，检测哪些方法运行后有显存泄漏问题。

## 🚀 快速开始

### 基本用法

```bash
# 测试所有方法（使用 GPU 0）
python test_gpu_memory_cleanup.py

# 指定 GPU 设备
python test_gpu_memory_cleanup.py --gpu 4

# 只测试特定方法
python test_gpu_memory_cleanup.py --methods AutoDAN PAIR TAP

# 自定义泄漏阈值（默认 1000 MB = 1 GB）
python test_gpu_memory_cleanup.py --threshold 500

# 指定输出文件
python test_gpu_memory_cleanup.py --output my_report.json
```

## 📊 工作流程

对每个攻击方法：

1. **运行前**：记录 GPU 显存占用
2. **运行中**：执行攻击方法（只运行1次迭代，1条数据）
3. **运行后**：记录显存占用
4. **清理**：执行 `gc.collect()` 和 `torch.cuda.empty_cache()`
5. **清理后**：再次记录显存占用
6. **分析**：计算显存泄漏量 = 清理后显存 - 运行前显存

## 📈 判断标准

- 🟢 **良好** (≤ 500 MB): 显存清理正常
- 🟡 **警告** (500 MB ~ 1000 MB): 显存清理不完全
- 🔴 **泄漏** (> 1000 MB): 显存泄漏严重，需要修复

## 📄 输出报告

### 1. 控制台输出

```
🔴 显存泄漏严重的方法:
   • MultiJail: 残留 2500 MB
   • PAP: 残留 1800 MB

🟡 显存清理不完全的方法:
   • TAP: 残留 750 MB
   • PAIR: 残留 600 MB

🟢 显存清理良好: 20 个方法
```

### 2. JSON 报告文件

生成的 `gpu_memory_report.json` 包含：

- **test_info**: 测试基本信息（时间、GPU、阈值等）
- **summary**: 统计摘要（成功、失败、泄漏数量）
- **results**: 每个方法的详细结果
  - 显存占用（运行前/后/清理后）
  - 峰值增加、残留显存
  - 运行耗时、是否成功
- **leaked_methods**: 有泄漏的方法列表
- **warning_methods**: 清理不完全的方法列表
- **good_methods**: 清理良好的方法列表

### JSON 结果示例

```json
{
  "test_info": {
    "timestamp": "2026-01-15T16:30:00",
    "gpu_id": 0,
    "threshold_mb": 1000,
    "initial_memory_mb": 3168
  },
  "summary": {
    "success_count": 22,
    "failed_count": 2,
    "leaked_count": 2,
    "warning_count": 3,
    "good_cleanup_count": 17
  },
  "results": [
    {
      "method": "MultiJail",
      "success": true,
      "duration_seconds": 45.2,
      "memory_before_mb": 3168,
      "memory_after_mb": 32000,
      "memory_after_cleanup_mb": 5668,
      "peak_increase_mb": 28832,
      "leaked_memory_mb": 2500
    }
  ],
  "leaked_methods": ["MultiJail", "PAP"],
  "warning_methods": ["TAP", "PAIR", "ReNeLLM"]
}
```

## 🛠️ 后续操作

检测到有泄漏的方法后：

1. 查看报告中的 `leaked_methods` 列表
2. 针对每个有泄漏的方法，在 `server.py` 的 `_cleanup_after_attack()` 函数中添加专门的清理逻辑
3. 重新运行该方法验证修复效果

## ⚠️ 注意事项

1. **测试时间较长**：24个方法全部测试可能需要数小时
2. **独占GPU**：测试期间建议不要在同一GPU上运行其他程序
3. **最小化测试**：脚本已设置为只运行1次迭代、1条数据，尽量减少测试时间
4. **可中断**：随时可以 Ctrl+C 中断，已完成的测试会保存到报告

## 🔍 检查特定方法

如果只想测试从日志中发现的可疑方法：

```bash
# 只测试 MultiJail 和 PAP（这两个在日志中有显存不足错误）
python test_gpu_memory_cleanup.py --methods MultiJail PAP
```

## 📞 问题排查

如果脚本运行失败：

1. 检查 `nvidia-smi` 是否可用
2. 确保在项目根目录运行
3. 检查是否激活了正确的 Python 环境
4. 查看具体错误信息

## 💡 示例运行命令

```bash
# 在 GPU 4 和 GPU 5 上测试（你的可用 GPU）
python test_gpu_memory_cleanup.py --gpu 4

# 或者
python test_gpu_memory_cleanup.py --gpu 5

# 快速测试几个可疑方法
python test_gpu_memory_cleanup.py --gpu 4 --methods MultiJail PAP TAP PAIR
```

