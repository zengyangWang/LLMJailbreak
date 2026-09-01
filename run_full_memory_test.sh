#!/bin/bash
# GPU 显存检测 - 测试所有方法（双GPU配置）

echo "========================================"
echo "  GPU 显存清理检测 - 完整测试"
echo "  (FastAPI 接口模式)"
echo "========================================"
echo ""
echo "⏰ 预计耗时: 数小时（24个方法）"
echo "🎯 GPU 配置: 4,5 (双GPU)"
echo "📝 测试方法: 全部 24 个方法"
echo "🌐 服务器: http://127.0.0.1:39000"
echo ""

# 检查 nvidia-smi 是否可用
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ 错误: nvidia-smi 未找到"
    exit 1
fi

# 检查服务器是否运行
echo "🔍 检查服务器状态..."
if ! curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:39000/ >/dev/null 2>&1; then
    echo "❌ 错误: 服务器未运行"
    echo ""
    echo "请先启动服务器:"
    echo "  CUDA_VISIBLE_DEVICES=4,5 nohup python -u server.py > logs/server_\$(date +%F_%H-%M-%S).log 2>&1 & echo \$! > server.pid"
    echo ""
    exit 1
fi
echo "✅ 服务器正在运行"
echo ""

# 显示当前 GPU 状态
echo "📊 当前 GPU 状态:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,memory.free --format=table --id=4,5
echo ""

# 确认开始
read -p "⚠️  这将测试所有 24 个方法，需要数小时。是否继续? (y/n): " CONFIRM
if [ "$CONFIRM" != "y" ] && [ "$CONFIRM" != "Y" ]; then
    echo "❌ 已取消"
    exit 0
fi

# 创建日志目录
LOG_DIR="memory_test_logs"
mkdir -p $LOG_DIR

# 生成时间戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/full_test_$TIMESTAMP.log"
REPORT_FILE="gpu_memory_report_full_$TIMESTAMP.json"

echo ""
echo "========================================"
echo "🚀 开始测试..."
echo "========================================"
echo "📄 日志文件: $LOG_FILE"
echo "📊 报告文件: $REPORT_FILE"
echo ""
echo "提示: 可以在另一个终端运行 'tail -f $LOG_FILE' 查看实时进度"
echo ""

# 运行测试（不需要设置 CUDA_VISIBLE_DEVICES，服务器已经在使用 GPU）
nohup python -u test_gpu_memory_cleanup.py \
    --gpus 4,5 \
    --server_url http://127.0.0.1:39000/attack_batch \
    --output $REPORT_FILE \
    --threshold 1000 \
    > $LOG_FILE 2>&1 &

PID=$!
echo "✅ 测试已在后台启动 (PID: $PID)"
echo ""
echo "监控命令:"
echo "  - 查看实时日志: tail -f $LOG_FILE"
echo "  - 查看进程状态: ps -p $PID"
echo "  - 监控GPU: watch -n 1 nvidia-smi"
echo "  - 停止测试: kill $PID"
echo ""
echo "测试完成后，报告将保存在: $REPORT_FILE"
echo ""

# 保存 PID
echo $PID > memory_test.pid
echo "💾 进程 ID 已保存到: memory_test.pid"

