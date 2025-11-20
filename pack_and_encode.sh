#!/bin/bash

# ==============================================================================
# pack_and_encode.sh
# 描述: 将指定目录(或当前目录)打包成 .tar.gz 压缩文件, 将其编码为 Base64,
#       并将结果保存到一个文本文件中, 以便复制粘贴。
# 使用方法: ./pack_and_encode.sh [目标目录]
# ==============================================================================

# --- 配置 ---
# 检查是否提供了目录参数
if [ $# -gt 0 ]; then
  TARGET_DIR="$1"
  # 转换为绝对路径
  TARGET_DIR=$(cd "$TARGET_DIR" && pwd)
  echo "目标目录: $TARGET_DIR"
else
  TARGET_DIR=$(pwd)
  echo "使用当前目录: $TARGET_DIR"
fi

# 切换到目标目录
cd "$TARGET_DIR" || {
  echo "错误: 无法进入目录 '$TARGET_DIR'"
  exit 1
}
# 定义输出编码后字符串的文件名
ENCODED_FILE="encoded_output.txt"
# 定义临时的压缩包文件名
ARCHIVE_FILE="project_temp.tar.gz"
# 定义需要从压缩包中排除的文件和目录
EXCLUDE_PATTERNS=(
  "--exclude=*.git*"
  "--exclude=*__pycache__*"
  "--exclude=*.DS_Store*"
  "--exclude=$ENCODED_FILE"
  "--exclude=$ARCHIVE_FILE"
  "--exclude=$(basename "$0")" # 排除此脚本自身
  "--exclude=docs/*" # 排除 docs 目录
)

# --- 主逻辑 ---

echo "步骤 1: 正在创建项目压缩包..."
# 使用 tar 命令创建压缩包
# -h 选项表示处理硬链接
# "${EXCLUDE_PATTERNS[@]}" 语法可以正确处理包含空格的文件名
tar "${EXCLUDE_PATTERNS[@]}" -czf "$ARCHIVE_FILE" .
if [ $? -ne 0 ]; then
  echo "错误: 创建压缩包失败, 操作中止。"
  exit 1
fi
echo "压缩包 '$ARCHIVE_FILE' 创建成功。"

echo "步骤 2: 正在将压缩包编码为 Base64..."
# 检测操作系统以兼容不同的 base64 命令
if [[ "$OSTYPE" == "darwin"* ]]; then
  # macOS
  base64 -i "$ARCHIVE_FILE" -o "$ENCODED_FILE"
else
  # Linux, WSL 等
  base64 -w 0 "$ARCHIVE_FILE" > "$ENCODED_FILE"
fi

if [ $? -ne 0 ]; then
  echo "错误: 文件编码失败, 操作中止。"
  rm -f "$ARCHIVE_FILE" # 清理临时文件
  exit 1
fi
echo "Base64 字符串已保存至 '$ENCODED_FILE'。"

echo "步骤 3: 清理临时文件..."
rm -f "$ARCHIVE_FILE"
echo "临时压缩包已删除。"
echo ""
echo "--- 全部完成 ---"
echo "目录 '$TARGET_DIR' 已被成功编码到文件: $ENCODED_FILE"
echo ""
echo "后续操作:"
echo "1. 打开 '$ENCODED_FILE' 文件。"
echo "2. 复制它的【全部】内容 (Ctrl+A, Ctrl+C)。"
echo "3. 将内容粘贴到其他平台上的 'decode_and_unpack.sh' 脚本中指定的位置。"
