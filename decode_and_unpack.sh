#!/bin/bash

# ==============================================================================
# decode_and_unpack.sh
# 描述: 从 'encoded_output.txt' 文件中读取 Base64 字符串, 解码并解压,
#       将解压后的内容覆盖到指定目录(或当前目录)中同名的文件和文件夹。
# 使用方法: ./decode_and_unpack.sh [目标目录]
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

INPUT_FILE="encoded_output.txt"

# --- 运行前检查 ---

echo "欢迎使用项目解包脚本!"
echo "---------------------------------"

# 检查输入文件是否存在
if [ ! -f "$INPUT_FILE" ]; then
  echo "错误: 未找到输入文件 '$INPUT_FILE'。"
  echo "请先创建该文件, 将 Base64 字符串粘贴进去, 然后重新运行此脚本。"
  exit 1
fi

# 检查输入文件是否为空
if [ ! -s "$INPUT_FILE" ]; then
  echo "错误: 输入文件 '$INPUT_FILE' 是空的。"
  echo "请将 Base64 字符串粘贴进去并保存后再运行。"
  exit 1
fi

# --- 主逻辑 ---

echo "步骤 1: 正在解码 Base64 并解压..."
# 从输入文件读取 -> base64解码 -> tar解压
# tar 的 '-' 参数表示从标准输入流读取数据
cat "$INPUT_FILE" | base64 -d | tar -xzf -

if [ $? -ne 0 ]; then
  echo "错误: 在解码或解压过程中发生错误。"
  echo "Base64 字符串可能已损坏或不完整。"
  echo "请检查输入文件后重新尝试。"
  exit 1
fi
echo "解压完成。"

echo "步骤 2: 清理工作..."
rm -f "$INPUT_FILE"
echo "输入文件 '$INPUT_FILE' 已被删除。"
echo ""
echo "--- 成功 ---"
echo "目录 '$TARGET_DIR' 的项目代码已成功更新!"