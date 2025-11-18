# RL_hok1v1
北京理工大学 23 级计科大三强化学习刘驰班腾讯开悟平台"智能体决策1V1"实验

## 将本地代码同步到开悟平台上

1. 使用 [pack_and_encode.sh](pack_and_encode.sh) 脚本本地项目打包编码，可能要多执行几次才能成功：
   ```bash
   sh ./pack_and_encode.sh /path/to/your/project/

   ```
2. 开悟平台新建文件 `encoded_output.txt`，将编码后的内容粘贴进去。同时新建脚本文件 `decode_and_unpack.sh`，将 [decode_and_unpack.sh](decode_and_unpack.sh) 复制进去。
3. 随后复制编码到代码实际存放位置当中去，并解码：
   ```bash
   cp ./encoded_output.txt /workspace/code/
   sh ./decode_and_unpack.sh /workspace/code/
   
   ```

同步完成。
