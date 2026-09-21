"""把测试期的日志重定向到临时文件。

`import app` 会在模块级建好 FileHandler，路径默认是项目根的 gateway.log —— 也就是
生产实例正在写的那一个。跑一次测试就会往里追加假 file_id（fid-boom / fid-err…）
和故意触发的栈回溯，让人误以为线上作业出了问题。必须在 app 被导入前设好环境变量，
所以放在 conftest 顶层。
"""
import os
import tempfile

os.environ.setdefault(
    "DOXIFY_LOG_FILE",
    os.path.join(tempfile.gettempdir(), "doxify-tests.log"),
)
