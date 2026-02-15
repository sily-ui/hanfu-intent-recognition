from modelscope.hub.snapshot_download import snapshot_download
import os

os.makedirs('./models', exist_ok=True)

print("正在从魔搭社区下载中文BERT模型...")
model_dir = snapshot_download(
    'AI-ModelScope/bert-base-chinese',
    cache_dir='./models'
)

print(f"\n模型已下载到: {model_dir}")
