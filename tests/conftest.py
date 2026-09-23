"""自动化测试运行时强制使用受控模型，禁止意外请求外部 API。"""

import os


os.environ["L4_LLM_PROVIDER"] = "mock"
os.environ["L4_REVIEW_MODE"] = "mock"
