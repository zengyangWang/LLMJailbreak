import os
import time
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer

from utils.test_utils import text_process


class Text_Embedding_Ada():
    """使用本地 BGE-M3 模型生成文本向量，不依赖外部 API 或 peft。"""

    def __init__(
        self,
        model_dir: Optional[str] = None,
        normalize_embeddings: bool = True,
        max_length: int = 512,
        device: Optional[str] = None,
    ):
        self.model_dir = model_dir or self._default_model_dir()
        self.normalize_embeddings = normalize_embeddings
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = None  # "sentence_transformers" or "transformers"
        self._load_model()
        self.load_saved_embeddings()

    def _default_model_dir(self) -> str:
        """默认模型路径为 ../../../llm_weights/bge-m3（相对于 DrAttack 根目录）。"""
        drattack_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        return os.path.abspath(os.path.join(drattack_root, "..", "..", "llm_weights", "bge-m3"))

    def _load_model(self):
        if not os.path.isdir(self.model_dir):
            raise FileNotFoundError(
                f"未找到本地向量模型目录：{self.model_dir}，请确认路径是否正确。"
            )
        # BGE-M3 是 SentenceTransformers 结构（带有 config_sentence_transformers.json 等），
        # 优先尝试从本地目录加载；若本地缺少权重文件，则退回到通过仓库名
        # "BAAI/bge-m3" 远程加载；再不满足时，退回到普通 Transformers 加载。
        st_config_path = os.path.join(self.model_dir, "config_sentence_transformers.json")
        # 判断本地目录下是否已经包含常见的权重文件
        has_local_weights = any(
            os.path.exists(os.path.join(self.model_dir, fname))
            for fname in ("pytorch_model.bin", "model.safetensors")
        )
        if os.path.exists(st_config_path) and has_local_weights:
            self.model_type = "sentence_transformers"
            self.model = SentenceTransformer(self.model_dir, device=self.device)
            # SentenceTransformer 自己内部管理 tokenizer，这里不单独暴露
            self.tokenizer = None
        elif os.path.exists(st_config_path):
            # 目录结构像 SentenceTransformers，但本地缺少权重文件
            #（例如只下载了部分文件），则直接通过仓库名从 HuggingFace
            # 拉取完整模型到缓存目录。
            self.model_type = "sentence_transformers"
            self.model = SentenceTransformer("BAAI/bge-m3", device=self.device)
            self.tokenizer = None
        else:
            self.model_type = "transformers"
            torch_dtype = (
                torch.float16 if self.device.startswith("cuda") else torch.float32
            )
            # 使用 AutoTokenizer/AutoModel 以兼容通用 Transformers 权重目录
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_dir,
                use_fast=True,
            )
            self.model = AutoModel.from_pretrained(
                self.model_dir,
                torch_dtype=torch_dtype,
            ).to(self.device)
        self.model.eval()

    def load_saved_embeddings(self):
        self.embedding_cache = {}

    def get_embedding(self, text: str):
        text = text.replace("\n", " ")

        if text in self.embedding_cache:
            return self.embedding_cache[text]

        retry_num = 0
        while True:
            try:
                processed_text = text_process(text)
                if self.model_type == "sentence_transformers":
                    # SentenceTransformer.encode 直接返回句向量
                    embedding = self.model.encode(
                        [processed_text],
                        convert_to_tensor=True,
                        normalize_embeddings=self.normalize_embeddings,
                    )  # [1, hidden]
                else:
                    inputs = self.tokenizer(
                        processed_text,
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors="pt",
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                    # 取 CLS 向量并归一化
                    embedding = outputs.last_hidden_state[:, 0, :]
                    if self.normalize_embeddings:
                        embedding = F.normalize(embedding, p=2, dim=-1)
                ret = embedding.detach().cpu().unsqueeze(0).unsqueeze(0).to(torch.float32)
                break
            except Exception as e:
                retry_num += 1
                print(e)
                if retry_num > 5:
                    print('Failed to get embedding for text: \n')
                    print("=*" * 20)
                    print(text)
                    print("=*" * 20)
                    raise
                print('Retrying... ', retry_num)
                time.sleep(1)

        self.embedding_cache[text] = ret
        return ret