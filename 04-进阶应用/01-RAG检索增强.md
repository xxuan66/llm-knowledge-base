# RAG 检索增强生成

## 学习目标
学完本节后，你将能够：
- 理解 RAG（Retrieval-Augmented Generation）的核心原理
- 掌握 RAG 系统的完整构建流程
- 了解向量数据库和检索技术
- 能够构建自己的 RAG 应用

## 核心知识点

### 1. RAG 基础

#### 1.1 什么是 RAG
RAG（检索增强生成）是一种结合检索系统和生成模型的技术，通过检索相关文档来增强模型的生成能力。

**核心思想**：
- **检索**：从知识库中找到与问题相关的文档
- **增强**：将检索到的文档作为上下文提供给 LLM
- **生成**：LLM 基于上下文生成准确的回答

#### 1.2 为什么需要 RAG
1. **知识时效性**：LLM 的知识有截止日期，RAG 可以获取最新信息
2. **领域专业性**：可以接入专业领域的私有知识
3. **可追溯性**：回答可以引用具体来源，提高可信度
4. **减少幻觉**：基于事实检索，减少模型编造

#### 1.3 RAG vs 微调
| 维度 | RAG | 微调 |
|------|------|------|
| **知识更新** | 实时更新 | 需要重新训练 |
| **成本** | 较低 | 较高 |
| **准确性** | 取决于检索质量 | 取决于训练数据 |
| **适用场景** | 知识密集型任务 | 任务适配 |

### 2. RAG 系统架构

#### 2.1 核心组件
```
用户问题 → 检索器 → 相关文档 → 生成器 → 回答
              ↑
         向量数据库
```

#### 2.2 检索器（Retriever）
- **功能**：从知识库中检索相关文档
- **方法**：
  - **稠密检索**：使用向量相似度（如 Sentence-BERT）
  - **稀疏检索**：使用关键词匹配（如 BM25）
  - **混合检索**：结合两者优点

#### 2.3 生成器（Generator）
- **功能**：基于检索到的文档生成回答
- **方法**：
  - **直接生成**：将文档作为上下文输入 LLM
  - **多步推理**：先分析文档，再生成回答
  - **引用生成**：在回答中标注引用来源

#### 2.4 向量数据库
- **作用**：存储文档的向量表示，支持高效相似度搜索
- **代表**：ChromaDB、Pinecone、Weaviate、FAISS
- **功能**：索引、检索、更新、删除

### 3. RAG 工作流程

#### 3.1 索引阶段
1. **文档加载**：读取各种格式的文档
2. **文档分割**：将长文档分割为适合检索的块
3. **向量化**：使用嵌入模型将文档块转换为向量
4. **索引存储**：将向量存储到向量数据库

#### 3.2 检索阶段
1. **查询向量化**：将用户问题转换为向量
2. **相似度搜索**：在向量数据库中找到最相似的文档块
3. **重排序**：使用更精细的模型重新排序检索结果
4. **上下文组装**：将检索到的文档组装为上下文

#### 3.3 生成阶段
1. **提示构建**：将检索到的文档和用户问题组合为提示
2. **LLM 生成**：将提示输入 LLM 生成回答
3. **后处理**：格式化回答，添加引用信息

### 4. 优化策略

#### 4.1 检索优化
- **分块策略**：调整文档块大小和重叠
- **嵌入模型选择**：选择适合领域的嵌入模型
- **混合检索**：结合语义和关键词检索
- **重排序**：使用更精确的模型重新排序

#### 4.2 生成优化
- **提示工程**：设计有效的提示模板
- **上下文管理**：控制上下文长度，避免超出限制
- **多轮对话**：维护对话历史，支持连续对话

#### 4.3 评估指标
- **检索准确率**：检索到的文档与问题的相关性
- **答案准确率**：生成答案的正确性
- **引用准确率**：引用来源的准确性
- **响应时间**：从问题到回答的时间

## 代码示例

### 完整 RAG 系统实现

```python
import os
from typing import List, Dict, Any
from dataclasses import dataclass
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import openai

@dataclass
class Document:
    """文档类"""
    content: str
    metadata: Dict[str, Any]
    embedding: List[float] = None

class RAGSystem:
    """RAG 系统"""
    
    def __init__(self, 
                 embedding_model_name: str = "all-MiniLM-L6-v2",
                 collection_name: str = "rag_collection",
                 persist_directory: str = "./chroma_db"):
        
        # 初始化嵌入模型
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # 初始化向量数据库
        self.chroma_client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=persist_directory
        ))
        
        # 创建或获取集合
        try:
            self.collection = self.chroma_client.get_collection(collection_name)
        except:
            self.collection = self.chroma_client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
    
    def chunk_document(self, text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
        """文档分块"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # 如果不是最后一块，尝试在句子边界分割
            if end < len(text):
                # 寻找最近的句号、问号、感叹号
                for punct in ['。', '？', '！', '.', '?', '!']:
                    punct_pos = text.rfind(punct, start, end)
                    if punct_pos != -1 and punct_pos > start + chunk_size // 2:
                        end = punct_pos + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # 移动到下一个块，考虑重叠
            start = end - overlap
            if start >= len(text):
                break
        
        return chunks
    
    def add_documents(self, documents: List[Document]):
        """添加文档到知识库"""
        # 分块
        all_chunks = []
        all_metadatas = []
        all_ids = []
        
        for doc_idx, doc in enumerate(documents):
            chunks = self.chunk_document(doc.content)
            
            for chunk_idx, chunk in enumerate(chunks):
                chunk_id = f"doc_{doc_idx}_chunk_{chunk_idx}"
                all_chunks.append(chunk)
                all_metadatas.append({
                    **doc.metadata,
                    "chunk_index": chunk_idx,
                    "doc_index": doc_idx
                })
                all_ids.append(chunk_id)
        
        # 生成嵌入
        embeddings = self.embedding_model.encode(all_chunks).tolist()
        
        # 添加到集合
        self.collection.add(
            documents=all_chunks,
            embeddings=embeddings,
            metadatas=all_metadatas,
            ids=all_ids
        )
        
        print(f"已添加 {len(all_chunks)} 个文档块")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """检索相关文档"""
        # 查询向量化
        query_embedding = self.embedding_model.encode(query).tolist()
        
        # 检索
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        # 格式化结果
        retrieved_docs = []
        for i in range(len(results['documents'][0])):
            retrieved_docs.append({
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i]
            })
        
        return retrieved_docs
    
    def build_prompt(self, query: str, retrieved_docs: List[Dict]) -> str:
        """构建提示"""
        context_parts = []
        
        for i, doc in enumerate(retrieved_docs):
            context_parts.append(f"[文档 {i+1}] {doc['content']}")
        
        context = "\n\n".join(context_parts)
        
        prompt = f"""请基于以下上下文回答问题。如果上下文没有相关信息，请说明。

上下文：
{context}

问题：{query}

回答："""
        
        return prompt
    
    def generate_answer(self, prompt: str, model: str = "gpt-3.5-turbo") -> str:
        """生成回答"""
        # 使用 OpenAI API（需要设置 API 密钥）
        # response = openai.ChatCompletion.create(
        #     model=model,
        #     messages=[
        #         {"role": "system", "content": "你是一个有用的助手，基于提供的上下文回答问题。"},
        #         {"role": "user", "content": prompt}
        #     ],
        #     temperature=0.7
        # )
        # return response.choices[0].message.content
        
        # 模拟生成（实际使用时替换为真实 API 调用）
        return f"基于检索到的文档，模拟生成的回答。实际使用时调用 LLM API。"
    
    def query(self, question: str, top_k: int = 5) -> Dict:
        """完整查询流程"""
        # 1. 检索
        retrieved_docs = self.retrieve(question, top_k)
        
        # 2. 构建提示
        prompt = self.build_prompt(question, retrieved_docs)
        
        # 3. 生成回答
        answer = self.generate_answer(prompt)
        
        # 4. 返回结果
        return {
            "question": question,
            "answer": answer,
            "retrieved_docs": retrieved_docs,
            "prompt": prompt
        }

# 使用示例
if __name__ == "__main__":
    # 创建 RAG 系统
    rag = RAGSystem()
    
    # 示例文档
    sample_documents = [
        Document(
            content="机器学习是人工智能的一个分支，它使计算机能够从数据中学习，而无需进行明确编程。机器学习算法通过训练数据构建数学模型，以便对新数据做出预测或决策。",
            metadata={"source": "机器学习入门", "type": "定义"}
        ),
        Document(
            content="深度学习是机器学习的一个子集，它使用多层神经网络来模拟人脑的工作方式。深度学习在图像识别、自然语言处理和语音识别等领域取得了突破性进展。",
            metadata={"source": "深度学习导论", "type": "定义"}
        ),
        Document(
            content="大语言模型（LLM）是基于 Transformer 架构的深度学习模型，通过在大规模文本数据上进行预训练，能够理解和生成人类语言。著名的 LLM 包括 GPT、BERT、LLaMA 等。",
            metadata={"source": "大语言模型概述", "type": "定义"}
        )
    ]
    
    # 添加文档
    rag.add_documents(sample_documents)
    
    # 查询
    result = rag.query("什么是机器学习？")
    
    print(f"问题: {result['question']}")
    print(f"检索到的文档数: {len(result['retrieved_docs'])}")
    print(f"回答: {result['answer']}")
    
    # 显示检索到的文档
    print("\n检索到的文档:")
    for i, doc in enumerate(result['retrieved_docs']):
        print(f"\n[文档 {i+1}] (距离: {doc['distance']:.4f})")
        print(f"内容: {doc['content'][:100]}...")
```

### 简易 RAG 演示（无外部依赖）

```python
import numpy as np
from typing import List, Dict

class SimpleRAG:
    """简易 RAG 实现（使用 NumPy 模拟向量搜索）"""
    
    def __init__(self):
        self.documents = []
        self.embeddings = []
    
    def simple_embedding(self, text: str) -> List[float]:
        """简单的嵌入模拟（实际应使用真实嵌入模型）"""
        # 基于字符的简单哈希嵌入（仅作演示）
        np.random.seed(hash(text) % 2**32)
        return np.random.randn(128).tolist()
    
    def add_document(self, content: str, metadata: Dict = None):
        """添加文档"""
        self.documents.append({
            'content': content,
            'metadata': metadata or {}
        })
        self.embeddings.append(self.simple_embedding(content))
    
    def cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """计算余弦相似度"""
        a = np.array(a)
        b = np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """检索文档"""
        query_embedding = self.simple_embedding(query)
        
        # 计算相似度
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            sim = self.cosine_similarity(query_embedding, doc_embedding)
            similarities.append((i, sim))
        
        # 排序并取 top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _ in similarities[:top_k]]
        
        # 返回结果
        results = []
        for idx in top_indices:
            results.append({
                'content': self.documents[idx]['content'],
                'metadata': self.documents[idx]['metadata'],
                'similarity': similarities[idx][1]
            })
        
        return results
    
    def generate_simple_answer(self, query: str, retrieved_docs: List[Dict]) -> str:
        """生成简单回答（实际应调用 LLM）"""
        if not retrieved_docs:
            return "抱歉，我没有找到相关信息。"
        
        # 简单拼接检索到的文档
        context = "\n".join([doc['content'] for doc in retrieved_docs[:2]])
        return f"根据检索到的信息：\n{context}\n\n关于'{query}'的更多信息请查阅相关资料。"

# 使用示例
if __name__ == "__main__":
    rag = SimpleRAG()
    
    # 添加文档
    rag.add_document("机器学习是人工智能的核心技术之一", {"领域": "AI"})
    rag.add_document("深度学习使用神经网络进行学习", {"领域": "AI"})
    rag.add_document("Python 是最流行的编程语言之一", {"领域": "编程"})
    
    # 查询
    query = "什么是人工智能的核心技术？"
    results = rag.retrieve(query, top_k=2)
    
    print(f"查询: {query}")
    print("\n检索结果:")
    for i, doc in enumerate(results):
        print(f"\n[文档 {i+1}] (相似度: {doc['similarity']:.4f})")
        print(f"内容: {doc['content']}")
    
    # 生成回答
    answer = rag.generate_simple_answer(query, results)
    print(f"\n回答: {answer}")
```

## 完整 RAG 工作流示例

### 1. 端到端 RAG 工作流

```python
"""
完整 RAG 工作流实现
包含文档处理、索引、检索、生成全流程
"""
import os
from typing import List, Dict, Optional
from dataclasses import dataclass
from pathlib import Path
import hashlib

@dataclass
class Document:
    """文档数据类"""
    content: str
    metadata: Dict
    doc_id: str = None
    
    def __post_init__(self):
        if self.doc_id is None:
            self.doc_id = hashlib.md5(self.content.encode()).hexdigest()

class DocumentProcessor:
    """文档处理器"""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def load_document(self, file_path: str) -> str:
        """加载文档"""
        path = Path(file_path)
        
        if path.suffix == '.txt':
            return path.read_text(encoding='utf-8')
        elif path.suffix == '.md':
            return path.read_text(encoding='utf-8')
        elif path.suffix == '.pdf':
            # 需要安装 PyPDF2
            try:
                from PyPDF2 import PdfReader
                reader = PdfReader(path)
                return "\n".join([page.extract_text() for page in reader.pages])
            except ImportError:
                raise ImportError("请安装 PyPDF2: pip install PyPDF2")
        else:
            raise ValueError(f"不支持的文件格式: {path.suffix}")
    
    def split_text(self, text: str) -> List[str]:
        """分割文本为块"""
        # 按段落分割
        paragraphs = text.split('\n\n')
        
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            # 如果当前块加上新段落超过限制，保存当前块
            if len(current_chunk) + len(para) > self.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                # 保留重叠部分
                overlap_start = max(0, len(current_chunk) - self.chunk_overlap)
                current_chunk = current_chunk[overlap_start:] + "\n\n" + para
            else:
                current_chunk += "\n\n" + para if current_chunk else para
        
        # 添加最后一块
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def process_document(self, file_path: str, metadata: Dict = None) -> List[Document]:
        """处理文档，返回文档块列表"""
        # 加载文档
        content = self.load_document(file_path)
        
        # 分割文本
        chunks = self.split_text(content)
        
        # 创建文档对象
        documents = []
        for i, chunk in enumerate(chunks):
            doc_metadata = metadata or {}
            doc_metadata.update({
                "source": file_path,
                "chunk_index": i,
                "total_chunks": len(chunks)
            })
            
            documents.append(Document(
                content=chunk,
                metadata=doc_metadata
            ))
        
        return documents

class EmbeddingGenerator:
    """嵌入向量生成器"""
    
    def __init__(self, model_name: str = "text-embedding-ada-002"):
        self.model_name = model_name
        # 实际实现应使用 OpenAI API 或其他嵌入模型
        # 这里使用简单的模拟实现
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """生成嵌入向量"""
        # 模拟嵌入生成（实际应调用嵌入模型）
        import random
        embeddings = []
        for text in texts:
            # 生成随机嵌入（实际应使用模型）
            random.seed(hash(text) % 2**32)
            embedding = [random.random() for _ in range(1536)]  # OpenAI ada-002 维度
            embeddings.append(embedding)
        return embeddings

class VectorStore:
    """向量存储"""
    
    def __init__(self, collection_name: str = "default"):
        self.collection_name = collection_name
        self.documents: List[Document] = []
        self.embeddings: List[List[float]] = []
    
    def add_documents(self, documents: List[Document], embeddings: List[List[float]]):
        """添加文档和嵌入向量"""
        self.documents.extend(documents)
        self.embeddings.extend(embeddings)
    
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict]:
        """搜索相似文档"""
        import numpy as np
        
        query_vec = np.array(query_embedding)
        similarities = []
        
        for i, doc_embedding in enumerate(self.embeddings):
            doc_vec = np.array(doc_embedding)
            # 计算余弦相似度
            similarity = np.dot(query_vec, doc_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(doc_vec))
            similarities.append((i, similarity))
        
        # 排序并取 top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for idx, score in similarities[:top_k]:
            results.append({
                "document": self.documents[idx],
                "score": float(score)
            })
        
        return results

class RAGWorkflow:
    """完整 RAG 工作流"""
    
    def __init__(self, 
                 embedding_model: str = "text-embedding-ada-002",
                 llm_model: str = "gpt-4",
                 chunk_size: int = 512,
                 chunk_overlap: int = 50):
        
        self.processor = DocumentProcessor(chunk_size, chunk_overlap)
        self.embedder = EmbeddingGenerator(embedding_model)
        self.vector_store = VectorStore()
        self.llm_model = llm_model
    
    def ingest_documents(self, file_paths: List[str], metadata: Dict = None):
        """摄入文档到知识库"""
        all_documents = []
        
        for file_path in file_paths:
            print(f"处理文档: {file_path}")
            documents = self.processor.process_document(file_path, metadata)
            all_documents.extend(documents)
        
        # 生成嵌入向量
        print(f"生成 {len(all_documents)} 个文档块的嵌入向量...")
        texts = [doc.content for doc in all_documents]
        embeddings = self.embedder.generate_embeddings(texts)
        
        # 存储到向量数据库
        self.vector_store.add_documents(all_documents, embeddings)
        print(f"成功添加 {len(all_documents)} 个文档块到知识库")
    
    def query(self, question: str, top_k: int = 3) -> Dict:
        """查询知识库并生成回答"""
        # 1. 生成查询嵌入
        query_embedding = self.embedder.generate_embeddings([question])[0]
        
        # 2. 检索相关文档
        retrieved_docs = self.vector_store.search(query_embedding, top_k)
        
        # 3. 构建上下文
        context = "\n\n".join([
            f"[文档 {i+1}] (相关度: {doc['score']:.4f})\n{doc['document'].content}"
            for i, doc in enumerate(retrieved_docs)
        ])
        
        # 4. 生成回答（模拟 LLM 调用）
        answer = self._generate_answer(question, context)
        
        return {
            "question": question,
            "answer": answer,
            "sources": [
                {
                    "content": doc["document"].content[:200] + "...",
                    "metadata": doc["document"].metadata,
                    "score": doc["score"]
                }
                for doc in retrieved_docs
            ],
            "context_length": len(context)
        }
    
    def _generate_answer(self, question: str, context: str) -> str:
        """生成回答（实际应调用 LLM）"""
        # 模拟 LLM 生成
        return f"""根据检索到的信息，我找到了以下相关内容：

{context[:500]}...

基于以上信息，关于"{question}"的回答是：
[这里应该是 LLM 生成的详细回答，基于检索到的上下文信息]

参考来源已标注在上方。"""

# 使用示例
if __name__ == "__main__":
    # 创建 RAG 工作流
    rag = RAGWorkflow(
        embedding_model="text-embedding-ada-002",
        llm_model="gpt-4",
        chunk_size=512,
        chunk_overlap=50
    )
    
    # 摄入文档
    rag.ingest_documents([
        "docs/introduction.md",
        "docs/technical_details.md",
        "docs/faq.md"
    ])
    
    # 查询
    result = rag.query("什么是 RAG 系统？")
    
    print(f"问题: {result['question']}")
    print(f"\n回答: {result['answer']}")
    print(f"\n来源:")
    for i, source in enumerate(result['sources']):
        print(f"  {i+1}. {source['content']} (相关度: {source['score']:.4f})")
```

### 2. RAG 工作流配置模板

**文件：** `rag_workflow_config.yaml`

```yaml
# RAG 工作流配置
workflow:
  name: "企业知识库 RAG"
  version: "1.0.0"
  description: "基于企业文档的 RAG 系统"

# 文档处理配置
document_processing:
  # 支持的文件格式
  supported_formats:
    - ".txt"
    - ".md"
    - ".pdf"
    - ".docx"
  
  # 分块策略
  chunking:
    strategy: "paragraph"  # paragraph, sentence, fixed
    chunk_size: 512
    chunk_overlap: 50
    min_chunk_size: 100
  
  # 元数据提取
  metadata_extraction:
    enabled: true
    extractors:
      - "filename"
      - "creation_date"
      - "author"
      - "section_headers"

# 嵌入配置
embedding:
  model: "text-embedding-ada-002"
  provider: "openai"
  batch_size: 100
  max_retries: 3

# 向量存储配置
vector_store:
  provider: "chroma"  # chroma, pinecone, weaviate, faiss
  collection_name: "enterprise_knowledge"
  persist_directory: "./data/vector_store"
  
  # Chroma 特定配置
  chroma:
    host: "localhost"
    port: 8000
  
  # Pinecone 特定配置
  pinecone:
    api_key: "${PINECONE_API_KEY}"
    environment: "us-west1-gcp"
    index_name: "enterprise-rag"

# 检索配置
retrieval:
  top_k: 5
  similarity_threshold: 0.7
  
  # 重排序配置
  reranking:
    enabled: true
    model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_n: 3
  
  # 混合检索
  hybrid_search:
    enabled: true
    semantic_weight: 0.7
    keyword_weight: 0.3

# 生成配置
generation:
  model: "gpt-4"
  provider: "openai"
  temperature: 0.7
  max_tokens: 2000
  
  # 提示模板
  prompt_template: |
    你是一个专业的企业知识助手。请基于以下参考文档回答用户问题。
    
    参考文档：
    {context}
    
    用户问题：{question}
    
    请遵循以下规则：
    1. 只基于提供的文档内容回答
    2. 如果文档中没有相关信息，请明确说明
    3. 引用具体的文档来源
    4. 使用专业但易懂的语言
    
    回答：

# 评估配置
evaluation:
  enabled: true
  metrics:
    - "retrieval_precision"
    - "retrieval_recall"
    - "answer_relevance"
    - "answer_faithfulness"
  
  test_set: "./data/eval/test_questions.json"

# 监控配置
monitoring:
  enabled: true
  log_queries: true
  log_retrievals: true
  log_generations: true
  metrics_port: 9090
```

### 3. RAG 工作流自动化脚本

**文件：** `rag_workflow_automation.py`

```python
"""
RAG 工作流自动化脚本
支持定时更新、增量索引、自动评估
"""
import schedule
import time
from datetime import datetime
from pathlib import Path
import json

class RAGWorkflowAutomation:
    """RAG 工作流自动化"""
    
    def __init__(self, config_path: str):
        self.config = self.load_config(config_path)
        self.rag_workflow = RAGWorkflow(**self.config["workflow"])
        self.last_update = None
    
    def load_config(self, config_path: str) -> dict:
        """加载配置"""
        import yaml
        with open(config_path) as f:
            return yaml.safe_load(f)
    
    def incremental_update(self):
        """增量更新知识库"""
        print(f"[{datetime.now()}] 开始增量更新...")
        
        # 获取新文档
        docs_dir = Path(self.config["document_processing"]["docs_directory"])
        new_docs = self.get_new_documents(docs_dir)
        
        if new_docs:
            print(f"发现 {len(new_docs)} 个新文档")
            self.rag_workflow.ingest_documents(new_docs)
            self.last_update = datetime.now()
            print("增量更新完成")
        else:
            print("没有新文档需要更新")
    
    def get_new_documents(self, docs_dir: Path) -> List[str]:
        """获取新文档列表"""
        if self.last_update is None:
            # 首次运行，返回所有文档
            return [str(p) for p in docs_dir.rglob("*") if p.is_file()]
        
        # 返回修改时间晚于上次更新的文档
        new_docs = []
        for file_path in docs_dir.rglob("*"):
            if file_path.is_file():
                mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                if mtime > self.last_update:
                    new_docs.append(str(file_path))
        
        return new_docs
    
    def run_evaluation(self):
        """运行评估"""
        print(f"[{datetime.now()}] 开始评估...")
        
        # 加载测试集
        test_set_path = self.config["evaluation"]["test_set"]
        with open(test_set_path) as f:
            test_questions = json.load(f)
        
        # 运行评估
        results = []
        for item in test_questions:
            result = self.rag_workflow.query(item["question"])
            results.append({
                "question": item["question"],
                "expected": item["answer"],
                "generated": result["answer"],
                "sources": result["sources"]
            })
        
        # 保存评估结果
        eval_dir = Path("./data/evaluation")
        eval_dir.mkdir(parents=True, exist_ok=True)
        
        eval_file = eval_dir / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(eval_file, 'w') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"评估结果已保存到: {eval_file}")
    
    def start_scheduler(self):
        """启动定时任务"""
        # 每小时增量更新
        schedule.every(1).hours.do(self.incremental_update)
        
        # 每天凌晨2点运行评估
        schedule.every().day.at("02:00").do(self.run_evaluation)
        
        print("定时任务已启动...")
        print("- 每小时增量更新")
        print("- 每天凌晨2点评估")
        
        while True:
            schedule.run_pending()
            time.sleep(60)

# 使用示例
if __name__ == "__main__":
    automation = RAGWorkflowAutomation("rag_workflow_config.yaml")
    automation.start_scheduler()
```

## 练习题

### 基础题
1. **概念理解**：解释 RAG 系统的三个核心组件及其作用。
2. **流程分析**：画出 RAG 的完整工作流程图，标注每个步骤。

### 进阶题
3. **系统设计**：设计一个用于企业内部知识库的 RAG 系统，考虑文档格式、检索策略、生成优化。
4. **性能优化**：分析如何优化 RAG 系统的检索速度和生成质量。

### 思考题
5. **前沿方向**：RAG 未来可能的发展方向是什么？多模态 RAG、Agentic RAG 等新范式有什么特点？

### GitHub 热门资源
1. **LangChain** - RAG 框架
2. **LlamaIndex** - 数据框架
3. **Chroma** - 向量数据库
4. **RAGAS** - RAG 评估

### 重要论文
- "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Lewis et al., 2020)
- "Dense Passage Retrieval for Open-Domain Question Answering" (Karpukhin et al., 2020)
- "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection" (Asai et al., 2023)

### 工具与框架
- Hugging Face - 嵌入模型和 LLM
- Pinecone - 托管向量数据库
- Weaviate - 向量搜索引擎
- FAISS - 向量相似度搜索

### 学习资源
- LangChain RAG Tutorial
- LlamaIndex RAG Guide
- RAG Best Practices - Pinecone 系列教程