from src.document_processor.file_handler import DocumentProcessor
from src.retriever.builder import RetrieverBuilder
import os 

processor = DocumentProcessor()
builder = RetrieverBuilder()

file_paths = [os.path.join("examples","gpt4o_report.pdf")]
chunks = processor.process(file_paths)

print(f"Chunks: {len(chunks)}")

print("---- Building hybrid retriever ----")
retriever = builder.build_hybrid_retrievers(chunks)
print("Retriever built successfully")

print("---- Testing retrieval ----")
question = "How does reinforcement learning from human feedback (RLHF) improve GPT-4’s safety and alignment?"
results = retriever.invoke(question)

print(f"Documents retrieved: {len(results)}")
for i, doc in enumerate(results[:3], 1):
    print(f"\n[Result {i}]")
    print(f"Metadata: {doc.metadata}")
    print(f"Content preview: {doc.page_content[:200]}")
