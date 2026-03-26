import os
from src.document_processor.file_handler import DocumentProcessor
import time

processor = DocumentProcessor()


file_paths = [(os.path.join("examples", "gpt4o_report.pdf"))]

print(f"File exists: {os.path.exists(file_paths)}")


print("--- First run (should process + cache) ---")
chunks = processor.process(file_paths)
print(f"Total chunks: {len(chunks)}")
print(f"\nFirst chunk content:\n{chunks[0].page_content[:300]}")
print(f"\nFirst chunk metadata:\n{chunks[0].metadata}")

print("\n--- Second run (should load from cache instantly) ---")
start = time.time()
chunks2 = processor.process(file_paths)
elapsed = time.time() - start
print(f"Total chunks: {len(chunks2)}")
print(f"Time taken: {elapsed:.2f}s  <- should be near 0 if cache hit")