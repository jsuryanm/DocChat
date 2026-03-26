import os
from src.document_processor.file_handler import DocumentProcessor
from src.retriever.builder import RetrieverBuilder
from src.agents.relevance_checker import RelevanceChecker

processor = DocumentProcessor()
builder = RetrieverBuilder()
checker = RelevanceChecker()

file_paths = [os.path.join("examples", "gpt4o_report.pdf")]

# Step 1: Build retriever
print("--- Building retriever ---")
chunks = processor.process(file_paths)
print(f"Chunks: {len(chunks)}")
retriever = builder.build_hybrid_retriever(chunks)
print("Retriever ready\n")

# Step 2: Test questions based on actual report content
test_questions = [

    # === Should return CAN_ANSWER ===
    # The report directly states GPT-4 passed the bar exam in the top 10%
    (
        "What percentile did GPT-4 achieve on the Uniform Bar Exam?",
        "CAN_ANSWER"
    ),
    # The report has a full table of exam results
    (
        "How did GPT-4 perform on the LSAT compared to GPT-3.5?",
        "CAN_ANSWER"
    ),
    # The report explicitly mentions HumanEval pass rate
    (
        "What was GPT-4's score on the HumanEval coding benchmark?",
        "CAN_ANSWER"
    ),
    # The report explicitly discusses RLHF and hallucinations
    (
        "How much did GPT-4 reduce hallucinations compared to GPT-3.5?",
        "CAN_ANSWER"
    ),
    # The report directly mentions RealToxicityPrompts results
    (
        "What percentage of the time does GPT-4 produce toxic content on RealToxicityPrompts?",
        "CAN_ANSWER"
    ),

    # === Should return PARTIAL ===
    # The report discusses safety but doesn't enumerate every jailbreak technique
    (
        "What specific jailbreak techniques were used against GPT-4 during red teaming?",
        "PARTIAL"
    ),
    # The report mentions MMLU multilingual results but at a high level
    (
        "What were GPT-4's exact accuracy scores for every language tested on MMLU?",
        "PARTIAL"
    ),

    # === Should return NO_MATCH ===
    # Completely unrelated to the report
    (
        "What is the weather forecast in Tokyo this weekend?",
        "NO_MATCH"
    ),
    # GPT-4o is a different model not covered in this report
    (
        "What are the new features of GPT-4o released in 2024?",
        "NO_MATCH"
    ),
    # The report does not cover Gemini at all
    (
        "How does Google Gemini compare to GPT-4 on coding benchmarks?",
        "NO_MATCH"
    ),
]

# Step 3: Run checks
print("--- Running relevance checks ---\n")
results = {"correct": 0, "wrong": 0}

for question, expected in test_questions:
    result = checker.check(question, retriever, k=5)
    correct = result == expected
    status = "✅" if correct else "❌"
    results["correct" if correct else "wrong"] += 1

    print(f"{status} [{expected}] → [{result}]")
    print(f"   Q: {question}\n")

# Step 4: Summary
total = len(test_questions)
print(f"--- Results: {results['correct']}/{total} correct ---")