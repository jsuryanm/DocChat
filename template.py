import os 
from pathlib import Path
import logging 

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s]: %(message)s")


list_of_files = [
    ".github/workflows/.gitkeep",
    "src/__init__.py",
    "src/agents/__init__.py",
    "src/agents/state.py",
    "src/agents/query_rewriter.py",
    "src/agents/answer_grader.py",
    "src/agents/reflexion_agent.py",
    "src/agents/relevance_checker.py",
    "src/agents/workflow.py",
    "src/agents/research_agent.py",
    "src/agents/verfication_agent.py",
    "src/config/__init__.py",
    "src/config/constants.py",
    "src/config/settings.py",
    "src/document_processor/__init__.py",
    "src/document_processor/file_handler.py",
    "src/retriever/__init__.py",
    "src/retriever/builder.py",
    "src/custom_logger/__init__.py",
    "src/custom_logger/logger.py",
    "tests/__init__.py",
    "tests/test_file_handler.py",
    "tests/test_builder.py",
    "tests/test_relevance_checker.py",
    "tests/test_verification_agent.py",
    "tests/test_research_agent.py",
    "app.py",
    ".env"
]

for file_path in list_of_files:
    file_path =  Path(file_path)
    file_dir,file_name = os.path.split(file_path)

    if file_dir != "":
        os.makedirs(file_dir,exist_ok=True)
        logging.info(f"Creating directory: {file_dir} for file: {file_name}")

    if (not os.path.exists(file_path)) or (os.path.getsize(file_path) == 0):
        with open(file_path,"w") as f:
            pass
            logging.info(f"Creating an empty file: {file_path}")
    
    else:
        logging.info(f"{file_name} already exists")


