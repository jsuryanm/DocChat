from a2a.server.apps import A2AStarletteApplication
from a2a.server.agent_execution import AgentExecutor,RequestContext
from a2a.server.tasks import TaskUpdater
from a2a.types import AgentCard,AgentCapabilities,AgentSkill,UnsupportedOperationError

from a2a.utils import new_agent_text_message

from src.config.settings import settings 
from src.custom_logger.logger import logger


# AgentCard: agent metadata definition
# A2AStarletteApplication: Creates HTTP server 
# AgentExecutor: Base class every A2A agent must implement
# RequestContext: Incoming request data
# TaskUpdater: Used to send progress updates
 
"""The  goal of this client.py is:
1. Creating an A2A executor runs the workflow
2. Define agent metadata
3. Creates A2A HTTP server

External Agent
      |
A2A HTTP Server
      |
DocChatA2AExecutor
      |
AgentWorkflow.run()
      |
Final Answer returned
"""

class DocChatA2AExector(AgentExecutor):

    def __init__(self,workflow):
        self.workflow = workflow

    
    async def execute(self,context: RequestContext, updater: TaskUpdater) -> None:
        """Called by A2A server for every incoming task 
        
        1. Extract User's question from the A2A message
        2. Run the full AgentWorkflow (rewrite -> retrieve -> verify)
        3. Push the final answer back as an A2A artifact"""

        question = self._extract_text(context)

        if not question:
            # fails task 
            updater.failed("No question found in request")
            return 
        
        logger.info(f"A2A task received | question: {question[:80]}...")

        try:
            updater.start_work() 
            # starts task useful for streaming,monitoring and progress tracking 

            result = await self.workflow.run(question)
            answer = result.get("final_answer","No answer produced.")
            logger.info(f"A2A task complete | delegate={result.get("delegated",False)}")

            await updater.add_artifact(parts=[new_agent_text_message(answer)],
                                       name="answer")
            # Sends result back to caller agent 
            
            updater.complete()
            # complete task 
        
        except Exception as e:
            logger.error(f"A2A executor error: {e}")
            updater.failed(str(e))

    
    async def cancel(self, context: RequestContext, updater: TaskUpdater) -> None:
        """Cancellation is not supported — raise to signal this to the caller."""
        raise UnsupportedOperationError("DocChat does not support task cancellation")
    
    @staticmethod 
    def _extract_text(context: RequestContext) -> str:
        """ Pull plain text from the first text part of the incoming A2A message.
        Returns an empty string if no text part is found."""
        try:
            for part in context.message.parts:
                root = getattr(part,"root",part)
                if hasattr(root,"text") and root.text:
                    return root.text.strip()
        except Exception as e:
            logger.warning(f"Could not extract text from A2A message: {e}")
        
        return ""

# agent card 

def build_agent_card() -> AgentCard:
    """Describes agent to A2A ecosystem 
    Served at /.well-known/agent.json by A2AStarletteApplication"""
    # Enables agent discovery
    base_url = f"http://{settings.A2A_HOST}:{settings.A2A_PORT}"

    return AgentCard(name="DocChat Agent",
                        description="Answers question grounded in uploaded documents using Agentic RAG with reflexion, verification and optional MCP tools",
                        url=base_url,
                        version="0.1.0",
                        capabilities=AgentCapabilities(streaming=False),
                        skills=[AgentSkill(id="document_qa",
                                        name="Document Q&A",
                                        description="Answer factural questions from a corpus of uploaded documents (PDF,DOCX,TXT,Markdown)",
                                        input_modes=["text"],
                                        output_modes=["text"])],
                            default_input_modes=["text"],
                            default_output_modes=["text"])

# factory 
def create_a2a_app(workflow) -> A2AStarletteApplication:
    """Build and return Starlette ASGI app that serves the DocChat agent over A2A protocol"""
    executor = DocChatA2AExector(workflow)
    card = build_agent_card()
    logger.info(f"A2A server ready | {card.name}  v{card.version} | {settings.A2A_HOST}:{settings.A2A_PORT}")
    
    return A2AStarletteApplication(agent_card=card,
                                   agent_executor=executor)