from a2a.server.apps import A2AStarletteApplication
from a2a.server.agent_execution import AgentExecutor,RequestContext
from a2a.server.tasks import TaskUpdater
from a2a.types import AgentCard,AgentCapabilities,AgentSkill,UnsupportedOperationError

from a2a.utils import new_agent_text_message

from src.config.settings import settings 
from src.custom_logger.logger import logger



class DocChatA2AExector(AgentExecutor):

    def __init__(self,workflow):
        self.workflow = workflow

    
    async def execute(self,context: RequestContext, updater: TaskUpdater) -> None:

        question = self._extract_text(context)

        if not question:
            # fails task 
            updater.failed("No question found in request")
            return 
        
        logger.info(f"A2A task received | question: {question[:80]}...")

        try:
            updater.start_work() 

            result = await self.workflow.run(question)
            answer = result.get("final_answer","No answer produced.")
            logger.info(f"A2A task complete | delegate={result.get("delegated",False)}")

            await updater.add_artifact(parts=[new_agent_text_message(answer)],
                                       name="answer")
            
            updater.complete()
        
        except Exception as e:
            logger.error(f"A2A executor error: {e}")
            updater.failed(str(e))

    
    async def cancel(self, context: RequestContext, updater: TaskUpdater) -> None:
        raise UnsupportedOperationError("DocChat does not support task cancellation")
    
    @staticmethod 
    def _extract_text(context: RequestContext) -> str:
        try:
            for part in context.message.parts:
                root = getattr(part,"root",part)
                if hasattr(root,"text") and root.text:
                    return root.text.strip()
        except Exception as e:
            logger.warning(f"Could not extract text from A2A message: {e}")
        
        return ""


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

def create_a2a_app(workflow) -> A2AStarletteApplication:
    """Build and return Starlette ASGI app that serves the DocChat agent over A2A protocol"""
    executor = DocChatA2AExector(workflow)
    card = build_agent_card()
    logger.info(f"A2A server ready | {card.name}  v{card.version} | {settings.A2A_HOST}:{settings.A2A_PORT}")
    
    return A2AStarletteApplication(agent_card=card,
                                   agent_executor=executor)