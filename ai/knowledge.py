from phi.knowledge import AssistantKnowledge
from phi.embedder.openai import OpenAIEmbedder
from phi.vectordb.pgvector import PgVector2

from ai.settings import ai_settings
from db.session import db_url

assistant_knowledge = AssistantKnowledge(
    vector_db=PgVector2(
        db_url=db_url,
        collection="personalized_assistant_documents",
        embedder=OpenAIEmbedder(model=ai_settings.embedding_model, dimensions=1536),
    ),
    # 3 references are added to the prompt
    num_documents=3,
)
