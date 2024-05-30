from phi.memory import AssistantMemory
from phi.memory.db.postgres import PgMemoryDb
from phi.storage.assistant.postgres import PgAssistantStorage

from db.session import db_url

assistant_memory = AssistantMemory(db=PgMemoryDb(table_name="personalized_assistant_memory", db_url=db_url))
assistant_storage = PgAssistantStorage(table_name="personalized_assistant_storage", db_url=db_url)
