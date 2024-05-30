from textwrap import dedent
from typing import Optional, List

from phi.assistant import Assistant
from phi.tools import Toolkit
from phi.tools.exa import ExaTools
from phi.tools.calculator import Calculator
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.yfinance import YFinanceTools
from phi.tools.file import FileTools
from phi.llm.openai import OpenAIChat
from phi.assistant.python import PythonAssistant

from ai.settings import ai_settings
from ai.storage import assistant_storage, assistant_memory
from ai.knowledge import assistant_knowledge
from workspace.settings import ws_settings


scratch_dir = ws_settings.ws_root.joinpath("scratch")
if not scratch_dir.exists():
    scratch_dir.mkdir(exist_ok=True, parents=True)


def get_personalized_assistant(
    calculator: bool = False,
    ddg_search: bool = False,
    file_tools: bool = False,
    finance_tools: bool = False,
    python_assistant: bool = False,
    research_assistant: bool = False,
    run_id: Optional[str] = None,
    user_id: Optional[str] = None,
    debug_mode: bool = True,
) -> Assistant:
    # LLM to use for the Assistant
    llm = OpenAIChat(
        model=ai_settings.gpt_4,
        max_tokens=ai_settings.default_max_tokens,
        temperature=ai_settings.default_temperature,
    )

    # Add tools available to the Personalized Assistant
    tools: List[Toolkit] = []
    # Extra instructions for using tools
    extra_instructions: List[str] = []
    if calculator:
        tools.append(
            Calculator(
                add=True,
                subtract=True,
                multiply=True,
                divide=True,
                exponentiate=True,
                factorial=True,
                is_prime=True,
                square_root=True,
            )
        )
    if ddg_search:
        tools.append(DuckDuckGo(fixed_max_results=3))
    if finance_tools:
        tools.append(
            YFinanceTools(
                stock_price=True, company_info=True, analyst_recommendations=True, company_news=True
            )
        )
    if file_tools:
        tools.append(FileTools(base_dir=ws_settings.ws_root))
        extra_instructions.append(
            "You can use the `read_file` tool to read a file, `save_file` to save a file, and `list_files` to list files in the working directory."
        )

    # Add team members available to the LLM OS
    team: List[Assistant] = []
    if python_assistant:
        _python_assistant = PythonAssistant(
            llm=llm,
            name="Python Assistant",
            role="Write and run python code",
            pip_install=True,
            charting_libraries=["streamlit"],
            base_dir=scratch_dir,
        )
        team.append(_python_assistant)
        extra_instructions.append(
            "To write and run python code, delegate the task to the `Python Assistant`."
        )
    if research_assistant:
        _research_assistant = Assistant(
            llm=llm,
            name="Research Assistant",
            role="Write a research report on a given topic",
            description="You are a Senior New York Times researcher tasked with writing a cover story research report.",
            instructions=[
                "For a given topic, use the `search_exa` to get the top 10 search results.",
                "Carefully read the results and generate a final - NYT cover story worthy report in the <report_format> provided below.",
                "Make your report engaging, informative, and well-structured.",
                "Remember: you are writing for the New York Times, so the quality of the report is important.",
            ],
            expected_output=dedent(
                """\
            An engaging, informative, and well-structured report in the following format:

            ## Title

            - **Overview** Brief introduction of the topic.
            - **Importance** Why is this topic significant now?

            ### Section 1
            - **Detail 1**
            - **Detail 2**

            ### Section 2
            - **Detail 1**
            - **Detail 2**

            ## Conclusion
            - **Summary of report:** Recap of the key findings from the report.
            - **Implications:** What these findings mean for the future.

            ## References
            - [Reference 1](Link to Source)
            - [Reference 2](Link to Source)
            """
            ),
            tools=[ExaTools(num_results=5, text_length_limit=1000)],
            # This setting tells the LLM to format messages in markdown
            markdown=True,
            add_datetime_to_instructions=True,
            debug_mode=debug_mode,
        )
        team.append(_research_assistant)
        extra_instructions.append(
            "To write a research report, delegate the task to the `Research Assistant`. "
            "Return the report in the <report_format> to the user as is, without any additional text like 'here is the report'."
        )

    return Assistant(
        llm=llm,
        name="Personalized Assistant",
        run_id=run_id,
        user_id=user_id,
        # Add personalization to the assistant by creating memories
        create_memories=True,
        # Update memory after each run
        update_memory_after_run=True,
        # Store the memories in a database
        memory=assistant_memory,
        # Store runs in a database
        storage=assistant_storage,
        # Store knowledge in a vector database
        knowledge_base=assistant_knowledge,
        description=dedent(
            """\
        You are the most advanced AI system in the world called `Optimus v7`.
        You have access to a set of tools and a team of AI Assistants at your disposal.
        Your goal is to assist the user in the best way possible.\
        """
        ),
        instructions=[
            "When the user sends a message, first **think** and determine if:\n"
            " - You can answer by using a tool available to you\n"
            " - You need to search the knowledge base\n"
            " - You need to search the internet\n"
            " - You need to delegate the task to a team member\n"
            " - You need to ask a clarifying question",
            "If the user asks about a topic, first ALWAYS search your knowledge base using the `search_knowledge_base` tool.",
            "If you dont find relevant information in your knowledge base, use the `duckduckgo_search` tool to search the internet.",
            "If the user asks to summarize the conversation, use the `get_chat_history` tool with None as the argument.",
            "If the users message is unclear, ask clarifying questions to get more information.",
            "Carefully read the information you have gathered and provide a clear and concise answer to the user.",
            "Do not use phrases like 'based on my knowledge' or 'depending on the information'.",
            "You can delegate tasks to an AI Assistant in your team depending of their role and the tools available to them.",
        ],
        # Add extra instructions for using tools
        extra_instructions=extra_instructions,
        # Add tools to the Assistant
        tools=tools,
        # Add team members to the Assistant
        team=team,
        # Show tool calls in the chat
        show_tool_calls=True,
        # This setting adds a tool to search the knowledge base for information
        search_knowledge=True,
        # This setting adds a tool to get chat history
        read_chat_history=True,
        # This setting tells the LLM to format messages in markdown
        markdown=True,
        # This setting adds chat history to the messages
        add_chat_history_to_messages=True,
        # This setting adds 6 previous messages from chat history to the messages sent to the LLM
        num_history_messages=6,
        # This setting adds the current datetime to the instructions
        add_datetime_to_instructions=True,
        # Add an introductory Assistant message
        introduction=dedent(
            """\
        Hi, I'm your personalized Assistant called `Optimus v7`.
        I can remember details about your preferences and solve problems using tools and other AI Assistants.
        Lets get started!\
        """
        ),
        debug_mode=debug_mode,
    )
