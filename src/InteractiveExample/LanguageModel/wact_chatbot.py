from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

from src.InteractiveExample.AgentHistory.AgentHistoryController import AgentHistoryController
from src.Predictors.PredictorPipeline import PredictorPipeline
from src.Utils.Colors import Colors

class WACTChatBot:
    singleton = None

    def __init__(self, agent_history_controller: AgentHistoryController, model_name: str = "qwen3:1.7b", extract_reasoning: bool = False):
        WACTChatBot.singleton = self
        self.agent_history_controller = agent_history_controller

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are an AI assistant supervising an autonomous agent. Your role is to help the user understand what the agent is up to and what happens in the scene."
                                       "You may call tools to help with the request, but you are allowed to answer normally if no tool is available for the job"),
                ("placeholder", "{chat_history}"),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )

        model = ChatOllama(model=model_name, extract_reasoning=extract_reasoning)

        tools = [self.get_execution_history, self.figure_out_all_phantom_actions, self.figure_a_specific_phantom_actions]
        self.chat_history = []

        agent = create_tool_calling_agent(model, tools, prompt)
        self.agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

    @tool
    def get_execution_history() -> str:
        """
        Returns the entire execution history of the agent, ready to be displayed or summarized for the user.
        The execution history is a sequence of events containing: the event's index, the action name, the confidence on the action name, the target objective id, the confidence on the target objective id.
        """
        return WACTChatBot.singleton.agent_history_controller.get_history()

    @staticmethod
    @tool
    def figure_out_all_phantom_actions():
        """Updates the execution history of the agent in such a way to figure out currently Unknown phantom actions, then returns the updated execution history"""
        WACTChatBot.singleton.agent_history_controller.analyze_all_phantom_actions()

        message = "Every unknown phantom action was analyzed. "
        message += WACTChatBot.singleton.agent_history_controller.get_history()
        return message

    @staticmethod
    @tool
    def figure_a_specific_phantom_actions(action_index: int) -> str:
        """Updates the execution history of the agent in such a way to figure out the specified Unknown phantom actions, then returns the updated action event"""
        WACTChatBot.singleton.agent_history_controller.figure_out_phantom_action(action_index)

        message = "The prediction model was executed on the given action. Here is the resulting prediction on said action:\n"
        message += WACTChatBot.singleton.agent_history_controller.agent_action_history[action_index].get_short_description()
        return message

    @staticmethod
    def _pretty_print_model_output(model_output: str):
        """
        Prints the thinking and non-thinking parts of the output in different colors
        :param model_output: The output of the model
        """

        END_THINK_STR = "</think>"

        string_index_after_thinking = 0

        # gets the position of the end of thinking mark
        if model_output.__contains__(END_THINK_STR):
            string_index_after_thinking = model_output.index(END_THINK_STR) + len(END_THINK_STR)

        # Get thinking and non-thinking substrings
        thinking_portion = model_output[: string_index_after_thinking]
        final_answer_portion = model_output[string_index_after_thinking:]

        # Finally, print them
        Colors.print_colored(thinking_portion, Colors.OK_BLUE)
        Colors.print_colored(final_answer_portion, Colors.OK_CYAN)


    def chat_now(self):
        """
        Initiates a chat session with the language model.
        """

        # Let the user continuously ask questions
        print("Ask some questions to the chatbot, use CTRL+C or leave empty to exit")

        while True:

            # Ask the user for input
            try:
                user_input = input(f"\n{Colors.OK_GREEN}You: {Colors.END_COLOR}")
            except KeyboardInterrupt:
                # If CTRL+C, quit the loop
                break

            # If no input, quit the loop
            if user_input.lower() == '':
                print("Exiting conversation. Goodbye!")
                break

            self.answer_question(user_input)


    def answer_question(self, question: str):
        """
        Invokes the LLM and answers your question
        :param question: The input to the LLM
        """

        # Print a temporary message to indicate that the model is processing
        print(f"{Colors.WARNING} Thinking... {Colors.END_COLOR}", end='')

        try:
            # Invoke the agent with the user's input

            response = self.agent_executor.invoke({
                "input": question,
                "chat_history": self.chat_history
            })

            # Bring the caret to the beginning of the line
            print("\r", end='')

            output: str = response["output"]

            self._pretty_print_model_output(output)

            self.chat_history.append(HumanMessage(question))
            self.chat_history.append(AIMessage(output))
        except Exception as e:
            print(f"An error occurred: {e}")

    def append_human_message(self, message: str):
        self.chat_history.append(HumanMessage(message))

    def append_ai_message(self, message: str):
        self.chat_history.append(AIMessage(message))

if __name__ == "__main__":
    chatbot = WACTChatBot()
    chatbot.chat_now()