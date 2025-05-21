from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.ollama import OllamaChatCompletionClient
# Define a tool
async def get_weather(city: str) -> str:
    return f"The weather in {city} is 73 degrees and Sunny."

async def greet_user(user: str) -> str:
    return f"Hello {user}! a warm welcome from your personal Weather agent\
        give me a city name to get the current weather."

async def main() -> None:
    # Define an agent
    weather_agent = AssistantAgent(
        name="weather_agent",
        model_client=OllamaChatCompletionClient(
            model="qwen3:0.6b",
            model_info={
                "vision": False,
                "function_calling": True,
                "json_output": True,
            },
            base_url="http://localhost:11434/v1"  # Ollama API endpoint
        ),
        tools=[get_weather, greet_user],
    )

    # Define a team with a single agent and maximum auto-gen turns of 1.
    agent_team = RoundRobinGroupChat([weather_agent], max_turns=1)

    while True:
        # Get user input from the console.
        user_input = input("Enter a message (type 'exit' to leave): ")
        if user_input.strip().lower() == "exit":
            break
        # Run the team and stream messages to the console.
        stream = agent_team.run_stream(task=user_input)
        await Console(stream)

# Run the script
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())