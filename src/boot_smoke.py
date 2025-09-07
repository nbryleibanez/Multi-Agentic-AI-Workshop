from pydantic_ai import Agent
from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    agent = Agent(
        "bedrock:anthropic.claude-3-haiku-20240307-v1:0", instructions="Be concise."
    )
    res = agent.run_sync("Say 'hello workshop' exactly.")
    print(res.output)


if __name__ == "__main__":
    main()

