import asyncio
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio

aws_knowledge = MCPServerStdio(
    command="uvx",
    args=[
        "mcp-proxy",
        "--transport",
        "streamablehttp",
        "https://knowledge-mcp.global.api.aws",
    ],
    tool_prefix="aws_knowledge",
    timeout=15,
)

agent = Agent(
    model="bedrock:anthropic.claude-3-haiku-20240307-v1:0",
    mcp_servers=[aws_knowledge],
    system_prompt=(
        "You are an AWS-savvy assistant. Prefer using the AWS Knowledge MCP tools "
        "to fetch answers from official AWS docs. Cite the doc title when possible."
    ),
)


async def main() -> None:
    async with agent:
        res = await agent.run(
            "What is Amazon S3 and what is it used for? Use the AWS Knowledge tool."
        )
        print(res.output)

        res2 = await agent.run(
            "What are the main types of Amazon EC2 instances (like general purpose "
            "and compute optimized)? Use the AWS Knowledge tool."
        )
        print(res2.output)


if __name__ == "__main__":
    asyncio.run(main())
