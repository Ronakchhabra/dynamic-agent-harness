"""
llm_client.py
─────────────
Universal async LLM caller via LiteLLM.
Swap the model string — no other code changes needed.

Supported providers (examples):
  Azure OpenAI  → "azure/gpt-4o-mini", "azure/gpt-4o"
  AWS Bedrock   → "bedrock/amazon.nova-lite-v1:0", "bedrock/amazon.nova-pro-v1:0"
  Vertex AI     → "vertex_ai/gemini-2.0-flash", "vertex_ai/gemini-2.5-pro"
  Ollama        → "ollama/qwen2.5:7b", "ollama/qwen2.5:32b"
  OpenAI        → "gpt-4o-mini", "gpt-4o"

Credentials — set via environment variables, LiteLLM picks them up automatically:
  Azure   : AZURE_API_KEY, AZURE_API_BASE, AZURE_API_VERSION
  Bedrock : AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION_NAME
  Vertex  : GOOGLE_APPLICATION_CREDENTIALS, VERTEXAI_PROJECT, VERTEXAI_LOCATION
  OpenAI  : OPENAI_API_KEY
"""

from litellm import acompletion


async def call_llm(
    system: str,
    user: str,
    model: str,
    max_tokens: int = 2048,
    temperature: float = 0.3,
) -> str:
    """
    Make a single async LLM call.

    Args:
        system:     System prompt — defines the agent's role and behaviour.
        user:       User message — the actual task or input.
        model:      LiteLLM model string (see provider examples above).
        max_tokens: Maximum tokens in the response.
        temperature: 0.0 = deterministic, 1.0 = creative. Keep low for structured tasks.

    Returns:
        The model's response as a plain string.
    """
    response = await acompletion(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content
