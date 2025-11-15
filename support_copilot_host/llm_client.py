"""
LLM client wrapper for making calls to the language model.
"""
import json
import base64
from pathlib import Path
from typing import Optional
import openai
from .config import OPENAI_API_KEY, LLM_MODEL_NAME, LLM_TEMPERATURE, LLM_MAX_TOKENS

# Initialize OpenAI client
client = openai.OpenAI(api_key=OPENAI_API_KEY)


def _encode_image(image_path: str) -> str:
    """Encode image to base64 for vision API."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def chat_with_vision(
    system_prompt: str,
    messages: list[dict],
    image_path: Optional[str] = None,
    temperature: float = LLM_TEMPERATURE,
    max_tokens: int = LLM_MAX_TOKENS
) -> str:
    """
    Call LLM with optional vision support.

    Args:
        system_prompt: System prompt to set context
        messages: List of message dicts with 'role' and 'content'
        image_path: Optional path to image file for vision
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate

    Returns:
        Text response from the LLM
    """
    # Build messages array
    api_messages = [{"role": "system", "content": system_prompt}]

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        # If this is the user message and we have an image, attach it
        if role == "user" and image_path and Path(image_path).exists():
            # For vision, content needs to be an array
            try:
                encoded_image = _encode_image(image_path)
                api_messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": content},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{encoded_image}"
                            }
                        }
                    ]
                })
                # Clear image_path so we don't add it again
                image_path = None
            except Exception as e:
                # If image fails to load, just use text
                print(f"Warning: Could not load image {image_path}: {e}")
                api_messages.append({"role": role, "content": content})
        else:
            api_messages.append({"role": role, "content": content})

    # Make API call
    response = client.chat.completions.create(
        model=LLM_MODEL_NAME,
        messages=api_messages,
        temperature=temperature,
        max_tokens=max_tokens
    )

    return response.choices[0].message.content


def call_llm_for_json(
    system_prompt: str,
    messages: list[dict],
    image_path: Optional[str] = None,
    temperature: float = LLM_TEMPERATURE,
    max_tokens: int = LLM_MAX_TOKENS
) -> dict:
    """
    Call LLM and parse response as JSON.

    Args:
        system_prompt: System prompt to set context
        messages: List of message dicts
        image_path: Optional path to image file
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate

    Returns:
        Parsed JSON response as dict
    """
    # Add instruction to output JSON
    enhanced_system_prompt = f"""{system_prompt}

IMPORTANT: You must respond with valid JSON only. Do not include any text before or after the JSON object.
Do not use markdown code blocks. Just return the raw JSON."""

    # Get text response
    text_response = chat_with_vision(
        system_prompt=enhanced_system_prompt,
        messages=messages,
        image_path=image_path,
        temperature=temperature,
        max_tokens=max_tokens
    )

    # Try to parse JSON
    try:
        # Remove markdown code blocks if present
        cleaned = text_response.strip()
        if cleaned.startswith("```"):
            # Remove opening ```json or ```
            lines = cleaned.split("\n")
            lines = lines[1:-1]  # Remove first and last lines
            cleaned = "\n".join(lines)

        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        # Retry once with explicit instruction
        print(f"Warning: Failed to parse JSON on first attempt: {e}")
        print(f"Response was: {text_response[:200]}")

        # Try one more time with even more explicit instruction
        retry_messages = messages + [{
            "role": "assistant",
            "content": text_response
        }, {
            "role": "user",
            "content": "Please reformat your response as valid JSON only, with no markdown or extra text."
        }]

        retry_response = chat_with_vision(
            system_prompt=enhanced_system_prompt,
            messages=retry_messages,
            temperature=0.0,  # Lower temperature for retry
            max_tokens=max_tokens
        )

        # Try to parse again
        cleaned = retry_response.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = lines[1:-1]
            cleaned = "\n".join(lines)

        return json.loads(cleaned)
