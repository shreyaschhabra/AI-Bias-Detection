import json
import os

from dotenv import load_dotenv
from google import genai

load_dotenv()


def generate_executive_summary(metrics: dict, top_features: list[dict]) -> str:
    """
    Calls the Google Gemini API to translate algorithmic fairness metrics
    into a non-technical executive summary.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return (
            "Warning: GEMINI_API_KEY not found in environment. "
            "Cannot generate executive summary."
        )

    # genai.Client uses GEMINI_API_KEY from os.environ natively if not passed, 
    # but we pass it explicitly to be safe
    client = genai.Client(api_key=api_key)

    prompt = f"""
Translate these ML fairness metrics into a 3-paragraph executive summary for a non-technical HR Manager.
Explain the legal risks and suggest the next steps.

Metrics:
{json.dumps(metrics, indent=2)}

Top Feature Proxies:
{json.dumps(top_features, indent=2)}
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                temperature=0.2,
            ),
        )
        return response.text
    except Exception as e:
        return f"Error communicating with LLM API: {e}"
