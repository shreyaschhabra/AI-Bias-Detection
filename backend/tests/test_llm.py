"""Tests for Phase 3 Data Storytelling (LLM Integration)."""

import os
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from main import app

client = TestClient(app)

class TestLLMEndpoint:
    @patch("llm_client.genai.Client")
    def test_generate_summary_endpoint(self, mock_client_class):
        # Setup mock behavior
        mock_response = MagicMock()
        mock_response.text = "Paragraph 1\n\nParagraph 2\n\nParagraph 3"
        
        mock_models = MagicMock()
        mock_models.generate_content.return_value = mock_response
        
        mock_client_instance = MagicMock()
        mock_client_instance.models = mock_models
        mock_client_class.return_value = mock_client_instance

        # Test request body
        payload = {
            "metrics": {
                "statistical_parity_difference": -0.5,
                "disparate_impact": 0.4,
                "equal_opportunity_difference": 0.1,
                "consistency_score": 0.9,
                "generalized_entropy_error": 0.2
            },
            "top_features": [
                {
                    "feature": "zip_code",
                    "importance": 0.8,
                    "corr_with_sensitive": 0.9
                }
            ]
        }

        # Inject fake env key to avoid early return
        with patch.dict(os.environ, {"GEMINI_API_KEY": "fake_key"}):
            response = client.post("/api/generate-summary", json=payload)
            
        assert response.status_code == 200
        assert "summary" in response.json()
        assert response.json()["summary"] == mock_response.text
        
        # Verify mock was called correctly
        mock_models.generate_content.assert_called_once()
        _, kwargs = mock_models.generate_content.call_args
        assert kwargs["model"] == "gemini-2.5-flash"
        assert "3-paragraph executive summary" in kwargs["contents"]
        assert "zip_code" in kwargs["contents"]

    def test_missing_api_key(self):
        payload = {
            "metrics": {
                "statistical_parity_difference": -0.5,
                "disparate_impact": 0.4,
                "equal_opportunity_difference": 0.1,
                "consistency_score": 0.9,
                "generalized_entropy_error": 0.2
            },
            "top_features": []
        }
        
        # Ensure GEMINI_API_KEY is not set
        with patch.dict(os.environ, clear=True):
            response = client.post("/api/generate-summary", json=payload)
            
        assert response.status_code == 200
        assert "Warning: GEMINI_API_KEY not found in environment" in response.json()["summary"]
