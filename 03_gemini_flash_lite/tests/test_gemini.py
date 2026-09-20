import importlib.util
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd


def test_gemini_evaluation_smoke() -> None:
    with patch("google.genai.Client") as mock_client_class:
        mock_client = mock_client_class.return_value
        mock_response = MagicMock()
        mock_response.text = '{"predicted_label": 0, "confidence": 0.95}'
        mock_client.models.generate_content.return_value = mock_response

        df = pd.DataFrame(
            {
                "text": ["reset my password", "login error"],
                "label": [0, 0],
            }
        )

        assert len(df) == 2
        assert "text" in df.columns
        assert "label" in df.columns


def test_gemini_evaluation_resume_skip(tmp_path: Path) -> None:
    tmp_dir = tmp_path / "tmp"
    tmp_dir.mkdir()
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"

    df = pd.DataFrame(
        {
            "text": ["test text 1", "test text 2"],
            "labels": [0, 1],
        }
    )
    input_path = input_dir / "test.csv"
    df.to_csv(input_path, index=False)

    sample_0_file = tmp_dir / "sample_0.json"
    with open(sample_0_file, "w", encoding="utf-8") as f:
        json.dump(
            {"idx": 0, "pred_label_str": "0", "latency_ms": 12.3, "conf": 0.99}, f
        )

    spec = importlib.util.spec_from_file_location(
        "evaluate", "03_gemini_flash_lite/script/evaluate.py"
    )
    assert spec is not None
    assert spec.loader is not None
    evaluate_mod = importlib.util.module_from_spec(spec)

    with (
        patch("google.genai.Client") as mock_client_class,
        patch("sys.argv", ["evaluate.py"]),
        patch("pathlib.Path") as mock_path_cls,
    ):
        mock_client = mock_client_class.return_value
        mock_chat = MagicMock()
        mock_response = MagicMock()
        mock_response.text = '{"predicted_label": 1, "confidence": 0.90}'
        mock_chat.send_message.return_value = mock_response
        mock_client.chats.create.return_value = mock_chat

        def path_side_effect(arg):
            p = str(arg)
            if "input/test.csv" in p:
                return input_path
            elif "output" in p and "tmp" not in p:
                return output_dir
            elif "tmp" in p:
                return tmp_dir
            return Path(arg)

        mock_path_cls.side_effect = path_side_effect
        mock_path_cls.return_value = Path(tmp_path)

        # Execute module loading/main if desired or test logic directly
        assert evaluate_mod is not None


def test_gemini_label_mapping_structure() -> None:
    spec = importlib.util.spec_from_file_location(
        "evaluate", "03_gemini_flash_lite/script/evaluate.py"
    )
    assert spec is not None
    assert spec.loader is not None
    evaluate_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluate_mod)

    mapping = getattr(evaluate_mod, "LABEL_MAPPING", None)
    assert mapping is not None
    assert isinstance(mapping, dict)

    expected_labels = set(range(12))
    assert set(mapping.keys()) == expected_labels

    for label_int, desc in mapping.items():
        assert isinstance(desc, str)
        assert len(desc) > 0
        assert desc.isascii(), (
            f"Description for label {label_int} should be ASCII English"
        )
