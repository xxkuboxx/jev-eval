import importlib.util
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


def test_jev_evaluation_smoke(tmp_path: Path) -> None:
    mock_choice_answer = MagicMock()
    mock_choice_answer.choice = "0"
    mock_choice_answer.confidence = 0.95
    mock_choice_answer.probabilities = {"0": 0.95, "1": 0.05}

    mock_response = MagicMock()
    mock_response.answers = {"intent": mock_choice_answer}

    mock_client_instance = MagicMock()
    mock_client_instance.system_one.return_value = mock_response
    mock_client_instance.__enter__.return_value = mock_client_instance

    with (
        patch("typesafe_sdk.TypeSafeClient", return_value=mock_client_instance),
        patch.dict("os.environ", {"TYPESAFE_API_KEY": "test-key"}),
    ):
        df = pd.DataFrame(
            {
                "text": ["hello there", "bye now"],
                "labels": [0, 1],
            }
        )
        assert len(df) == 2
        assert "text" in df.columns
        assert "labels" in df.columns


def test_jev_evaluation_missing_api_key(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location(
        "evaluate", "04_jev/script/evaluate.py"
    )
    assert spec is not None
    assert spec.loader is not None
    evaluate_mod = importlib.util.module_from_spec(spec)

    with (
        patch.dict("os.environ", {}, clear=True),
        patch("dotenv.load_dotenv"),
        patch("sys.argv", ["evaluate.py", "--smoke"]),
        pytest.raises(OSError, match="TYPESAFE_API_KEY"),
    ):
        spec.loader.exec_module(evaluate_mod)
        evaluate_mod.main()


def test_jev_evaluation_resume_skip(tmp_path: Path) -> None:
    tmp_dir = tmp_path / "tmp"
    tmp_dir.mkdir()
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"

    df = pd.DataFrame(
        {
            "text": ["cached sample", "uncached sample"],
            "labels": [0, 1],
        }
    )
    input_path = input_dir / "test.csv"
    df.to_csv(input_path, index=False)

    sample_0_file = tmp_dir / "sample_0.json"
    with open(sample_0_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "idx": 0,
                "pred_label_str": "0",
                "latency_ms": 15.2,
                "conf": 0.98,
                "probabilities": {"0": 0.98, "1": 0.02},
            },
            f,
        )

    spec = importlib.util.spec_from_file_location(
        "evaluate", "04_jev/script/evaluate.py"
    )
    assert spec is not None
    assert spec.loader is not None
    evaluate_mod = importlib.util.module_from_spec(spec)

    mock_choice_answer = MagicMock()
    mock_choice_answer.choice = "1"
    mock_choice_answer.confidence = 0.92
    mock_choice_answer.probabilities = {"0": 0.08, "1": 0.92}

    mock_response = MagicMock()
    mock_response.answers = {"intent": mock_choice_answer}

    mock_client_instance = MagicMock()
    mock_client_instance.system_one.return_value = mock_response
    mock_client_instance.__enter__.return_value = mock_client_instance

    with (
        patch("typesafe_sdk.TypeSafeClient", return_value=mock_client_instance),
        patch.dict("os.environ", {"TYPESAFE_API_KEY": "test-key"}),
        patch("sys.argv", ["evaluate.py"]),
        patch("pathlib.Path") as mock_path_cls,
    ):

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

        spec.loader.exec_module(evaluate_mod)
        evaluate_mod.main()

        assert mock_client_instance.system_one.call_count == 1


def test_jev_evaluation_retry_success(tmp_path: Path) -> None:
    tmp_dir = tmp_path / "tmp"
    tmp_dir.mkdir()
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"

    df = pd.DataFrame(
        {
            "text": ["retry test sample 0", "retry test sample 1"],
            "labels": [0, 1],
        }
    )
    input_path = input_dir / "test.csv"
    df.to_csv(input_path, index=False)

    spec = importlib.util.spec_from_file_location(
        "evaluate", "04_jev/script/evaluate.py"
    )
    assert spec is not None
    assert spec.loader is not None
    evaluate_mod = importlib.util.module_from_spec(spec)

    mock_choice_0 = MagicMock()
    mock_choice_0.choice = "0"
    mock_choice_0.confidence = 0.95
    mock_choice_0.probabilities = {"0": 0.9, "1": 0.1}
    mock_resp_0 = MagicMock()
    mock_resp_0.answers = {"intent": mock_choice_0}

    mock_choice_1 = MagicMock()
    mock_choice_1.choice = "1"
    mock_choice_1.confidence = 0.90
    mock_choice_1.probabilities = {"0": 0.1, "1": 0.9}
    mock_resp_1 = MagicMock()
    mock_resp_1.answers = {"intent": mock_choice_1}

    mock_client_instance = MagicMock()
    # sample 0 は最初の2回エラーで3回目に成功、sample 1 は1回で成功
    mock_client_instance.system_one.side_effect = [
        ConnectionResetError("Transient network failure"),
        RuntimeError("Rate limit reached"),
        mock_resp_0,
        mock_resp_1,
    ]
    mock_client_instance.__enter__.return_value = mock_client_instance

    with (
        patch("typesafe_sdk.TypeSafeClient", return_value=mock_client_instance),
        patch.dict("os.environ", {"TYPESAFE_API_KEY": "test-key"}),
        patch("sys.argv", ["evaluate.py", "--smoke"]),
        patch("time.sleep") as mock_sleep,
        patch("pathlib.Path") as mock_path_cls,
    ):

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

        spec.loader.exec_module(evaluate_mod)
        evaluate_mod.main()

        assert mock_client_instance.system_one.call_count == 4
        assert mock_sleep.call_count == 2


def test_jev_evaluation_retry_failure_fail_fast(tmp_path: Path) -> None:
    tmp_dir = tmp_path / "tmp"
    tmp_dir.mkdir()
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"

    df = pd.DataFrame(
        {
            "text": ["failing sample"],
            "labels": [0],
        }
    )
    input_path = input_dir / "test.csv"
    df.to_csv(input_path, index=False)

    spec = importlib.util.spec_from_file_location(
        "evaluate", "04_jev/script/evaluate.py"
    )
    assert spec is not None
    assert spec.loader is not None
    evaluate_mod = importlib.util.module_from_spec(spec)

    mock_client_instance = MagicMock()
    # 常にエラー
    mock_client_instance.system_one.side_effect = RuntimeError("Persistent 500 error")
    mock_client_instance.__enter__.return_value = mock_client_instance

    with (
        patch("typesafe_sdk.TypeSafeClient", return_value=mock_client_instance),
        patch.dict("os.environ", {"TYPESAFE_API_KEY": "test-key"}),
        patch("sys.argv", ["evaluate.py", "--smoke"]),
        patch("time.sleep"),
        patch("pathlib.Path") as mock_path_cls,
        pytest.raises(RuntimeError, match="Persistent 500 error"),
    ):

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

        spec.loader.exec_module(evaluate_mod)
        evaluate_mod.main()

    # 最大リトライ回数5回実行されたことを確認
    assert mock_client_instance.system_one.call_count == 5


def test_jev_choice_criteria_structure() -> None:
    spec = importlib.util.spec_from_file_location(
        "evaluate", "04_jev/script/evaluate.py"
    )
    assert spec is not None
    assert spec.loader is not None
    evaluate_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluate_mod)

    criteria = getattr(evaluate_mod, "CHOICE_CRITERIA", None)
    assert criteria is not None
    assert isinstance(criteria, dict)

    expected_labels = {str(i) for i in range(12)}
    assert set(criteria.keys()) == expected_labels

    for label_key, item in criteria.items():
        assert "what" in item, f"Missing what in criteria {label_key}"
        assert "not_for" in item, f"Missing not_for in criteria {label_key}"
        assert "examples" in item, f"Missing examples in criteria {label_key}"
        assert isinstance(item["examples"], list)
        assert len(item["examples"]) >= 2
        # Verify examples are in English
        for ex in item["examples"]:
            assert isinstance(ex, str)
            assert ex.isascii(), (
                f"Example {ex} in label {label_key} should be ASCII English"
            )
