import sys

import pytest

from rag_dataprep_agent.cli import main


def test_cli_rejects_missing_input_path(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["rag-dataprep", "does-not-exist"])

    with pytest.raises(SystemExit) as error:
        main()

    assert error.value.code == 2
    assert "input path does not exist: does-not-exist" in capsys.readouterr().err
