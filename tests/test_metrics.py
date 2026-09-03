import sys
from pathlib import Path

# Ensure project root is on sys.path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.evaluation.metrics import exact_match, f1_score, abstention_accuracy


class TestExactMatch:
    def test_exact_match_identical(self):
        assert exact_match("Hello World", "hello world") == 1

    def test_exact_match_different(self):
        assert exact_match("Hello World", "Goodbye World") == 0

    def test_exact_match_whitespace(self):
        assert exact_match("  Hello  ", "hello") == 1

    def test_exact_match_empty(self):
        assert exact_match("", "") == 1


class TestF1Score:
    def test_f1_perfect_match(self):
        assert f1_score("hello world", "hello world") == 1.0

    def test_f1_no_common_tokens(self):
        assert f1_score("hello", "goodbye") == 0.0

    def test_f1_partial_match(self):
        # 1 token común de 3 en pred y 1 de 3 en ref => F1 = 1/3
        assert f1_score("hello world foo", "hello goodbye bar") == 1 / 3


class TestAbstentionAccuracy:
    def test_abstention_accuracy_all_correct(self):
        predictions = {"answer": "No se menciona en el documento."}
        references = {"answer": "No se menciona en el documento."}
        result = abstention_accuracy([predictions], [references])
        assert result == 1.0

    def test_abstention_accuracy_all_wrong(self):
        predictions = {"answer": "Some answer"}
        references = {"answer": "No se menciona en el documento."}
        result = abstention_accuracy([predictions], [references])
        assert result == 0.0

    def test_abstention_accuracy_mixed(self):
        predictions = [
            {"answer": "No se menciona en el documento."},
            {"answer": "Some answer"},
        ]
        references = [
            {"answer": "No se menciona en el documento."},
            {"answer": "Other answer"},
        ]
        result = abstention_accuracy(predictions, references)
        assert result == 0.5

    def test_abstention_accuracy_english_token(self):
        predictions = {"answer": "it is not mentioned in the document."}
        references = {"answer": "it is not mentioned in the document."}
        result = abstention_accuracy([predictions], [references], abstention_token="it is not mentioned in the document.")
        assert result == 1.0