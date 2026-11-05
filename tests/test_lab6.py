import importlib.util
import math
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab6 import HMMPOSTagger, assign_unk, read_tagged_corpus, evaluate_tagger, evaluate_nltk_baseline

TRAIN = [
    [("The", "DT"), ("cat", "NN"), ("sleeps", "VBZ"), (".", ".")],
    [("The", "DT"), ("dog", "NN"), ("runs", "VBZ"), (".", ".")],
    [("A", "DT"), ("cat", "NN"), ("runs", "VBZ"), (".", ".")],
    [("They", "PRP"), ("run", "VBP"), (".", ".")],
    [("A", "DT"), ("dog", "NN"), ("sleeps", "VBZ"), (".", ".")],
]


def test_unknown_categories():
    assert assign_unk("1900") == "--unk_digit--"
    assert assign_unk("!") == "--unk_punct--"
    assert assign_unk("London") == "--unk_upper--"
    assert assign_unk("happiness") == "--unk_noun--"
    assert assign_unk("beautiful") == "--unk_adj--"
    assert assign_unk("slowly") == "--unk_adv--"
    assert assign_unk("tree") == "--unk--"


def test_corpus_reader(tmp_path):
    fn = tmp_path / "tiny.pos"
    fn.write_text("The\tDT\ncat\tNN\n\nRuns\tVBZ\n", encoding="utf-8")
    assert read_tagged_corpus(fn) == [[("The", "DT"), ("cat", "NN")], [("Runs", "VBZ")]]


def test_counts():
    model = HMMPOSTagger(alpha=0.1).fit(TRAIN)
    assert model.transition_counts[("<START>", "DT")] == 4
    assert model.transition_counts[(".", "<END>")] == len(TRAIN)
    assert model.tag_counts["NN"] == 4
    assert model.emission_counts[("NN", "cat")] == 2


def test_a_and_b_are_valid():
    model = HMMPOSTagger(alpha=0.1).fit(TRAIN)
    assert model.A.shape == (len(model.tags) + 2, len(model.tags) + 2)
    assert model.B.shape == (len(model.tags), len(model.words))
    assert np.all(model.A >= 0) and np.all(model.B > 0)
    np.testing.assert_allclose(model.A.sum(axis=1), 1.0, atol=1e-10)
    np.testing.assert_allclose(model.B.sum(axis=1), 1.0, atol=1e-10)
    assert model.A[model.state_to_id["<START>"], model.state_to_id["<START>"]] == 0


def test_viterbi_known():
    model = HMMPOSTagger(alpha=0.1).fit(TRAIN)
    assert model.viterbi(["The", "cat", "runs", "."]) == ["DT", "NN", "VBZ", "."]


def test_viterbi_unknown():
    model = HMMPOSTagger(alpha=0.1).fit(TRAIN)
    result = model.viterbi(["A", "giraffe", "sleeps", "."])
    assert len(result) == 4 and all(tag in model.tags for tag in result)


def test_viterbi_empty_and_single():
    model = HMMPOSTagger(alpha=0.1).fit(TRAIN)
    assert model.viterbi([]) == []
    assert len(model.viterbi(["cat"])) == 1


def test_evaluate_shared_test():
    model = HMMPOSTagger(alpha=0.1).fit(TRAIN)
    m = evaluate_tagger(model.viterbi, TRAIN)
    assert m["correct"] == m["total"]
    assert m["accuracy"] == 1


def test_nltk_same_sentences():
    metrics, _ = evaluate_nltk_baseline(TRAIN, TRAIN)
    assert metrics["total"] == sum(map(len, TRAIN))
    assert 0 <= metrics["accuracy"] <= 1


def test_unseen_rare_class():
    train = [[("Jamie", "NNP"), ("runs", "VBZ")], [("Mary", "NNP"), ("runs", "VBZ")]]
    model = HMMPOSTagger(alpha=0.1, min_freq=2).fit(train)
    assert "Jamie" not in model.vocab
    assert model.emission_counts[("NNP", "--unk_upper--")] == 2
