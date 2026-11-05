"""Лабораторна робота №6: HMM POS-тегер, Вітербі та NLTK.

Використовує Penn Treebank WSJ: sections 02-21 (train), 24 (test).
Інсталяція: pip install -r requirements.txt
Запуск: python lab6.py --data-dir data --output-dir results
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import time
import urllib.request
from pathlib import Path

import numpy as np

START = "<START>"
END = "<END>"
UNKNOWN = ("--unk--", "--unk_digit--", "--unk_punct--", "--unk_upper--",
           "--unk_noun--", "--unk_verb--", "--unk_adj--", "--unk_adv--")
DATA_URL_BASE = (
    "https://raw.githubusercontent.com/anishLearnsToCode/"
    "nlp-probabilistic-models/master/week_2/assignment/"
)
DATA_FILES = ("WSJ_02-21.pos", "WSJ_24.pos")


def assign_unk(word: str) -> str:
    """Морфологічна класифікація невідомого англійського слова."""
    if any(c.isdigit() for c in word):
        return "--unk_digit--"
    if not any(c.isalnum() for c in word):
        return "--unk_punct--"
    if any(c.isupper() for c in word):
        return "--unk_upper--"
    lower = word.lower()
    if lower.endswith(("tion", "ment", "ness", "ity", "ship", "hood", "ism", "ist", "ance", "ence")):
        return "--unk_noun--"
    if lower.endswith(("ize", "ise", "ify", "ate", "ing", "ed")):
        return "--unk_verb--"
    if lower.endswith(("able", "ible", "less", "ful", "ous", "ive", "ic", "ish", "al")):
        return "--unk_adj--"
    if lower.endswith(("ly", "wise", "ward", "wards")):
        return "--unk_adv--"
    return "--unk--"


def read_tagged_corpus(path: str | Path) -> list[list[tuple[str, str]]]:
    """WSJ .pos: один токен + POS у рядку; порожній рядок відділяє речення."""
    sentences, current = [], []
    with Path(path).open(encoding="utf-8") as stream:
        for num, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                if current:
                    sentences.append(current)
                    current = []
                continue
            parts = line.rsplit(None, 1)
            if len(parts) != 2:
                raise ValueError(f"Неочікуваний рядок {num} у {path}: {line!r}")
            current.append((parts[0], parts[1]))
    if current:
        sentences.append(current)
    if not sentences:
        raise ValueError(f"Порожній корпус: {path}")
    return sentences


def download_data(data_dir: str | Path) -> tuple[Path, Path]:
    """Скачує два WSJ файли. Можна заздалегідь покласти свої .pos у data/."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    for name in DATA_FILES:
        target = data_dir / name
        if target.exists() and target.stat().st_size > 50:
            continue
        print(f"Завантаження {name} ...")
        url = DATA_URL_BASE + name
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (educational POS lab)"})
        try:
            with urllib.request.urlopen(req, timeout=90) as response:
                content = response.read()
            if not content or content[:20].lower().startswith(b"<!doctype html"):
                raise ValueError("Сервер повернув HTML замість корпусу")
            target.write_bytes(content)
        except Exception as exc:
            raise RuntimeError(
                f"Не вдалося завантажити {url}. Поклади файли {', '.join(DATA_FILES)} "
                f"до {data_dir} і запусти повторно. Причина: {exc}"
            ) from exc
    return tuple(data_dir / name for name in DATA_FILES)


class HMMPOSTagger:
    """Біграмна HMM з add-alpha, лексичними UNK-класами та лог-Вітербі."""

    def __init__(self, alpha: float = 0.1, min_freq: int = 2):
        if alpha <= 0 or min_freq < 1:
            raise ValueError("alpha > 0, min_freq >= 1")
        self.alpha = float(alpha)
        self.min_freq = int(min_freq)
        self.fitted = False

    def fit(self, tagged_sentences: list[list[tuple[str, str]]]):
        if not tagged_sentences:
            raise ValueError("Немає навчальних речень")
        word_freq = collections.Counter(w for sent in tagged_sentences for w, _ in sent)
        self.vocab = {w for w, count in word_freq.items() if count >= self.min_freq}
        self.words = sorted(self.vocab | set(UNKNOWN))
        self.word_to_id = {word: i for i, word in enumerate(self.words)}
        self.tags = sorted({tag for sent in tagged_sentences for _, tag in sent})
        self.tag_to_id = {tag: i for i, tag in enumerate(self.tags)}
        self.states = self.tags + [START, END]
        self.state_to_id = {tag: i for i, tag in enumerate(self.states)}
        self.transition_counts = collections.Counter()
        self.emission_counts = collections.Counter()
        self.tag_counts = collections.Counter()

        for sent in tagged_sentences:
            if not sent:
                continue
            prev = START
            for word, tag in sent:
                norm_word = self.normalize(word)
                self.transition_counts[(prev, tag)] += 1
                self.emission_counts[(tag, norm_word)] += 1
                self.tag_counts[tag] += 1
                prev = tag
            self.transition_counts[(prev, END)] += 1

        T, V = len(self.tags), len(self.words)
        self.A = np.zeros((T + 2, T + 2), dtype=np.float64)
        # Рядки: реальні POS-теги та START. Стовпці: реальні POS-теги та END.
        for prev in self.tags + [START]:
            row = self.state_to_id[prev]
            total = sum(self.transition_counts[(prev, next_tag)] for next_tag in self.tags + [END])
            # ADD-alpha, також згладжується перехід до END.
            denom = total + self.alpha * (T + 1)
            for next_tag in self.tags + [END]:
                self.A[row, self.state_to_id[next_tag]] = (
                    self.transition_counts[(prev, next_tag)] + self.alpha
                ) / denom
        self.A[self.state_to_id[END], self.state_to_id[END]] = 1.0

        self.B = np.empty((T, V), dtype=np.float64)
        for i, tag in enumerate(self.tags):
            denom = self.tag_counts[tag] + self.alpha * V
            self.B[i, :] = self.alpha / denom
        for (tag, word), count in self.emission_counts.items():
            self.B[self.tag_to_id[tag], self.word_to_id[word]] += (
                count / (self.tag_counts[tag] + self.alpha * V)
            )

        self.log_A = np.full_like(self.A, -np.inf)
        self.log_A[self.A > 0] = np.log(self.A[self.A > 0])
        self.log_B = np.log(self.B)
        self.fitted = True
        return self

    def normalize(self, word: str) -> str:
        return word if word in self.vocab else assign_unk(word)

    def viterbi(self, sentence: list[str]) -> list[str]:
        if not self.fitted:
            raise RuntimeError("Спочатку викличте fit()")
        if not sentence:
            return []
        indices = [self.word_to_id[self.normalize(word)] for word in sentence]
        T, N = len(self.tags), len(indices)
        back = np.zeros((N, T), dtype=np.int32)
        start_row = self.state_to_id[START]
        end_col = self.state_to_id[END]
        scores = self.log_A[start_row, :T] + self.log_B[:, indices[0]]
        for n in range(1, N):
            candidate_scores = scores[:, None] + self.log_A[:T, :T]
            back[n, :] = np.argmax(candidate_scores, axis=0)
            scores = candidate_scores[back[n, :], np.arange(T)] + self.log_B[:, indices[n]]
        last = int(np.argmax(scores + self.log_A[:T, end_col]))
        out = [last]
        for n in range(N - 1, 0, -1):
            last = int(back[n, last])
            out.append(last)
        return [self.tags[i] for i in reversed(out)]

    def tag(self, words: list[str]) -> list[tuple[str, str]]:
        return list(zip(words, self.viterbi(words)))


def evaluate_tagger(tag_method, sentences):
    """Однаковий тест і знаменник для різних тегерів."""
    correct = total = unseen_correct = unseen_total = 0
    examples = []
    for sent_id, sent in enumerate(sentences):
        words = [w for w, _ in sent]
        predicted = tag_method(words)
        if len(predicted) != len(sent):
            raise ValueError("Некоректна кількість тегів")
        for idx, ((word, gold), pred) in enumerate(zip(sent, predicted)):
            okay = gold == pred
            correct += okay
            total += 1
            if len(examples) < 50 and not okay:
                examples.append({"sentence": sent_id, "position": idx, "word": word,
                                 "gold": gold, "prediction": pred})
    return {"correct": int(correct), "total": int(total), "accuracy": correct / max(1, total),
            "errors_sample": examples}


def evaluate_hmm(hmm, sentences):
    result = evaluate_tagger(hmm.viterbi, sentences)
    oov_total = oov_correct = 0
    for sent in sentences:
        tags = hmm.viterbi([w for w, _ in sent])
        for (word, gold), pred in zip(sent, tags):
            if word not in hmm.vocab:
                oov_total += 1
                oov_correct += (gold == pred)
    result["oov_total"] = oov_total
    result["oov_accuracy"] = oov_correct / oov_total if oov_total else None
    return result


def evaluate_nltk_baseline(train, test):
    """NLTK BigramTagger -> UnigramTagger -> NN, навчений на тих самих train."""
    from nltk.tag import BigramTagger, UnigramTagger, DefaultTagger
    unigram = UnigramTagger(train, backoff=DefaultTagger("NN"))
    bigram = BigramTagger(train, backoff=unigram)
    return evaluate_tagger(lambda words: [tag for _, tag in bigram.tag(words)], test), bigram


def save_confusion_and_plots(hmm, test, out_dir, limit=15):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tag_freq = collections.Counter(tag for sent in test for _, tag in sent)
    tags = [tag for tag, _ in tag_freq.most_common(limit)]
    ix = {t: i for i, t in enumerate(tags)}
    confusion = np.zeros((len(tags), len(tags)), dtype=int)
    for sent in test:
        pred = hmm.viterbi([w for w, _ in sent])
        for (_, gold), p in zip(sent, pred):
            if gold in ix and p in ix:
                confusion[ix[gold], ix[p]] += 1
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(confusion, cmap="Blues")
    ax.set_xticks(range(len(tags)), tags, rotation=55, ha="right")
    ax.set_yticks(range(len(tags)), tags)
    ax.set_xlabel("Передбачений POS-тег")
    ax.set_ylabel("Правильний POS-тег")
    ax.set_title("HMM: матриця помилок (найпоширеніші теги)")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    return tags


def run_lab(data_dir="data", output_dir="results", max_train=None, max_test=None, alpha=0.1):
    """Повний експеримент. max_train/max_test тільки для швидкої демонстрації."""
    started = time.perf_counter()
    train_path, test_path = download_data(data_dir)
    train = read_tagged_corpus(train_path)
    test = read_tagged_corpus(test_path)
    if max_train is not None:
        train = train[:max_train]
    if max_test is not None:
        test = test[:max_test]
    if not train or not test:
        raise ValueError("Вибірка порожня")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Train: {len(train)} речень, test: {len(test)} речень")
    hmm = HMMPOSTagger(alpha=alpha, min_freq=2).fit(train)
    t0 = time.perf_counter()
    hmm_metrics = evaluate_hmm(hmm, test)
    hmm_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    baseline_metrics, baseline = evaluate_nltk_baseline(train, test)
    nltk_seconds = time.perf_counter() - t0
    examples = []
    for sentence in test[:5]:
        words = [w for w, _ in sentence]
        hp = hmm.viterbi(words)
        npred = [tag for _, tag in baseline.tag(words)]
        examples.append([{"word": word, "gold": gold, "hmm": h, "nltk": n}
                         for ((word, gold), h, n) in zip(sentence, hp, npred)])
    tags = save_confusion_and_plots(hmm, test, out_dir)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(["HMM + Вітербі", "NLTK Bigram"],
                  [100 * hmm_metrics["accuracy"], 100 * baseline_metrics["accuracy"]])
    ax.set_ylim(0, 105)
    ax.set_ylabel("Точність, %")
    ax.set_title("Порівняння на тому самому тестовому наборі WSJ 24")
    for rect in bars:
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 1,
                f"{rect.get_height():.2f}%", ha="center", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "accuracy_comparison.png", dpi=150)
    plt.close(fig)
    rows = []
    for sent_id, sent in enumerate(test):
        words = [w for w, _ in sent]
        hpred = hmm.viterbi(words)
        npred = [tag for _, tag in baseline.tag(words)]
        for position, ((word, gold), hp, np_) in enumerate(zip(sent, hpred, npred)):
            rows.append((sent_id, position, word, gold, hp, np_, int(hp == gold), int(np_ == gold)))
    with (out_dir / "predictions.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["sentence_id", "position", "word", "gold", "hmm", "nltk", "hmm_correct", "nltk_correct"])
        writer.writerows(rows)
    metrics = {
        "dataset": "WSJ_02-21.pos train / WSJ_24.pos test",
        "train_sentences": len(train), "test_sentences": len(test),
        "train_tokens": sum(map(len, train)), "test_tokens": sum(map(len, test)),
        "tags": len(hmm.tags), "vocabulary": len(hmm.vocab),
        "transition_matrix_shape": list(hmm.A.shape),
        "emission_matrix_shape": list(hmm.B.shape),
        "alpha": alpha, "min_frequency": 2,
        "hmm": {k:v for k,v in hmm_metrics.items() if k != "errors_sample"},
        "nltk_bigram": {k:v for k,v in baseline_metrics.items() if k != "errors_sample"},
        "delta_accuracy_pp": round(100 * (hmm_metrics["accuracy"] - baseline_metrics["accuracy"]), 4),
        "hmm_evaluation_seconds": round(hmm_seconds, 3),
        "nltk_train_evaluate_seconds": round(nltk_seconds, 3),
        "total_seconds": round(time.perf_counter() - started, 3),
        "comparison_note": "Обидва тегери навчені на WSJ_02-21.pos та оцінені на WSJ_24.pos; POS-теги однакові.",
        "limited": max_train is not None or max_test is not None,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "examples.json").write_text(json.dumps(examples, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "errors_hmm.json").write_text(json.dumps(hmm_metrics["errors_sample"], indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "errors_nltk.json").write_text(json.dumps(baseline_metrics["errors_sample"], indent=2, ensure_ascii=False), encoding="utf-8")
    report = f"""# Звіт: лабораторна робота №6

**Тема:** Частиномовний аналіз тексту з використанням прихованих марковських моделей.

## Мета
Реалізувати HMM-тегер, алгоритм Вітербі, порівняти з NLTK на тому самому наборі.

## Дані та метод
- Навчання: `WSJ_02-21.pos` ({len(train)} речень, {metrics['train_tokens']} токенів).
- Тест: `WSJ_24.pos` ({len(test)} речень, {metrics['test_tokens']} токенів).
- Розмір словника: {len(hmm.vocab)} відомих слів; спеціальні UNK-класи для рідкісних слів.
- Кількість POS-тегів: {len(hmm.tags)}.
- Матриця A: {hmm.A.shape}; матриця B: {hmm.B.shape}; add-alpha = {alpha}.
- HMM: Viterbi у логарифмах; речення декодуються окремо, START/END враховано.
- NLTK: BigramTagger з backoff до UnigramTagger і DefaultTagger(NN).

## Результати
| Показник | Результат |
|---|---:|
| HMM correct | {hmm_metrics['correct']} / {hmm_metrics['total']} |
| HMM accuracy | {100*hmm_metrics['accuracy']:.2f}% |
| NLTK correct | {baseline_metrics['correct']} / {baseline_metrics['total']} |
| NLTK accuracy | {100*baseline_metrics['accuracy']:.2f}% |
| Різниця HMM − NLTK | {metrics['delta_accuracy_pp']:+.2f} в. п. |
| HMM OOV accuracy | {('%.2f%%' % (100*hmm_metrics['oov_accuracy'])) if hmm_metrics['oov_accuracy'] is not None else 'N/A'} |

## Аналіз
Обидві системи оцінено на **однакових реченнях** із незмінними правильними POS-тегами.
Результати показують, як врахування контексту через переходи HMM співвідноситься з NLTK BigramTagger.
Головні обмеження: невідомі слова та спрощене припущення Маркова.

## Висновок
Реалізовано частотні словники, матриці переходів і емісій, декодування Вітербі та оцінку точності; виконано коректне порівняння з NLTK.

**Обмеження експерименту:** {'використано обмежену кількість речень (демонстрація), потрібно повторити повний тест.' if metrics['limited'] else 'використано всі доступні речення з обох обраних файлів.'}
"""
    (out_dir / "REPORT.md").write_text(report, encoding="utf-8")
    print(f"HMM: {100*hmm_metrics['accuracy']:.2f}% | NLTK: {100*baseline_metrics['accuracy']:.2f}%")
    print("Результати збережено у", out_dir)
    return metrics


def main():
    p = argparse.ArgumentParser(description="ЛР6 HMM POS tagging vs NLTK")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--output-dir", default="results")
    p.add_argument("--max-train", type=int, default=None, help="Тільки для швидкої перевірки")
    p.add_argument("--max-test", type=int, default=None, help="Тільки для швидкої перевірки")
    p.add_argument("--alpha", type=float, default=0.1)
    args = p.parse_args()
    run_lab(args.data_dir, args.output_dir, args.max_train, args.max_test, args.alpha)


if __name__ == "__main__":
    main()
