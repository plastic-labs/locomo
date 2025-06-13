import regex
import json
import string
import unicodedata
from typing import List
import numpy as np
from collections import Counter
import os
from bert_score import score
from nltk.stem import PorterStemmer
ps = PorterStemmer()

LENGTH_THRESHOLD = 5

class SimpleTokenizer(object):
    ALPHA_NUM = r'[\p{L}\p{N}\p{M}]+'
    NON_WS = r'[^\p{Z}\p{C}]'

    def __init__(self):
        """
        Args:
            annotators: None or empty set (only tokenizes).
        """
        self._regexp = regex.compile(
            '(%s)|(%s)' % (self.ALPHA_NUM, self.NON_WS),
            flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE
        )

    def tokenize(self, text, uncased=False):
        matches = [m for m in self._regexp.finditer(text)]
        if uncased:
            tokens = [m.group().lower() for m in matches]
        else:
            tokens = [m.group() for m in matches]
        return tokens


def check_answer(example, tokenizer) -> List[bool]:
    """Search through all the top docs to see if they have any of the answers."""
    answers = example['answers']
    ctxs = example['ctxs']

    hits = []

    for _, doc in enumerate(ctxs):
        text = doc['text']

        if text is None:  # cannot find the document for some reason
            hits.append(False)
            continue

        hits.append(has_answer(answers, text, tokenizer))

    return hits


def has_answer(answers, text, tokenizer=SimpleTokenizer()) -> bool:
    """Check if a document contains an answer string."""
    text = _normalize(text)
    text = tokenizer.tokenize(text, uncased=True)

    for answer in answers:
        answer = _normalize(answer)
        answer = tokenizer.tokenize(answer, uncased=True)
        for i in range(0, len(text) - len(answer) + 1):
            if answer == text[i: i + len(answer)]:
                return True
    return False


def _normalize(text):
    return unicodedata.normalize('NFD', text)


def normalize_answer(s):

    s = s.replace(',', "")
    def remove_articles(text):
        # return regex.sub(r'\b(a|an|the)\b', ' ', text)
        return regex.sub(r'\b(a|an|the|and)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction, ground_truth):

    prediction = normalize_answer(prediction)
    ground_truth = normalize_answer(ground_truth)
    # print('# EM #', prediction, ' | ', ground_truth, ' #', set(prediction.split()) == set(ground_truth.split()))
    # return normalize_answer(prediction) == normalize_answer(ground_truth)
    return set(prediction.split()) == set(ground_truth.split())
    
# def bert_score(prediction, ground_truths):
#     prediction = normalize_answer(prediction)
#     values = []
#     for ground_truth in ground_truths:
#         ground_truth = normalize_answer(ground_truth)
#         P, R, F1 = score([prediction], [ground_truth], lang='en', verbose=False, rescale_with_baseline=True)
#         values.append(R[0].item())
#     print('# BERT # ', normalize_answer(prediction), ' | ', normalize_answer(ground_truth), ' #', P, R, F1)
#     return max(0, max(values))


def bert_score(prediction, ground_truth):
    prediction = normalize_answer(prediction)
    ground_truth = normalize_answer(ground_truth)
    P, R, F1 = score([prediction], [ground_truth], lang='en', verbose=False, rescale_with_baseline=True)
    # print('# BERT # ', normalize_answer(prediction), ' | ', normalize_answer(ground_truth), ' #', P, R, F1)
    return max(0, F1[0].item())


def ems(prediction, ground_truths):
    return max([exact_match_score(prediction, gt) for gt in ground_truths])


def f1_score(prediction, ground_truth):
    prediction_tokens = [ps.stem(w) for w in normalize_answer(prediction).split()]
    ground_truth_tokens = [ps.stem(w) for w in normalize_answer(ground_truth).split()]
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    # print('# F1 #', prediction, ' | ', ground_truth, ' #', precision, recall, f1)
    # return recall
    return f1


def f1(prediction, ground_truth):
    predictions = [p.strip() for p in prediction.split(',')]
    ground_truths = [g.strip() for g in ground_truth.split(',')]
    # print('# F1 [multi-answer]#', predictions, ' | ', ground_truths, ' #', np.mean([max([f1_score(prediction, gt) for prediction in predictions]) for gt in ground_truths]))
    return np.mean([max([f1_score(prediction, gt) for prediction in predictions]) for gt in ground_truths])


def rougel_score(prediction, ground_truth):
    from rouge import Rouge
    rouge = Rouge()
    prediction = ' '.join([ps.stem(w) for w in normalize_answer(prediction).split()])
    ground_truth = ' '.join([ps.stem(w) for w in normalize_answer(ground_truth).split()])
    # no normalization
    try:
        scores = rouge.get_scores(prediction, ground_truth, avg=True)
    except ValueError:  # "Hypothesis is empty."
        return 0.0
    return scores["rouge-1"]["f"]


def rl(prediction, ground_truths):
    return max([rougel_score(prediction, gt) for gt in ground_truths])


## file-level evaluation ... ### 
def eval_recall(infile):

    tokenizer = SimpleTokenizer()
    lines = open(infile, 'r').readlines()[1:]

    has_answer_count = 0
    answer_lengths = []
    for line in lines:
        line = json.loads(line)  # type: ignore[arg-type]
        answer = line['answer']
        output = ' || '.join(line['output'])

        if has_answer(answer, output, tokenizer):
            has_answer_count += 1

        answer_lengths.append(len(output.split()))

    recall = round(has_answer_count/len(lines), 4)
    lens = round(np.mean(answer_lengths), 4)

    return recall, lens


def rule_based_eval_question_answering(qas, eval_key='prediction', metric='f1'):

    all_ems = []
    all_recall = []
    exact_match_count = 0
    f1_count = 0
    answer_lengths = []
    
    skipped_questions = []
    for i, line in enumerate(qas):
        # Check if we have the required fields
        if 'answer' not in line:
            print(f"Warning: Question {i} missing 'answer' field. Skipping.")
            skipped_questions.append(i)
            continue
            
        if eval_key not in line:
            print(f"Warning: Question {i} missing prediction field '{eval_key}'. Skipping.")
            skipped_questions.append(i)
            continue
        
        # line = json.loads(line)
        if type(line[eval_key]) == list:
            answer = line['answer']
        else:
            answer = str(line['answer'])
        if line['category'] == 3:
            answer = answer.split(';')[0].strip()
        
        output = line[eval_key]
        
        # Handle None or empty predictions
        if output is None or output == '':
            print(f"Warning: Question {i} has empty prediction. Using default.")
            if line['category'] == 5:
                output = "No information available"
            else:
                output = "No answer found"
        
        output = str(output).strip()
        
        # single-hop, temporal, open-domain eval without splitting for sub-answers 
        if line['category'] in [2, 3, 4]:
            all_ems.append(f1_score(output, answer))
        
        # multi-hop eval by splitting entire phrase into sub-answers and computing partial F1 for each
        elif line['category'] in [1]:
            all_ems.append(f1(output, answer))

        # adversarial eval --> check for selection of correct option
        elif line['category'] in [5]:
            if 'no information available' in output.lower() or 'not mentioned' in output.lower():
                all_ems.append(1)
            else:
                all_ems.append(0)
        else:
            print(line)
            raise ValueError
        
        assert len(all_ems) == i+1-len(skipped_questions), f"Mismatch: {len(all_ems)} scores for {i+1-len(skipped_questions)} valid questions"

        if eval_key + '_context' in line and 'evidence' in line and len(line['evidence']) > 0:
            # recall_acc for dialog
            if line[eval_key + '_context'][0].startswith('S'):
                sessions = [e[1:] for e in line[eval_key + '_context']]
                recall_acc = float(sum([ev.split(':')[0][1:] in sessions for ev in line["evidence"]]))/len(line['evidence'])
            else:
                recall_acc = float(sum([ev in line[eval_key + '_context'] for ev in line["evidence"]]))/len(line['evidence'])
            all_recall.append(recall_acc)
        else:
            all_recall.append(1)

    valid_questions = len(qas) - len(skipped_questions)
    print("{} QA samples total, {} valid questions evaluated, {} accuracy values, {} skipped".format(
        len(qas), valid_questions, len(all_ems), len(skipped_questions)))
    
    if skipped_questions:
        print(f"Skipped questions: {skipped_questions}")
    
    lens = 0.0
    return all_ems, lens, all_recall


def eval_fact_checking(infile):

    tokenizer = SimpleTokenizer()
    lines = open(infile, 'r').readlines()[1:]

    exact_match_count = 0
    answer_lengths = []
    for line in lines:
        line = json.loads(line)  # type: ignore[arg-type]
        answer = line['answer']
        output = line['output'][0]

        if answer == ["refutes"]:
            answer = ["refutes", "no", "false"]
        if answer == ["supports"]:
            answer = ["supports", "yes", "true"]

        if has_answer(answer, output, tokenizer):
            exact_match_count += 1
        
        answer_lengths.append(len(output.split()))

    em = round(exact_match_count/len(lines), 4)
    lens = round(np.mean(answer_lengths), 4)

    return em, lens


def eval_dialogue_system(infile):

    lines = open(infile, 'r').readlines()[1:]

    f1_scores = []
    rl_scores = []
    answer_lengths = []
    for line in lines:
        line = json.loads(line)  # type: ignore[arg-type]
        answer = line['answer']
        output = line['output'][0]

        f1_scores.append(f1(output, answer))
        rl_scores.append(rl(output, answer))
        answer_lengths.append(len(output.split()))

    F1 = round(np.mean(f1_scores), 4)
    RL = round(np.mean(rl_scores), 4)
    lens = round(np.mean(answer_lengths), 4)

    return F1, RL, lens


# ==========================================================
#  LLM-as-Judge scoring
#
#  Uses an LLM (default GPT-4o) to grade the generated answer
#  CORRECT / WRONG based on the prompt from mem0 paper.
#  – Caches scores by writing them back onto the QA dict under
#    `{model_key}_f1`, so the evaluation script can skip them
#    next time it runs.
#  – Skips category 5 (adversarial yes/no) and applies the
#    simple heuristic as in the rule-based version.
# ==========================================================

import os, json, time, re

try:
    import openai
    from dotenv import load_dotenv
    load_dotenv(override=True)
    openai.api_key = os.environ["OPENAI_API_KEY"]
except ImportError:
    openai = None  # openai might not be available in some environments


def _call_openai_chat(prompt: str, model: str = "gpt-4o-mini", max_retries: int = 5, backoff_base: float = 1.5):
    """Call OpenAI ChatCompletion with retry/back-off."""
    if openai is None:
        raise RuntimeError("openai python package is not installed – required for LLM-as-judge scoring")

    wait = 1.0
    for attempt in range(max_retries):
        try:
            response = openai.ChatCompletion.create(  # type: ignore
                model=model,
                temperature=0.0,
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}],
            )
            return response["choices"][0]["message"]["content"].strip()  # type: ignore[index]
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(wait)
            wait *= backoff_base


_LLM_JUDGE_PROMPT_TEMPLATE = (
    "Your task is to label an answer to a question as \"CORRECT\" or \"WRONG\". "
    "You will be given the following data: (1) a question (posed by one user to another user), "
    "(2) a 'gold' (ground truth) answer, (3) a generated answer which you will score as CORRECT/WRONG.\n"
    "The point of the question is to ask about something one user should know about the other user based on their prior conversations. "
    "The gold answer will usually be a concise and short answer that includes the referenced topic.\n"
    "For time related questions, the gold answer will be a specific date, month, year, etc. "
    "The generated answer might be much longer or use relative time references (like 'last Tuesday' or 'next month'), "
    "but you should be generous with your grading – as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT.\n"
    "Now it's time for the real question:\n"
    "Question: {question}\n"
    "Gold answer: {gold_answer}\n"
    "Generated answer: {generated_answer}\n"
    "First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG. "
    "Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.\n"
    "Just return the label CORRECT or WRONG in a json format with the key as \"label\"."
)


def _extract_label_from_response(text) -> str:
    """Return upper-case 'CORRECT' or 'WRONG' string from LLM response."""
    text = str(text)
    try:
        j = json.loads(text)  # type: ignore[arg-type]
        label = j.get("label", "").strip().upper()
        if label in {"CORRECT", "WRONG"}:
            return label
    except json.JSONDecodeError:
        pass
    # Fallback: regex search
    match = re.search(r"CORRECT|WRONG", text, re.IGNORECASE)  # type: ignore[arg-type]
    if match:
        return match.group(0).upper()
    return "WRONG"  # safe default


def llm_as_judge_eval_question_answering(
    qas,
    eval_key: str = "prediction",
    model_key: str | None = None,
    openai_model: str = "gpt-4o-mini",
    override_cached: bool = False,
):
    """Evaluate QA using an LLM judge.

    Parameters
    ----------
    qas : list[dict]
        QA entries (modified in-place to cache scores).
    eval_key : str
        Key containing the generated answer.
    model_key : str | None
        Model identifier used for caching (e.g. 'honcho'). If provided, the
        cached score will be written/read from `{model_key}_f1`.
    openai_model : str
        Which OpenAI model to use.
    """

    # Ensure OpenAI key set
    if openai is not None and not hasattr(openai, "api_key"):
        if os.getenv("OPENAI_API_KEY"):
            openai.api_key = os.environ["OPENAI_API_KEY"]

    all_scores = []
    all_recall = []

    score_field = f"{model_key}_llm" if model_key else "llm_judge"

    print(f"[LLM-JUDGE] Starting evaluation with cache field '{score_field}' on {len(qas)} questions")

    for idx, qa in enumerate(qas):
        # Skip if missing prediction
        if eval_key not in qa:
            continue

        # Use cached value if present
        if score_field in qa and not override_cached:
            print(f"[LLM-JUDGE] Using cached score for QA {idx}")
            all_scores.append(qa[score_field])
            all_recall.append(1)
            continue
        elif score_field in qa and override_cached:
            print(f"[LLM-JUDGE] Overriding cached score for QA {idx}")

        category = qa.get("category", 0)
        generated = str(qa[eval_key]).strip()

        # Fallback for empty generations
        if not generated:
            generated = "No answer found"

        if category == 5:
            # Use simple heuristic from rule-based version
            score = 1 if ("no information available" in generated.lower() or "not mentioned" in generated.lower()) else 0
        else:
            prompt = _LLM_JUDGE_PROMPT_TEMPLATE.format(
                question=qa.get("question", ""),
                gold_answer=str(qa.get("answer", "")),
                generated_answer=generated,
            )

            try:
                print(f"[LLM-JUDGE] Calling OpenAI for QA {idx}...")
                response_text = _call_openai_chat(prompt, model=openai_model)
                label = _extract_label_from_response(response_text)
                score = 1 if label == "CORRECT" else 0
            except Exception as e:
                # Propagate error after exhausting retries
                raise RuntimeError(f"LLM judge failed on QA index {idx}: {e}") from e

        # Cache on QA dict
        qa[score_field] = score
        all_scores.append(score)
        all_recall.append(1)

    lens = 0.0
    return all_scores, lens, all_recall

