# Self-Verifying Knowledge Assistant

**Author:** [Moksh Jain](https://github.com/MokshJainrock) ([@MokshJainrock](https://github.com/MokshJainrock))

A question-answering system that only answers from a trusted set of documents,
checks its own answers with a second independent AI, and says "I don't know" when
it isn't sure instead of making something up.

## The problem it solves

Normal AI chatbots have a dangerous habit: when they don't know something, they
often make up a confident, wrong answer. For real business use (healthcare, finance,
support, legal), a confident wrong answer is worse than no answer at all.

This system is built so that does not happen. It only answers using the documents it
was given, it shows which document each statement came from, a separate model
fact-checks every claim, and if anything is unsupported it refuses or sends the case
to a human instead of guessing.

## How it works

A question passes through four steps:

1. Retriever - searches the document collection and pulls the most relevant passages.
2. Responder - writes an answer using only those passages, with citations. If the
   answer is not in the passages, it declines.
3. Verifier - a different AI model (Claude) independently re-reads the passages and
   checks, claim by claim, whether each statement is actually supported. It does not
   trust the Responder's own citations - it checks for itself.
4. Confidence Gate - looks at all the signals and makes one final decision:
   - ANSWER: grounded and confident, return it to the user.
   - REFUSE: the answer is not in the documents, decline.
   - ESCALATE: an answer was drafted but a claim could not be verified, send it to a
     human reviewer.

The most important design choice: the Responder and the Verifier are two different
AI models from two different companies (OpenAI writes the answer, Anthropic's Claude
checks it). Because they do not share the same training, they do not share the same
blind spots. A single model checking its own work just agrees with itself; two
independent models catch each other's mistakes. That is what makes the verification
meaningful.

## What is in the knowledge base

The documents come from SQuAD 2.0, a public dataset built from Wikipedia paragraphs.
By default the system indexes 5,000 paragraphs covering about 95 Wikipedia topics
(for example: the Alps, Beyonce, the iPod, solar energy, To Kill a Mockingbird,
Alexander Graham Bell, Buddhism, New York City).

This is why the system answers questions about those topics and refuses everything
else. If you ask "Who is the president?", it refuses, because that is not in its
documents. That refusal is the system working correctly, not a bug.

You can index more topics by raising MAX_DOCS (see Configuration below), or point it
at your own documents instead.

## Setup

You need Python 3, an OpenAI API key (for the Responder), and an Anthropic API key
(for the Verifier). Both keys cost a small amount per use; a few dollars on each is
plenty for this project.

```
cd self-verifying-assistant
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
```

Then open .env and fill in your two keys:

```
LLM_API_KEY=sk-...            your OpenAI key (Responder = gpt-4o-mini)
ANTHROPIC_API_KEY=sk-ant-...  your Anthropic key (Verifier = Claude)
VERIFIER_MODEL=claude-sonnet-4-6
```

## Build the index (run once)

This downloads the documents, turns them into searchable vectors, and saves them to
disk. It runs locally and is free (no API calls). It takes a few minutes.

```
python -m scripts.build_index
```

## Run the app

```
streamlit run app.py
```

This opens a dashboard in your browser with four tabs:

- Ask: type a question and see the decision, the answer, the claim-by-claim check,
  and the source passages. A sidebar lets you restrict which documents can be used
  (access control).
- Evaluation: the test results - accuracy, refusal rate, and every question scored.
- Query log: every question the system has answered, with its decision, speed, and
  estimated cost.
- Knowledge base: which topics are indexed and how much of each.

Note: if you change a file in src/ while the app is running, stop it fully with
Ctrl+C and start it again. A browser refresh is not enough.

## Measuring it (the evaluation)

The honest way to judge this kind of system is to test two things, not one:

- Can it answer correctly when the answer is available?
- Does it correctly refuse questions that have no answer in the documents?

The test set uses real SQuAD 2.0 questions: answerable ones (whose answers are in the
documents) and unanswerable ones (deliberately written so the answer is not there).

```
python -m scripts.evaluate --n 50
```

This runs the full system on 50 answerable plus 50 unanswerable questions, prints a
report, and saves data/eval_results.csv and data/eval_summary.json.

## Results

Measured on 50 answerable and 50 unanswerable questions:

| Metric                  | Meaning                                              | Result |
|-------------------------|------------------------------------------------------|--------|
| Correct-refusal rate    | unanswerable questions it correctly declined         | 88%    |
| Hallucination rate      | unanswerable questions it answered anyway (want low) | 12%    |
| Accuracy when answered  | of the answers it gave, how many were correct        | 92%    |
| Answer rate             | how often it chose to answer an answerable question  | 78%    |
| Cost                    | per question (selective verification keeps it low)   | ~$0.002|

How those numbers were reached: the first run scored 82% correct-refusal and 18%
hallucination. Looking at the failures showed they were almost all "false premise"
questions - questions that twist a real fact (wrong date, wrong number, wrong name)
so the question cannot actually be answered, and the system was answering the nearest
true fact instead of refusing. Adding one rule to the Responder (check that the
question's specifics match the sources before answering) dropped hallucination from
18% to 12% and raised correct-refusal from 82% to 88%, with no loss in accuracy. That
before-and-after is a clean, measured improvement.

## How it scales

The system is small to run but built like a production one:

- Larger corpus: MAX_DOCS controls how many documents are indexed (default 5,000).
- Access control: retrieval can be limited to the documents a given user is allowed
  to see, enforced at search time so forbidden documents never reach the model.
- Selective verification: the expensive Verifier call is skipped when retrieval is
  extremely strong, which is how cost stays low at high volume. Set VERIFY_ALWAYS=true
  to force verification on every answer for high-stakes use.
- Observability: every query is logged to data/query_log.jsonl with its decision,
  speed, and estimated cost - an audit trail.

## Project structure

```
self-verifying-assistant/
  requirements.txt
  .env.example          copy to .env and add your keys
  data/                 the saved index, logs, and eval output
  src/
    config.py           all settings in one place
    llm.py              the Responder's model client (OpenAI)
    ingest.py           load and split the documents
    embed_store.py      turn text into vectors, store and search them
    retriever.py        the Retriever
    responder.py        the Responder (writes cited answers, can decline)
    verifier.py         the Verifier (Claude, checks each claim)
    gate.py             the Confidence Gate (final decision)
    pipeline.py         runs the whole flow end to end
    metrics.py          scoring helpers for the evaluation
  scripts/
    build_index.py      builds the searchable index (run once)
    evaluate.py         runs the test and prints the results
  app.py                the dashboard
```

## Configuration

All settings live in src/config.py (some can be set in .env):

- MAX_DOCS: how many documents to index.
- TOP_K: how many passages to retrieve per question.
- RETRIEVAL_MAX_DISTANCE: how close a passage must be to count as relevant.
- GROUNDING_FULL: the share of claims that must be supported to answer automatically.
- VERIFY_ALWAYS / STRONG_RETRIEVAL_DISTANCE: controls selective verification.
- VERIFIER_MODEL: which Claude model checks the answers.
