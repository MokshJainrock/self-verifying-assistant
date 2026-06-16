# Self-Verifying Knowledge Assistant

**Author:** [Moksh Jain](https://github.com/MokshJainrock) ([@MokshJainrock](https://github.com/MokshJainrock))
live demo: https://self-verifying-assistant.streamlit.app

A question-answering app that only answers from a fixed set of documents, has a second
AI fact-check every answer, and just says "I don't know" when the answer isn't there
instead of making something up.

## Why I built it

LLM chatbots have one habit that makes them hard to trust for anything serious: when
they don't actually know something, they'll often hand you a confident, wrong answer
anyway. In a hospital, a bank, or a support queue, a confident wrong answer is worse
than no answer.

I wanted to build the opposite. This one only answers from documents it was given, it
shows you which document each sentence came from, it runs every claim past a separate
fact-checker, and it refuses (or flags the case for a human) when it can't back
something up.

## How it works

Every question goes through four stages:

1. **Retriever** searches the documents and pulls the most relevant passages.
2. **Responder** (OpenAI gpt-4o-mini) writes an answer using only those passages, with
   citations. If the answer isn't in them, it declines.
3. **Verifier** (Anthropic Claude) re-reads the passages on its own and checks the
   answer claim by claim. It doesn't trust the Responder's citations; it checks for
   itself.
4. **Confidence Gate** weighs all of that and makes the final call: ANSWER, REFUSE, or
   ESCALATE to a human reviewer.

The part I care about most is that the Responder and the Verifier are different models
from different companies. OpenAI writes the answer, Claude checks it. They don't share
training data, so they don't share blind spots. A model checking its own work mostly
just agrees with itself; two independent models actually catch each other's mistakes.
That's the whole reason the fact-checking step means anything.

## The agentic layer

This was the last thing I added. When the system refuses, it's sometimes not because
the answer doesn't exist, but because the question was worded badly and retrieval
grabbed the wrong passages. So I built a small controller that, on a refusal, rewrites
the query, searches again, and tries once or twice more before giving up. You turn it on
with the "Agentic mode" toggle in the sidebar.

The honest part: my first version of this made things worse. It retried too eagerly and
started talking itself into answering questions it should have refused, so hallucination
went up and correct-refusal dropped. I ran a proper A/B test (single pass vs the loop),
saw the regression in the numbers, and went through the failing cases one by one. The
damage all came from retrying "partially grounded" answers on trick questions. So I
changed the rule to only retry flat refusals, never partial ones. After that the loop
started recovering answerable questions it used to miss, with zero new hallucinations
and no drop in refusal. It's off by default; set AGENT_ENABLED to turn it on.

## What's in the knowledge base

The documents are Wikipedia paragraphs from SQuAD 2.0, a public dataset. By default it
indexes 5,000 paragraphs across roughly 95 topics: the Alps, Beyoncé, the iPod, solar
energy, To Kill a Mockingbird, Buddhism, New York City, and so on.

So it answers questions about those topics and refuses everything else. Ask it "who's
the president?" and it refuses, because that isn't in its documents. That's the system
working correctly, not a bug. You can point it at your own documents instead, or raise
MAX_DOCS to cover more.

## Results

I tested it on 50 answerable questions and 50 deliberately unanswerable ones (SQuAD's
"impossible" questions, written to bait a model into answering). Two numbers matter
here, not one: can it answer correctly, and does it correctly refuse what it can't
answer.

| What it measures | Result |
|---|---|
| Correct-refusal rate (declines unanswerable questions) | 88% |
| Hallucination rate (answers them anyway, lower is better) | 12% |
| Accuracy when it does answer | ~92% |
| Cost per question | a fraction of a cent |

Getting there took a few rounds. My first run was 82% refusal and 18% hallucination.
When I looked at what was failing, almost all of it was "false premise" questions:
ones that twist a real fact (wrong year, wrong number, wrong name) so they can't
actually be answered, and the system was answering the nearest real fact instead of
refusing. I added one rule telling the Responder to check the question's specifics
against the sources first, and that alone took hallucination from 18% to 12% and
refusal from 82% to 88% with no loss in accuracy. Most of this project was that loop:
measure, look at what broke, fix one thing, measure again.

## Running it yourself

You'll need Python 3, an OpenAI key (for the Responder), and an Anthropic key (for the
Verifier). A few dollars on each is plenty.

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Put your keys in `.env`:

```
LLM_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
VERIFIER_MODEL=claude-sonnet-4-6
```

Build the search index once. This runs locally, costs nothing, and takes a few minutes:

```
python -m scripts.build_index
```

Run the app:

```
streamlit run app.py
```

The dashboard has four tabs. **Ask** is where you type a question and see the decision,
the claim-by-claim check, and the source passages (the sidebar has access control and
the agentic toggle). **Evaluation** shows the test scores and every scored question.
**Query log** lists every question served with its cost and speed. **Knowledge base**
shows what's indexed.

One gotcha: if you edit anything in `src/` while the app is running, stop it fully with
Ctrl+C and start it again. A browser refresh isn't enough, since Streamlit keeps the old
modules loaded.

## Testing it

```
python -m scripts.evaluate --n 50      # accuracy and refusal on the full pipeline
python -m scripts.ab_eval --n 50       # single pass vs the agentic loop, head to head
```

## Things I built in for scale

- **Access control.** Retrieval can be limited to the documents a given user is allowed
  to see, enforced at search time so a document they shouldn't see never reaches the
  model in the first place.
- **Selective verification.** The Verifier call is the expensive part, so it's skipped
  when retrieval is very strong and the answer is almost certainly fine. That keeps cost
  down at volume. Set VERIFY_ALWAYS to force it on everywhere for high-stakes use.
- **Logging.** Every query is written to `data/query_log.jsonl` with its decision, cost,
  and latency, so there's an audit trail.

## Project layout

```
src/
  config.py        every setting in one place
  llm.py           Responder client (OpenAI)
  ingest.py        load and chunk the documents
  embed_store.py   embeddings and the vector store
  retriever.py     the Retriever
  responder.py     the Responder
  verifier.py      the Verifier (Claude)
  gate.py          the Confidence Gate
  reformulator.py  query rewriting for the agentic loop
  agent.py         the agentic controller
  pipeline.py      ties the four stages together
  metrics.py       scoring for the evaluation
scripts/
  build_index.py   build the search index (run once)
  evaluate.py      accuracy and refusal test
  ab_eval.py       single pass vs agentic comparison
app.py             the dashboard
```

## Settings worth knowing (src/config.py)

- `MAX_DOCS` — how many documents to index
- `TOP_K` — passages retrieved per question
- `VERIFIER_MODEL` — which Claude model checks the answers
- `AGENT_ENABLED` — turn the agentic retry loop on or off
- `RETRIEVAL_MAX_DISTANCE`, `GROUNDING_FULL` — the gate's thresholds
