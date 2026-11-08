# Customer Support Agent

A LangGraph workflow that classifies support tickets, drafts a reply with a billing, technical, or general handler, and reviews the draft. Low-scoring replies can be retried up to three times. Billing tickets pause for human approval.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirement.txt
```

Add your OpenAI API key to a `.env` file in the project directory:

```text
OPENAI_API_KEY=your-api-key
```

Run the sample tickets with:

```bash
python Agent.py
```

## Model

The workflow uses **`gpt-4.1-mini`** for classification, response drafting, and review.

## Tricky part

Review retries need conditional routing: a low score returns the ticket through classification to the right handler, up to three attempts. Human approval is only on the billing route; other tickets go to the final reply.
