import argparse
import json

import llm_client

IN_LAW_ROLES = {"mother-in-law", "father-in-law", "sister-in-law"}

EXTRACT_PROMPT = """You are given a block of text from an online post or forum thread. It may or may not involve a household in-law (mother-in-law, father-in-law, or sister-in-law). Your job is to find DIRECT QUOTES that are spoken BY an in-law, and nothing else.

Rules:
1. A quote qualifies ONLY if the speaker is a mother-in-law, father-in-law, or sister-in-law. Quotes spoken by the narrator, spouse, husband, wife, parent, child, friend, or anyone else must be REJECTED, even if the post is about an in-law.
2. Do not treat the narrator's summary or paraphrase as a quote. Only extract when the in-law's own words are quoted or very closely reported.
3. For every quote you keep, you must identify who is speaking. If you cannot confidently tell that the speaker is an in-law, REJECT it.
4. Rewrite each kept quote as a clean standalone first-person statement, as if the in-law said it themselves.
5. If there are no qualifying in-law quotes, return an empty list.

Here are examples of correct behavior:

EXAMPLE 1
Text: My mother-in-law came into the kitchen and said "you should know how to cook by now, what will people think." I just stayed quiet.
Output: {{"quotes": [{{"text": "You should know how to cook by now, what will people think.", "speaker": "mother-in-law", "addressee": "daughter-in-law"}}]}}

EXAMPLE 2
Text: My husband got so angry he said "I wanted to relax tonight, not deal with this." Honestly his mom stays out of it mostly.
Output: {{"quotes": []}}
(Reason: the only quote is spoken by the husband, not an in-law.)

EXAMPLE 3
Text: FIL keeps saying "in our house men don't enter the kitchen" while my SIL muttered "let her handle it, that's her job." My own mother would never.
Output: {{"quotes": [{{"text": "In our house, men don't enter the kitchen.", "speaker": "father-in-law", "addressee": "family"}}, {{"text": "Let her handle it, that's her job.", "speaker": "sister-in-law", "addressee": "family"}}]}}

EXAMPLE 4
Text: The doctor told me "you have a mild deficiency, nothing serious." My mother-in-law overheard and said "see, this is what happens when you don't eat properly at home."
Output: {{"quotes": [{{"text": "See, this is what happens when you don't eat properly at home.", "speaker": "mother-in-law", "addressee": "daughter-in-law"}}]}}
(Reason: the doctor's quote is rejected; only the mother-in-law's quote is kept.)

Now process this text.

Text: {text}

Respond with ONLY a JSON object, no preamble, following exactly this schema:
{{"quotes": [{{"text": "<in-law quote as standalone first-person statement>", "speaker": "<mother-in-law|father-in-law|sister-in-law>", "addressee": "<who they are speaking to, or 'unknown'>"}}]}}
"""


def extract_from_document(client, provider, model, text, max_chars=1500):
    prompt = EXTRACT_PROMPT.format(text=text[:max_chars])
    result = llm_client.call_llm_json(client, provider, prompt, model=model, max_tokens=400)
    if result is None:
        return []
    kept = []
    for q in result.get("quotes", []):
        if not isinstance(q, dict):
            continue
        speaker = str(q.get("speaker", "")).strip().lower()
        quote_text = str(q.get("text", "")).strip()
        if speaker in IN_LAW_ROLES and quote_text:
            kept.append({
                "text": quote_text,
                "speaker": speaker,
                "addressee": str(q.get("addressee", "unknown")).strip() or "unknown",
            })
    return kept


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--documents-files", nargs="+", required=True)
    parser.add_argument("--provider", choices=["anthropic", "openai", "deepseek"], default="anthropic")
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--out", default="extracted_quotes.json")
    args = parser.parse_args()

    client = llm_client.get_client(args.provider, base_url=args.base_url)

    source_documents = []
    for path in args.documents_files:
        with open(path, "r", encoding="utf-8") as f:
            source_documents.extend(json.load(f)["documents"])

    print(f"Scanning {len(source_documents)} documents for embedded in-law quotes...")

    extracted = []
    for i, doc in enumerate(source_documents):
        quotes = extract_from_document(client, args.provider, args.model, doc["text"])
        for q in quotes:
            extracted.append({
                "text": q["text"],
                "speaker": q["speaker"],
                "addressee": q["addressee"],
                "subreddit": f"extracted_quote:{doc['subreddit']}",
            })
        if (i + 1) % 50 == 0:
            print(f"  scanned {i + 1}/{len(source_documents)}, found {len(extracted)} quotes so far")

    print(f"\nExtracted {len(extracted)} in-law quotes from {len(source_documents)} documents.")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"documents": extracted}, f, indent=2, ensure_ascii=False)

    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
