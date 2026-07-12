import argparse
import json

import llm_client

EXTRACT_PROMPT = """You are given a block of text from an online post, usually written by a wife describing a household or marital conflict. Your job is to pull out things the HUSBAND said out loud, as quoted or closely reported by the author.

Keep a line only when it is the husband's own speech - what HE said in the situation. Skip lines spoken by the wife/author, the mother-in-law, father-in-law, sister-in-law, or anyone else. When a line is plausibly the husband's own words (often introduced by "he said", "my husband said/told me", "he shouted", "he replied"), keep it.

Also keep closely-reported husband speech without quotation marks - e.g. "he told me to adjust" yields "You should adjust." Rewrite each kept line as a clean standalone first-person statement in the husband's own voice.

Return every husband line you find; a post may contain several. Only return an empty list if the husband never speaks in the text.

Examples:

EXAMPLE 1 (keep husband, drop the author)
Text: I told him I was tired, and he said "you should manage the house better, my mother did everything alone."
Output: {{"quotes": [{{"text": "You should manage the house better, my mother did everything alone.", "speaker": "husband"}}]}}

EXAMPLE 2 (husband never speaks -> empty)
Text: My mother-in-law said "you should know how to cook by now." I stayed silent and cried later.
Output: {{"quotes": []}}
(The quote is the mother-in-law's, not the husband's.)

EXAMPLE 3 (multiple husband lines)
Text: He shouted "why is dinner late again" and later told me I don't respect his parents.
Output: {{"quotes": [{{"text": "Why is dinner late again?", "speaker": "husband"}}, {{"text": "You don't respect my parents.", "speaker": "husband"}}]}}

Now process this text.

Text: {text}

Respond with ONLY a JSON object, no preamble, following exactly this schema:
{{"quotes": [{{"text": "<husband's own quote as standalone first-person statement>", "speaker": "husband"}}]}}
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
        if speaker == "husband" and quote_text:
            kept.append(quote_text)
    return kept


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--documents-files", nargs="+", required=True)
    parser.add_argument("--provider", choices=["anthropic", "openai", "deepseek"], default="anthropic")
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--out", default="extracted_husband_quotes.json")
    args = parser.parse_args()

    client = llm_client.get_client(args.provider, base_url=args.base_url)

    source_documents = []
    for path in args.documents_files:
        with open(path, "r", encoding="utf-8") as f:
            source_documents.extend(json.load(f)["documents"])

    print(f"Scanning {len(source_documents)} documents for husband's quoted speech...")

    extracted = []
    for i, doc in enumerate(source_documents):
        quotes = extract_from_document(client, args.provider, args.model, doc["text"])
        for q in quotes:
            extracted.append({
                "text": q,
                "speaker": "husband",
                "subreddit": f"husband_quote:{doc['subreddit']}",
            })
        if (i + 1) % 50 == 0:
            print(f"  scanned {i + 1}/{len(source_documents)}, found {len(extracted)} quotes so far")

    print(f"\nExtracted {len(extracted)} husband quotes from {len(source_documents)} documents.")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"documents": extracted}, f, indent=2, ensure_ascii=False)

    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
