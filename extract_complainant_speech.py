import argparse
import json

import llm_client

EXTRACT_PROMPT = """You are given a block of text from an online post, usually written in the first person by someone describing a household or family conflict (a daughter-in-law, wife, or younger family member — the "complainant").

Your job is to pull out things the POST AUTHOR said out loud themselves — their own dialogue in the situation they describe. This is the author's first-person speech, typically introduced by "I said", "I told him/her", "I asked", "I replied", "so I told them", etc.

Keep by default. If a quoted or closely-reported line is the author's own speech, extract it. Only skip a line when it is clearly spoken by someone else (their mother-in-law, father-in-law, husband, wife, sister-in-law, or another person the author is reporting). When a line is plausibly the author's own words, keep it.

Also extract closely-reported author speech even without quotation marks — e.g. "I told her I was doing my best" yields "I was doing my best." Rewrite each extracted line as a clean standalone first-person statement in the author's voice.

Return every author line you find. A single post often contains several. Only return an empty list if the author genuinely never speaks in the text (pure narration, or the post only reports other people's words).

Examples:

EXAMPLE 1 (keep author, drop the other speaker)
Text: My mother-in-law said "you should know how to cook by now." I replied "I am doing my best, please give me time."
Output: {{"quotes": [{{"text": "I am doing my best, please give me time.", "speaker": "author"}}]}}

EXAMPLE 2 (multiple author lines, keep all)
Text: When they blamed me I said "what choice do I have" and later I told him won't your dad bear your expenses while you live there. Finally I just said fine, I'll manage.
Output: {{"quotes": [{{"text": "What choice do I have?", "speaker": "author"}}, {{"text": "Won't your dad bear your expenses while you live there?", "speaker": "author"}}, {{"text": "Fine, I'll manage.", "speaker": "author"}}]}}

EXAMPLE 3 (author never speaks -> empty)
Text: He kept insisting and finally shouted "I am a man, I won't cook." She just stayed silent the whole time.
Output: {{"quotes": []}}

Now process this text.

Text: {text}

Respond with ONLY a JSON object, no preamble, following exactly this schema:
{{"quotes": [{{"text": "<author's own quote as standalone first-person statement>", "speaker": "author"}}]}}
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
        if speaker == "author" and quote_text:
            kept.append(quote_text)
    return kept


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--documents-files", nargs="+", required=True)
    parser.add_argument("--provider", choices=["anthropic", "openai", "deepseek"], default="anthropic")
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--out", default="extracted_complainant_quotes.json")
    args = parser.parse_args()

    client = llm_client.get_client(args.provider, base_url=args.base_url)

    source_documents = []
    for path in args.documents_files:
        with open(path, "r", encoding="utf-8") as f:
            source_documents.extend(json.load(f)["documents"])

    print(f"Scanning {len(source_documents)} documents for author's own quoted speech...")

    extracted = []
    for i, doc in enumerate(source_documents):
        quotes = extract_from_document(client, args.provider, args.model, doc["text"])
        for q in quotes:
            extracted.append({
                "text": q,
                "speaker": "author",
                "subreddit": f"complainant_quote:{doc['subreddit']}",
            })
        if (i + 1) % 50 == 0:
            print(f"  scanned {i + 1}/{len(source_documents)}, found {len(extracted)} quotes so far")

    print(f"\nExtracted {len(extracted)} complainant quotes from {len(source_documents)} documents.")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"documents": extracted}, f, indent=2, ensure_ascii=False)

    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
