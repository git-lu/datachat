"""CLI entrypoint for the Iris Ollama chatbot."""

import argparse
import os
import sys

from app.iris_agent import AGENT_MODEL, FORMATTER_MODEL, ask_iris, ask_iris_friendly


def _answer(question: str, raw: bool, verbose: bool) -> None:
    if raw:
        result = ask_iris(question, verbose=verbose)
        print("\n--- draft ---")
        print(result["draft"])
        if result["tool_trace"]:
            print("\n--- tool trace ---")
            print("\n\n".join(result["tool_trace"]))
    else:
        print(ask_iris_friendly(question, verbose=verbose))


def _interactive(raw: bool, verbose: bool) -> None:
    host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    print(
        f"Iris chatbot (interactive)\n"
        f"  Ollama: {host}\n"
        f"  Agent: {AGENT_MODEL}  |  Formatter: {FORMATTER_MODEL}\n"
        f"  Commands: /help  /verbose  /raw  /quit\n",
        file=sys.stderr,
    )

    while True:
        try:
            question = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not question:
            continue
        lower = question.lower()
        if lower in {"/quit", "/exit", "/q", "quit", "exit"}:
            print("Bye.")
            break
        if lower == "/help":
            print(
                "Ask anything about the Iris dataset in plain English.\n"
                "  /verbose — toggle tool-call logging\n"
                "  /raw     — toggle agent-only output (no friendly formatter)\n"
                "  /quit    — exit"
            )
            continue
        if lower == "/verbose":
            verbose = not verbose
            print(f"verbose = {verbose}")
            continue
        if lower == "/raw":
            raw = not raw
            print(f"raw = {raw}")
            continue

        try:
            _answer(question, raw=raw, verbose=verbose)
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ask questions about the Iris dataset.",
        epilog="Tip: run with -i for a multi-question chat session (Ollama stays warm).",
    )
    parser.add_argument("question", nargs="?", help="Single question (non-interactive)")
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Chat loop: ask many questions without restarting",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Skip formatter; print agent draft + tool trace only",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Print tool calls")
    args = parser.parse_args()

    if args.interactive or args.question is None:
        _interactive(raw=args.raw, verbose=args.verbose)
        return

    host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    print(f"Ollama: {host}\nQuestion: {args.question}\n", file=sys.stderr)
    _answer(args.question, raw=args.raw, verbose=args.verbose)


if __name__ == "__main__":
    main()
