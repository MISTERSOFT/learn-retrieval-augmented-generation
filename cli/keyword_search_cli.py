#!/usr/bin/env python3

import argparse
from core.commands import build_command, idf_command, search_command, tf_command, tfidf_command


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Build inverted indexes")

    tf_parser = subparsers.add_parser("tf", help="Display term frequency for a specific movie")
    tf_parser.add_argument("id", type=int, help="Movie ID")
    tf_parser.add_argument("term", type=str, help="Term")

    idf_parser = subparsers.add_parser("idf", help="Calculate the Inverse Document Frequency (IDF) of a term")
    idf_parser.add_argument("term", type=str, help="Term")

    tfidf_parser = subparsers.add_parser("tfidf", help="Calculate the TF-IDF (Term Frequency-Inverse Document Frequency) of a term")
    tfidf_parser.add_argument("doc_id", type=int, help="Movie ID")
    tfidf_parser.add_argument("term", type=str, help="Term")

    args = parser.parse_args()

    match args.command:
        case "search":
            results = search_command(args.query)
            print(f"Searching for: {args.query}")
            for result in results:
                print(f"{result['id']}. {result['title']}")

        case "build":
            build_command()

        case "tf":
            frequency = tf_command(args.id, args.term)
            print(f'Term: {args.term}')
            print(f'Frequency: {frequency}')

        case "idf":
            idf = idf_command(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")

        case "tfidf":
            tf_idf = tfidf_command(args.doc_id, args.term)
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
