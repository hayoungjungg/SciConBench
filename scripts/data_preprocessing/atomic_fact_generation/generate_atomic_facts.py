"""
Atomic Fact Generator

Breaks complex text into independent, verifiable atomic facts using an LLM pipeline.

Pipeline stages (each stage is optional except decomposition and decontextualization):
    0. Sentence preprocessing  — split, fix, filter using NLTK and other standard NLP libraries
    1. Decomposition           — extract initial atomic facts per sentence using LLM
    2. Decontextualization     — resolve vague references using LLM
    3. Incomplete detection    — rewrite dependent/incomplete claims using LLM (optional)
    4. Irrelevant filtering    — drop off-topic facts using LLM (optional)
    5. Redundant filtering     — keep the most atomic, non-duplicate set using LLM (optional)

Parts of this code were adapted from FActScore (EMNLP 2023) and VeriFact (EMNLP 2025).
FActScore: https://github.com/shmsw25/FActScore/blob/main/factscore/atomic_facts.py
VeriFact:  https://aclanthology.org/2025.emnlp-main.905/
"""

import argparse
import json
import os
import re
import string
import sys
import warnings
from pathlib import Path

import nltk
import spacy
from dotenv import load_dotenv
from nltk.tokenize import sent_tokenize

warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*found in sys.modules.*")

# Make the package importable whether this file is run directly or as a module.
_pkg_parent = str(Path(__file__).resolve().parent.parent)
if _pkg_parent not in sys.path:
    sys.path.insert(0, _pkg_parent)

from atomic_fact_generation.config.model_config import (
    ModelConfig,
    UnifiedLLMClient,
    create_default_configs,
)
from atomic_fact_generation.utils.calculate_pricing import calculate_pricing, format_cost_report
from atomic_fact_generation.utils.helper_atomic_facts import (
    combine_bullet_point_lists,
    convert_metadata_for_json,
    detect_initials,
    fix_sentence_splitter,
    is_non_content_sentence,
    parse_decontextualization_response,
    parse_filter_irrelevant_facts_response,
    parse_filter_redundant_facts_batch_response,
    parse_incomplete_fact_response,
    print_verbose_comparisons,
    text_to_sentences,
)
from atomic_fact_generation.utils.prompts import (
    decomposition_persona_description,
    decomposition_prompt,
    decontextualization_persona_description,
    decontextualization_prompt,
    detect_incomplete_facts_prompt,
    filter_irrelevant_facts_prompt,
    filter_redundant_facts_batch_prompt,
)

load_dotenv()

nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)


class AtomicFactGenerator:
    """Extract atomic facts from text using a configurable LLM pipeline.

    Args:
        model_configs: Optional dict mapping component names to
            :class:`~atomic_fact_generation.config.model_config.ModelConfig` objects.
            Missing components fall back to defaults from ``config/model_config.yaml``.
            Component names: ``decomposition``, ``decontextualization``,
            ``incomplete_detection``, ``irrelevant_filtering``, ``redundant_filtering``.

    Example::

        from atomic_fact_generation import AtomicFactGenerator
        from atomic_fact_generation.config.model_config import ModelConfig

        generator = AtomicFactGenerator()
        facts, para_breaks, metadata = generator.run(
            generation="Some scientific text ...",
            question="What are the effects of X on Y?",
        )
    """

    def __init__(self, model_configs: dict = None):
        self.model_configs = create_default_configs()
        if model_configs:
            self.model_configs.update(model_configs)

        self.clients = {
            component: UnifiedLLMClient(config)
            for component, config in self.model_configs.items()
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_client(self, component: str) -> UnifiedLLMClient:
        return self.clients[component]

    def _get_model_config(self, component: str) -> ModelConfig:
        return self.model_configs[component]

    def _extract_token_usage(self, response, component: str) -> dict:
        return self._get_client(component).extract_token_usage(response)

    def _accumulate_token_usage(self, token_usage: dict, step_name: str, step_usage: dict):
        token_usage[step_name] = step_usage
        for key in ("prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens"):
            token_usage["total"][key] += step_usage.get(key, 0)

    def _initialize_token_usage(self) -> dict:
        empty = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cached_tokens": 0}
        return {
            "decomposition": dict(empty),
            "decontextualization": dict(empty),
            "incomplete_detection": dict(empty),
            "irrelevant_filtering": dict(empty),
            "redundant_filtering": dict(empty),
            "total": dict(empty),
        }

    def _preprocess_sentences(self, paragraphs: list):
        """Split paragraphs into content sentences, handling initials and bullet points."""
        sentences = []
        sentences_to_para = {}
        para_breaks = []

        for idx, para in enumerate(paragraphs):
            if idx > 0:
                para_breaks.append(len(sentences))
            initials = detect_initials(para)
            curr = fix_sentence_splitter(sent_tokenize(para), initials)
            sentences.extend(curr)
            for s in curr:
                sentences_to_para[s] = para

        sentences, sentences_to_para = combine_bullet_point_lists(sentences, sentences_to_para)

        content, content_to_para = [], {}
        for s in sentences:
            if is_non_content_sentence(s):
                print(f"Filtering out non-content sentence: '{s}'")
            else:
                content.append(s)
                content_to_para[s] = sentences_to_para[s]

        print(
            f"Filtered {len(sentences) - len(content)} non-content sentences "
            f"out of {len(sentences)} total"
        )
        return content, content_to_para, para_breaks

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        generation: str,
        question: str,
        verbose: bool = False,
        enable_incomplete_detection: bool = True,
        enable_irrelevant_filtering: bool = True,
        enable_redundant_filtering: bool = True,
    ):
        """Run the full atomic-fact extraction pipeline.

        Args:
            generation: Input text (one or more paragraphs) to decompose.
            question: The question the generation is answering. Used for
                irrelevant-fact filtering and decontextualization context.
            verbose: If True, print detailed per-step comparisons.
            enable_incomplete_detection: Detect and rewrite incomplete claims.
            enable_irrelevant_filtering: Drop facts irrelevant to *question*.
            enable_redundant_filtering: Remove redundant / duplicate facts.

        Returns:
            Tuple of ``(final_atomic_facts_pairs, paragraph_breaks, metadata)``:

            * ``final_atomic_facts_pairs``: ``list[(sentence, [fact, ...])]``
            * ``paragraph_breaks``: ``list[int]`` — indices of paragraph boundaries
            * ``metadata``: dict with intermediate results and token usage
        """
        assert isinstance(generation, str), "generation must be a string"
        assert isinstance(question, str) and question.strip(), "question must be a non-empty string"

        paragraphs = [p.strip() for p in generation.split("\n") if p.strip()]
        return self._process_atomic_facts(
            paragraphs, generation, question, verbose,
            enable_incomplete_detection, enable_irrelevant_filtering, enable_redundant_filtering,
        )

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------

    def _process_atomic_facts(
        self,
        paragraphs,
        original_generation,
        question,
        verbose=False,
        enable_incomplete_detection=True,
        enable_irrelevant_filtering=True,
        enable_redundant_filtering=True,
    ):
        # Stage 0 — preprocess
        content_sentences, sentences_to_para, para_breaks = self._preprocess_sentences(paragraphs)

        # Stage 1 — decompose
        atoms, decomp_usage = self.get_init_atomic_facts_from_sentence(
            content_sentences, sentences_to_para, question
        )
        initial_pairs = [(s, atoms[s]) for s in content_sentences]

        # Stage 2 — decontextualize
        print("Decontextualizing atomic facts...")
        dectx_pairs, dectx_usage = self.decontextualize_atomic_facts(
            initial_pairs, original_generation, question
        )

        current = dectx_pairs
        dep_meta, irr_meta, red_meta = {}, {}, {}
        rewritten_pairs = None

        # Stage 3 — incomplete detection (optional)
        if enable_incomplete_detection:
            print("Detecting incomplete facts...")
            rewritten_pairs, dep_meta, incomplete_usage = self.detect_incomplete_facts(
                current, original_generation
            )
            current = rewritten_pairs
        else:
            rewritten_pairs = current
            incomplete_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        # Stage 4 — irrelevant filtering (optional)
        if enable_irrelevant_filtering:
            print("Filtering irrelevant facts...")
            current, irr_meta, filter_usage = self.filter_irrelevant_facts(
                current, original_generation, question
            )
        else:
            filter_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        # Stage 5 — redundant filtering (optional)
        if enable_redundant_filtering:
            print("Filtering redundant facts...")
            final_pairs, red_meta, redundant_usage = self.filter_redundant_facts(
                current, original_generation
            )
        else:
            final_pairs = current
            redundant_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        # Accumulate token usage
        token_usage = self._initialize_token_usage()
        self._accumulate_token_usage(token_usage, "decomposition", decomp_usage)
        self._accumulate_token_usage(token_usage, "decontextualization", dectx_usage)
        self._accumulate_token_usage(token_usage, "incomplete_detection", incomplete_usage)
        self._accumulate_token_usage(token_usage, "irrelevant_filtering", filter_usage)
        self._accumulate_token_usage(token_usage, "redundant_filtering", redundant_usage)

        metadata = {
            "initial_atomic_facts_pairs": initial_pairs,
            "decontextualized_atomic_facts_pairs": dectx_pairs,
            "rewritten_missing_atomic_facts_pairs": rewritten_pairs,
            "dependent_facts_metadata": dep_meta,
            "irrelevant_facts_metadata": irr_meta,
            "redundant_facts_metadata": red_meta,
            "token_usage": token_usage,
        }

        if verbose:
            print_verbose_comparisons(
                initial_pairs, dectx_pairs, rewritten_pairs, final_pairs,
                dep_meta, irr_meta, red_meta, token_usage,
            )

        return final_pairs, para_breaks, metadata

    def get_init_atomic_facts_from_sentence(self, sentences, sentences_to_para, question):
        """Decompose a list of sentences into initial atomic facts.

        Returns:
            Tuple of ``(atoms, token_usage)`` where *atoms* maps each sentence
            to a list of atomic-fact strings.
        """
        prompts = []
        prompt_to_sent = {}
        atoms = {}
        token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        for sentence in sentences:
            para = sentences_to_para[sentence]
            content = decomposition_prompt.format(
                question=question, paragraph=para, sentence=sentence
            )
            prompt = [
                {"role": "system", "content": decomposition_persona_description},
                {"role": "user", "content": content},
            ]
            prompts.append(prompt)
            prompt_to_sent[tuple((m["role"], m["content"]) for m in prompt)] = sentence

        print(f"Number of prompts: {len(prompts)}")
        client = self._get_client("decomposition")

        for prompt in prompts:
            response = client.create_completion(messages=prompt)
            normalized = client.normalize_response(response)

            usage = self._extract_token_usage(response, "decomposition")
            for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                token_usage[k] += usage[k]

            key = tuple((m["role"], m["content"]) for m in prompt)
            atoms[prompt_to_sent[key]] = text_to_sentences(normalized)

        return atoms, token_usage

    def decontextualize_atomic_facts(self, atomic_facts_pairs, original_paragraph, question):
        """Replace vague references in each fact with specific entities from context.

        Returns:
            Tuple of ``(decontextualized_pairs, token_usage)``.
        """
        result_pairs = []
        total = sum(len(facts) for _, facts in atomic_facts_pairs)
        processed = 0
        token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        client = self._get_client("decontextualization")

        for sentence, facts in atomic_facts_pairs:
            dectx_facts = []
            for fact in facts:
                content = (
                    decontextualization_prompt
                    .replace("{question}", question)
                    .replace("[INDIVIDUAL_FACT]", fact)
                    .replace("[RESPONSE]", original_paragraph)
                )
                prompt = [
                    {"role": "system", "content": decontextualization_persona_description},
                    {"role": "user", "content": content},
                ]
                response = client.create_completion(messages=prompt)
                normalized = client.normalize_response(response)

                usage = self._extract_token_usage(response, "decontextualization")
                for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    token_usage[k] += usage[k]

                dectx_fact, _ = parse_decontextualization_response(normalized)
                if dectx_fact:
                    dectx_facts.append(dectx_fact)
                else:
                    print(f"Warning: decontextualization failed for '{fact[:50]}...', using original")
                    dectx_facts.append(fact)

                processed += 1
                if processed % 10 == 0:
                    print(f"Decontextualized {processed}/{total} atomic facts...")

            result_pairs.append((sentence, dectx_facts))

        print(f"Completed decontextualization of {processed} atomic facts")
        return result_pairs, token_usage

    def detect_incomplete_facts(self, atomic_facts_pairs, original_paragraph):
        """Classify facts as Independent/Dependent; rewrite dependent ones.

        Returns:
            Tuple of ``(rewritten_pairs, dependent_metadata, token_usage)``.
        """
        dep_meta = {}
        rewritten_pairs = []
        total = sum(len(facts) for _, facts in atomic_facts_pairs)
        processed = 0
        token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        client = self._get_client("incomplete_detection")

        for sentence, facts in atomic_facts_pairs:
            modified = []
            for fact in facts:
                content = (
                    detect_incomplete_facts_prompt
                    .replace("[CONTEXT]", original_paragraph)
                    .replace("[CLAIM]", fact)
                )
                response = client.create_completion(messages=[{"role": "user", "content": content}])
                normalized = client.normalize_response(response)

                usage = self._extract_token_usage(response, "incomplete_detection")
                for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    token_usage[k] += usage[k]

                result = parse_incomplete_fact_response(normalized)
                is_dependent = (
                    result["classification"]
                    and "independent" not in result["classification"].lower()
                )
                rewritten = result.get("rewritten_claim", fact)

                if is_dependent:
                    modified.append(rewritten)
                    dep_meta[(sentence, fact)] = [
                        result["dependent_type"],
                        result["explanation"],
                        result["classification"],
                        rewritten,
                    ]
                else:
                    modified.append(fact)

                processed += 1
                if processed % 10 == 0:
                    print(f"Processed {processed}/{total} facts for incomplete detection...")

            rewritten_pairs.append((sentence, modified))

        print(f"Completed incomplete fact detection for {processed} atomic facts")
        return rewritten_pairs, dep_meta, token_usage

    def filter_irrelevant_facts(self, atomic_facts_pairs, original_paragraph, question):
        """Remove facts not relevant to *question*.

        Returns:
            Tuple of ``(filtered_pairs, irrelevant_metadata, token_usage)``.
        """
        irr_meta = {}
        filtered_pairs = []
        total = sum(len(facts) for _, facts in atomic_facts_pairs)
        processed = 0
        token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        client = self._get_client("irrelevant_filtering")

        for sentence, facts in atomic_facts_pairs:
            kept = []
            for fact in facts:
                content = (
                    filter_irrelevant_facts_prompt
                    .replace("[PROMPT]", question)
                    .replace("[RESPONSE]", original_paragraph)
                    .replace("[INDIVIDUAL FACT]", fact)
                )
                response = client.create_completion(messages=[{"role": "user", "content": content}])
                normalized = client.normalize_response(response)

                usage = self._extract_token_usage(response, "irrelevant_filtering")
                for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    token_usage[k] += usage[k]

                result = parse_filter_irrelevant_facts_response(normalized)
                if result["classification"] and "[Foo]" in result["classification"]:
                    kept.append(fact)
                else:
                    irr_meta[(sentence, fact)] = [
                        question, fact, result["reasoning"], result["classification"]
                    ]

                processed += 1
                if processed % 10 == 0:
                    print(f"Processed {processed}/{total} facts for irrelevant filtering...")

            filtered_pairs.append((sentence, kept))

        print(f"Completed irrelevant fact filtering for {processed} atomic facts")
        return filtered_pairs, irr_meta, token_usage

    def filter_redundant_facts(self, atomic_facts_pairs, original_paragraph):
        """Select the most atomic, non-redundant set of facts per sentence.

        Returns:
            Tuple of ``(filtered_pairs, redundant_metadata, token_usage)``.
        """
        red_meta = {}
        filtered_pairs = []
        token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        client = self._get_client("redundant_filtering")

        for sentence, facts in atomic_facts_pairs:
            if not facts:
                filtered_pairs.append((sentence, []))
                continue
            if len(facts) == 1:
                filtered_pairs.append((sentence, facts))
                continue

            facts_list = "\n".join(f"{i+1}. {f}" for i, f in enumerate(facts))
            content = (
                filter_redundant_facts_batch_prompt
                .replace("[RESPONSE]", original_paragraph)
                .replace("[ALL_FACTS]", facts_list)
            )
            response = client.create_completion(messages=[{"role": "user", "content": content}])
            normalized = client.normalize_response(response)

            usage = self._extract_token_usage(response, "redundant_filtering")
            for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                token_usage[k] += usage[k]

            result = parse_filter_redundant_facts_batch_response(normalized)
            selected = set(result.get("selected_statements", []))
            idx_to_fact = {i + 1: f for i, f in enumerate(facts)}
            selected_facts = {idx_to_fact[i] for i in selected if i in idx_to_fact}

            kept = []
            for fact in facts:
                if fact in selected_facts:
                    kept.append(fact)
                else:
                    red_meta[(sentence, fact)] = [
                        fact,
                        result.get("reasoning", "Filtered as redundant"),
                        "[Foo]",
                        list(selected_facts),
                    ]

            filtered_pairs.append((sentence, kept))

        total_in = sum(len(f) for _, f in atomic_facts_pairs)
        total_out = sum(len(f) for _, f in filtered_pairs)
        print(f"Completed redundant filtering: {total_out}/{total_in} facts kept")
        return filtered_pairs, red_meta, token_usage


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def format_output_for_json(final_atomic_facts_pairs, para_breaks, metadata) -> dict:
    """Serialize pipeline output to a JSON-compatible dict.

    Args:
        final_atomic_facts_pairs: ``list[(sentence, [facts])]``
        para_breaks: ``list[int]``
        metadata: metadata dict from :meth:`AtomicFactGenerator.run`

    Returns:
        JSON-serializable dict.
    """
    return {
        "final_atomic_facts_pairs": final_atomic_facts_pairs,
        "paragraph_breaks": para_breaks,
        "metadata": {
            "initial_atomic_facts_pairs": metadata.get("initial_atomic_facts_pairs", []),
            "decontextualized_atomic_facts_pairs": metadata.get("decontextualized_atomic_facts_pairs", []),
            "rewritten_missing_atomic_facts_pairs": metadata.get("rewritten_missing_atomic_facts_pairs", []),
            "dependent_facts_metadata": convert_metadata_for_json(metadata.get("dependent_facts_metadata", {})),
            "irrelevant_facts_metadata": convert_metadata_for_json(metadata.get("irrelevant_facts_metadata", {})),
            "redundant_facts_metadata": convert_metadata_for_json(metadata.get("redundant_facts_metadata", {})),
            "token_usage": metadata.get("token_usage", {}),
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract atomic facts from text using an LLM pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--text", required=True, help="Input text to decompose.")
    parser.add_argument("--question", required=True, help="Question the text is answering.")
    parser.add_argument("--output-file", required=True, help="Output JSON file path.")
    parser.add_argument("--verbose", action="store_true", help="Print detailed step comparisons.")
    parser.add_argument(
        "--model-configs",
        default=None,
        help=(
            "JSON string or path to a JSON/YAML file with per-component model configurations. "
            "Omit to use defaults from config/model_config.yaml."
        ),
    )
    parser.add_argument("--disable-incomplete-detection", action="store_true")
    parser.add_argument("--disable-irrelevant-filtering", action="store_true")
    parser.add_argument("--disable-redundant-filtering", action="store_true")

    args = parser.parse_args()

    # Parse --model-configs
    model_configs = None
    if args.model_configs:
        try:
            config_dict = json.loads(args.model_configs)
        except json.JSONDecodeError:
            cfg_path = Path(args.model_configs)
            if not cfg_path.exists():
                raise ValueError(f"Cannot parse --model-configs as JSON or find file: {args.model_configs}")
            if cfg_path.suffix.lower() == ".json":
                from atomic_fact_generation.config.model_config import load_configs_from_json
                model_configs = load_configs_from_json(cfg_path)
            else:
                from atomic_fact_generation.config.model_config import load_configs_from_yaml
                model_configs = load_configs_from_yaml(cfg_path)
        else:
            model_configs = {
                comp: ModelConfig(**cfg) if isinstance(cfg, dict) else cfg
                for comp, cfg in config_dict.items()
            }

    generator = AtomicFactGenerator(model_configs=model_configs)
    final_facts, para_breaks, metadata = generator.run(
        args.text,
        args.question,
        verbose=args.verbose,
        enable_incomplete_detection=not args.disable_incomplete_detection,
        enable_irrelevant_filtering=not args.disable_irrelevant_filtering,
        enable_redundant_filtering=not args.disable_redundant_filtering,
    )

    output_data = format_output_for_json(final_facts, para_breaks, metadata)

    output_path = Path(args.output_file).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    # Cost report
    print(f"\n{'='*80}\nCOST BREAKDOWN\n{'='*80}")
    try:
        used_configs = model_configs or create_default_configs()
        cost_data = calculate_pricing(
            output_data=output_data,
            model_configs_dict={
                comp: {"model": cfg.model, "provider": cfg.provider}
                for comp, cfg in used_configs.items()
            },
        )
        print(format_cost_report(cost_data))
    except Exception as e:
        print(f"Warning: could not calculate pricing: {e}")

    # Summary
    print(f"\n{'='*80}\nSUMMARY\n{'='*80}")
    print(f"Sentences processed:    {len(final_facts)}")
    print(f"Total atomic facts:     {sum(len(f) for _, f in final_facts)}")
    print(f"Dependent facts:        {len(metadata['dependent_facts_metadata'])}")
    print(f"Irrelevant filtered:    {len(metadata['irrelevant_facts_metadata'])}")
    print(f"Redundant filtered:     {len(metadata['redundant_facts_metadata'])}")
    print(f"Results saved to:       {output_path}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
