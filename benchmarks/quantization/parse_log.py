#!/usr/bin/env python3
"""
Parse measure_accuracy_and_performance.sh log file and extract metrics.

Usage:
    python parse_log.py <log_file> [--format csv|json|table]

Example:
    python parse_log.py benchmarks/data/measure_accuracy_and_performance_log.txt --format csv
"""

import argparse
import re
import sys
from typing import Dict, List, Optional

from tabulate import tabulate


def parse_header(lines: List[str]) -> Dict[str, str]:
    """Parse the first 4 lines to extract library versions."""
    header = {}
    for line in lines[:4]:
        # Format: torch.__version__='2.9.0+cu128'
        # or: torch.cuda.get_device_name()='NVIDIA H100'
        match = re.match(r"([^=]+)='([^']+)'", line)
        if match:
            key = match.group(1)
            value = match.group(2)
            header[key] = value
    return header


def parse_accuracy_metrics(section_lines: List[str]) -> Dict[str, Optional[float]]:
    """Parse accuracy metrics from lm_eval output table."""
    metrics = {
        "wikitext_word_perplexity": None,
        "winogrande_acc": None,
        "winogrande_acc_stderr": None,
    }

    # Each entry maps a metric key to the ``(label, offset)`` column in the
    # lm_eval table row identified by ``row_marker``.
    # Format of a matched row, e.g.:
    #   |          |    |none|0|word_perplexity|↓|7.5435|±|   N/A|
    #   |winogrande|1   |none|0|acc            |↑|0.7419|±|0.0123|
    row_specs = [
        ("|word_perplexity", [("wikitext_word_perplexity", "word_perplexity", 2)]),
        (
            "|winogrande",
            [
                ("winogrande_acc", "acc", 2),
                ("winogrande_acc_stderr", "acc", 4),
            ],
        ),
    ]

    for line in section_lines:
        for row_marker, columns in row_specs:
            if row_marker in line:
                _extract_row_metrics(line, columns, metrics)

    return metrics


def _column_value(parts: List[str], label: str, offset: int) -> Optional[float]:
    """Return the float ``offset`` columns after ``label`` in ``parts`` if valid."""
    try:
        idx = parts.index(label)
    except ValueError:
        return None
    if idx + offset >= len(parts):
        return None
    value = parts[idx + offset]
    if value == "N/A":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _extract_row_metrics(
    line: str,
    columns: List[tuple],
    metrics: Dict[str, Optional[float]],
) -> None:
    """Extract each ``(metric_key, label, offset)`` column from a table ``line``."""
    parts = [p.strip() for p in line.split("|")]
    for metric_key, label, offset in columns:
        value = _column_value(parts, label, offset)
        if value is not None:
            metrics[metric_key] = value


def _parse_throughput_line(line: str) -> Optional[Dict[str, float]]:
    """Return the throughput data for a ``Throughput:`` ``line``, or ``None``."""
    # Format: Throughput: 7.50 requests/s, 30939.86 total tokens/s, 239.84 output tokens/s
    match = re.match(
        r"Throughput:\s+([\d.]+)\s+requests/s,\s+([\d.]+)\s+total tokens/s,\s+([\d.]+)\s+output tokens/s",
        line,
    )
    if not match:
        return None
    return {
        "requests_per_sec": float(match.group(1)),
        "total_tokens_per_sec": float(match.group(2)),
        "output_tokens_per_sec": float(match.group(3)),
    }


def _classify_throughput(section_lines: List[str], index: int) -> Optional[str]:
    """Return ``"prefill"``/``"decode"`` for the throughput line at ``index``.

    Determines the phase by looking backwards for the benchmark command:
    prefill has ``--input_len 4096 --output_len 32`` and decode has
    ``--input_len 32 --output_len 2048``. Returns ``None`` if no marker is found.
    """
    for j in range(max(0, index - 50), index):
        if "benchmarking vllm prefill performance" in section_lines[j]:
            return "prefill"
        if "benchmarking vllm decode performance" in section_lines[j]:
            return "decode"
    return None


def _assign_throughput(
    metrics: Dict[str, Optional[Dict[str, float]]],
    phase: Optional[str],
    throughput_data: Dict[str, float],
) -> None:
    """Store ``throughput_data`` under its ``phase`` (or by order if unknown)."""
    if phase is not None:
        metrics[phase] = throughput_data
    elif metrics["prefill"] is None:
        # If we can't find the marker, assign based on order.
        metrics["prefill"] = throughput_data
    elif metrics["decode"] is None:
        metrics["decode"] = throughput_data


def parse_throughput_metrics(
    section_lines: List[str],
) -> Dict[str, Optional[Dict[str, float]]]:
    """Parse vLLM throughput metrics."""
    metrics = {
        "prefill": None,
        "decode": None,
    }

    for i, line in enumerate(section_lines):
        if not line.startswith("Throughput:"):
            continue
        throughput_data = _parse_throughput_line(line)
        if throughput_data is None:
            continue

        phase = _classify_throughput(section_lines, i)
        _assign_throughput(metrics, phase, throughput_data)

    return metrics


def parse_recipe_section(section_lines: List[str], recipe_name: str) -> Dict:
    """Parse a single recipe section."""
    result = {
        "recipe": recipe_name,
    }

    # Parse checkpoint size
    checkpoint_size_gb = None
    for line in section_lines:
        # Format: checkpoint size: 16.077915292 GB
        match = re.match(r"checkpoint size:\s+([\d.]+)\s+GB", line)
        if match:
            checkpoint_size_gb = float(match.group(1))
            break
    result["checkpoint_size_gb"] = checkpoint_size_gb

    # Parse accuracy metrics
    accuracy = parse_accuracy_metrics(section_lines)
    result.update(accuracy)

    # Parse throughput metrics
    throughput = parse_throughput_metrics(section_lines)

    # Flatten throughput metrics
    if throughput["prefill"]:
        result["prefill_total_tokens_per_sec"] = throughput["prefill"][
            "total_tokens_per_sec"
        ]
    else:
        result["prefill_total_tokens_per_sec"] = None

    if throughput["decode"]:
        result["decode_total_tokens_per_sec"] = throughput["decode"][
            "total_tokens_per_sec"
        ]
    else:
        result["decode_total_tokens_per_sec"] = None

    return result


def parse_log_file(log_file_path: str) -> Dict:
    """Parse the entire log file."""
    with open(log_file_path, "r") as f:
        lines = [line.rstrip("\n") for line in f]

    # Parse header
    header = parse_header(lines)

    # Find recipe sections
    recipe_sections = []
    current_section = None
    current_recipe = None

    for i, line in enumerate(lines):
        # Look for recipe markers
        match = re.match(r"processing quant_recipe (.+)", line)
        if match:
            # Save previous section if exists
            if current_section is not None:
                recipe_sections.append((current_recipe, current_section))

            # Start new section
            current_recipe = match.group(1).strip()
            current_section = []
        elif current_section is not None:
            current_section.append(line)

    # Don't forget the last section
    if current_section is not None:
        recipe_sections.append((current_recipe, current_section))

    # Parse each recipe section
    results = []
    for recipe_name, section_lines in recipe_sections:
        result = parse_recipe_section(section_lines, recipe_name)
        results.append(result)

    # Calculate speedups relative to baseline ("None")
    baseline = None
    for result in results:
        if result["recipe"] == "None":
            baseline = result
            break

    if baseline:
        baseline_prefill = baseline["prefill_total_tokens_per_sec"]
        baseline_decode = baseline["decode_total_tokens_per_sec"]

        for result in results:
            # Calculate prefill speedup
            if (
                result["prefill_total_tokens_per_sec"] is not None
                and baseline_prefill is not None
            ):
                result["speedup_prefill"] = (
                    result["prefill_total_tokens_per_sec"] / baseline_prefill
                )
            else:
                result["speedup_prefill"] = None

            # Calculate decode speedup
            if (
                result["decode_total_tokens_per_sec"] is not None
                and baseline_decode is not None
            ):
                result["speedup_decode"] = (
                    result["decode_total_tokens_per_sec"] / baseline_decode
                )
            else:
                result["speedup_decode"] = None
    else:
        # No baseline found, set all speedups to None
        for result in results:
            result["speedup_prefill"] = None
            result["speedup_decode"] = None

    return {
        "header": header,
        "results": results,
    }


def format_as_csv(data: Dict) -> str:
    """Format parsed data as CSV."""
    lines = []

    # Header comment with library versions
    lines.append("# Library Versions:")
    for key, value in data["header"].items():
        lines.append(f"# {key}={value}")
    lines.append("")

    # CSV header
    fieldnames = [
        "recipe",
        "checkpoint_size_gb",
        "wikitext_word_perplexity",
        "winogrande_acc",
        "winogrande_acc_stderr",
        "prefill_total_tokens_per_sec",
        "decode_total_tokens_per_sec",
        "speedup_prefill",
        "speedup_decode",
    ]
    lines.append(",".join(fieldnames))

    # Data rows
    for result in data["results"]:
        row = [
            str(result.get(field, "")) if result.get(field) is not None else ""
            for field in fieldnames
        ]
        lines.append(",".join(row))

    return "\n".join(lines)


def format_as_table(data: Dict) -> str:
    """Format parsed data as a human-readable table using tabulate."""
    lines = []

    # Header with library versions
    lines.append("Library Versions:")
    lines.append("=" * 80)
    for key, value in data["header"].items():
        lines.append(f"{key}: {value}")
    lines.append("")

    # Prepare table data
    table_data = []
    headers = [
        "Recipe",
        "Checkpoint\n(GB)",
        "Wikitext\nPerplexity",
        "Winogrande\nAcc",
        "Winogrande\nStderr",
        "Prefill\ntoks/s",
        "Decode\ntoks/s",
        "Speedup\nPrefill",
        "Speedup\nDecode",
    ]

    for result in data["results"]:
        row = [
            result["recipe"],
            f"{result['checkpoint_size_gb']:.2f}"
            if result["checkpoint_size_gb"] is not None
            else None,
            f"{result['wikitext_word_perplexity']:.4f}"
            if result["wikitext_word_perplexity"] is not None
            else None,
            f"{result['winogrande_acc']:.4f}"
            if result["winogrande_acc"] is not None
            else None,
            f"{result['winogrande_acc_stderr']:.4f}"
            if result["winogrande_acc_stderr"] is not None
            else None,
            f"{result['prefill_total_tokens_per_sec']:.2f}"
            if result["prefill_total_tokens_per_sec"] is not None
            else None,
            f"{result['decode_total_tokens_per_sec']:.2f}"
            if result["decode_total_tokens_per_sec"] is not None
            else None,
            f"{result['speedup_prefill']:.3f}"
            if result["speedup_prefill"] is not None
            else None,
            f"{result['speedup_decode']:.3f}"
            if result["speedup_decode"] is not None
            else None,
        ]
        table_data.append(row)

    # Generate table
    lines.append("Quantization Recipe Results:")
    lines.append("=" * 80)
    lines.append(tabulate(table_data, headers=headers, tablefmt="grid"))

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Parse measure_accuracy_and_performance.sh log file"
    )
    parser.add_argument("log_file", help="Path to the log file to parse")
    parser.add_argument(
        "--format",
        choices=["csv", "table"],
        default="table",
        help="Output format (default: table)",
    )
    parser.add_argument("--output", "-o", help="Output file (default: stdout)")

    args = parser.parse_args()

    # Parse the log file
    data = parse_log_file(args.log_file)

    # Format output
    if args.format == "csv":
        output = format_as_csv(data)
    else:  # table
        output = format_as_table(data)

    # Write output
    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Output written to {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
