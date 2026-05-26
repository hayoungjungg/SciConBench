#!/usr/bin/env python3
"""
Calculate pricing for atomic fact generation based on token usage.

Reads output.json and model_configs.json to calculate the total cost based on
input/output tokens and model pricing.
"""

import json
import argparse
import os
from typing import Dict, Any, Optional
from pathlib import Path


# Pricing per 1M tokens (in USD)
PRICING = {
    "gpt-5-mini": {"input": 0.25, "cached_input": 0.025, "output": 2.00},
    "gpt-5.1": {"input": 1.25, "cached_input": 0.125, "output": 10.00},
}


def normalize_model_name(model: str) -> str:
    """
    Normalize model name to match pricing table keys.
    
    Args:
        model: Model name (e.g., 'gpt-5-mini', 'gpt-5-chat')
    
    Returns:
        Normalized model name
    """
    model_lower = model.lower()
    
    # Direct match
    if model_lower in PRICING:
        return model_lower
    
    # Handle variations
    if model_lower.startswith("gpt-5.2"):
        return "gpt-5.2"
    elif model_lower.startswith("gpt-5.1"):
        return "gpt-5.1"
    elif model_lower.startswith("gpt-5-mini"):
        return "gpt-5-mini"
    elif model_lower.startswith("gpt-5-nano"):
        return "gpt-5-nano"
    elif model_lower.startswith("gpt-5-chat"):
        return "gpt-5-chat"
    elif model_lower.startswith("gpt-5"):
        return "gpt-5"
    
    # Default fallback
    return model_lower


def get_model_for_component(component: str, model_configs: Dict[str, Any]) -> Optional[str]:
    """
    Get the model name used for a specific component.
    
    Args:
        component: Component name (e.g., 'decomposition')
        model_configs: Dictionary of model configurations
    
    Returns:
        Model name or None if not found
    """
    if component in model_configs:
        return model_configs[component].get("model")
    return None


def calculate_component_cost(
    component: str,
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int,
    model: str
) -> Dict[str, Any]:
    """
    Calculate cost for a single component.
    
    Args:
        component: Component name
        prompt_tokens: Number of prompt/input tokens
        completion_tokens: Number of completion/output tokens
        cached_tokens: Number of cached input tokens (default: 0)
        model: Model name
    
    Returns:
        Dictionary with cost breakdown
    """
    normalized_model = normalize_model_name(model)
    
    if normalized_model not in PRICING:
        print(f"Warning: Model '{model}' (normalized: '{normalized_model}') not found in pricing table. Using gpt-5 pricing.")
        normalized_model = "gpt-5"
    
    pricing = PRICING[normalized_model]
    
    # Calculate costs (pricing is per 1M tokens)
    input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
    cached_input_cost = (cached_tokens / 1_000_000) * pricing["cached_input"]
    output_cost = (completion_tokens / 1_000_000) * pricing["output"]
    
    # Non-cached input tokens
    non_cached_input_tokens = prompt_tokens - cached_tokens
    non_cached_input_cost = (non_cached_input_tokens / 1_000_000) * pricing["input"]
    
    total_cost = non_cached_input_cost + cached_input_cost + output_cost
    
    return {
        "component": component,
        "model": model,
        "normalized_model": normalized_model,
        "tokens": {
            "input": prompt_tokens,
            "cached_input": cached_tokens,
            "non_cached_input": non_cached_input_tokens,
            "output": completion_tokens,
            "total": prompt_tokens + completion_tokens
        },
        "costs": {
            "input": input_cost,
            "cached_input": cached_input_cost,
            "non_cached_input": non_cached_input_cost,
            "output": output_cost,
            "total": total_cost
        },
        "pricing_per_1M": {
            "input": pricing["input"],
            "cached_input": pricing["cached_input"],
            "output": pricing["output"]
        }
    }


def calculate_pricing(
    output_file: str = None,
    output_data: Optional[Dict[str, Any]] = None,
    model_configs_file: Optional[str] = None,
    model_configs_dict: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Calculate total pricing from output.json and model configs.
    
    Args:
        output_file: Optional path to output.json file (if output_data not provided)
        output_data: Optional dictionary with output data (takes precedence over output_file)
        model_configs_file: Optional path to model_configs.json file
        model_configs_dict: Optional dictionary of model configs (takes precedence over file)
    
    Returns:
        Dictionary with detailed cost breakdown
    """
    # Load output.json if output_data not provided
    if output_data is None:
        if output_file is None:
            raise ValueError("Either output_file or output_data must be provided")
        with open(output_file, 'r', encoding='utf-8') as f:
            output_data = json.load(f)
    
    # Load model configs
    model_configs = {}
    if model_configs_dict:
        model_configs = model_configs_dict
    elif model_configs_file and os.path.exists(model_configs_file):
        with open(model_configs_file, 'r', encoding='utf-8') as f:
            model_configs = json.load(f)
    else:
        # Try to infer from defaults or use gpt-5-chat as fallback
        print("Warning: No model configs provided. Using gpt-5-chat as default for all components.")
        model_configs = {
            "decomposition": {"model": "gpt-5-chat"},
            "decontextualization": {"model": "gpt-5-chat"},
            "incomplete_detection": {"model": "gpt-5-chat"},
            "irrelevant_filtering": {"model": "gpt-5-chat"},
            "redundant_filtering": {"model": "gpt-5-chat"}
        }
    
    # Get token usage from metadata
    token_usage = output_data.get("metadata", {}).get("token_usage", {})
    
    if not token_usage:
        raise ValueError("No token_usage found in output.json metadata")
    
    # Calculate costs for each component
    component_costs = []
    total_cost = 0.0
    
    components = ["decomposition", "decontextualization", "incomplete_detection", 
                  "irrelevant_filtering", "redundant_filtering"]
    
    for component in components:
        if component not in token_usage:
            continue
        
        usage = token_usage[component]
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        cached_tokens = usage.get("cached_tokens", 0)  # May not be present
        
        # Get model for this component
        model = get_model_for_component(component, model_configs)
        if not model:
            print(f"Warning: No model found for component '{component}'. Using gpt-5-chat as default.")
            model = "gpt-5-chat"
        
        # Calculate cost
        cost_breakdown = calculate_component_cost(
            component, prompt_tokens, completion_tokens, cached_tokens, model
        )
        
        component_costs.append(cost_breakdown)
        total_cost += cost_breakdown["costs"]["total"]
    
    # Get total token usage
    total_usage = token_usage.get("total", {})
    total_cached_tokens = total_usage.get("cached_tokens", 0)
    total_input_tokens = total_usage.get("prompt_tokens", 0)
    
    return {
        "components": component_costs,
        "summary": {
            "total_cost_usd": total_cost,
            "total_tokens": {
                "input": total_input_tokens,
                "cached_input": total_cached_tokens,
                "non_cached_input": total_input_tokens - total_cached_tokens,
                "output": total_usage.get("completion_tokens", 0),
                "total": total_usage.get("total_tokens", 0)
            }
        }
    }


def format_cost_report(cost_data: Dict[str, Any]) -> str:
    """
    Format cost data as a readable report.
    
    Args:
        cost_data: Cost data dictionary from calculate_pricing
    
    Returns:
        Formatted string report
    """
    lines = []
    lines.append("=" * 80)
    lines.append("ATOMIC FACT GENERATION - COST BREAKDOWN")
    lines.append("=" * 80)
    lines.append("")
    
    # Component breakdown
    lines.append("COMPONENT BREAKDOWN:")
    lines.append("-" * 80)
    
    for comp in cost_data["components"]:
        lines.append(f"\n{comp['component'].upper()}:")
        lines.append(f"  Model: {comp['model']} ({comp['normalized_model']})")
        lines.append(f"  Tokens:")
        lines.append(f"    Input:        {comp['tokens']['input']:,} ({comp['tokens']['non_cached_input']:,} non-cached, {comp['tokens']['cached_input']:,} cached)")
        lines.append(f"    Output:       {comp['tokens']['output']:,}")
        lines.append(f"    Total:        {comp['tokens']['total']:,}")
        lines.append(f"  Pricing (per 1M tokens):")
        lines.append(f"    Input:        ${comp['pricing_per_1M']['input']:.3f}")
        lines.append(f"    Cached Input: ${comp['pricing_per_1M']['cached_input']:.3f}")
        lines.append(f"    Output:       ${comp['pricing_per_1M']['output']:.2f}")
        lines.append(f"  Costs:")
        lines.append(f"    Non-cached Input: ${comp['costs']['non_cached_input']:.6f}")
        if comp['tokens']['cached_input'] > 0:
            lines.append(f"    Cached Input:     ${comp['costs']['cached_input']:.6f}")
        lines.append(f"    Output:           ${comp['costs']['output']:.6f}")
        lines.append(f"    Total:             ${comp['costs']['total']:.6f}")
    
    # Summary
    lines.append("")
    lines.append("=" * 80)
    lines.append("SUMMARY")
    lines.append("=" * 80)
    lines.append(f"Total Cost: ${cost_data['summary']['total_cost_usd']:.6f} USD")
    lines.append(f"Total Tokens:")
    lines.append(f"  Input:        {cost_data['summary']['total_tokens']['input']:,}")
    if cost_data['summary']['total_tokens'].get('cached_input', 0) > 0:
        lines.append(f"    Non-cached: {cost_data['summary']['total_tokens']['non_cached_input']:,}")
        lines.append(f"    Cached:     {cost_data['summary']['total_tokens']['cached_input']:,}")
    lines.append(f"  Output:       {cost_data['summary']['total_tokens']['output']:,}")
    lines.append(f"  Total:        {cost_data['summary']['total_tokens']['total']:,}")
    lines.append("=" * 80)
    
    return "\n".join(lines)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Calculate pricing for atomic fact generation based on token usage',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--output-file',
        type=str,
        required=True,
        help='Path to output.json file from atomic fact generation'
    )
    
    parser.add_argument(
        '--model-configs',
        type=str,
        default=None,
        help='Path to model_configs.json file used for generation (optional, will use defaults if not provided)'
    )
    
    parser.add_argument(
        '--output-report',
        type=str,
        default=None,
        help='Optional path to save cost report as text file'
    )
    
    parser.add_argument(
        '--json-output',
        type=str,
        default=None,
        help='Optional path to save cost breakdown as JSON file'
    )
    
    args = parser.parse_args()
    
    # Calculate pricing
    cost_data = calculate_pricing(
        output_file=args.output_file,
        model_configs_file=args.model_configs
    )
    
    # Print report
    report = format_cost_report(cost_data)
    print(report)
    
    # Save report if requested
    if args.output_report:
        with open(args.output_report, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\nReport saved to: {args.output_report}")
    
    # Save JSON if requested
    if args.json_output:
        with open(args.json_output, 'w', encoding='utf-8') as f:
            json.dump(cost_data, f, indent=2)
        print(f"JSON saved to: {args.json_output}")


if __name__ == "__main__":
    main()

