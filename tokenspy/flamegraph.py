"""
flamegraph.py — Renders cost data as text summary or HTML/SVG flame graph.
"""

from __future__ import annotations

import html
import webbrowser
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tokenspy.tracker import CallRecord, Tracker


def render_text(tracker: Tracker) -> str:
    """Return a human-readable cost report string."""
    records = tracker.records()
    if not records:
        return "tokenspy: no LLM calls recorded.\n"

    total_cost = tracker.total_cost()
    total_tokens = tracker.total_tokens()
    total_calls = tracker.total_calls()

    lines: list[str] = []
    bar_width = 16

    lines.append("")
    lines.append(
        f"tokenspy cost report — total: ${total_cost:.4f}  "
        f"({total_tokens:,} tokens, {total_calls} call{'s' if total_calls != 1 else ''})"
    )
    lines.append("─" * 70)

    # Group by function
    by_fn: dict[str, list[CallRecord]] = {}
    for r in records:
        by_fn.setdefault(r.function_name, []).append(r)

    fn_costs = {fn: sum(r.cost_usd for r in recs) for fn, recs in by_fn.items()}
    sorted_fns = sorted(fn_costs, key=lambda f: fn_costs[f], reverse=True)

    for fn in sorted_fns:
        fn_cost = fn_costs[fn]
        pct = (fn_cost / total_cost * 100) if total_cost > 0 else 0
        bar = _bar(pct, bar_width)
        lines.append(f"  {fn:<35}  ${fn_cost:.4f}  {bar}  {pct:.0f}%")

        # Per-model breakdown within function
        model_costs: dict[str, tuple[float, int]] = {}
        for r in by_fn[fn]:
            prev_cost, prev_tok = model_costs.get(r.model, (0.0, 0))
            model_costs[r.model] = (prev_cost + r.cost_usd, prev_tok + r.total_tokens)

        for model, (mc, mt) in sorted(model_costs.items(), key=lambda x: x[1][0], reverse=True):
            m_pct = (mc / total_cost * 100) if total_cost > 0 else 0
            m_bar = _bar(m_pct, bar_width)
            lines.append(
                f"    {'└─ ' + model:<33}  ${mc:.4f}  {m_bar}  {m_pct:.0f}%  [{mt:,} tokens]"
            )

    lines.append("")
    return "\n".join(lines)


def render_html(tracker: Tracker, output_path: Path | None = None) -> str:
    """Render an HTML flame graph. Returns the HTML string.

    If output_path is given, writes the file there.
    """
    records = tracker.records()
    if not records:
        return "<p>No LLM calls recorded.</p>"

    total_cost = tracker.total_cost()
    total_tokens = tracker.total_tokens()

    # Aggregate by function for the flame graph
    by_fn: dict[str, list[CallRecord]] = {}
    for r in records:
        by_fn.setdefault(r.function_name, []).append(r)

    fn_costs = {fn: sum(r.cost_usd for r in recs) for fn, recs in by_fn.items()}
    sorted_fns = sorted(fn_costs, key=lambda f: fn_costs[f], reverse=True)

    # Build SVG bars
    svg_width = 800
    bar_max_width = 580
    bars_svg = []
    y = 40

    for fn in sorted_fns:
        cost = fn_costs[fn]
        pct = cost / total_cost if total_cost > 0 else 0
        bar_w = int(pct * bar_max_width)
        color = _cost_color(pct)
        label = html.escape(fn)
        cost_str = f"${cost:.4f}"
        pct_str = f"{pct * 100:.1f}%"
        tooltip = f"{label}: {cost_str} ({pct_str})"

        bars_svg.append(f"""
          <g transform="translate(160, {y})">
            <rect x="0" y="0" width="{bar_w}" height="24" fill="{color}" rx="3" opacity="0.85">
              <title>{html.escape(tooltip)}</title>
            </rect>
            <text x="-5" y="17" text-anchor="end" font-size="12" fill="#333">{label[:25]}</text>
            <text x="{bar_w + 6}" y="17" font-size="12" fill="#555">{cost_str} ({pct_str})</text>
          </g>""")
        y += 36

    svg = f"""<svg width="{svg_width}" height="{y + 20}" xmlns="http://www.w3.org/2000/svg" font-family="monospace">
      <text x="10" y="22" font-size="14" font-weight="bold" fill="#222">
        tokenspy — Total: ${total_cost:.4f} ({total_tokens:,} tokens)
      </text>
      {''.join(bars_svg)}
    </svg>"""

    # Build model breakdown table
    model_rows = []
    model_agg: dict[str, tuple[float, int, int]] = {}
    for r in records:
        c, it, ot = model_agg.get(r.model, (0.0, 0, 0))
        model_agg[r.model] = (c + r.cost_usd, it + r.input_tokens, ot + r.output_tokens)

    for model, (mc, mit, mot) in sorted(model_agg.items(), key=lambda x: x[1][0], reverse=True):
        pct = mc / total_cost * 100 if total_cost > 0 else 0
        model_rows.append(
            f"<tr><td>{html.escape(model)}</td><td>${mc:.4f}</td>"
            f"<td>{pct:.1f}%</td><td>{mit:,}</td><td>{mot:,}</td></tr>"
        )

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>tokenspy — LLM Cost Report</title>
  <style>
    body {{ font-family: 'Courier New', monospace; background: #f8f9fa; padding: 24px; color: #222; }}
    h1 {{ font-size: 20px; margin-bottom: 4px; }}
    .subtitle {{ color: #666; font-size: 13px; margin-bottom: 24px; }}
    .card {{ background: white; border-radius: 8px; padding: 20px; margin-bottom: 20px;
             box-shadow: 0 1px 4px rgba(0,0,0,0.08); }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th {{ text-align: left; padding: 8px 12px; background: #f0f0f0; border-bottom: 2px solid #ddd; }}
    td {{ padding: 8px 12px; border-bottom: 1px solid #eee; }}
    tr:hover td {{ background: #fafafa; }}
    .badge {{ display: inline-block; background: #e8f4fd; color: #1a73e8;
              border-radius: 4px; padding: 2px 8px; font-size: 12px; }}
  </style>
</head>
<body>
  <h1>tokenspy — LLM Cost Report</h1>
  <div class="subtitle">
    Total: <strong>${total_cost:.4f}</strong> &nbsp;|&nbsp;
    Tokens: <strong>{total_tokens:,}</strong> &nbsp;|&nbsp;
    Calls: <strong>{len(records)}</strong>
  </div>

  <div class="card">
    <h2 style="font-size:15px;margin-top:0">Cost by Function</h2>
    {svg}
  </div>

  <div class="card">
    <h2 style="font-size:15px;margin-top:0">Breakdown by Model</h2>
    <table>
      <thead>
        <tr><th>Model</th><th>Cost (USD)</th><th>% of Total</th>
            <th>Input Tokens</th><th>Output Tokens</th></tr>
      </thead>
      <tbody>
        {''.join(model_rows)}
      </tbody>
    </table>
  </div>

  <div class="card">
    <h2 style="font-size:15px;margin-top:0">All Calls ({len(records)})</h2>
    <table>
      <thead>
        <tr><th>#</th><th>Function</th><th>Model</th><th>Cost</th>
            <th>Tokens (in/out)</th><th>Duration</th></tr>
      </thead>
      <tbody>
        {''.join(
            f"<tr><td>{i+1}</td><td>{html.escape(r.function_name)}</td>"
            f"<td>{html.escape(r.model)}</td><td>${r.cost_usd:.4f}</td>"
            f"<td>{r.input_tokens:,} / {r.output_tokens:,}</td>"
            f"<td>{r.duration_ms:.0f}ms</td></tr>"
            for i, r in enumerate(records)
        )}
      </tbody>
    </table>
  </div>
</body>
</html>"""

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html_doc, encoding="utf-8")

    return html_doc


def open_html_report(tracker: Tracker, output_path: Path | None = None) -> Path:
    """Write the HTML report and open it in the default browser."""
    if output_path is None:
        output_path = Path("tokenspy_report.html")
    render_html(tracker, output_path=output_path)
    webbrowser.open(output_path.resolve().as_uri())
    return output_path


# ── Helpers ────────────────────────────────────────────────────────────────────

def _bar(pct: float, width: int) -> str:
    filled = int(pct / 100 * width)
    return "█" * filled + "░" * (width - filled)


def _cost_color(fraction: float) -> str:
    """Returns a color from green (cheap) to red (expensive)."""
    if fraction > 0.5:
        return "#e74c3c"   # red
    if fraction > 0.25:
        return "#e67e22"   # orange
    if fraction > 0.1:
        return "#f1c40f"   # yellow
    return "#2ecc71"       # green
