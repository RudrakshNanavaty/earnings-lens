import html
import os
import sys
import json
import gradio as gr
from gradio.themes.utils import colors, sizes, fonts
from typing import Any, Optional
from dotenv import load_dotenv
from openai import AsyncOpenAI
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

# Ensure the project root is in sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

try:
    from earnings_analyst.server.earnings_analyst_environment import (
        EarningsAnalystEnvironment,
    )
    from earnings_analyst.server.episode_index import (
        RANDOM_EPISODE_LABEL,
        format_episode_id,
        get_episode_index,
    )
    from earnings_analyst.models import (
        EarningsAnalystAction,
        EarningsAnalystObservation,
    )
    from earnings_analyst.tasks.exceptions import TaskNotImplementedError
    from earnings_analyst.tasks.registry import TASK_IDS
except (ImportError, ModuleNotFoundError):
    from server.earnings_analyst_environment import EarningsAnalystEnvironment
    from server.episode_index import (
        RANDOM_EPISODE_LABEL,
        format_episode_id,
        get_episode_index,
    )
    from models import EarningsAnalystAction, EarningsAnalystObservation
    from tasks.exceptions import TaskNotImplementedError
    from tasks.registry import TASK_IDS

load_dotenv()

# One accordion per possible text column (task specs use these keys).
TEXT_KEYS = (
    "earnings_transcript",
    "press_release_8k_body",
    "press_release_ex991",
    "press_release_ex992",
    "press_release_sources",
)

# ---------------------------------------------------------------------------
# Theme — Zerodha/Kite-inspired
# ---------------------------------------------------------------------------

zerodha_blue = colors.Color(
    c50="#eff6ff",
    c100="#dbeafe",
    c200="#bfdbfe",
    c300="#93c5fd",
    c400="#60a5fa",
    c500="#387ED1",
    c600="#2563eb",
    c700="#1d4ed8",
    c800="#1e40af",
    c900="#1e3a8a",
    c950="#172554",
)

theme = gr.themes.Base(  # type: ignore[attr-defined]
    primary_hue=zerodha_blue,
    secondary_hue=zerodha_blue,
    neutral_hue=colors.slate,
    radius_size=sizes.radius_md,
    font=(fonts.GoogleFont("Inter"), "system-ui", "sans-serif"),
    font_mono=(fonts.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"),
)

theme.set(
    # Body
    body_background_fill="#F8F9FA",
    body_background_fill_dark="#18181b",
    body_text_color="#1a1a1a",
    body_text_color_dark="#f4f4f5",
    body_text_color_subdued="#6b7280",
    body_text_color_subdued_dark="#a1a1aa",
    # Blocks / cards
    block_background_fill="#ffffff",
    block_background_fill_dark="#27272a",
    block_border_color="#e5e7eb",
    block_border_color_dark="#3f3f46",
    block_border_width="1px",
    block_radius="12px",
    block_shadow="none",
    block_shadow_dark="none",
    block_label_background_fill="#f9fafb",
    block_label_background_fill_dark="#3f3f46",
    block_label_border_color="#e5e7eb",
    block_label_border_color_dark="#52525b",
    block_label_text_color="#374151",
    block_label_text_color_dark="#d4d4d8",
    # Inputs
    input_background_fill="#ffffff",
    input_background_fill_dark="#3f3f46",
    input_background_fill_focus="#ffffff",
    input_background_fill_focus_dark="#52525b",
    input_border_color="#d1d5db",
    input_border_color_dark="#52525b",
    input_border_color_focus="#387ED1",
    input_border_color_focus_dark="#60a5fa",
    input_border_width="1px",
    input_radius="8px",
    input_shadow="none",
    input_shadow_dark="none",
    input_shadow_focus="0 0 0 3px rgba(56, 126, 209, 0.15)",
    input_shadow_focus_dark="0 0 0 3px rgba(96, 165, 250, 0.2)",
    input_placeholder_color="#9ca3af",
    input_placeholder_color_dark="#71717a",
    # Primary buttons — solid Zerodha blue
    button_primary_background_fill="#387ED1",
    button_primary_background_fill_dark="#387ED1",
    button_primary_background_fill_hover="#2563eb",
    button_primary_background_fill_hover_dark="#2563eb",
    button_primary_text_color="#ffffff",
    button_primary_text_color_dark="#ffffff",
    button_primary_text_color_hover="#ffffff",
    button_primary_text_color_hover_dark="#ffffff",
    button_primary_border_color="#387ED1",
    button_primary_border_color_dark="#387ED1",
    button_primary_border_color_hover="#2563eb",
    button_primary_border_color_hover_dark="#2563eb",
    button_primary_shadow="none",
    button_primary_shadow_dark="none",
    button_primary_shadow_hover="none",
    button_primary_shadow_hover_dark="none",
    # Secondary buttons — subtle outline
    button_secondary_background_fill="#ffffff",
    button_secondary_background_fill_dark="#3f3f46",
    button_secondary_background_fill_hover="#eff6ff",
    button_secondary_background_fill_hover_dark="#52525b",
    button_secondary_text_color="#387ED1",
    button_secondary_text_color_dark="#93c5fd",
    button_secondary_text_color_hover="#2563eb",
    button_secondary_text_color_hover_dark="#bfdbfe",
    button_secondary_border_color="#d1d5db",
    button_secondary_border_color_dark="#52525b",
    button_secondary_border_color_hover="#387ED1",
    button_secondary_border_color_hover_dark="#60a5fa",
    button_secondary_shadow="none",
    button_secondary_shadow_dark="none",
    button_secondary_shadow_hover="none",
    button_secondary_shadow_hover_dark="none",
    # Links
    link_text_color="#387ED1",
    link_text_color_dark="#60a5fa",
    link_text_color_hover="#2563eb",
    link_text_color_hover_dark="#93c5fd",
    link_text_color_active="#1d4ed8",
    link_text_color_active_dark="#bfdbfe",
    link_text_color_visited="#387ED1",
    link_text_color_visited_dark="#60a5fa",
    # Accent
    color_accent="#387ED1",
    color_accent_soft="#eff6ff",
    color_accent_soft_dark="#1e3a8a",
    border_color_accent="#387ED1",
    border_color_accent_dark="#60a5fa",
    border_color_accent_subdued="#bfdbfe",
    border_color_accent_subdued_dark="#1d4ed8",
)

# ---------------------------------------------------------------------------
# CSS overrides
# ---------------------------------------------------------------------------

custom_css = """
/* Align custom panels with theme block/input radii */
:root {
    --ea-radius-block: 12px;
    --ea-radius-control: 8px;
}

footer { visibility: hidden; }

/* Page shell */
.app-shell {
    max-width: 1140px;
    margin: 0 auto;
    padding: 1.25rem 1rem 2rem;
}

/* Header strip */
.header-row {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    margin-bottom: 1.25rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid #e5e7eb;
}
.dark .header-row { border-color: #3f3f46; }

/* Status badge */
.status-badge > .prose p {
    margin: 0;
    font-size: 0.82rem;
    color: #6b7280;
    text-align: right;
}
.dark .status-badge > .prose p { color: #a1a1aa; }

/* Theme toggle — rounding only on outer frame; Gradio may add per-button radii (fix asymmetry) */
.theme-toggle .wrap {
    gap: 0 !important;
    display: inline-flex !important;
    flex-direction: row !important;
    align-items: stretch;
    border-radius: var(--ea-radius-control);
    overflow: hidden;
    box-sizing: border-box;
}
/* Prefer fieldset / radiogroup as the stroked frame; otherwise stroke the wrap (e.g. button UI) */
.theme-toggle fieldset,
.theme-toggle [role="radiogroup"] {
    display: inline-flex !important;
    flex-direction: row !important;
    align-items: stretch;
    padding: 0 !important;
    margin: 0 !important;
    border: 1px solid #d1d5db !important;
    border-radius: var(--ea-radius-control);
    overflow: hidden;
    gap: 0 !important;
    box-sizing: border-box;
}
.theme-toggle .wrap:not(:has(fieldset)):not(:has([role="radiogroup"])) {
    border: 1px solid #d1d5db;
}
.theme-toggle .gap-2 { gap: 0 !important; }
.theme-toggle input[type=radio] { display: none; }
.theme-toggle label {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 4px 12px;
    font-size: 0.78rem;
    font-weight: 500;
    cursor: pointer;
    border: none !important;
    border-radius: 0 !important;
    border-right: 1px solid #d1d5db !important;
    background: #fff;
    color: #374151;
    transition: background 0.15s, color 0.15s, border-color 0.15s;
    white-space: nowrap;
}
.theme-toggle label:last-of-type { border-right: none !important; }
/* Segments: zero radius so only the group clips to --ea-radius-control (overrides Gradio/Tailwind per-side radii) */
.theme-toggle button,
.theme-toggle [role="radiogroup"] button,
.theme-toggle button:first-child,
.theme-toggle button:last-child {
    border: none !important;
    border-radius: 0 !important;
    border-top-left-radius: 0 !important;
    border-top-right-radius: 0 !important;
    border-bottom-left-radius: 0 !important;
    border-bottom-right-radius: 0 !important;
    border-right: 1px solid #d1d5db !important;
    margin: 0 !important;
    box-shadow: none !important;
}
.theme-toggle button:last-child { border-right: none !important; }
/* Gradio 4+ uses buttons; Zerodha blue for resolved/selected segment */
.theme-toggle button.selected,
.theme-toggle button.ea-theme-active,
.theme-toggle button[aria-checked="true"],
.theme-toggle button[data-state="checked"] {
    background: #387ED1 !important;
    color: #fff !important;
    border-color: #387ED1 !important;
    z-index: 1;
}
.theme-toggle input[type=radio]:checked + label,
.theme-toggle label.selected {
    background: #387ED1;
    color: #fff;
    border-color: #387ED1;
    z-index: 1;
}
/* Same highlight when .dark is on body/html (do not rely on html.dark alone) */
.dark .theme-toggle button.selected,
.dark .theme-toggle button.ea-theme-active,
.dark .theme-toggle button[aria-checked="true"],
.dark .theme-toggle button[data-state="checked"] {
    background: #387ED1 !important;
    color: #fff !important;
    border-color: #387ED1 !important;
}
body.dark .theme-toggle button.selected,
body.dark .theme-toggle button.ea-theme-active,
body.dark .theme-toggle button[aria-checked="true"],
body.dark .theme-toggle button[data-state="checked"],
html.dark .theme-toggle button.selected,
html.dark .theme-toggle button.ea-theme-active,
html.dark .theme-toggle button[aria-checked="true"],
html.dark .theme-toggle button[data-state="checked"] {
    background: #387ED1 !important;
    color: #fff !important;
    border-color: #387ED1 !important;
}
/* System: Gradio keeps "System" selected — neutralize that segment; .ea-theme-active marks resolved Light/Dark */
html[data-ea-theme="system"][data-ea-resolved="light"] .theme-toggle button.selected:not(.ea-theme-active) {
    background: #fff !important;
    color: #374151 !important;
    border-right-color: #d1d5db !important;
}
html[data-ea-theme="system"][data-ea-resolved="dark"] .theme-toggle button.selected:not(.ea-theme-active) {
    background: #3f3f46 !important;
    color: #d4d4d8 !important;
    border-right-color: #52525b !important;
}
.dark .theme-toggle .wrap:not(:has(fieldset)):not(:has([role="radiogroup"])) {
    border-color: #52525b;
}
.dark .theme-toggle fieldset,
.dark .theme-toggle [role="radiogroup"] {
    border-color: #52525b !important;
}
.dark .theme-toggle button { border-right-color: #52525b !important; }
.dark .theme-toggle label {
    background: #3f3f46;
    color: #d4d4d8;
    border-right-color: #52525b !important;
}
.dark .theme-toggle input[type=radio]:checked + label,
.dark .theme-toggle label.selected {
    background: #387ED1;
    color: #fff;
    border-color: #387ED1;
}

/* Gradio index.html uses @media (prefers-color-scheme: dark) on body — override when forcing light */
@media (prefers-color-scheme: dark) {
    html[data-ea-theme="light"] body {
        background: var(--bg, #f8f9fa) !important;
        color: var(--col, #1a1a1a) !important;
    }
}

/* Task instruction callout — top accent + even border (no heavy left margin) */
.task-instruction {
    border: 1px solid #bfdbfe;
    border-top: 3px solid #387ED1;
    background: #f0f6ff;
    border-radius: var(--ea-radius-block);
    padding: 0.9rem 1.1rem;
    margin-bottom: 1rem;
    box-sizing: border-box;
}
.dark .task-instruction {
    background: rgba(30, 58, 138, 0.14);
    border-color: #3f3f46;
    border-top-color: #60a5fa;
}

/* Market Data JSON — full width, aligned with section (no label-column indent) */
.market-data-json {
    width: 100% !important;
    border: 1px solid #e5e7eb;
    border-radius: var(--ea-radius-block);
    overflow: hidden;
    box-sizing: border-box;
}
.dark .market-data-json {
    border-color: #3f3f46;
}
.market-data-json > .wrap {
    padding: 0 !important;
    gap: 0 !important;
}
.market-data-json pre {
    margin: 0 !important;
    width: 100%;
    box-sizing: border-box;
    border-radius: 0 !important;
}

/* Accordion content scroll */
.accordion-scroll > .prose {
    max-height: 280px;
    overflow-y: auto;
    padding-right: 4px;
}

/* Tab active indicator */
.tabs > .tab-nav button.selected {
    border-bottom: 2px solid #387ED1 !important;
    color: #387ED1 !important;
    font-weight: 600;
}
.dark .tabs > .tab-nav button.selected {
    border-bottom-color: #60a5fa !important;
    color: #60a5fa !important;
}

/* Result / error cards */
.result-card {
    border-radius: var(--ea-radius-block);
    padding: 0.9rem 1.1rem;
    border-width: 1.5px;
    border-style: solid;
    background: #fff;
    margin-top: 0.5rem;
}
.dark .result-card { background: #27272a; }
.result-card .rc-title {
    font-weight: 600;
    font-size: 0.92rem;
    margin-bottom: 0.4rem;
    color: #1a1a1a;
}
.dark .result-card .rc-title { color: #f4f4f5; }
.result-card .rc-body {
    font-size: 0.9rem;
    color: #374151;
    line-height: 1.6;
}
.dark .result-card .rc-body { color: #d4d4d8; }

/* Result variants (Analysis tab HTML) — avoid inline light bg + .dark text */
.result-card.result-ok {
    background: #f0fdf4;
    border-color: #bbf7d0;
}
.result-card.result-ok .rc-title { color: #14532d; }
.result-card.result-ok .rc-body { color: #374151; }
.dark .result-card.result-ok {
    background: rgba(20, 83, 45, 0.42);
    border-color: #15803d;
}
.dark .result-card.result-ok .rc-title { color: #bbf7d0; }
.dark .result-card.result-ok .rc-body { color: #d4d4d8; }

.result-card.result-bad {
    background: #fff1f2;
    border-color: #fecaca;
}
.result-card.result-bad .rc-title { color: #991b1b; }
.result-card.result-bad .rc-body { color: #374151; }
.dark .result-card.result-bad {
    background: rgba(127, 29, 29, 0.38);
    border-color: #b91c1c;
}
.dark .result-card.result-bad .rc-title { color: #fecaca; }
.dark .result-card.result-bad .rc-body { color: #d4d4d8; }

.result-card.result-notice {
    background: #fffbeb;
    border-color: #fde68a;
}
.result-card.result-notice .rc-title { color: #92400e; }
.result-card.result-notice .rc-body { color: #374151; }
.dark .result-card.result-notice {
    background: rgba(120, 53, 15, 0.42);
    border-color: #d97706;
}
.dark .result-card.result-notice .rc-title { color: #fde68a; }
.dark .result-card.result-notice .rc-body { color: #d4d4d8; }

/* Sidebar section labels */
.section-label {
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #9ca3af;
    margin: 1rem 0 0.4rem;
}
.dark .section-label { color: #71717a; }
"""

# ---------------------------------------------------------------------------
# JS — dark / light / system toggle
# ---------------------------------------------------------------------------

theme_toggle_js = """
(() => {
    const KEY = "ea-theme";
    const PREF_INDEX = { system: 0, light: 1, dark: 2 };

    function wantsDark(pref) {
        if (pref === "dark") return true;
        if (pref === "light") return false;
        return window.matchMedia("(prefers-color-scheme: dark)").matches;
    }

    function getToggleRoot() {
        return document.querySelector(".theme-toggle");
    }

    function getToggleButtons() {
        const root = getToggleRoot();
        if (!root) return [];
        return [...root.querySelectorAll("button")];
    }

    function isButtonSelected(b) {
        return (
            b.classList.contains("selected") ||
            b.getAttribute("aria-checked") === "true" ||
            b.dataset.state === "checked"
        );
    }

    /** Gradio Radio value ↔ saved preference (fixes default "System" after reload). */
    function syncRadioToStorage() {
        const pref = localStorage.getItem(KEY) || "system";
        const want = PREF_INDEX[pref];
        if (want === undefined) return;
        const buttons = getToggleButtons();
        if (buttons.length < 3) return;
        const selected = buttons.findIndex(isButtonSelected);
        if (selected !== want) buttons[want].click();
    }

    /** Highlight effective Light/Dark when preference is system; match explicit light/dark. */
    function updateThemeToggleVisual() {
        const pref = localStorage.getItem(KEY) || "system";
        const buttons = getToggleButtons();
        if (buttons.length < 3) return;

        // Reset all buttons: remove class, clear highlight, and force radius to 0 via inline
        // !important (beats Gradio's Tailwind rounded-l-lg / rounded-r-lg cascade)
        buttons.forEach((b) => {
            b.classList.remove("ea-theme-active");
            b.style.removeProperty("background");
            b.style.removeProperty("color");
            b.style.removeProperty("border-color");
            ["border-radius",
             "border-top-left-radius", "border-top-right-radius",
             "border-bottom-left-radius", "border-bottom-right-radius",
            ].forEach((p) => b.style.setProperty(p, "0", "important"));
        });

        const resolvedDark = wantsDark(pref);
        let activeBtn = null;
        if (pref === "system") {
            activeBtn = resolvedDark ? buttons[2] : buttons[1];
        } else if (pref === "dark") {
            activeBtn = buttons[2];
        } else {
            activeBtn = buttons[1];
        }
        if (activeBtn) {
            activeBtn.classList.add("ea-theme-active");
            // Inline !important overrides Gradio's dark-mode stylesheet — no cascade battle
            activeBtn.style.setProperty("background", "#387ED1", "important");
            activeBtn.style.setProperty("color", "#fff", "important");
            activeBtn.style.setProperty("border-color", "#387ED1", "important");
        }
    }

    function applyTheme(pref) {
        const dark = wantsDark(pref);
        const html = document.documentElement;
        html.classList.toggle("dark", dark);
        if (document.body) {
            document.body.classList.toggle("dark", dark);
        }
        html.setAttribute("data-ea-theme", pref);
        html.setAttribute("data-ea-resolved", dark ? "dark" : "light");
        if (pref === "system") {
            html.style.removeProperty("color-scheme");
        } else {
            html.style.colorScheme = dark ? "dark" : "light";
        }
        queueMicrotask(() => {
            syncRadioToStorage();
            updateThemeToggleVisual();
        });
    }

    let toggleSyncScheduled = false;
    function scheduleToggleSync() {
        if (toggleSyncScheduled) return;
        toggleSyncScheduled = true;
        queueMicrotask(() => {
            toggleSyncScheduled = false;
            syncRadioToStorage();
            updateThemeToggleVisual();
        });
    }

    let themeToggleOuterObs = false;
    function attachThemeToggleObserver() {
        const hook = (root) => {
            if (!root || root.__eaToggleObserved) return;
            root.__eaToggleObserved = true;
            const mo = new MutationObserver(() => scheduleToggleSync());
            mo.observe(root, {
                childList: true,
                subtree: true,
                attributes: true,
                attributeFilter: ["class", "aria-checked", "data-state"],
            });
            scheduleToggleSync();
        };
        hook(getToggleRoot());
        if (themeToggleOuterObs) return;
        themeToggleOuterObs = true;
        const outer = new MutationObserver(() => {
            const r = getToggleRoot();
            if (r && !r.__eaToggleObserved) hook(r);
        });
        if (document.body) {
            outer.observe(document.body, { childList: true, subtree: true });
        }
    }

    function syncFromStorage() {
        applyTheme(localStorage.getItem(KEY) || "system");
    }

    function attachBodyObserver() {
        if (!document.body) return;
        new MutationObserver(() => {
            const pref = localStorage.getItem(KEY) || "system";
            const need = wantsDark(pref);
            if (document.body.classList.contains("dark") !== need) {
                applyTheme(pref);
            }
        }).observe(document.body, { attributes: true, attributeFilter: ["class"] });
    }

    syncFromStorage();
    if (document.body) {
        attachBodyObserver();
        attachThemeToggleObserver();
    } else {
        document.addEventListener("DOMContentLoaded", () => {
            attachBodyObserver();
            attachThemeToggleObserver();
        });
    }

    function beatGradioInit() {
        syncFromStorage();
        queueMicrotask(syncFromStorage);
        requestAnimationFrame(syncFromStorage);
    }
    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", beatGradioInit);
    } else {
        beatGradioInit();
    }
    window.addEventListener("load", () => {
        syncFromStorage();
        attachThemeToggleObserver();
        [0, 50, 200, 500, 1000].forEach((ms) => setTimeout(syncFromStorage, ms));
    });

    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
        if ((localStorage.getItem(KEY) || "system") === "system") {
            applyTheme("system");
        }
    });

    window.__setEATheme = (pref) => {
        localStorage.setItem(KEY, pref);
        applyTheme(pref);
    };
})();
"""

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _format_result_html(reward: float, ground_truth: Any) -> str:
    variant = "result-ok" if reward >= 0 else "result-bad"
    icon = "+" if reward >= 0 else ""
    gt = html.escape(str(ground_truth))
    return (
        f'<div class="result-card {variant}">'
        f'<div class="rc-title">Result {icon}</div>'
        f'<div class="rc-body">'
        f"<strong>Reward:</strong> {reward:.4f}<br/>"
        f"<strong>Ground truth:</strong> {gt}"
        f"</div></div>"
    )


def _format_error_html(message: str) -> str:
    msg = html.escape(message)
    return (
        '<div class="result-card result-notice">'
        '<div class="rc-title">Notice</div>'
        f'<div class="rc-body">{msg}</div></div>'
    )


def _episode_status_markdown(
    task_id: str,
    company: str | None = None,
    year: str | None = None,
    quarter: str | None = None,
) -> str:
    use_specific = bool(
        company and year and quarter and company != RANDOM_EPISODE_LABEL
    )
    if use_specific:
        assert company is not None and year is not None and quarter is not None
        idx = get_episode_index()
        sym = idx.symbol_for_display(company)
        yi = int(year)
        qi = int(quarter)
        eid = format_episode_id(sym, yi, qi)
        row_line = f"**Row:** `{html.escape(eid)}`  \n"
    else:
        row_line = "**Row:** random sample  \n"
    return (
        f"**Episode loaded** &mdash; `{html.escape(task_id)}`  \n"
        f"{row_line}"
        "Use **Observation** / **Analysis** tabs."
    )


def _reset_failed_outputs(message: str) -> list[Any]:
    texts, acc_updates = _text_rows_and_accordion_updates(None)
    err_status = f"**Episode error** &mdash; {html.escape(message)}"
    idle = "*Could not load episode. Fix the selection or try again.*"
    return [
        err_status,
        idle,
        *texts,
        *acc_updates,
        {},
        gr.update(visible=False),
        gr.update(visible=False, value=""),
        "",
        message,
    ]


def _init_episode_dropdowns() -> tuple[Any, Any, Any]:
    idx = get_episode_index()
    displays = idx.sorted_company_displays()
    choices = [RANDOM_EPISODE_LABEL] + displays
    return (
        gr.update(choices=choices, value=RANDOM_EPISODE_LABEL),
        gr.update(choices=[], value=None, interactive=False),
        gr.update(choices=[], value=None, interactive=False),
    )


def _on_company_change(company: str | None) -> tuple[Any, Any]:
    if not company or company == RANDOM_EPISODE_LABEL:
        return (
            gr.update(choices=[], value=None, interactive=False),
            gr.update(choices=[], value=None, interactive=False),
        )
    idx = get_episode_index()
    sym = idx.symbol_for_display(company)
    years = idx.years_for_symbol(sym)
    year_choices = [str(y) for y in years]
    y0 = year_choices[0] if year_choices else None
    if y0 is not None:
        qs = idx.quarters_for(sym, int(y0))
        q_choices = [str(q) for q in qs]
        q0 = q_choices[0] if q_choices else None
    else:
        q_choices, q0 = [], None
    return (
        gr.update(choices=year_choices, value=y0, interactive=bool(year_choices)),
        gr.update(choices=q_choices, value=q0, interactive=bool(q_choices)),
    )


def _on_year_change(company: str | None, year_str: str | None) -> Any:
    if not company or company == RANDOM_EPISODE_LABEL or not year_str:
        return gr.update(choices=[], value=None, interactive=False)
    idx = get_episode_index()
    sym = idx.symbol_for_display(company)
    qs = idx.quarters_for(sym, int(year_str))
    q_choices = [str(q) for q in qs]
    q0 = q_choices[0] if q_choices else None
    return gr.update(choices=q_choices, value=q0, interactive=bool(q_choices))


def _text_rows_and_accordion_updates(
    obs: Optional[EarningsAnalystObservation],
) -> tuple[list[str], list[dict[str, Any]]]:
    texts: list[str] = []
    updates: list[dict[str, Any]] = []
    ctx = obs.text_context if obs and obs.text_context else {}
    for key in TEXT_KEYS:
        raw = ctx.get(key, "") if ctx else ""
        texts.append(raw if isinstance(raw, str) else str(raw))
        updates.append(gr.update(visible=(key in ctx)))
    return texts, updates


# ---------------------------------------------------------------------------
# State & environment helpers
# ---------------------------------------------------------------------------


class State:
    def __init__(self) -> None:
        self.env: Optional[EarningsAnalystEnvironment] = None
        self.obs: Optional[EarningsAnalystObservation] = None
        self.task_id: str = "sentiment_label"


state = State()


async def reset_env(
    task_id: str,
    company: str | None,
    year: str | None,
    quarter: str | None,
) -> list[Any]:
    state.task_id = task_id
    state.env = EarningsAnalystEnvironment(task_id=task_id)
    try:
        use_specific = bool(
            company and year and quarter and company != RANDOM_EPISODE_LABEL
        )
        if use_specific:
            assert company is not None and year is not None and quarter is not None
            idx = get_episode_index()
            sym = idx.symbol_for_display(company)
            state.obs = state.env.reset(
                pick_symbol=sym,
                pick_year=int(year),
                pick_quarter=int(quarter),
            )
        else:
            state.obs = state.env.reset()
    except (ValueError, TaskNotImplementedError) as e:
        state.obs = None
        return _reset_failed_outputs(str(e))

    texts, acc_updates = _text_rows_and_accordion_updates(state.obs)

    numerical_value: Any = (
        state.obs.numerical_context if state.obs.numerical_context else {}
    )

    return [
        _episode_status_markdown(task_id, company, year, quarter),
        state.obs.task_instruction,
        *texts,
        *acc_updates,
        numerical_value,
        gr.update(visible=True),  # prediction_row
        gr.update(visible=False, value=""),  # result_view
        "",  # pred_input
        "",  # message_view
    ]


async def step_env(prediction: str) -> list[Any]:
    if not state.env or not state.obs:
        return [
            gr.update(),
            gr.update(
                visible=True,
                value=_format_error_html(
                    "Environment not initialized. Click New Episode."
                ),
            ),
        ]

    action = EarningsAnalystAction(prediction=prediction)
    state.obs = state.env.step(action)

    raw_reward = state.obs.reward
    reward_f = float(raw_reward) if raw_reward is not None else 0.0
    ground_truth = getattr(state.obs, "ground_truth", "N/A")

    return [
        gr.update(visible=False),
        gr.update(visible=True, value=_format_result_html(reward_f, ground_truth)),
    ]


async def run_agent(
    task_id: str,
    api_key: str,
    model: str,
    base_url: str,
    company: str | None,
    year: str | None,
    quarter: str | None,
) -> list[Any]:
    n_out = 17
    if not api_key:
        return [gr.update()] * (n_out - 1) + ["Please provide an API Key."]

    out = await reset_env(task_id, company, year, quarter)
    if state.obs is None:
        return out

    assert state.obs is not None

    if base_url:
        client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    else:
        client = AsyncOpenAI(api_key=api_key)

    user_content = f"{state.obs.task_instruction}\n\n"
    if state.obs.text_context:
        user_content += "## Text context\n"
        for name, text in sorted(state.obs.text_context.items()):
            user_content += f"### {name}\n{text}\n"
    if state.obs.numerical_context:
        user_content += (
            f"\n## Numerical context\n{json.dumps(state.obs.numerical_context)}\n"
        )

    system_prompt = (
        "You are a financial analyst assistant. "
        "Analyze the data and respond EXACTLY as instructed. "
        "Reply with a single JSON object containing 'prediction' key."
    )

    try:
        completion = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            response_format={"type": "json_object"},
        )
        response_text = completion.choices[0].message.content or "{}"
        parsed = json.loads(response_text)
        prediction = str(parsed.get("prediction", response_text))

        step_out = await step_env(prediction)

        return [
            *out[:13],
            step_out[0],
            step_out[1],
            prediction,
            f"Agent used {model}. Raw response: {response_text}",
        ]

    except Exception as e:
        return [*out[:13], out[13], out[14], "", f"Error: {str(e)}"]


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------


def _accordion_title(key: str) -> str:
    labels = {
        "earnings_transcript": "Earnings Call Transcript",
        "press_release_8k_body": "Press Release — 8-K Body",
        "press_release_ex991": "Exhibit 99.1",
        "press_release_ex992": "Exhibit 99.2",
        "press_release_sources": "Press Release Sources",
    }
    return labels.get(key, key.replace("_", " ").title())


# ---------------------------------------------------------------------------
# Gradio layout
# ---------------------------------------------------------------------------

with gr.Blocks(title="Earnings Analyst - OpenEnv") as demo:
    with gr.Column(elem_classes="app-shell"):
        # ── Header ──────────────────────────────────────────────────────────
        with gr.Row(elem_classes="header-row"):
            with gr.Column(scale=5, min_width=0):
                gr.Markdown("## Earnings Analyst")
                gr.Markdown(
                    "Interactive environment for financial analysis using "
                    "[OpenEnv](https://github.com/meta-pytorch/OpenEnv). "
                    "Evaluate agents or your own predictions on earnings call data."
                )
            with gr.Column(scale=2, min_width=0):
                theme_toggle = gr.Radio(
                    choices=["System", "Light", "Dark"],
                    value="System",
                    label="Theme",
                    elem_classes="theme-toggle",
                    container=False,
                )
                status_badge = gr.Markdown(
                    "No episode loaded. Click **New Episode**.",
                    elem_classes="status-badge",
                )

        # ── Main content ─────────────────────────────────────────────────────
        with gr.Row():
            # Sidebar
            with gr.Column(scale=1, min_width=220):
                with gr.Group():
                    company_dd = gr.Dropdown(
                        label="Company",
                        choices=[RANDOM_EPISODE_LABEL],
                        value=RANDOM_EPISODE_LABEL,
                    )
                    with gr.Row():
                        year_dd = gr.Dropdown(
                            label="Year",
                            choices=[],
                            interactive=False,
                        )
                        quarter_dd = gr.Dropdown(
                            label="Quarter",
                            choices=[],
                            interactive=False,
                        )

                with gr.Group():
                    task_select = gr.Dropdown(
                        choices=TASK_IDS,
                        value="sentiment_label",
                        label="Active Task",
                    )
                    reset_btn = gr.Button("New Episode", variant="primary")

                gr.Markdown("Auto-Agent", elem_classes="section-label")
                with gr.Group():
                    api_key = gr.Textbox(
                        label="OpenAI API Key",
                        type="password",
                        placeholder="sk-...",
                        value=os.environ.get("OPENAI_API_KEY", ""),
                    )
                    model_name = gr.Textbox(label="Model", value="gpt-4o-mini")
                    base_url = gr.Textbox(
                        label="Base URL (optional)",
                        placeholder="https://api.openai.com/v1",
                        value=os.environ.get("API_BASE_URL", ""),
                    )
                    agent_btn = gr.Button("Run LLM Agent", variant="secondary")

            # Main panel
            with gr.Column(scale=2):
                with gr.Column(elem_classes="task-instruction"):
                    instr_view = gr.Markdown("*No task loaded. Click **New Episode**.*")

                with gr.Tabs():
                    with gr.Tab("Observation"):
                        gr.Markdown("**Documents** — expand a section to read.")

                        with gr.Accordion(
                            _accordion_title(TEXT_KEYS[0]),
                            open=True,
                            visible=False,
                        ) as acc_earnings_transcript:
                            md_earnings_transcript = gr.Markdown(
                                "", elem_classes="accordion-scroll"
                            )

                        with gr.Accordion(
                            _accordion_title(TEXT_KEYS[1]),
                            open=False,
                            visible=False,
                        ) as acc_press_release_8k_body:
                            md_press_release_8k_body = gr.Markdown(
                                "", elem_classes="accordion-scroll"
                            )

                        with gr.Accordion(
                            _accordion_title(TEXT_KEYS[2]),
                            open=False,
                            visible=False,
                        ) as acc_press_release_ex991:
                            md_press_release_ex991 = gr.Markdown(
                                "", elem_classes="accordion-scroll"
                            )

                        with gr.Accordion(
                            _accordion_title(TEXT_KEYS[3]),
                            open=False,
                            visible=False,
                        ) as acc_press_release_ex992:
                            md_press_release_ex992 = gr.Markdown(
                                "", elem_classes="accordion-scroll"
                            )

                        with gr.Accordion(
                            _accordion_title(TEXT_KEYS[4]),
                            open=False,
                            visible=False,
                        ) as acc_press_release_sources:
                            md_press_release_sources = gr.Markdown(
                                "", elem_classes="accordion-scroll"
                            )

                        gr.Markdown("**Market Data**", elem_classes="section-label")
                        num_view = gr.JSON(
                            value={},
                            label=None,
                            show_label=False,
                            container=False,
                            elem_classes="market-data-json",
                        )

                    with gr.Tab("Analysis"):
                        with gr.Column(visible=False) as prediction_row:
                            pred_input = gr.Textbox(
                                label="Your prediction / analysis output",
                                placeholder="e.g. bullish, or 0.05",
                            )
                            submit_btn = gr.Button("Submit analysis", variant="primary")

                        result_view = gr.HTML(visible=False, value="")
                        message_view = gr.Textbox(
                            label="Agent log / messages",
                            interactive=False,
                            buttons=["copy"],
                            lines=5,
                            max_lines=12,
                            autoscroll=True,
                        )

    # ── Event outputs list ───────────────────────────────────────────────────
    reset_outputs = [
        status_badge,
        instr_view,
        md_earnings_transcript,
        md_press_release_8k_body,
        md_press_release_ex991,
        md_press_release_ex992,
        md_press_release_sources,
        acc_earnings_transcript,
        acc_press_release_8k_body,
        acc_press_release_ex991,
        acc_press_release_ex992,
        acc_press_release_sources,
        num_view,
        prediction_row,
        result_view,
        pred_input,
        message_view,
    ]

    demo.load(
        _init_episode_dropdowns,
        outputs=[company_dd, year_dd, quarter_dd],
    )
    company_dd.change(
        _on_company_change,
        inputs=[company_dd],
        outputs=[year_dd, quarter_dd],
    )
    year_dd.change(
        _on_year_change,
        inputs=[company_dd, year_dd],
        outputs=[quarter_dd],
    )

    reset_btn.click(
        fn=reset_env,
        inputs=[task_select, company_dd, year_dd, quarter_dd],
        outputs=reset_outputs,
    )
    submit_btn.click(
        fn=step_env, inputs=[pred_input], outputs=[prediction_row, result_view]
    )
    agent_btn.click(
        fn=run_agent,
        inputs=[
            task_select,
            api_key,
            model_name,
            base_url,
            company_dd,
            year_dd,
            quarter_dd,
        ],
        outputs=reset_outputs,
    )

    # Theme toggle — client-side only, no server round-trip
    theme_toggle.change(
        fn=None,
        inputs=[theme_toggle],
        js=(
            "(choice) => { "
            "const v = Array.isArray(choice) ? choice[0] : choice; "
            "const key = String(v ?? 'system').toLowerCase(); "
            "window.__setEATheme?.(key); "
            "return choice; "
            "}"
        ),
    )

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

api_app = FastAPI(title="Earnings Analyst API")

api_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@api_app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "environment": "earnings_analyst"}


@api_app.get("/web")
async def web_redirect() -> RedirectResponse:
    return RedirectResponse(url="/")


try:
    from earnings_analyst.server.app import app as env_app
except (ImportError, ModuleNotFoundError):
    from server.app import app as env_app

api_app.include_router(env_app.router)
api_app.mount("/api", env_app)

# Mount Gradio — theme, css, and js passed here (Gradio 6 API)
app = gr.mount_gradio_app(
    api_app,
    demo,
    path="/",
    theme=theme,
    css=custom_css,
    js=theme_toggle_js,
)


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
