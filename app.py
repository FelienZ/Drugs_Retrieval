#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import sys
import json
import webbrowser
import threading
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

from interactive_cli import (
    load_system,
    run_hqe,
    run_ntr,
    retrieve_bm25,
    reciprocal_rank_fusion,
    sigmoid,
    RRF_K
)

SYSTEM_STATE = {
    "ready": False,
    "status": "Loading models...",
    "error": None,
    "nlp": None,
    "passages": None,
    "passage_dict": None,
    "bm25": None,
    "t5_tokenizer": None,
    "t5_model": None,
    "cross_encoder": None,
    "device": "cpu",
    "drug_catalog": []
}

PREFIX_PATTERN = re.compile(r"^\[Drug:[^\]]*\]\s*", re.IGNORECASE)


def clean_passage_text(text):
    if not text:
        return ""
    return PREFIX_PATTERN.sub("", text).strip()


def make_snippet(text, max_len=140):
    cleaned = re.sub(r"\s+", " ", clean_passage_text(text)).strip()
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[:max_len].rstrip() + "..."


def normalize_name(name):
    return re.sub(r"\s+", " ", (name or "").strip()).lower()


def parse_bm25_candidates(raw_candidates):
    parsed = []
    for rank, entry in enumerate(raw_candidates, start=1):
        if isinstance(entry, str):
            pid, score = entry, None
        elif isinstance(entry, (tuple, list)) and len(entry) >= 2:
            pid, score = str(entry[0]), float(entry[1])
        elif isinstance(entry, dict):
            pid = str(entry.get("passage_id") or entry.get("pid") or "")
            raw_score = entry.get("score")
            score = float(raw_score) if raw_score is not None else None
        else:
            pid, score = str(entry), None
        if pid:
            parsed.append((rank, pid, score))
    return parsed


def build_drug_catalog(passage_dict):
    catalog = {}
    for item in passage_dict.values():
        brand = (item.get("brand_name") or "").strip()
        generic = (item.get("generic_name") or "").strip()

        if generic == "-":
            generic = ""
        if not brand and not generic:
            continue

        display_brand = brand or generic
        same_name = normalize_name(generic) == normalize_name(display_brand)
        clean_generic = "" if same_name else generic

        key = (normalize_name(display_brand), normalize_name(clean_generic))
        if key not in catalog:
            catalog[key] = {
                "brand_name": display_brand,
                "generic_name": clean_generic
            }

    drugs = list(catalog.values())
    drugs.sort(key=lambda d: d["brand_name"].lower())
    return drugs


def init_system_background():
    try:
        SYSTEM_STATE["status"] = "Loading drug corpus and retrieval models..."
        nlp, passages, passage_dict, bm25, t5_tokenizer, t5_model, cross_encoder, device = load_system()
        SYSTEM_STATE["nlp"] = nlp
        SYSTEM_STATE["passages"] = passages
        SYSTEM_STATE["passage_dict"] = passage_dict
        SYSTEM_STATE["bm25"] = bm25
        SYSTEM_STATE["t5_tokenizer"] = t5_tokenizer
        SYSTEM_STATE["t5_model"] = t5_model
        SYSTEM_STATE["cross_encoder"] = cross_encoder
        SYSTEM_STATE["device"] = device
        SYSTEM_STATE["drug_catalog"] = build_drug_catalog(passage_dict)
        SYSTEM_STATE["ready"] = True
        SYSTEM_STATE["status"] = "Ready"
        print("\n[OK] Drugs IR System ready at http://127.0.0.1:7860\n")
    except Exception as e:
        SYSTEM_STATE["error"] = str(e)
        SYSTEM_STATE["status"] = f"Initialization failed: {e}"
        print(f"[Error] Initialization failed: {e}", file=sys.stderr)


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Drugs IR System</title>
  <style>
    :root {
      --bg-page: #f8fafc;
      --surface: #ffffff;
      --surface-subtle: #f1f5f9;
      --border: #e2e8f0;
      --border-strong: #cbd5e1;
      --text-primary: #0f172a;
      --text-secondary: #334155;
      --text-muted: #64748b;
      --accent: #0284c7;
      --accent-hover: #0369a1;
      --accent-soft: #f0f9ff;
      --accent-border: #bae6fd;
      --ok: #047857;
      --ok-bg: #ecfdf5;
      --warn: #b45309;
      --warn-bg: #fffbeb;
      --warn-border: #fde68a;
      --down: #b91c1c;
      --radius-sm: 6px;
      --radius-md: 10px;
      --radius-lg: 14px;
      --font-sans: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      --font-mono: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }

    html, body {
      height: 100%;
    }

    body {
      font-family: var(--font-sans);
      background: var(--bg-page);
      color: var(--text-primary);
      font-size: 14px;
      line-height: 1.55;
      display: flex;
      flex-direction: column;
      -webkit-font-smoothing: antialiased;
    }

    header {
      background: var(--surface);
      border-bottom: 1px solid var(--border);
      padding: 12px 24px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      flex-shrink: 0;
      z-index: 10;
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 10px;
    }

    .brand-mark {
      width: 34px;
      height: 34px;
      border-radius: 9px;
      background: var(--accent-soft);
      border: 1px solid var(--accent-border);
      color: var(--accent);
      display: flex;
      align-items: center;
      justify-content: center;
      flex-shrink: 0;
    }

    .brand-name {
      font-size: 15px;
      font-weight: 700;
      letter-spacing: -0.01em;
      color: var(--text-primary);
    }

    .brand-meta {
      font-size: 12px;
      color: var(--text-muted);
    }

    .header-actions {
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .status-pill {
      display: inline-flex;
      align-items: center;
      gap: 7px;
      padding: 5px 10px;
      border-radius: 999px;
      border: 1px solid var(--border);
      background: var(--surface-subtle);
      color: var(--text-secondary);
      font-size: 12px;
      font-weight: 500;
    }

    .status-pill.hidden {
      display: none;
    }

    .status-dot {
      width: 7px;
      height: 7px;
      border-radius: 50%;
      background: #94a3b8;
    }

    .status-dot.error {
      background: #dc2626;
    }

    .btn-ghost {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 6px 12px;
      border-radius: var(--radius-sm);
      border: 1px solid var(--border);
      background: var(--surface);
      color: var(--text-secondary);
      font-size: 13px;
      font-weight: 500;
      cursor: pointer;
      transition: background 0.15s, border-color 0.15s, color 0.15s;
    }

    .btn-ghost:hover:not(:disabled) {
      background: var(--surface-subtle);
      border-color: var(--border-strong);
      color: var(--text-primary);
    }

    .btn-ghost:disabled {
      opacity: 0.5;
      cursor: not-allowed;
    }

    .chat-scroll {
      flex: 1;
      overflow-y: auto;
      padding: 28px 20px 32px;
    }

    .chat-container {
      max-width: 860px;
      margin: 0 auto;
      display: flex;
      flex-direction: column;
      gap: 22px;
    }

    .empty-view {
      margin: auto;
      padding: 64px 16px 32px;
      text-align: center;
      max-width: 520px;
    }

    .empty-icon {
      width: 48px;
      height: 48px;
      margin: 0 auto 16px;
      border-radius: 12px;
      background: var(--surface);
      border: 1px solid var(--border);
      color: var(--accent);
      display: flex;
      align-items: center;
      justify-content: center;
      box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }

    .empty-view h1 {
      font-size: 22px;
      font-weight: 700;
      letter-spacing: -0.02em;
      color: var(--text-primary);
      margin-bottom: 8px;
    }

    .empty-view p {
      font-size: 14px;
      color: var(--text-muted);
      line-height: 1.6;
    }

    .empty-action {
      margin-top: 18px;
    }

    .message-group {
      display: flex;
      flex-direction: column;
      gap: 12px;
    }

    .user-row {
      display: flex;
      justify-content: flex-end;
    }

    .user-bubble {
      background: var(--text-primary);
      color: #ffffff;
      padding: 10px 16px;
      border-radius: 16px 16px 4px 16px;
      max-width: 76%;
      font-size: 14px;
      line-height: 1.5;
      white-space: pre-wrap;
      word-break: break-word;
    }

    .bot-row {
      display: flex;
      align-items: flex-start;
      gap: 10px;
    }

    .bot-avatar {
      width: 30px;
      height: 30px;
      border-radius: 8px;
      background: var(--accent-soft);
      border: 1px solid var(--accent-border);
      color: var(--accent);
      display: flex;
      align-items: center;
      justify-content: center;
      flex-shrink: 0;
      margin-top: 2px;
    }

    .bot-bubble {
      flex: 1;
      min-width: 0;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 4px 16px 16px 16px;
      padding: 16px 18px;
      box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03);
      display: flex;
      flex-direction: column;
      gap: 12px;
    }

    .bot-bubble.oot {
      border-color: var(--warn-border);
    }

    .answer-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      padding-bottom: 10px;
      border-bottom: 1px solid var(--border);
    }

    .drug-identity {
      display: flex;
      flex-direction: column;
      gap: 2px;
      min-width: 0;
    }

    .drug-brand {
      font-size: 16px;
      font-weight: 700;
      color: var(--text-primary);
      letter-spacing: -0.01em;
    }

    .drug-generic {
      font-size: 12.5px;
      color: var(--text-muted);
    }

    .btn-copy {
      flex-shrink: 0;
      padding: 4px 10px;
      border-radius: var(--radius-sm);
      border: 1px solid var(--border);
      background: var(--surface);
      color: var(--text-muted);
      font-size: 12px;
      font-weight: 500;
      cursor: pointer;
      transition: all 0.15s;
    }

    .btn-copy:hover {
      background: var(--surface-subtle);
      color: var(--text-primary);
    }

    .passage-text {
      font-size: 14px;
      line-height: 1.7;
      color: var(--text-secondary);
      white-space: pre-wrap;
      word-break: break-word;
    }

    .oot-header {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 14px;
      font-weight: 600;
      color: var(--warn);
    }

    .oot-body {
      font-size: 14px;
      color: var(--text-secondary);
      line-height: 1.6;
    }

    /* Search Details & Multi-Stage Pipeline Inspector */
    .details-toggle {
      padding-top: 10px;
      border-top: 1px solid var(--border);
    }

    .details-toggle summary {
      font-size: 12px;
      font-weight: 500;
      color: var(--text-muted);
      cursor: pointer;
      user-select: none;
      list-style: none;
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }

    .details-toggle summary::-webkit-details-marker {
      display: none;
    }

    .details-toggle summary:hover {
      color: var(--text-primary);
    }

    .details-toggle summary::before {
      content: "";
      width: 6px;
      height: 6px;
      border-right: 1.5px solid currentColor;
      border-bottom: 1.5px solid currentColor;
      transform: rotate(-45deg);
      transition: transform 0.15s ease;
    }

    .details-toggle[open] summary::before {
      transform: rotate(45deg);
    }

    .details-body {
      margin-top: 12px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }

    .stage-box {
      padding: 12px;
      background: var(--bg-page);
      border: 1px solid var(--border);
      border-radius: var(--radius-sm);
      display: flex;
      flex-direction: column;
      gap: 8px;
    }

    .stage-title {
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      color: var(--text-muted);
    }

    .detail-row {
      display: grid;
      grid-template-columns: 155px 1fr;
      gap: 8px;
      font-size: 12px;
    }

    .detail-key {
      color: var(--text-muted);
      font-weight: 600;
    }

    .detail-val {
      color: var(--text-secondary);
      font-family: var(--font-mono);
      word-break: break-word;
    }

    .metrics-grid {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 8px;
    }

    .metric-cell {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius-sm);
      padding: 8px 10px;
    }

    .metric-label {
      font-size: 10.5px;
      color: var(--text-muted);
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.03em;
    }

    .metric-value {
      font-size: 13px;
      font-weight: 700;
      font-family: var(--font-mono);
      color: var(--text-primary);
      margin-top: 2px;
    }

    .gate-ok {
      color: var(--ok);
    }

    .gate-oot {
      color: var(--warn);
    }

    .pool-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 8px;
    }

    .pool-tabs {
      display: inline-flex;
      flex-wrap: wrap;
      gap: 4px;
      background: var(--surface-subtle);
      padding: 3px;
      border-radius: var(--radius-sm);
      border: 1px solid var(--border);
    }

    .pool-tab {
      border: none;
      background: transparent;
      color: var(--text-muted);
      font-size: 11.5px;
      font-weight: 500;
      padding: 4px 9px;
      border-radius: 4px;
      cursor: pointer;
      transition: all 0.12s;
    }

    .pool-tab:hover {
      color: var(--text-primary);
    }

    .pool-tab.active {
      background: var(--surface);
      color: var(--text-primary);
      font-weight: 600;
      box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06);
    }

    .pool-table-wrap {
      max-height: 340px;
      overflow: auto;
      border: 1px solid var(--border);
      border-radius: var(--radius-sm);
      background: var(--surface);
    }

    .pool-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 11.5px;
      font-variant-numeric: tabular-nums;
    }

    .pool-table th {
      position: sticky;
      top: 0;
      z-index: 1;
      text-align: left;
      padding: 7px 9px;
      background: var(--surface-subtle);
      color: var(--text-muted);
      font-weight: 600;
      border-bottom: 1px solid var(--border);
      white-space: nowrap;
    }

    .pool-table td {
      padding: 7px 9px;
      border-bottom: 1px solid var(--border);
      color: var(--text-secondary);
      vertical-align: top;
    }

    .pool-table tr:last-child td {
      border-bottom: none;
    }

    .pool-table tr.is-top1 td {
      background: var(--accent-soft);
    }

    .col-mono {
      font-family: var(--font-mono);
      white-space: nowrap;
    }

    .col-right {
      text-align: right;
    }

    .cell-drug {
      font-weight: 600;
      color: var(--text-primary);
    }

    .cell-sec {
      font-size: 10.5px;
      color: var(--text-muted);
    }

    .cell-snip {
      font-size: 11px;
      color: var(--text-muted);
      margin-top: 2px;
      line-height: 1.4;
      max-width: 300px;
    }

    .shift-up {
      color: var(--ok);
      font-weight: 600;
    }

    .shift-down {
      color: var(--down);
      font-weight: 600;
    }

    .shift-same {
      color: var(--text-muted);
    }

    .loading-bubble {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 4px 16px 16px 16px;
      padding: 12px 16px;
      display: inline-flex;
      align-items: center;
      gap: 10px;
      color: var(--text-muted);
      font-size: 13.5px;
    }

    .spinner {
      width: 15px;
      height: 15px;
      border: 2px solid var(--border-strong);
      border-top-color: var(--accent);
      border-radius: 50%;
      animation: spin 0.75s linear infinite;
    }

    @keyframes spin {
      to { transform: rotate(360deg); }
    }

    .composer-wrap {
      background: var(--surface);
      border-top: 1px solid var(--border);
      padding: 14px 20px 18px;
      flex-shrink: 0;
    }

    .composer-inner {
      max-width: 860px;
      margin: 0 auto;
    }

    .composer-box {
      display: flex;
      align-items: flex-end;
      gap: 10px;
      background: var(--bg-page);
      border: 1px solid var(--border-strong);
      border-radius: var(--radius-lg);
      padding: 8px 10px 8px 14px;
      transition: border-color 0.15s, box-shadow 0.15s;
    }

    .composer-box:focus-within {
      border-color: var(--accent);
      background: var(--surface);
      box-shadow: 0 0 0 3px rgba(2, 132, 199, 0.12);
    }

    textarea {
      flex: 1;
      border: none;
      background: transparent;
      resize: none;
      outline: none;
      font-family: var(--font-sans);
      font-size: 14px;
      line-height: 1.5;
      color: var(--text-primary);
      max-height: 160px;
      min-height: 24px;
      padding: 4px 0;
    }

    textarea::placeholder {
      color: var(--text-muted);
    }

    .btn-send {
      width: 36px;
      height: 36px;
      border-radius: var(--radius-md);
      border: none;
      background: var(--accent);
      color: #ffffff;
      display: flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      flex-shrink: 0;
      transition: background 0.15s;
    }

    .btn-send:hover:not(:disabled) {
      background: var(--accent-hover);
    }

    .btn-send:disabled {
      background: var(--border-strong);
      cursor: not-allowed;
    }

    .composer-footer {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-top: 8px;
      padding: 0 4px;
      font-size: 11.5px;
      color: var(--text-muted);
    }

    .drawer-backdrop {
      position: fixed;
      inset: 0;
      background: rgba(15, 23, 42, 0.32);
      z-index: 40;
      opacity: 0;
      pointer-events: none;
      transition: opacity 0.2s ease;
    }

    .drawer-backdrop.open {
      opacity: 1;
      pointer-events: auto;
    }

    .drawer {
      position: fixed;
      top: 0;
      right: 0;
      bottom: 0;
      width: 400px;
      max-width: 100vw;
      background: var(--surface);
      border-left: 1px solid var(--border);
      z-index: 50;
      display: flex;
      flex-direction: column;
      transform: translateX(100%);
      transition: transform 0.22s ease;
      box-shadow: -8px 0 24px rgba(15, 23, 42, 0.08);
    }

    .drawer.open {
      transform: translateX(0);
    }

    .drawer-header {
      padding: 16px 20px;
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }

    .drawer-title {
      font-size: 15px;
      font-weight: 700;
      color: var(--text-primary);
    }

    .drawer-subtitle {
      font-size: 12px;
      color: var(--text-muted);
      margin-top: 2px;
    }

    .btn-icon-close {
      width: 30px;
      height: 30px;
      border-radius: var(--radius-sm);
      border: 1px solid var(--border);
      background: var(--surface);
      color: var(--text-secondary);
      display: flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
    }

    .btn-icon-close:hover {
      background: var(--surface-subtle);
      color: var(--text-primary);
    }

    .drawer-controls {
      padding: 12px 16px;
      border-bottom: 1px solid var(--border);
      background: var(--bg-page);
    }

    .drawer-controls input {
      width: 100%;
      padding: 8px 12px;
      border-radius: var(--radius-sm);
      border: 1px solid var(--border-strong);
      background: var(--surface);
      font-family: var(--font-sans);
      font-size: 13px;
      color: var(--text-primary);
      outline: none;
    }

    .drawer-controls input:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 2px rgba(2, 132, 199, 0.14);
    }

    .drawer-list {
      flex: 1;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
    }

    .drug-row {
      width: 100%;
      text-align: left;
      padding: 11px 18px;
      border: none;
      border-bottom: 1px solid var(--border);
      background: var(--surface);
      cursor: pointer;
      display: flex;
      flex-direction: column;
      gap: 2px;
      transition: background 0.12s;
    }

    .drug-row:hover {
      background: var(--bg-page);
    }

    .drug-row-brand {
      font-size: 13.5px;
      font-weight: 600;
      color: var(--text-primary);
      line-height: 1.35;
      display: -webkit-box;
      -webkit-line-clamp: 2;
      -webkit-box-orient: vertical;
      overflow: hidden;
    }

    .drug-row-generic {
      font-size: 12px;
      color: var(--text-muted);
      line-height: 1.35;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .drawer-empty {
      text-align: center;
      padding: 36px 16px;
      color: var(--text-muted);
      font-size: 13px;
    }

    .btn-load-more {
      margin: 14px auto;
      padding: 7px 14px;
      border-radius: var(--radius-sm);
      border: 1px solid var(--border);
      background: var(--surface);
      color: var(--text-secondary);
      font-size: 12.5px;
      font-weight: 500;
      cursor: pointer;
    }

    .btn-load-more:hover {
      background: var(--surface-subtle);
      color: var(--text-primary);
    }

    @media (max-width: 720px) {
      header {
        padding: 12px 16px;
      }
      .brand-meta, .composer-footer {
        display: none;
      }
      .detail-row {
        grid-template-columns: 1fr;
        gap: 2px;
      }
      .metrics-grid {
        grid-template-columns: repeat(2, 1fr);
      }
      .user-bubble {
        max-width: 88%;
      }
    }
  </style>
</head>
<body>

  <header>
    <div class="brand">
      <div class="brand-mark" aria-hidden="true">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="m10.5 20.5 10-10a4.95 4.95 0 1 0-7-7l-10 10a4.95 4.95 0 1 0 7 7Z"/>
          <path d="m8.5 8.5 7 7"/>
        </svg>
      </div>
      <div>
        <div class="brand-name">Drugs IR System</div>
        <div class="brand-meta">Conversational Drug Search</div>
      </div>
    </div>

    <div class="header-actions">
      <div class="status-pill" id="status-pill">
        <span class="status-dot" id="status-dot"></span>
        <span id="status-text">Starting up...</span>
      </div>
      <button class="btn-ghost" id="btn-drugs" disabled>
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1 0-5H20"/>
        </svg>
        <span>Drug Directory</span>
      </button>
      <button class="btn-ghost" id="btn-new-chat" disabled>New Chat</button>
    </div>
  </header>

  <main class="chat-scroll" id="chat-scroll">
    <div class="chat-container" id="chat-container">
      <div class="empty-view" id="empty-view">
        <div class="empty-icon" aria-hidden="true">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <circle cx="11" cy="11" r="8"/>
            <path d="m21 21-4.3-4.3"/>
          </svg>
        </div>
        <h1>Ask about any medication</h1>
        <p>Look up indications, dosing, contraindications, warnings, and adverse reactions from openFDA drug labels. Follow-up questions automatically keep track of your conversation.</p>
        <div class="empty-action">
          <button class="btn-ghost" id="btn-browse-empty" disabled>Browse available drugs</button>
        </div>
      </div>
    </div>
  </main>

  <footer class="composer-wrap">
    <div class="composer-inner">
      <div class="composer-box">
        <textarea
          id="query-input"
          rows="1"
          placeholder="Ask about a medication..."
          disabled
        ></textarea>
        <button class="btn-send" id="btn-send" title="Send" disabled>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
            <path d="m22 2-7 20-4-9-9-4Z"/>
            <path d="M22 2 11 13"/>
          </svg>
        </button>
      </div>
      <div class="composer-footer">
        <span id="context-indicator">No active conversation context</span>
        <span>Press Enter to send, Shift+Enter for new line</span>
      </div>
    </div>
  </footer>

  <div class="drawer-backdrop" id="drawer-backdrop"></div>
  <aside class="drawer" id="drug-drawer" aria-hidden="true">
    <div class="drawer-header">
      <div>
        <div class="drawer-title">Drug Directory</div>
        <div class="drawer-subtitle" id="drawer-count">Available medications</div>
      </div>
      <button class="btn-icon-close" id="btn-close-drawer" aria-label="Close">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M18 6 6 18"/>
          <path d="m6 6 12 12"/>
        </svg>
      </button>
    </div>
    <div class="drawer-controls">
      <input type="text" id="drug-filter-input" placeholder="Search brand or generic name..." autocomplete="off">
    </div>
    <div class="drawer-list" id="drug-list">
      <div class="drawer-empty">Loading medications...</div>
    </div>
  </aside>

  <script>
    let systemReady = false;
    let isProcessing = false;
    let sessionHistory = [];
    let turnIndex = 1;
    let copyStore = new Map();
    let detailStore = new Map();
    let msgCounter = 0;

    let allDrugs = [];
    let filteredDrugs = [];
    let renderedLimit = 100;

    const BOT_ICON_SVG = `
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="m10.5 20.5 10-10a4.95 4.95 0 1 0-7-7l-10 10a4.95 4.95 0 1 0 7 7Z"/>
        <path d="m8.5 8.5 7 7"/>
      </svg>
    `;

    const chatScroll = document.getElementById('chat-scroll');
    const chatContainer = document.getElementById('chat-container');
    const emptyView = document.getElementById('empty-view');
    const queryInput = document.getElementById('query-input');
    const btnSend = document.getElementById('btn-send');
    const btnNewChat = document.getElementById('btn-new-chat');
    const btnDrugs = document.getElementById('btn-drugs');
    const btnBrowseEmpty = document.getElementById('btn-browse-empty');
    const statusPill = document.getElementById('status-pill');
    const statusDot = document.getElementById('status-dot');
    const statusText = document.getElementById('status-text');
    const contextIndicator = document.getElementById('context-indicator');

    const drawer = document.getElementById('drug-drawer');
    const drawerBackdrop = document.getElementById('drawer-backdrop');
    const btnCloseDrawer = document.getElementById('btn-close-drawer');
    const drugFilterInput = document.getElementById('drug-filter-input');
    const drugListEl = document.getElementById('drug-list');
    const drawerCountEl = document.getElementById('drawer-count');

    function escapeHtml(str) {
      if (!str) return '';
      return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
    }

    function normalizeKey(str) {
      return String(str || '').trim().replace(/\s+/g, ' ').toLowerCase();
    }

    function toTitleCase(str) {
      if (!str) return '';
      return String(str)
        .toLowerCase()
        .replace(/\b\w/g, (c) => c.toUpperCase());
    }

    function formatSection(raw) {
      if (!raw) return 'General';
      return toTitleCase(String(raw).replace(/_/g, ' '));
    }

    function formatBrandDisplay(name) {
      if (!name) return '';
      const trimmed = String(name).trim();
      if (trimmed === trimmed.toLowerCase()) {
        return toTitleCase(trimmed);
      }
      return trimmed;
    }

    function formatGenericDisplay(name) {
      if (!name || name === '-') return '';
      return toTitleCase(String(name).trim());
    }

    function formatShift(shift) {
      if (shift > 0) return `<span class="shift-up">+${shift}</span>`;
      if (shift < 0) return `<span class="shift-down">${shift}</span>`;
      return `<span class="shift-same">0</span>`;
    }

    function autoResize() {
      queryInput.style.height = 'auto';
      queryInput.style.height = Math.min(queryInput.scrollHeight, 160) + 'px';
    }

    function scrollToBottom() {
      chatScroll.scrollTop = chatScroll.scrollHeight;
    }

    function updateContextIndicator() {
      if (sessionHistory.length === 0) {
        contextIndicator.textContent = 'No active conversation context';
        btnNewChat.disabled = true;
      } else {
        const count = sessionHistory.length;
        contextIndicator.textContent = `Active context: ${count} ${count === 1 ? 'turn' : 'turns'}`;
        btnNewChat.disabled = false;
      }
    }

    async function loadDrugCatalog() {
      try {
        const res = await fetch('/api/drugs');
        if (!res.ok) return;
        const data = await res.json();
        allDrugs = data.drugs || [];
        applyDrugFilter();
      } catch (err) {
        drugListEl.innerHTML = '<div class="drawer-empty">Unable to load drug list.</div>';
      }
    }

    function applyDrugFilter() {
      const term = drugFilterInput.value.trim().toLowerCase();
      if (!term) {
        filteredDrugs = allDrugs;
      } else {
        filteredDrugs = allDrugs.filter((d) =>
          d.brand_name.toLowerCase().includes(term) ||
          (d.generic_name && d.generic_name.toLowerCase().includes(term))
        );
      }
      renderedLimit = 100;
      renderDrugList();
    }

    function renderDrugList() {
      const total = filteredDrugs.length;
      const query = drugFilterInput.value.trim();
      drawerCountEl.textContent = query
        ? `${total.toLocaleString()} matching medications`
        : `${total.toLocaleString()} medications available`;

      if (total === 0) {
        drugListEl.innerHTML = '<div class="drawer-empty">No matching medications found.</div>';
        return;
      }

      const slice = filteredDrugs.slice(0, renderedLimit);
      let html = slice.map((d) => {
        const displayBrand = formatBrandDisplay(d.brand_name);
        const displayGeneric = formatGenericDisplay(d.generic_name);
        const showGeneric = displayGeneric && normalizeKey(displayGeneric) !== normalizeKey(displayBrand);

        return `
          <button type="button" class="drug-row" data-drug-name="${escapeHtml(d.brand_name)}">
            <span class="drug-row-brand">${escapeHtml(displayBrand)}</span>
            ${showGeneric ? `<span class="drug-row-generic">${escapeHtml(displayGeneric)}</span>` : ''}
          </button>
        `;
      }).join('');

      if (total > renderedLimit) {
        html += `<button type="button" class="btn-load-more" id="btn-load-more">Show more (${total - renderedLimit} remaining)</button>`;
      }

      drugListEl.innerHTML = html;
    }

    function openDrawer() {
      drawer.classList.add('open');
      drawerBackdrop.classList.add('open');
      drawer.setAttribute('aria-hidden', 'false');
      if (allDrugs.length === 0 && systemReady) {
        loadDrugCatalog();
      }
      setTimeout(() => drugFilterInput.focus(), 180);
    }

    function closeDrawer() {
      drawer.classList.remove('open');
      drawerBackdrop.classList.remove('open');
      drawer.setAttribute('aria-hidden', 'true');
    }

    async function pollStatus() {
      try {
        const res = await fetch('/api/status');
        const data = await res.json();

        if (data.error) {
          statusPill.classList.remove('hidden');
          statusDot.className = 'status-dot error';
          statusText.textContent = 'Load failed';
          return;
        }

        if (data.ready) {
          systemReady = true;
          statusPill.classList.add('hidden');
          queryInput.disabled = false;
          btnSend.disabled = false;
          btnDrugs.disabled = false;
          btnBrowseEmpty.disabled = false;
          loadDrugCatalog();
          queryInput.focus();
          return;
        }

        statusText.textContent = data.status || 'Loading...';
        setTimeout(pollStatus, 2000);
      } catch (err) {
        statusText.textContent = 'Connecting...';
        setTimeout(pollStatus, 2500);
      }
    }

    function renderPoolTable(record, activeTab) {
      const p = record.data.pipeline || {};
      const isOot = Boolean(record.data.is_oot);

      if (activeTab === 'reranked' || activeTab === 'rrf') {
        const baseList = p.reranked_pool || [];
        if (baseList.length === 0) {
          return '<div class="drawer-empty">No candidates retrieved.</div>';
        }
        const rows = activeTab === 'rrf'
          ? [...baseList].sort((a, b) => a.rrf_rank - b.rrf_rank)
          : baseList;

        const bodyHtml = rows.map((item) => {
          const isTop1 = !isOot && item.final_rank === 1;
          const hqeText = item.hqe_rank ? `#${item.hqe_rank}` : '-';
          const ntrText = item.ntr_rank ? `#${item.ntr_rank}` : '-';
          const primaryRank = activeTab === 'rrf' ? `#${item.rrf_rank}` : `#${item.final_rank}`;
          const secondaryRank = activeTab === 'rrf' ? `#${item.final_rank}` : `#${item.rrf_rank}`;

          return `
            <tr class="${isTop1 ? 'is-top1' : ''}">
              <td class="col-mono"><strong>${primaryRank}</strong></td>
              <td class="col-mono">${secondaryRank}</td>
              <td class="col-mono">${formatShift(item.rank_shift)}</td>
              <td class="col-mono">${escapeHtml(item.passage_id)}</td>
              <td>
                <div class="cell-drug">${escapeHtml(formatBrandDisplay(item.brand_name))}</div>
                <div class="cell-sec">${escapeHtml(formatSection(item.section))}</div>
                <div class="cell-snip">${escapeHtml(item.snippet || '')}</div>
              </td>
              <td class="col-mono">${hqeText} / ${ntrText}</td>
              <td class="col-mono col-right">${item.rrf_score.toFixed(4)}</td>
              <td class="col-mono col-right">
                <div>${item.ce_score.toFixed(3)}</div>
                <div class="cell-sec">${(item.ce_prob * 100).toFixed(1)}%</div>
              </td>
            </tr>
          `;
        }).join('');

        return `
          <table class="pool-table">
            <thead>
              <tr>
                <th>${activeTab === 'rrf' ? 'RRF' : 'CE Rank'}</th>
                <th>${activeTab === 'rrf' ? 'CE Rank' : 'RRF'}</th>
                <th>Shift</th>
                <th>Passage ID</th>
                <th>Drug & Passage Preview</th>
                <th>HQE / NTR</th>
                <th class="col-right">RRF Score</th>
                <th class="col-right">CE Logit / Prob</th>
              </tr>
            </thead>
            <tbody>${bodyHtml}</tbody>
          </table>
        `;
      }

      const bm25List = activeTab === 'hqe' ? (p.bm25_hqe || []) : (p.bm25_ntr || []);
      if (bm25List.length === 0) {
        return '<div class="drawer-empty">No BM25 candidates retrieved.</div>';
      }

      const hasScores = bm25List.some((x) => typeof x.bm25_score === 'number');
      const bodyHtml = bm25List.map((item) => {
        const isTop1 = !isOot && item.final_rank === 1;
        return `
          <tr class="${isTop1 ? 'is-top1' : ''}">
            <td class="col-mono"><strong>#${item.rank}</strong></td>
            <td class="col-mono">${escapeHtml(item.passage_id)}</td>
            <td>
              <div class="cell-drug">${escapeHtml(formatBrandDisplay(item.brand_name))}</div>
              <div class="cell-sec">${escapeHtml(formatSection(item.section))}</div>
              <div class="cell-snip">${escapeHtml(item.snippet || '')}</div>
            </td>
            ${hasScores ? `<td class="col-mono col-right">${typeof item.bm25_score === 'number' ? item.bm25_score.toFixed(3) : '-'}</td>` : ''}
            <td class="col-mono col-right">${item.rrf_rank ? `#${item.rrf_rank}` : '-'}</td>
            <td class="col-mono col-right">${item.final_rank ? `#${item.final_rank}` : '-'}</td>
          </tr>
        `;
      }).join('');

      return `
        <table class="pool-table">
          <thead>
            <tr>
              <th>BM25 Rank</th>
              <th>Passage ID</th>
              <th>Drug & Passage Preview</th>
              ${hasScores ? '<th class="col-right">BM25 Score</th>' : ''}
              <th class="col-right">RRF Rank</th>
              <th class="col-right">Final CE</th>
            </tr>
          </thead>
          <tbody>${bodyHtml}</tbody>
        </table>
      `;
    }

    function renderDetailsBody(msgId) {
      const record = detailStore.get(msgId);
      if (!record) return '';
      const data = record.data;
      const p = data.pipeline || {};
      const activeTab = record.activeTab || 'reranked';

      const statusLabel = data.is_oot
        ? 'Rejected (Out of Topic, history preserved)'
        : 'Accepted (In Domain, added to history)';
      const statusClass = data.is_oot ? 'gate-oot' : 'gate-ok';
      const probText = typeof p.top1_prob === 'number' ? `${(p.top1_prob * 100).toFixed(2)}%` : '-';
      const logitText = typeof p.top1_score === 'number' ? p.top1_score.toFixed(4) : '-';
      const aiText = typeof p.a_i === 'number' ? p.a_i.toFixed(2) : '-';
      const ruleText = p.turn === 1
        ? 'Turn 1: BM25 QPP (A_i) >= 4.00 and P >= 18.00%'
        : 'Turn > 1: P >= 18.00% & In-Domain Entity Verification';
      const historyText = (record.history && record.history.length > 0)
        ? record.history.join(' ||| ')
        : '(none)';

      const rerankedCount = (p.reranked_pool || []).length;
      const hqeCount = (p.bm25_hqe || []).length;
      const ntrCount = (p.bm25_ntr || []).length;

      return `
        <div class="stage-box">
          <div class="stage-title">1. Parallel Conversational Query Reformulation (CQR)</div>
          <div class="detail-row">
            <span class="detail-key">Turn & Input</span>
            <span class="detail-val">Turn ${p.turn ?? '-'} | ${escapeHtml(record.query)}</span>
          </div>
          <div class="detail-row">
            <span class="detail-key">Prior Context</span>
            <span class="detail-val">${escapeHtml(historyText)}</span>
          </div>
          <div class="detail-row">
            <span class="detail-key">HQE (+POS) Query</span>
            <span class="detail-val">${escapeHtml(p.q_hqe || '-')}</span>
          </div>
          <div class="detail-row">
            <span class="detail-key">NTR (T5-CANARD)</span>
            <span class="detail-val">${escapeHtml(p.q_ntr || '-')}</span>
          </div>
        </div>

        <div class="stage-box">
          <div class="stage-title">2. Retrieval, Early Fusion (RRF k=${p.rrf_k ?? 60}) & Gatekeeper</div>
          <div class="metrics-grid">
            <div class="metric-cell">
              <div class="metric-label">BM25 QPP (A_i)</div>
              <div class="metric-value">${aiText}</div>
            </div>
            <div class="metric-cell">
              <div class="metric-label">HQE / NTR Overlap</div>
              <div class="metric-value">${p.overlap_count ?? 0} / ${rerankedCount}</div>
            </div>
            <div class="metric-cell">
              <div class="metric-label">Top-1 CE Logit</div>
              <div class="metric-value">${logitText}</div>
            </div>
            <div class="metric-cell">
              <div class="metric-label">Sigmoid Prob (P)</div>
              <div class="metric-value">${probText}</div>
            </div>
          </div>
          <div class="detail-row">
            <span class="detail-key">Gate Rule</span>
            <span class="detail-val">${escapeHtml(ruleText)}</span>
          </div>
          <div class="detail-row">
            <span class="detail-key">Gate Decision</span>
            <span class="detail-val ${statusClass}">${escapeHtml(statusLabel)}</span>
          </div>
        </div>

        <div class="stage-box">
          <div class="pool-header">
            <div class="stage-title">3. Candidate Pool & Cross-Encoder Re-ranking (${rerankedCount} Passages)</div>
            <div class="pool-tabs">
              <button type="button" class="pool-tab ${activeTab === 'reranked' ? 'active' : ''}" data-detail-msg="${msgId}" data-pool-tab="reranked">Re-ranked (${rerankedCount})</button>
              <button type="button" class="pool-tab ${activeTab === 'rrf' ? 'active' : ''}" data-detail-msg="${msgId}" data-pool-tab="rrf">RRF Fusion (${rerankedCount})</button>
              <button type="button" class="pool-tab ${activeTab === 'hqe' ? 'active' : ''}" data-detail-msg="${msgId}" data-pool-tab="hqe">BM25 HQE (${hqeCount})</button>
              <button type="button" class="pool-tab ${activeTab === 'ntr' ? 'active' : ''}" data-detail-msg="${msgId}" data-pool-tab="ntr">BM25 NTR (${ntrCount})</button>
            </div>
          </div>
          <div class="pool-table-wrap">
            ${renderPoolTable(record, activeTab)}
          </div>
        </div>
      `;
    }

    function renderBotReply(msgId, data, query, usedHistory) {
      detailStore.set(msgId, {
        data,
        query,
        history: usedHistory,
        activeTab: 'reranked',
        rendered: false
      });

      const detailsShell = `
        <details class="details-toggle" data-detail-msg="${msgId}">
          <summary>Search details</summary>
          <div class="details-body" id="details-body-${msgId}"></div>
        </details>
      `;

      if (data.is_oot || !data.top_passage) {
        return `
          <div class="bot-row">
            <div class="bot-avatar" aria-hidden="true">${BOT_ICON_SVG}</div>
            <div class="bot-bubble oot">
              <div class="oot-header">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                  <circle cx="12" cy="12" r="10"/>
                  <line x1="12" y1="8" x2="12" y2="12"/>
                  <line x1="12" y1="16" x2="12.01" y2="16"/>
                </svg>
                <span>No matching drug information found</span>
              </div>
              <div class="oot-body">
                This query does not appear to match any medication in the drug label database. Open the Drug Directory in the top bar to see available medications.
              </div>
              ${detailsShell}
            </div>
          </div>
        `;
      }

      const item = data.top_passage;
      const brand = formatBrandDisplay(item.brand_name) || 'Unknown Drug';
      const generic = formatGenericDisplay(item.generic_name);
      const showGeneric = generic && normalizeKey(generic) !== normalizeKey(brand);
      const passage = item.passage_text || '';

      copyStore.set(msgId, `${brand}\n\n${passage}`);

      return `
        <div class="bot-row">
          <div class="bot-avatar" aria-hidden="true">${BOT_ICON_SVG}</div>
          <div class="bot-bubble">
            <div class="answer-header">
              <div class="drug-identity">
                <div class="drug-brand">${escapeHtml(brand)}</div>
                ${showGeneric ? `<div class="drug-generic">${escapeHtml(generic)}</div>` : ''}
              </div>
              <button type="button" class="btn-copy" data-copy-msg="${msgId}">Copy</button>
            </div>
            <div class="passage-text">${escapeHtml(passage)}</div>
            ${detailsShell}
          </div>
        </div>
      `;
    }

    async function handleSend() {
      const text = queryInput.value.trim();
      if (!text || !systemReady || isProcessing) return;

      isProcessing = true;
      queryInput.value = '';
      autoResize();
      queryInput.disabled = true;
      btnSend.disabled = true;

      if (emptyView) {
        emptyView.style.display = 'none';
      }

      const historySnapshot = [...sessionHistory];
      const msgId = `msg-${++msgCounter}`;
      const group = document.createElement('div');
      group.className = 'message-group';
      group.id = msgId;
      group.innerHTML = `
        <div class="user-row">
          <div class="user-bubble">${escapeHtml(text)}</div>
        </div>
        <div class="answer-slot">
          <div class="bot-row">
            <div class="bot-avatar" aria-hidden="true">${BOT_ICON_SVG}</div>
            <div class="loading-bubble">
              <span class="spinner"></span>
              <span>Searching drug labels...</span>
            </div>
          </div>
        </div>
      `;
      chatContainer.appendChild(group);
      scrollToBottom();

      try {
        const res = await fetch('/api/query', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            utterance: text,
            history: historySnapshot,
            turn_idx: turnIndex
          })
        });

        const data = await res.json();
        const slot = group.querySelector('.answer-slot');

        if (!res.ok || data.error) {
          slot.innerHTML = `
            <div class="bot-row">
              <div class="bot-avatar" aria-hidden="true">${BOT_ICON_SVG}</div>
              <div class="bot-bubble oot">
                <div class="oot-header"><span>Request failed</span></div>
                <div class="oot-body">${escapeHtml(data.error || 'Unable to process your query.')}</div>
              </div>
            </div>
          `;
        } else {
          slot.innerHTML = renderBotReply(msgId, data, text, historySnapshot);

          if (!data.is_oot) {
            sessionHistory.push(text);
            turnIndex += 1;
            updateContextIndicator();
          }
        }
      } catch (err) {
        const slot = group.querySelector('.answer-slot');
        slot.innerHTML = `
          <div class="bot-row">
            <div class="bot-avatar" aria-hidden="true">${BOT_ICON_SVG}</div>
            <div class="bot-bubble oot">
              <div class="oot-header"><span>Connection error</span></div>
              <div class="oot-body">${escapeHtml(err.message)}</div>
            </div>
          </div>
        `;
      } finally {
        isProcessing = false;
        queryInput.disabled = false;
        btnSend.disabled = false;
        queryInput.focus();
        scrollToBottom();
      }
    }

    function resetChat() {
      sessionHistory = [];
      turnIndex = 1;
      copyStore.clear();
      detailStore.clear();
      chatContainer.querySelectorAll('.message-group').forEach((el) => el.remove());
      if (emptyView) {
        emptyView.style.display = 'block';
      }
      updateContextIndicator();
      queryInput.focus();
    }

    chatContainer.addEventListener('toggle', (e) => {
      const detailsEl = e.target;
      if (!detailsEl.matches || !detailsEl.matches('.details-toggle') || !detailsEl.open) {
        return;
      }
      const msgId = detailsEl.dataset.detailMsg;
      const record = detailStore.get(msgId);
      if (record && !record.rendered) {
        const bodyEl = document.getElementById(`details-body-${msgId}`);
        if (bodyEl) {
          bodyEl.innerHTML = renderDetailsBody(msgId);
          record.rendered = true;
        }
      }
    }, true);

    chatContainer.addEventListener('click', (e) => {
      const copyBtn = e.target.closest('.btn-copy');
      if (copyBtn) {
        const msgId = copyBtn.dataset.copyMsg;
        const textToCopy = copyStore.get(msgId);
        if (textToCopy) {
          navigator.clipboard.writeText(textToCopy).then(() => {
            copyBtn.textContent = 'Copied';
            setTimeout(() => { copyBtn.textContent = 'Copy'; }, 1500);
          });
        }
        return;
      }

      const poolTab = e.target.closest('.pool-tab');
      if (poolTab) {
        const msgId = poolTab.dataset.detailMsg;
        const nextTab = poolTab.dataset.poolTab;
        const record = detailStore.get(msgId);
        if (record && record.activeTab !== nextTab) {
          record.activeTab = nextTab;
          const bodyEl = document.getElementById(`details-body-${msgId}`);
          if (bodyEl) {
            bodyEl.innerHTML = renderDetailsBody(msgId);
          }
        }
      }
    });

    drugFilterInput.addEventListener('input', applyDrugFilter);

    drugListEl.addEventListener('click', (e) => {
      const loadMore = e.target.closest('#btn-load-more');
      if (loadMore) {
        renderedLimit += 100;
        renderDrugList();
        return;
      }

      const row = e.target.closest('.drug-row');
      if (row) {
        const drugName = row.dataset.drugName || '';
        const current = queryInput.value.trim();
        queryInput.value = current ? `${current} ${drugName} ` : `${drugName} `;
        autoResize();
        closeDrawer();
        queryInput.focus();
      }
    });

    btnDrugs.addEventListener('click', openDrawer);
    btnBrowseEmpty.addEventListener('click', openDrawer);
    btnCloseDrawer.addEventListener('click', closeDrawer);
    drawerBackdrop.addEventListener('click', closeDrawer);

    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && drawer.classList.contains('open')) {
        closeDrawer();
      }
    });

    queryInput.addEventListener('input', autoResize);
    queryInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    });

    btnSend.addEventListener('click', handleSend);
    btnNewChat.addEventListener('click', resetChat);

    pollStatus();
  </script>
</body>
</html>
"""


class ClinicalAppHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        sys.stderr.write(f"[HTTP] {self.address_string()} - {format % args}\n")

    def _send_json(self, status_code, payload):
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode("utf-8"))

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
        elif parsed.path == "/api/status":
            self._send_json(200, {
                "ready": SYSTEM_STATE["ready"],
                "status": SYSTEM_STATE["status"],
                "error": SYSTEM_STATE["error"],
                "device": SYSTEM_STATE["device"]
            })
        elif parsed.path == "/api/drugs":
            if not SYSTEM_STATE["ready"]:
                self._send_json(503, {"error": "System is still loading."})
                return
            self._send_json(200, {
                "drugs": SYSTEM_STATE["drug_catalog"]
            })
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/api/query":
            self.send_response(404)
            self.end_headers()
            return

        if not SYSTEM_STATE["ready"]:
            self._send_json(503, {"error": "System is still loading models. Please try again shortly."})
            return

        try:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8")
            req_data = json.loads(body)
            utterance = req_data.get("utterance", "").strip()
            history = req_data.get("history", [])
            turn_idx = int(req_data.get("turn_idx", 1))

            if not utterance:
                self._send_json(400, {"error": "Query cannot be empty."})
                return

            nlp = SYSTEM_STATE["nlp"]
            passages = SYSTEM_STATE["passages"]
            passage_dict = SYSTEM_STATE["passage_dict"]
            bm25 = SYSTEM_STATE["bm25"]
            t5_tok = SYSTEM_STATE["t5_tokenizer"]
            t5_model = SYSTEM_STATE["t5_model"]
            ce = SYSTEM_STATE["cross_encoder"]
            device = SYSTEM_STATE["device"]

            q_hqe, a_i = run_hqe(nlp, bm25, history, utterance)
            q_ntr = run_ntr(t5_tok, t5_model, device, history, utterance)

            cand_hqe = retrieve_bm25(bm25, passages, q_hqe, top_n=50)
            cand_ntr = retrieve_bm25(bm25, passages, q_ntr, top_n=50)

            parsed_hqe = parse_bm25_candidates(cand_hqe)
            parsed_ntr = parse_bm25_candidates(cand_ntr)
            hqe_rank_map = {pid: rank for rank, pid, _ in parsed_hqe}
            ntr_rank_map = {pid: rank for rank, pid, _ in parsed_ntr}
            overlap_count = len(set(hqe_rank_map.keys()) & set(ntr_rank_map.keys()))

            fused = reciprocal_rank_fusion(cand_hqe, cand_ntr, k_const=RRF_K)[:50]
            rrf_rank_map = {pid: idx + 1 for idx, pid in enumerate(fused)}

            if not fused:
                self._send_json(200, {
                    "is_oot": True,
                    "top_passage": None,
                    "pipeline": {
                        "turn": turn_idx,
                        "a_i": float(a_i),
                        "q_hqe": q_hqe,
                        "q_ntr": q_ntr,
                        "rrf_k": RRF_K,
                        "rrf_pool_size": 0,
                        "overlap_count": overlap_count,
                        "top1_score": 0.0,
                        "top1_prob": 0.0,
                        "is_oot": True,
                        "reranked_pool": [],
                        "bm25_hqe": [],
                        "bm25_ntr": []
                    }
                })
                return

            ce_pairs = [[q_ntr, passage_dict[pid]["passage_text"]] for pid in fused]
            ce_scores = ce.predict(ce_pairs)
            sorted_idx = sorted(range(len(ce_scores)), key=lambda k: ce_scores[k], reverse=True)

            final_rank_map = {}
            reranked_pool = []
            for final_rank, k in enumerate(sorted_idx, start=1):
                pid = fused[k]
                rrf_rank = k + 1
                final_rank_map[pid] = final_rank
                item = passage_dict.get(pid, {})
                r_hqe = hqe_rank_map.get(pid)
                r_ntr = ntr_rank_map.get(pid)
                rrf_score = (1.0 / (RRF_K + r_hqe) if r_hqe else 0.0) + (
                    1.0 / (RRF_K + r_ntr) if r_ntr else 0.0
                )
                ce_score = float(ce_scores[k])
                reranked_pool.append({
                    "final_rank": final_rank,
                    "rrf_rank": rrf_rank,
                    "rank_shift": rrf_rank - final_rank,
                    "hqe_rank": r_hqe,
                    "ntr_rank": r_ntr,
                    "rrf_score": rrf_score,
                    "ce_score": ce_score,
                    "ce_prob": float(sigmoid(ce_score)),
                    "passage_id": pid,
                    "brand_name": item.get("brand_name", "Unknown Drug"),
                    "section": item.get("section", "general"),
                    "snippet": make_snippet(item.get("passage_text", ""))
                })

            bm25_hqe_detail = []
            for rank, pid, score in parsed_hqe:
                item = passage_dict.get(pid, {})
                bm25_hqe_detail.append({
                    "rank": rank,
                    "passage_id": pid,
                    "brand_name": item.get("brand_name", "Unknown Drug"),
                    "section": item.get("section", "general"),
                    "snippet": make_snippet(item.get("passage_text", "")),
                    "bm25_score": score,
                    "rrf_rank": rrf_rank_map.get(pid),
                    "final_rank": final_rank_map.get(pid)
                })

            bm25_ntr_detail = []
            for rank, pid, score in parsed_ntr:
                item = passage_dict.get(pid, {})
                bm25_ntr_detail.append({
                    "rank": rank,
                    "passage_id": pid,
                    "brand_name": item.get("brand_name", "Unknown Drug"),
                    "section": item.get("section", "general"),
                    "snippet": make_snippet(item.get("passage_text", "")),
                    "bm25_score": score,
                    "rrf_rank": rrf_rank_map.get(pid),
                    "final_rank": final_rank_map.get(pid)
                })

            top1_idx = sorted_idx[0]
            top1_score = float(ce_scores[top1_idx])
            top1_prob = float(sigmoid(top1_score))
            top1_pid = fused[top1_idx]
            top1_p = passage_dict[top1_pid]

            doc_curr = nlp(utterance)
            content_tokens = [
                t.text.lower() for t in doc_curr 
                if not t.is_stop and t.is_alpha and len(t.text) > 1
            ]

            if content_tokens:
                content_bm25_scores = [
                    float(max(bm25.get_scores([t]))) if len(bm25.get_scores([t])) > 0 else 0.0
                    for t in content_tokens
                ]
                max_content_bm25 = max(content_bm25_scores)
                zero_bm25_tokens = [
                    t for t, s in zip(content_tokens, content_bm25_scores) if s <= 0.0
                ]
            else:
                max_content_bm25 = 0.0
                zero_bm25_tokens = []

            q_ntr_lower = q_ntr.lower()
            dropped_zero_tokens = [t for t in zero_bm25_tokens if t not in q_ntr_lower]

            if turn_idx == 1:
                is_oot = (a_i < 4.0) or (top1_prob < 0.18) or (len(zero_bm25_tokens) == len(content_tokens) and len(content_tokens) > 0)
            else:
                has_only_zero_bm25 = (len(zero_bm25_tokens) > 0 and len(zero_bm25_tokens) == len(content_tokens))
                has_degenerate_collapse = (len(dropped_zero_tokens) > 0 and (max_content_bm25 < 3.0))
                is_oot = (top1_prob < 0.18) or has_only_zero_bm25 or has_degenerate_collapse

            self._send_json(200, {
                "is_oot": is_oot,
                "top_passage": None if is_oot else {
                    "passage_id": top1_pid,
                    "brand_name": top1_p.get("brand_name", "Unknown Drug"),
                    "generic_name": top1_p.get("generic_name", "-"),
                    "section": top1_p.get("section", "general"),
                    "passage_text": clean_passage_text(top1_p.get("passage_text", ""))
                },
                "pipeline": {
                    "turn": turn_idx,
                    "a_i": float(a_i),
                    "q_hqe": q_hqe,
                    "q_ntr": q_ntr,
                    "rrf_k": RRF_K,
                    "rrf_pool_size": len(fused),
                    "overlap_count": overlap_count,
                    "top1_score": top1_score,
                    "top1_prob": top1_prob,
                    "is_oot": is_oot,
                    "reranked_pool": reranked_pool,
                    "bm25_hqe": bm25_hqe_detail,
                    "bm25_ntr": bm25_ntr_detail
                }
            })
        except Exception as e:
            self._send_json(500, {"error": str(e)})


def run_server(host="127.0.0.1", port=7860):
    server = ThreadingHTTPServer((host, port), ClinicalAppHandler)
    url = f"http://{host}:{port}"
    print(f"Drugs IR System running at {url}")

    init_thread = threading.Thread(target=init_system_background, daemon=True)
    init_thread.start()

    try:
        webbrowser.open(url)
    except Exception:
        pass

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        server.server_close()


if __name__ == "__main__":
    host = "127.0.0.1"
    port = 7860
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass
    run_server(host=host, port=port)