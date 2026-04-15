#!/usr/bin/env python3
"""
AWS-ready reconstructed v5.2 pipeline for:
LLM-Assisted RAG for Domain-Specific QA over Canadian Corporate Tax Fact Documents

Important note:
This script was reconstructed from the observed v5.2 artifact bundle
(chunks.jsonl, chunk_embeddings.npy, evaluation_results.csv, run_config.json, etc.).
It preserves the observed model stack and scoring weights where possible, but some
internal helper logic and prompt text had to be rebuilt because the original source
notebook / .py code was not included in the upload.

What this script does well:
- Runs the observed v5.2 retrieval stack on AWS-friendly paths
- Supports local disk or S3-backed artifact storage
- Preserves the observed embedding model, reranker, LLM, and hybrid weights
- Provides a notebook-friendly class API plus CLI entry points
- Reproduces the shape of the v5.2 outputs (ask / evaluate)

What this script does not fully reconstruct:
- The original PDF ingestion and chunk-construction notebook logic
- Every exact-mode heuristic used in the original v5.2 source

Recommended AWS runtime:
- EC2 GPU instance (for local Qwen 4-bit generation)
- or SageMaker AI JupyterLab GPU space
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import logging
import math
import os
import pickle
import random
import re
import shutil
import sys
import textwrap
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None

if load_dotenv:
    load_dotenv()

LOGGER = logging.getLogger("kpmg_tax_rag_v52_aws")
if not LOGGER.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

RECONSTRUCTION_NOTICE = (
    "This AWS-ready v5.2 code was reconstructed from the observed artifact bundle. "
    "It preserves the observed retrieval stack and defaults, but some internal helper "
    "logic had to be rebuilt because the original source notebook was not included."
)

DEFAULT_OUTPUT_SUBDIR = "kpmg_tax_rag_outputs_v52_corporate_50q"
DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_LLM = "Qwen/Qwen2.5-1.5B-Instruct"

PERCENT_RE = re.compile(r"(?<!\d)(?:\(?\d{1,3}(?:,\d{3})*(?:\.\d+)?\)?%|\d{1,3}(?:\.\d+)?)")
MONEY_RE = re.compile(r"\$\s?\d{1,3}(?:,\d{3})*(?:\.\d+)?")
YEAR_RE = re.compile(r"\b20\d{2}\b")
DAY_RE = re.compile(
    r"\b(?:first|second|third|fourth|fifth|\d{1,2}(?:st|nd|rd|th)?)\s+(?:working\s+day|day)\b",
    flags=re.IGNORECASE,
)
MONTHS_AFTER_YEAR_END_RE = re.compile(r"\b(?:two|three|six|\d+)\s+months?\s+after\s+year-?end\b", re.I)
MONTHLY_DUE_RE = re.compile(r"\b15th day of the following month\b", re.I)
QUARTERLY_DUE_RE = re.compile(r"\b15th day of the month following the end of each calendar quarter\b", re.I)
WEEKLY_DUE_RE = re.compile(r"\bthird working day after the end of each weekly period\b", re.I)
FREQUENCY_RE = re.compile(r"\b(monthly|quarterly|weekly)\b", re.I)

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "for", "to", "in", "on", "by", "is", "are", "what", "when",
    "how", "do", "does", "with", "after", "up", "from", "earned", "income", "tax", "rate", "rates",
    "corporate", "corporation", "corporations", "federal", "provincial", "territorial", "combined",
}

JURISDICTIONS = [
    "british columbia", "alberta", "saskatchewan", "manitoba", "ontario", "quebec",
    "new brunswick", "nova scotia", "prince edward island", "newfoundland and labrador",
    "northwest territories", "nunavut", "yukon", "federal",
]

ENTITY_HINTS = {
    "ccpc": ["ccpc", "canadian-controlled private corporation", "qualifying ccpc", "eligible ccpc"],
    "general corporation": ["general corporation", "general corporations"],
    "banks": ["banks", "bank", "loan or trust corporations", "trading in securities"],
    "credit unions": ["credit and savings unions", "credit union", "credit unions"],
}

QUESTION_TYPE_HINTS = {
    "deadline": ["deadline", "due date", "due", "when are", "when is", "filing deadline", "balance of taxes payable"],
    "frequency": ["how often", "monthly", "quarterly", "weekly", "instalments", "installments", "remittances"],
    "threshold": ["threshold", "limit", "above the small business limit", "expenditure limit", "after-tax cost"],
    "rate": ["rate", "%", "tax rate", "combined tax rate", "investment tax credit"],
}

PREFER_TYPES_BY_QUESTION_TYPE = {
    "rate": ["row", "table", "text", "heading"],
    "threshold": ["row", "table", "text", "heading"],
    "deadline": ["text", "table", "heading", "row"],
    "frequency": ["text", "table", "heading", "row"],
    "unknown": ["text", "table", "row", "heading"],
}


def bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def str_env(name: str, default: str = "") -> str:
    raw = os.getenv(name)
    return raw if raw not in {None, ""} else default


def int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def safe_parse_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            return parsed if isinstance(parsed, list) else [parsed]
        except Exception:
            text = value.strip()
            if not text:
                return []
            return [text]
    return [value]


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def normalize_for_match(text: str) -> str:
    text = (text or "").lower()
    text = text.replace("—", "-").replace("–", "-")
    return normalize_whitespace(text)


def tokenize(text: str) -> List[str]:
    text = normalize_for_match(text)
    return re.findall(r"[a-z0-9%$\.\-]+", text)


def minmax(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    vmin = float(values.min())
    vmax = float(values.max())
    if abs(vmax - vmin) < 1e-12:
        return np.full_like(values, 0.5, dtype=np.float32)
    return ((values - vmin) / (vmax - vmin)).astype(np.float32)


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def cosine_sim_matrix(query: np.ndarray, docs: np.ndarray) -> np.ndarray:
    query = query.astype(np.float32)
    docs = docs.astype(np.float32)
    qn = np.linalg.norm(query)
    dn = np.linalg.norm(docs, axis=1)
    qn = qn if qn > 0 else 1.0
    dn = np.where(dn == 0, 1.0, dn)
    return (docs @ query) / (dn * qn)


class SimpleBM25:
    """Tiny BM25 implementation to avoid extra package dependencies."""

    def __init__(self, tokenized_corpus: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(tokenized_corpus)
        self.doc_len = np.array([len(doc) for doc in tokenized_corpus], dtype=np.float32)
        self.avgdl = float(self.doc_len.mean()) if self.corpus_size else 0.0
        self.term_freqs: List[Dict[str, int]] = []
        df: Dict[str, int] = {}

        for doc in tokenized_corpus:
            tf: Dict[str, int] = {}
            for term in doc:
                tf[term] = tf.get(term, 0) + 1
            self.term_freqs.append(tf)
            for term in tf:
                df[term] = df.get(term, 0) + 1

        self.idf = {
            term: math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)
            for term, freq in df.items()
        }

    def get_scores(self, query_tokens: Sequence[str]) -> np.ndarray:
        scores = np.zeros(self.corpus_size, dtype=np.float32)
        if self.corpus_size == 0:
            return scores
        for i, tf in enumerate(self.term_freqs):
            dl = self.doc_len[i]
            denom_const = self.k1 * (1.0 - self.b + self.b * dl / max(self.avgdl, 1e-9))
            score = 0.0
            for term in query_tokens:
                f = tf.get(term, 0)
                if f <= 0:
                    continue
                idf = self.idf.get(term, 0.0)
                denom = f + denom_const
                score += idf * (f * (self.k1 + 1.0)) / max(denom, 1e-9)
            scores[i] = score
        return scores


@dataclass
class Settings:
    base_dir: Path = field(default_factory=lambda: Path(str_env("BASE_DIR", os.getcwd())).expanduser())
    output_subdir: str = field(default_factory=lambda: str_env("OUTPUT_SUBDIR", DEFAULT_OUTPUT_SUBDIR))
    artifacts_dir: Path = field(init=False)
    data_dir: Path = field(init=False)
    raw_dir: Path = field(init=False)
    outputs_dir: Path = field(init=False)
    logs_dir: Path = field(init=False)
    run_config_path: Optional[Path] = None
    bundle_zip_path: Optional[Path] = field(default_factory=lambda: Path(str_env("BUNDLE_ZIP_PATH", "")).expanduser() if str_env("BUNDLE_ZIP_PATH", "") else None)
    s3_bucket: str = field(default_factory=lambda: str_env("S3_BUCKET", ""))
    s3_prefix: str = field(default_factory=lambda: str_env("S3_PREFIX", "kpmg-tax-rag-v52"))
    aws_region: str = field(default_factory=lambda: str_env("AWS_REGION", "us-east-1"))

    pdf_filename: Optional[Path] = None
    downloaded_pdf_name: str = field(default_factory=lambda: str_env("DOWNLOADED_PDF_NAME", "kpmg_tax_facts_public.pdf"))

    embed_model_name: str = field(default_factory=lambda: str_env("EMBED_MODEL_NAME", DEFAULT_EMBED_MODEL))
    reranker_model_name: str = field(default_factory=lambda: str_env("RERANKER_MODEL_NAME", DEFAULT_RERANKER))
    llm_model_name: str = field(default_factory=lambda: str_env("LLM_MODEL_NAME", DEFAULT_LLM))
    llm_load_in_4bit: bool = field(default_factory=lambda: bool_env("LLM_LOAD_IN_4BIT", True))

    dense_weight: float = field(default_factory=lambda: float_env("DENSE_WEIGHT", 0.44))
    bm25_weight: float = field(default_factory=lambda: float_env("BM25_WEIGHT", 0.18))
    metadata_weight: float = field(default_factory=lambda: float_env("METADATA_WEIGHT", 0.23))
    rerank_weight: float = field(default_factory=lambda: float_env("RERANK_WEIGHT", 0.15))
    dense_top_k: int = field(default_factory=lambda: int_env("DENSE_TOP_K", 28))
    bm25_top_k: int = field(default_factory=lambda: int_env("BM25_TOP_K", 28))
    rerank_top_k: int = field(default_factory=lambda: int_env("RERANK_TOP_K", 14))
    final_top_k: int = field(default_factory=lambda: int_env("FINAL_TOP_K", 6))

    use_reranker: bool = field(default_factory=lambda: bool_env("USE_RERANKER", True))
    use_llm_planner: bool = field(default_factory=lambda: bool_env("USE_LLM_PLANNER", True))
    use_llm_verifier: bool = field(default_factory=lambda: bool_env("USE_LLM_VERIFIER", True))
    use_llm_answer: bool = field(default_factory=lambda: bool_env("USE_LLM_ANSWER", True))

    llm_max_input_tokens: int = field(default_factory=lambda: int_env("LLM_MAX_INPUT_TOKENS", 4096))
    planner_max_new_tokens: int = field(default_factory=lambda: int_env("PLANNER_MAX_NEW_TOKENS", 160))
    answer_max_new_tokens: int = field(default_factory=lambda: int_env("ANSWER_MAX_NEW_TOKENS", 220))
    verifier_max_new_tokens: int = field(default_factory=lambda: int_env("VERIFIER_MAX_NEW_TOKENS", 160))
    planner_do_sample: bool = field(default_factory=lambda: bool_env("PLANNER_DO_SAMPLE", False))
    answer_do_sample: bool = field(default_factory=lambda: bool_env("ANSWER_DO_SAMPLE", False))
    verifier_do_sample: bool = field(default_factory=lambda: bool_env("VERIFIER_DO_SAMPLE", False))
    max_candidate_values: int = field(default_factory=lambda: int_env("MAX_CANDIDATE_VALUES", 48))
    seed: int = field(default_factory=lambda: int_env("SEED", 42))

    hf_home: Path = field(default_factory=lambda: Path(str_env("HF_HOME", str(Path.home() / ".cache" / "huggingface"))).expanduser())
    device: str = field(default_factory=lambda: str_env("DEVICE", "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"))

    def __post_init__(self) -> None:
        self.base_dir = Path(self.base_dir).expanduser()
        self.data_dir = self.base_dir / "data"
        self.raw_dir = self.data_dir / "raw"
        self.outputs_dir = self.base_dir / "outputs"
        self.logs_dir = self.base_dir / "logs"
        self.artifacts_dir = self.base_dir / "artifacts" / self.output_subdir
        if self.pdf_filename is None:
            self.pdf_filename = self.raw_dir / self.downloaded_pdf_name
        if self.run_config_path is None:
            default_cfg = self.artifacts_dir / "run_config.json"
            self.run_config_path = default_cfg if default_cfg.exists() else None
        for path in [self.base_dir, self.data_dir, self.raw_dir, self.outputs_dir, self.logs_dir, self.artifacts_dir, self.hf_home]:
            path.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(self.hf_home))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(self.hf_home / "transformers"))
        os.environ.setdefault("HF_HUB_CACHE", str(self.hf_home / "hub"))
        random.seed(self.seed)
        np.random.seed(self.seed)
        if torch is not None:
            torch.manual_seed(self.seed)

    def apply_run_config(self, cfg: Dict[str, Any]) -> None:
        if not cfg:
            return
        self.embed_model_name = cfg.get("embed_model_name", self.embed_model_name)
        self.reranker_model_name = cfg.get("reranker_model_name", self.reranker_model_name)
        self.llm_model_name = cfg.get("llm_model_name", self.llm_model_name)
        self.llm_load_in_4bit = bool(cfg.get("llm_load_in_4bit", self.llm_load_in_4bit))
        self.dense_weight = float(cfg.get("dense_weight", self.dense_weight))
        self.bm25_weight = float(cfg.get("bm25_weight", self.bm25_weight))
        self.metadata_weight = float(cfg.get("metadata_weight", self.metadata_weight))
        self.rerank_weight = float(cfg.get("rerank_weight", self.rerank_weight))
        self.dense_top_k = int(cfg.get("dense_top_k", self.dense_top_k))
        self.bm25_top_k = int(cfg.get("bm25_top_k", self.bm25_top_k))
        self.rerank_top_k = int(cfg.get("rerank_top_k", self.rerank_top_k))
        self.final_top_k = int(cfg.get("final_top_k", self.final_top_k))
        self.use_reranker = bool(cfg.get("use_reranker", self.use_reranker))
        self.use_llm_planner = bool(cfg.get("use_llm_planner", self.use_llm_planner))
        self.use_llm_verifier = bool(cfg.get("use_llm_verifier", self.use_llm_verifier))
        self.llm_max_input_tokens = int(cfg.get("llm_max_input_tokens", self.llm_max_input_tokens))
        self.planner_max_new_tokens = int(cfg.get("planner_max_new_tokens", self.planner_max_new_tokens))
        self.answer_max_new_tokens = int(cfg.get("answer_max_new_tokens", self.answer_max_new_tokens))
        self.verifier_max_new_tokens = int(cfg.get("verifier_max_new_tokens", self.verifier_max_new_tokens))
        self.planner_do_sample = bool(cfg.get("planner_do_sample", self.planner_do_sample))
        self.answer_do_sample = bool(cfg.get("answer_do_sample", self.answer_do_sample))
        self.verifier_do_sample = bool(cfg.get("verifier_do_sample", self.verifier_do_sample))
        self.max_candidate_values = int(cfg.get("max_candidate_values", self.max_candidate_values))
        self.seed = int(cfg.get("seed", self.seed))
        if cfg.get("downloaded_pdf_name"):
            self.downloaded_pdf_name = cfg["downloaded_pdf_name"]
            self.pdf_filename = self.raw_dir / self.downloaded_pdf_name
        if cfg.get("pdf_filename"):
            self.pdf_filename = Path(cfg["pdf_filename"]).expanduser()


class S3Helper:
    def __init__(self, bucket: str, region: str = "us-east-1") -> None:
        self.bucket = bucket
        self.region = region
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import boto3  # lazy import
            self._client = boto3.client("s3", region_name=self.region)
        return self._client

    def exists(self) -> bool:
        if not self.bucket:
            return False
        try:
            self.client.head_bucket(Bucket=self.bucket)
            return True
        except Exception:
            return False

    def download_prefix(self, prefix: str, local_dir: Path) -> int:
        local_dir.mkdir(parents=True, exist_ok=True)
        paginator = self.client.get_paginator("list_objects_v2")
        count = 0
        prefix = prefix.rstrip("/") + "/"
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith("/"):
                    continue
                rel = key[len(prefix):]
                target = local_dir / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                self.client.download_file(self.bucket, key, str(target))
                count += 1
        return count

    def upload_dir(self, local_dir: Path, prefix: str) -> int:
        count = 0
        prefix = prefix.rstrip("/")
        for path in local_dir.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(local_dir).as_posix()
            key = f"{prefix}/{rel}" if prefix else rel
            self.client.upload_file(str(path), self.bucket, key)
            count += 1
        return count


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def strip_code_fences(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_\-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_first_json_object(text: str) -> Dict[str, Any]:
    text = strip_code_fences(text)
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
        snippet = match.group(0)
        try:
            return json.loads(snippet)
        except Exception:
            pass
    return {}


def choose_best_match(matches: List[Tuple[float, Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    if not matches:
        return None
    matches.sort(key=lambda x: x[0], reverse=True)
    return matches[0][1]


def truncate_text(text: str, limit: int = 280) -> str:
    text = normalize_whitespace(text)
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def extract_structured_values(text: str) -> List[str]:
    text = text or ""
    candidates: List[str] = []
    for pattern in [MONTHLY_DUE_RE, QUARTERLY_DUE_RE, WEEKLY_DUE_RE, MONTHS_AFTER_YEAR_END_RE, DAY_RE, FREQUENCY_RE, MONEY_RE, PERCENT_RE]:
        for match in pattern.finditer(text):
            value = normalize_whitespace(match.group(0))
            if value and value not in candidates:
                candidates.append(value)
    return candidates


def build_support_snippets(chunk: Dict[str, Any], limit: int = 240) -> List[str]:
    text = chunk.get("text") or chunk.get("retrieval_text") or ""
    row_label = chunk.get("row_label", "")
    values = extract_structured_values(text)
    snippets: List[str] = []
    if row_label:
        for value in values[:4]:
            snippets.append(f"{row_label} -> {value}")
    if not snippets:
        snippets.append(truncate_text(text, limit=limit))
    return snippets[:6]


def heuristic_plan(question: str) -> Dict[str, Any]:
    q_norm = normalize_for_match(question)
    jurisdiction = None
    for item in sorted(JURISDICTIONS, key=len, reverse=True):
        if item in q_norm:
            jurisdiction = item
            break

    entity = None
    for key, aliases in ENTITY_HINTS.items():
        if any(alias in q_norm for alias in aliases):
            entity = key
            break
    if entity is None and "corporation" in q_norm:
        entity = "corporation"

    question_type = "unknown"
    for qtype, hints in QUESTION_TYPE_HINTS.items():
        if any(hint in q_norm for hint in hints):
            question_type = qtype
            break

    year_match = YEAR_RE.search(q_norm)
    year = year_match.group(0) if year_match else None

    keywords = [tok for tok in tokenize(question) if tok not in STOPWORDS][:8]
    prefer_types = PREFER_TYPES_BY_QUESTION_TYPE.get(question_type, PREFER_TYPES_BY_QUESTION_TYPE["unknown"])

    rewrite_queries = [question]
    pieces = [p for p in [jurisdiction, entity, question_type, year] if p]
    if pieces:
        rewrite_queries.append(" ".join(pieces + [question]))
    if jurisdiction and entity:
        rewrite_queries.append(f"{jurisdiction} {entity} {question}")
    if question_type != "unknown":
        rewrite_queries.append(f"{question} exact {question_type}")

    topic = question_type if question_type != "unknown" else (keywords[0] if keywords else "unknown")

    return {
        "jurisdiction": jurisdiction,
        "entity": entity,
        "topic": topic,
        "year": year,
        "question_type": question_type,
        "prefer_types": prefer_types,
        "keywords": keywords,
        "rewrite_queries": rewrite_queries,
        "planner_raw": "",
        "original_question": question,
    }


class KPMGTaxRAGV52AWS:
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings()
        self.chunks: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self.tokenized_corpus: Optional[List[List[str]]] = None
        self.bm25: Optional[SimpleBM25] = None
        self.embedder = None
        self.reranker = None
        self.tokenizer = None
        self.llm = None
        self.device = self.settings.device
        self.reconstruction_notice = RECONSTRUCTION_NOTICE

    # ---------- artifact management ----------
    def extract_bundle_if_needed(self) -> None:
        if self.settings.bundle_zip_path is None:
            return
        zip_path = Path(self.settings.bundle_zip_path)
        if not zip_path.exists():
            return
        marker = self.settings.artifacts_dir / "chunks.jsonl"
        if marker.exists():
            return
        LOGGER.info("Extracting artifact bundle from %s", zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(self.settings.artifacts_dir.parent)

    def sync_from_s3_if_needed(self) -> None:
        marker = self.settings.artifacts_dir / "chunks.jsonl"
        if marker.exists() or not self.settings.s3_bucket:
            return
        helper = S3Helper(self.settings.s3_bucket, self.settings.aws_region)
        prefix = f"{self.settings.s3_prefix.rstrip('/')}/{self.settings.output_subdir}"
        LOGGER.info("Attempting to download artifacts from s3://%s/%s", self.settings.s3_bucket, prefix)
        try:
            count = helper.download_prefix(prefix, self.settings.artifacts_dir)
            LOGGER.info("Downloaded %s artifact files from S3", count)
        except Exception as exc:
            LOGGER.warning("S3 sync skipped: %s", exc)

    def sync_to_s3(self) -> None:
        if not self.settings.s3_bucket:
            LOGGER.info("No S3 bucket configured; skipping upload")
            return
        helper = S3Helper(self.settings.s3_bucket, self.settings.aws_region)
        prefix = f"{self.settings.s3_prefix.rstrip('/')}/{self.settings.output_subdir}"
        count = helper.upload_dir(self.settings.artifacts_dir, prefix)
        LOGGER.info("Uploaded %s files to s3://%s/%s", count, self.settings.s3_bucket, prefix)

    def load_artifacts(self) -> None:
        self.extract_bundle_if_needed()
        self.sync_from_s3_if_needed()

        run_cfg_path = self.settings.artifacts_dir / "run_config.json"
        if run_cfg_path.exists():
            observed = load_json(run_cfg_path)
            self.settings.apply_run_config(observed)
            LOGGER.info("Loaded observed run config from %s", run_cfg_path)

        chunks_path = self.settings.artifacts_dir / "chunks.jsonl"
        embed_path = self.settings.artifacts_dir / "chunk_embeddings.npy"
        cache_path = self.settings.artifacts_dir / "retrieval_cache.pkl"

        if not chunks_path.exists() or not embed_path.exists():
            raise FileNotFoundError(
                f"Expected artifacts at {self.settings.artifacts_dir}, but chunks.jsonl or chunk_embeddings.npy is missing."
            )

        self.chunks = load_jsonl(chunks_path)
        self.embeddings = np.load(embed_path).astype(np.float32)
        if self.embeddings.ndim != 2:
            raise ValueError(f"Expected 2-D embeddings array, got shape {self.embeddings.shape}")
        if len(self.chunks) != self.embeddings.shape[0]:
            raise ValueError("Mismatch between number of chunks and embedding rows")

        if cache_path.exists():
            with cache_path.open("rb") as f:
                cache_obj = pickle.load(f)
            tokenized = cache_obj.get("tokenized_corpus") if isinstance(cache_obj, dict) else None
            if tokenized and len(tokenized) == len(self.chunks):
                self.tokenized_corpus = tokenized
        if self.tokenized_corpus is None:
            self.tokenized_corpus = [tokenize(chunk.get("retrieval_text", chunk.get("text", ""))) for chunk in self.chunks]

        self.bm25 = SimpleBM25(self.tokenized_corpus)
        LOGGER.info("Loaded %s chunks and embedding matrix %s", len(self.chunks), self.embeddings.shape)

    # ---------- model loading ----------
    def load_embedder(self):
        if self.embedder is not None:
            return self.embedder
        from sentence_transformers import SentenceTransformer
        LOGGER.info("Loading embed model: %s", self.settings.embed_model_name)
        self.embedder = SentenceTransformer(self.settings.embed_model_name, device=self.device)
        return self.embedder

    def load_reranker(self):
        if not self.settings.use_reranker:
            return None
        if self.reranker is not None:
            return self.reranker
        from sentence_transformers import CrossEncoder
        LOGGER.info("Loading reranker: %s", self.settings.reranker_model_name)
        self.reranker = CrossEncoder(self.settings.reranker_model_name, device=self.device)
        return self.reranker

    def load_llm(self):
        if self.llm is not None and self.tokenizer is not None:
            return self.tokenizer, self.llm
        from transformers import AutoModelForCausalLM, AutoTokenizer
        LOGGER.info("Loading LLM: %s", self.settings.llm_model_name)

        quantization_config = None
        kwargs: Dict[str, Any] = {"trust_remote_code": True}
        if torch is not None and torch.cuda.is_available():
            kwargs["device_map"] = "auto"
            kwargs["torch_dtype"] = torch.float16
        else:
            kwargs["torch_dtype"] = torch.float32

        if self.settings.llm_load_in_4bit and torch is not None and torch.cuda.is_available():
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
                kwargs["quantization_config"] = quantization_config
            except Exception as exc:
                LOGGER.warning("4-bit load requested but BitsAndBytesConfig failed: %s", exc)

        self.tokenizer = AutoTokenizer.from_pretrained(self.settings.llm_model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.llm = AutoModelForCausalLM.from_pretrained(self.settings.llm_model_name, **kwargs)
        return self.tokenizer, self.llm

    # ---------- core retrieval ----------
    def plan_question(self, question: str) -> Dict[str, Any]:
        plan = heuristic_plan(question)
        if not self.settings.use_llm_planner:
            return plan
        try:
            llm_plan = self._llm_plan(question, plan)
            if llm_plan:
                merged = {**plan, **llm_plan}
                merged["original_question"] = question
                return merged
        except Exception as exc:
            LOGGER.warning("LLM planner failed; using heuristic plan instead: %s", exc)
        return plan

    def _llm_plan(self, question: str, heuristic: Dict[str, Any]) -> Dict[str, Any]:
        self.load_llm()
        system = (
            "You are a query planner for domain-specific tax QA. "
            "Return only JSON with keys: jurisdiction, entity, topic, question_type, year, prefer_types, keywords, rewrite_queries. "
            "Use lowercase strings where practical. prefer_types must be an ordered list using only row, table, text, heading."
        )
        user = (
            f"Question: {question}\n"
            f"Heuristic guess: {json.dumps(heuristic, ensure_ascii=False)}\n"
            "Return JSON only."
        )
        raw = self._generate_json(system, user, max_new_tokens=self.settings.planner_max_new_tokens, do_sample=self.settings.planner_do_sample)
        parsed = parse_first_json_object(raw)
        if parsed:
            parsed["planner_raw"] = raw
        return parsed

    def retrieve(self, question: str, plan: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not self.chunks or self.embeddings is None or self.bm25 is None:
            self.load_artifacts()
        plan = plan or self.plan_question(question)
        rewrite_queries = list(dict.fromkeys(plan.get("rewrite_queries") or [question]))
        queries = rewrite_queries[:4] if rewrite_queries else [question]

        dense_norm = np.zeros(len(self.chunks), dtype=np.float32)
        if self.load_embedder() is not None:
            q_embs = self.embedder.encode(queries, normalize_embeddings=False, show_progress_bar=False)
            if isinstance(q_embs, list):
                q_embs = np.array(q_embs, dtype=np.float32)
            for q_emb in q_embs:
                sims = cosine_sim_matrix(np.array(q_emb, dtype=np.float32), self.embeddings)
                dense_norm = np.maximum(dense_norm, minmax(sims.astype(np.float32)))

        bm25_norm = np.zeros(len(self.chunks), dtype=np.float32)
        for q in queries:
            tokens = tokenize(q)
            scores = self.bm25.get_scores(tokens)
            bm25_norm = np.maximum(bm25_norm, minmax(scores))

        metadata_scores = np.array([self.metadata_boost(plan, chunk) for chunk in self.chunks], dtype=np.float32)
        pre_scores = (
            self.settings.dense_weight * dense_norm
            + self.settings.bm25_weight * bm25_norm
            + self.settings.metadata_weight * metadata_scores
        ).astype(np.float32)

        dense_top = np.argsort(-dense_norm)[: self.settings.dense_top_k]
        bm25_top = np.argsort(-bm25_norm)[: self.settings.bm25_top_k]
        candidate_idx = sorted(set(map(int, dense_top)) | set(map(int, bm25_top)))
        if not candidate_idx:
            candidate_idx = list(np.argsort(-pre_scores)[: max(self.settings.final_top_k, 8)])

        rerank_scores_norm = {idx: 0.5 for idx in candidate_idx}
        if self.settings.use_reranker and candidate_idx:
            try:
                reranker = self.load_reranker()
                pairs = [[question, self.chunks[idx].get("retrieval_text", self.chunks[idx].get("text", ""))] for idx in candidate_idx]
                raw_scores = np.array(reranker.predict(pairs, show_progress_bar=False), dtype=np.float32)
                raw_norm = minmax(raw_scores)
                for idx, score in zip(candidate_idx, raw_norm):
                    rerank_scores_norm[idx] = float(score)
            except Exception as exc:
                LOGGER.warning("Reranker failed; using pre-scores only: %s", exc)

        final_rows = []
        for idx in candidate_idx:
            final_score = (1.0 - self.settings.rerank_weight) * float(pre_scores[idx]) + self.settings.rerank_weight * float(rerank_scores_norm[idx])
            chunk = self.chunks[idx]
            row = {
                "idx": idx,
                "chunk": chunk,
                "dense_score": float(dense_norm[idx]),
                "bm25_score": float(bm25_norm[idx]),
                "metadata_boost": float(metadata_scores[idx]),
                "pre_score": float(pre_scores[idx]),
                "rerank_score": float(rerank_scores_norm[idx]),
                "final_score": float(final_score),
            }
            final_rows.append(row)

        final_rows.sort(key=lambda r: r["final_score"], reverse=True)
        reranked = final_rows[: self.settings.rerank_top_k]
        final_top = reranked[: self.settings.final_top_k]
        candidate_pool = self.build_candidate_pool(final_rows, plan)
        return {
            "plan": plan,
            "all_candidates": final_rows,
            "selected": final_top,
            "candidate_pool": candidate_pool,
        }

    def metadata_boost(self, plan: Dict[str, Any], chunk: Dict[str, Any]) -> float:
        score = 0.0
        jurs = [normalize_for_match(j) for j in chunk.get("jurisdictions", [])]
        ents = [normalize_for_match(e) for e in chunk.get("entities", [])]
        tops = [normalize_for_match(t) for t in chunk.get("topics", [])]
        ctype = normalize_for_match(chunk.get("content_type", ""))
        row_label = normalize_for_match(chunk.get("row_label", ""))
        text = normalize_for_match(chunk.get("text", ""))

        jurisdiction = normalize_for_match(plan.get("jurisdiction") or "")
        entity = normalize_for_match(plan.get("entity") or "")
        topic = normalize_for_match(plan.get("topic") or "")
        question_type = normalize_for_match(plan.get("question_type") or "unknown")
        year = normalize_for_match(plan.get("year") or "")
        prefer_types = [normalize_for_match(x) for x in plan.get("prefer_types") or []]

        if jurisdiction:
            if jurisdiction in jurs:
                score += 0.28
            if jurisdiction in row_label or jurisdiction in text:
                score += 0.10
        if entity:
            if entity in ents:
                score += 0.22
            if entity in text:
                score += 0.08
        if topic:
            if topic in tops:
                score += 0.18
            if topic in text:
                score += 0.06
        if year and year in text:
            score += 0.06
        if prefer_types:
            if ctype == prefer_types[0]:
                score += 0.10
            elif ctype in prefer_types:
                score += 0.06
        if question_type in {"deadline", "frequency"} and ctype == "text":
            score += 0.05
        return clamp01(score)

    def build_candidate_pool(self, ranked_rows: List[Dict[str, Any]], plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        values: List[Dict[str, Any]] = []
        question_type = plan.get("question_type")
        for source_rank, row in enumerate(ranked_rows[: self.settings.rerank_top_k], start=1):
            chunk = row["chunk"]
            snippets = build_support_snippets(chunk)
            extracted = extract_structured_values(chunk.get("text", ""))
            if question_type == "frequency":
                extracted = [v.lower() for v in extracted if v.lower() in {"monthly", "quarterly", "weekly"}] or extracted
            if question_type == "deadline":
                extracted = [v for v in extracted if any(s in normalize_for_match(v) for s in ["month", "working day", "following month", "calendar quarter", "year-end"])] or extracted
            if not extracted:
                continue
            for value, support in zip(extracted[:4], snippets[:4]):
                candidate = {
                    "source_id": source_rank,
                    "printed_page": chunk.get("printed_page"),
                    "content_type": chunk.get("content_type"),
                    "section_title": chunk.get("section_title"),
                    "source_rank": source_rank,
                    "row_label": chunk.get("row_label", ""),
                    "axis_label": "",
                    "structural_score": 0.25 if chunk.get("row_label") else 0.0,
                    "value": value,
                    "support": support,
                    "candidate_score": float(row["final_score"]),
                }
                values.append(candidate)
                if len(values) >= self.settings.max_candidate_values:
                    return values
        return values

    # ---------- answer generation ----------
    def answer_question(self, question: str) -> Dict[str, Any]:
        retrieval = self.retrieve(question)
        plan = retrieval["plan"]
        selected = retrieval["selected"]
        candidate_pool = retrieval["candidate_pool"]
        baseline_answer = self.extractive_baseline(candidate_pool)
        exact = self.try_exact_mode(question, plan, selected, candidate_pool)

        if exact is not None:
            answer_payload = {
                "answer": exact,
                "selected_candidate_ids": [],
                "confidence": "high",
                "reason": f"reconstructed_exact_{plan.get('question_type', 'unknown')}",
                "raw": "",
            }
            exact_mode_used = True
        else:
            answer_payload = self.llm_answer(question, plan, selected, candidate_pool, baseline_answer)
            exact_mode_used = False

        verifier_payload = self.verify_answer(question, answer_payload.get("answer", ""), selected)
        final_answer_short = verifier_payload.get("corrected_answer") or answer_payload.get("answer") or baseline_answer
        full_answer = self.format_full_answer(final_answer_short, selected, verifier_payload)

        result = {
            "question": question,
            "query_plan": plan,
            "question_type": plan.get("question_type", "unknown"),
            "baseline_answer": baseline_answer,
            "deterministic_answer": exact,
            "llm_answer_payload": answer_payload,
            "verifier_payload": verifier_payload,
            "candidate_pool": candidate_pool,
            "retrieved": selected,
            "exact_mode_used": exact_mode_used,
            "final_answer_short": final_answer_short,
            "full_answer": full_answer,
            "retrieved_pages": [row["chunk"].get("printed_page") for row in selected],
            "top1_final_score": float(selected[0]["final_score"]) if selected else None,
        }
        return result

    def extractive_baseline(self, candidate_pool: List[Dict[str, Any]]) -> str:
        if not candidate_pool:
            return ""
        return str(candidate_pool[0].get("value", "")).strip()

    def try_exact_mode(
        self,
        question: str,
        plan: Dict[str, Any],
        selected: List[Dict[str, Any]],
        candidate_pool: List[Dict[str, Any]],
    ) -> Optional[str]:
        qtype = plan.get("question_type")
        if qtype not in {"rate", "threshold", "deadline", "frequency"}:
            return None
        jurisdiction = normalize_for_match(plan.get("jurisdiction") or "")
        entity = normalize_for_match(plan.get("entity") or "")
        topic = normalize_for_match(plan.get("topic") or "")
        q_norm = normalize_for_match(question)

        ranked_matches: List[Tuple[float, Dict[str, Any]]] = []
        for cand in candidate_pool:
            score = float(cand.get("candidate_score", 0.0))
            row_label = normalize_for_match(cand.get("row_label", ""))
            support = normalize_for_match(cand.get("support", ""))
            value = str(cand.get("value", "")).strip()
            if jurisdiction and (jurisdiction in row_label or jurisdiction in support):
                score += 0.35
            if entity and (entity in row_label or entity in support or entity in q_norm):
                score += 0.15
            if topic and topic in support:
                score += 0.08
            if qtype == "frequency" and value.lower() in {"monthly", "quarterly", "weekly"}:
                score += 0.25
            if qtype == "deadline" and any(term in value.lower() for term in ["month", "working day", "following month", "year-end", "calendar quarter"]):
                score += 0.25
            if qtype in {"rate", "threshold"}:
                if re.search(r"\d", value):
                    score += 0.12
                if qtype == "rate" and "%" in value:
                    score += 0.12
            ranked_matches.append((score, cand))

        best = choose_best_match(ranked_matches)
        if best is not None:
            return str(best.get("value", "")).strip()

        # Last fallback: mine the first selected chunk directly.
        if selected:
            text = selected[0]["chunk"].get("text", "")
            values = extract_structured_values(text)
            if values:
                return values[0]
        return None

    def llm_answer(
        self,
        question: str,
        plan: Dict[str, Any],
        selected: List[Dict[str, Any]],
        candidate_pool: List[Dict[str, Any]],
        baseline_answer: str,
    ) -> Dict[str, Any]:
        if not self.settings.use_llm_answer:
            return {
                "answer": baseline_answer,
                "selected_candidate_ids": [],
                "confidence": "medium" if baseline_answer else "low",
                "reason": "extractive_baseline_only",
                "raw": "",
            }
        try:
            self.load_llm()
            evidence = self._make_evidence_block(selected, max_items=6)
            candidate_text = json.dumps(candidate_pool[:12], ensure_ascii=False)
            system = (
                "You answer domain-specific Canadian corporate tax questions using only the provided evidence. "
                "Return JSON only with keys: answer, confidence, reason. "
                "If the evidence is weak, say that the answer cannot be determined from the provided evidence."
            )
            user = (
                f"Question: {question}\n"
                f"Plan: {json.dumps(plan, ensure_ascii=False)}\n"
                f"Baseline answer: {baseline_answer!r}\n"
                f"Candidate values: {candidate_text}\n\n"
                f"Evidence:\n{evidence}\n"
            )
            raw = self._generate_json(system, user, max_new_tokens=self.settings.answer_max_new_tokens, do_sample=self.settings.answer_do_sample)
            parsed = parse_first_json_object(raw)
            if parsed.get("answer"):
                parsed.setdefault("raw", raw)
                parsed.setdefault("selected_candidate_ids", [])
                return parsed
        except Exception as exc:
            LOGGER.warning("LLM answer failed; falling back to extractive baseline: %s", exc)
        return {
            "answer": baseline_answer,
            "selected_candidate_ids": [],
            "confidence": "medium" if baseline_answer else "low",
            "reason": "fallback_extractive_baseline",
            "raw": "",
        }

    def verify_answer(self, question: str, answer: str, selected: List[Dict[str, Any]]) -> Dict[str, Any]:
        pages = sorted({int(row["chunk"].get("printed_page", -1)) for row in selected if row["chunk"].get("printed_page") is not None})
        if not self.settings.use_llm_verifier:
            return {
                "verdict": "supported" if answer else "unsupported",
                "corrected_answer": answer,
                "confidence": "medium" if answer else "low",
                "pages": pages,
                "reason": "verifier_disabled",
            }
        try:
            self.load_llm()
            evidence = self._make_evidence_block(selected, max_items=6)
            system = (
                "You verify whether an answer is fully supported by evidence. Return JSON only with keys: "
                "verdict, corrected_answer, confidence, pages, reason. "
                "Use verdict supported_exact, supported_partial, or unsupported."
            )
            user = (
                f"Question: {question}\n"
                f"Draft answer: {answer}\n"
                f"Evidence:\n{evidence}\n"
            )
            raw = self._generate_json(system, user, max_new_tokens=self.settings.verifier_max_new_tokens, do_sample=self.settings.verifier_do_sample)
            parsed = parse_first_json_object(raw)
            if parsed:
                parsed.setdefault("corrected_answer", answer)
                parsed.setdefault("pages", pages)
                return parsed
        except Exception as exc:
            LOGGER.warning("Verifier failed; using heuristic verifier: %s", exc)
        return {
            "verdict": "supported_exact" if answer else "unsupported",
            "corrected_answer": answer,
            "confidence": "high" if answer else "low",
            "pages": pages,
            "reason": "heuristic_verifier_fallback",
        }

    def format_full_answer(self, answer: str, selected: List[Dict[str, Any]], verifier_payload: Dict[str, Any]) -> str:
        evidence_lines = []
        for i, row in enumerate(selected[:3], start=1):
            chunk = row["chunk"]
            title = chunk.get("section_title", "Untitled")
            page = chunk.get("printed_page", "?")
            excerpt = truncate_text(chunk.get("text", ""), limit=220)
            evidence_lines.append(f"- [Source {i} | p. {page}] {title} — {excerpt}")
        pages = verifier_payload.get("pages") or []
        citations = ", ".join(f"p. {p}" for p in pages) if pages else "n/a"
        confidence = verifier_payload.get("confidence", "unknown")
        verdict = verifier_payload.get("verdict", "unknown")
        return (
            f"{answer}\n\n"
            f"Evidence\n" + "\n".join(evidence_lines) + "\n\n"
            f"Citations: ({citations})\n\n"
            f"Confidence: {confidence}\n"
            f"Verifier: {verdict}"
        )

    def _make_evidence_block(self, selected: List[Dict[str, Any]], max_items: int = 6) -> str:
        blocks = []
        for i, row in enumerate(selected[:max_items], start=1):
            chunk = row["chunk"]
            blocks.append(
                f"[Source {i}] page={chunk.get('printed_page')} type={chunk.get('content_type')} "
                f"section={chunk.get('section_title')} row_label={chunk.get('row_label', '')}\n"
                f"{chunk.get('text', '')}\n"
            )
        return "\n".join(blocks)

    def _generate_json(self, system_prompt: str, user_prompt: str, max_new_tokens: int, do_sample: bool = False) -> str:
        tokenizer, llm = self.load_llm()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if hasattr(tokenizer, "apply_chat_template"):
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = system_prompt + "\n\n" + user_prompt
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=self.settings.llm_max_input_tokens)
        if torch is not None and hasattr(llm, "device"):
            inputs = {k: v.to(llm.device) for k, v in inputs.items()}
        gen = llm.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=0.0 if not do_sample else 0.2,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        new_tokens = gen[0][inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        return text.strip()

    # ---------- evaluation ----------
    def evaluate_csv(self, eval_csv: Path, output_csv: Optional[Path] = None) -> pd.DataFrame:
        df = pd.read_csv(eval_csv)
        rows: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            question = str(row["question"])
            result = self.answer_question(question)
            expected_pages = safe_parse_list(row.get("expected_pages"))
            expected_substrings = [str(x) for x in safe_parse_list(row.get("expected_substrings"))]
            retrieved_pages = result.get("retrieved_pages", [])
            top1_page = retrieved_pages[0] if retrieved_pages else None
            retrieval_hit_top1 = bool(top1_page in expected_pages) if expected_pages else False
            retrieval_hit_topk = bool(set(retrieved_pages) & set(expected_pages)) if expected_pages else False
            baseline_answer = result.get("baseline_answer", "")
            final_answer_short = result.get("final_answer_short", "")
            baseline_hit = self.substrings_hit(baseline_answer, expected_substrings)
            answer_hit = self.substrings_hit(final_answer_short, expected_substrings)
            citation_hit = retrieval_hit_topk
            verifier = result.get("verifier_payload", {})
            plan = result.get("query_plan", {})
            rows.append({
                "question": question,
                "question_type": row.get("question_type", result.get("question_type", "unknown")),
                "expected_pages": json.dumps(expected_pages),
                "expected_substrings": json.dumps(expected_substrings),
                "retrieved_pages": json.dumps(retrieved_pages),
                "retrieval_hit_top1": retrieval_hit_top1,
                "retrieval_hit_topk": retrieval_hit_topk,
                "baseline_answer": baseline_answer,
                "baseline_hit": baseline_hit,
                "final_answer_short": final_answer_short,
                "answer_hit": answer_hit,
                "citation_hit": citation_hit,
                "verifier_verdict": verifier.get("verdict"),
                "planner_jurisdiction": plan.get("jurisdiction"),
                "planner_entity": plan.get("entity"),
                "planner_topic": plan.get("topic"),
                "n_candidates": len(result.get("candidate_pool", [])),
                "top1_final_score": result.get("top1_final_score"),
                "exact_mode_used": result.get("exact_mode_used", False),
                "full_answer": result.get("full_answer", ""),
            })
        out_df = pd.DataFrame(rows)
        if output_csv is not None:
            output_csv = Path(output_csv)
            output_csv.parent.mkdir(parents=True, exist_ok=True)
            out_df.to_csv(output_csv, index=False)
            LOGGER.info("Wrote evaluation CSV to %s", output_csv)
        return out_df

    @staticmethod
    def substrings_hit(answer: str, expected_substrings: List[str]) -> bool:
        ans_norm = normalize_for_match(answer)
        return all(normalize_for_match(s) in ans_norm for s in expected_substrings if s)

    def summarize_evaluation(self, df: pd.DataFrame) -> Dict[str, Any]:
        n = len(df)
        if n == 0:
            return {"n_questions": 0}
        verifier_supported = df["verifier_verdict"].fillna("").astype(str).str.startswith("supported")
        return {
            "n_questions": int(n),
            "retrieval_hit_rate_top1": float(df["retrieval_hit_top1"].mean()),
            "retrieval_hit_rate_topk": float(df["retrieval_hit_topk"].mean()),
            "baseline_answer_hit_rate": float(df["baseline_hit"].mean()),
            "answer_hit_rate": float(df["answer_hit"].mean()),
            "citation_hit_rate": float(df["citation_hit"].mean()),
            "exact_mode_rate": float(df["exact_mode_used"].mean()),
            "verifier_supported_rate": float(verifier_supported.mean()),
        }


# ---------- convenience helpers ----------
def make_settings_from_env(base_dir: Optional[str] = None, bundle_zip_path: Optional[str] = None) -> Settings:
    settings = Settings(base_dir=Path(base_dir).expanduser() if base_dir else Settings().base_dir)
    if bundle_zip_path:
        settings.bundle_zip_path = Path(bundle_zip_path).expanduser()
    return settings


def reconstruct_eval_csv_from_observed(observed_eval_csv: Path, output_path: Path) -> Path:
    df = pd.read_csv(observed_eval_csv)
    keep = ["question", "question_type", "expected_pages", "expected_substrings"]
    df = df[keep].copy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def preflight_summary(artifact_dir: Path) -> Dict[str, Any]:
    artifact_dir = Path(artifact_dir)
    summary: Dict[str, Any] = {"artifact_dir": str(artifact_dir)}
    run_cfg = artifact_dir / "run_config.json"
    if run_cfg.exists():
        cfg = load_json(run_cfg)
        summary["embed_model_name"] = cfg.get("embed_model_name")
        summary["reranker_model_name"] = cfg.get("reranker_model_name")
        summary["llm_model_name"] = cfg.get("llm_model_name")
        summary["llm_load_in_4bit"] = cfg.get("llm_load_in_4bit")
        summary["dense_weight"] = cfg.get("dense_weight")
        summary["bm25_weight"] = cfg.get("bm25_weight")
        summary["metadata_weight"] = cfg.get("metadata_weight")
        summary["rerank_weight"] = cfg.get("rerank_weight")
        summary["dense_top_k"] = cfg.get("dense_top_k")
        summary["bm25_top_k"] = cfg.get("bm25_top_k")
        summary["rerank_top_k"] = cfg.get("rerank_top_k")
        summary["final_top_k"] = cfg.get("final_top_k")
    chunks_path = artifact_dir / "chunks.jsonl"
    if chunks_path.exists():
        chunks = load_jsonl(chunks_path)
        summary["n_chunks"] = len(chunks)
        counts: Dict[str, int] = {}
        for chunk in chunks:
            counts[chunk.get("content_type", "unknown")] = counts.get(chunk.get("content_type", "unknown"), 0) + 1
        summary["chunk_type_counts"] = counts
    emb_path = artifact_dir / "chunk_embeddings.npy"
    if emb_path.exists():
        summary["embedding_shape"] = list(np.load(emb_path).shape)
    eval_summary_path = artifact_dir / "evaluation_summary.json"
    if eval_summary_path.exists():
        summary["evaluation_summary"] = load_json(eval_summary_path)
    return summary


# ---------- CLI ----------
def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AWS-ready reconstructed v5.2 KPMG tax RAG")
    parser.add_argument("--base-dir", default=os.getenv("BASE_DIR", os.getcwd()))
    parser.add_argument("--bundle-zip", default=os.getenv("BUNDLE_ZIP_PATH", ""))
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("preflight", help="Summarize observed artifact files")

    ask = sub.add_parser("ask", help="Answer a single question")
    ask.add_argument("question")
    ask.add_argument("--json-out", default="")

    ev = sub.add_parser("evaluate", help="Evaluate a CSV of questions")
    ev.add_argument("eval_csv")
    ev.add_argument("--output-csv", default="")
    ev.add_argument("--summary-json", default="")

    recon = sub.add_parser("reconstruct-eval", help="Rebuild the 50-question eval CSV from observed evaluation_results.csv")
    recon.add_argument("observed_eval_csv")
    recon.add_argument("output_csv")

    upload = sub.add_parser("sync-to-s3", help="Upload artifacts directory to S3")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_cli()
    args = parser.parse_args(argv)
    settings = make_settings_from_env(base_dir=args.base_dir, bundle_zip_path=args.bundle_zip or None)
    rag = KPMGTaxRAGV52AWS(settings)

    if args.command == "preflight":
        rag.extract_bundle_if_needed()
        rag.sync_from_s3_if_needed()
        summary = preflight_summary(rag.settings.artifacts_dir)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0

    if args.command == "reconstruct-eval":
        out = reconstruct_eval_csv_from_observed(Path(args.observed_eval_csv), Path(args.output_csv))
        print(out)
        return 0

    if args.command == "sync-to-s3":
        rag.sync_to_s3()
        return 0

    rag.load_artifacts()

    if args.command == "ask":
        result = rag.answer_question(args.question)
        payload = json.dumps(result, indent=2, ensure_ascii=False)
        if args.json_out:
            out = Path(args.json_out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(payload, encoding="utf-8")
        print(payload)
        return 0

    if args.command == "evaluate":
        out_csv = Path(args.output_csv) if args.output_csv else rag.settings.outputs_dir / f"{rag.settings.output_subdir}_eval_reconstructed.csv"
        out_df = rag.evaluate_csv(Path(args.eval_csv), output_csv=out_csv)
        summary = rag.summarize_evaluation(out_df)
        if args.summary_json:
            Path(args.summary_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return 0

    parser.error("Unknown command")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
