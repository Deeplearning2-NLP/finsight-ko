"""Minimal cached OpenDART client.

Every response is cached under ``data/raw/opendart/`` so repeated runs make no
extra calls (daily quota). The API key is read from ``.env`` and never logged.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import requests
from dotenv import dotenv_values

BASE = "https://opendart.fss.or.kr/api"
ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "raw" / "opendart"

# status codes worth naming; "013" means "no data", not "no consolidated statements"
STATUS = {"000": "정상", "013": "조회된 데이터 없음", "020": "요청 제한 초과", "010": "미등록 키"}


def _key() -> str:
    key = os.environ.get("OPENDART_API_KEY") or dotenv_values(ROOT / ".env").get("OPENDART_API_KEY")
    if not key:
        raise RuntimeError("OPENDART_API_KEY missing in .env")
    return key


def _cache_path(endpoint: str, params: dict, ext: str) -> Path:
    tag = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:10]
    name = "_".join(str(v) for _, v in sorted(params.items()))[:80]
    return CACHE / endpoint / f"{name}_{tag}.{ext}"


def _get(endpoint: str, params: dict, ext: str) -> bytes:
    path = _cache_path(endpoint, params, ext)
    if path.exists():
        return path.read_bytes()
    r = requests.get(f"{BASE}/{endpoint}.{'json' if ext == 'json' else 'xml'}",
                     params={"crtfc_key": _key(), **params}, timeout=120)
    r.raise_for_status()
    if ext == "json":
        status = r.json().get("status")
        if status not in ("000", "013"):
            raise RuntimeError(f"OpenDART {endpoint} status {status}: {STATUS.get(status, r.json().get('message'))}")
    elif not r.content.startswith(b"PK"):
        raise RuntimeError(f"OpenDART {endpoint} returned non-zip: {r.content[:200]!r}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(r.content)
    return r.content


def corp_codes() -> dict[str, dict]:
    """corp_name -> {corp_code, stock_code, modify_date}.

    Several codes can share a name (e.g. 국민은행: two dormant 2017 entries and the active bank).
    The most recently modified entry wins; listed status breaks ties.
    """
    xml_path = CACHE / "CORPCODE.xml"
    if not xml_path.exists():
        zipfile.ZipFile(io.BytesIO(_get("corpCode", {}, "zip"))).extractall(CACHE)
    out: dict[str, dict] = {}
    for el in ET.parse(xml_path).getroot().iter("list"):
        name = el.findtext("corp_name").strip()
        rec = {"corp_code": el.findtext("corp_code").strip(), "stock_code": el.findtext("stock_code").strip(),
               "modify_date": el.findtext("modify_date").strip()}
        if name not in out or (rec["modify_date"], bool(rec["stock_code"])) > \
                (out[name]["modify_date"], bool(out[name]["stock_code"])):
            out[name] = rec
    return out


def list_filings(corp_code: str, bgn_de: str, end_de: str, detail: str = "A001") -> list[dict]:
    """Filings list (A001 = 사업보고서)."""
    js = json.loads(_get("list", {"corp_code": corp_code, "bgn_de": bgn_de, "end_de": end_de,
                                  "pblntf_detail_ty": detail, "page_count": 100}, "json"))
    return js.get("list", [])


def document(rcept_no: str) -> dict[str, bytes]:
    """Original filing XML files (name -> bytes) from document.xml (a ZIP)."""
    zf = zipfile.ZipFile(io.BytesIO(_get("document", {"rcept_no": rcept_no}, "zip")))
    return {n: zf.read(n) for n in zf.namelist()}


def fin_statements(corp_code: str, year: int, reprt_code: str = "11011", fs_div: str = "CFS") -> dict:
    """fnlttSinglAcntAll: full financial statements. Returns the raw JSON (check 'status')."""
    return json.loads(_get("fnlttSinglAcntAll", {"corp_code": corp_code, "bsns_year": str(year),
                                                 "reprt_code": reprt_code, "fs_div": fs_div}, "json"))
