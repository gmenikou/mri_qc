import re
import io
import json
import time
import uuid
import base64
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    HRFlowable,
    Image as RLImage,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

APP_TITLE = "ACR MRI Large Phantom QC Reporter"

# =========================================================
# EDIT THESE ONCE
# =========================================================
DEFAULT_GITHUB_OWNER = "YOUR_GITHUB_USERNAME"
DEFAULT_GITHUB_REPO = "YOUR_REPO_NAME"
DEFAULT_GITHUB_BRANCH = "main"
DEFAULT_GITHUB_CSV_PATH = "acr_qc_data/acr_qc_history.csv"

DATA_DIR = Path("acr_qc_data")
LOCAL_HISTORY_CSV = DATA_DIR / "acr_qc_history.csv"
LOCAL_LOCK_FILE = DATA_DIR / "acr_qc_history.lock"
REPORTS_DIR = DATA_DIR / "reports"
CHARTS_DIR = DATA_DIR / "charts"
LOGO_PATH = DATA_DIR / "logo.png"  # optional

DATA_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)
CHARTS_DIR.mkdir(exist_ok=True)


# =========================================================
# HELPERS
# =========================================================
def safe_float(text):
    try:
        return float(text)
    except Exception:
        return None


def validate_iso_timestamp(ts: str) -> bool:
    try:
        datetime.fromisoformat(ts)
        return True
    except Exception:
        return False


def build_scanner_id(site_name: str, scanner_name: str) -> str:
    raw = f"{str(site_name).strip()}__{str(scanner_name).strip()}".lower()
    raw = re.sub(r"[^a-z0-9]+", "_", raw).strip("_")
    return raw or "unknown_scanner"


def sanitize_filename(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(text or "").strip()) or "file"


def get_history_columns():
    return [
        "timestamp",
        "session_label",
        "site_name",
        "scanner_name",
        "scanner_id",
        "test_name",
        "value",
        "unit",
        "criteria",
        "status",
        "details",
        "source_file",
        "sequence_label",
    ]


def read_text_file(uploaded_file):
    raw = uploaded_file.read()
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return raw.decode(enc)
        except Exception:
            continue
    return raw.decode("utf-8", errors="ignore")


def empty_history_df():
    return pd.DataFrame(columns=get_history_columns())


def detect_sequence_label(filename: str, text: str = "") -> str:
    s = f"{filename}\n{text}".lower()
    if "t1" in s or "t1-weighted" in s:
        return "T1"
    if "t2" in s or "t2-weighted" in s:
        return "T2"
    return ""


def normalize_history_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return empty_history_df()

    df = df.copy()

    for col in get_history_columns():
        if col not in df.columns:
            df[col] = None

    text_cols = [
        "timestamp",
        "session_label",
        "site_name",
        "scanner_name",
        "scanner_id",
        "test_name",
        "unit",
        "criteria",
        "status",
        "details",
        "source_file",
        "sequence_label",
    ]
    for col in text_cols:
        df[col] = df[col].fillna("").astype(str)

    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    missing_id = df["scanner_id"].str.strip() == ""
    if missing_id.any():
        df.loc[missing_id, "scanner_id"] = df.loc[missing_id].apply(
            lambda r: build_scanner_id(r["site_name"], r["scanner_name"]),
            axis=1,
        )

    return df[get_history_columns()]


def github_is_ready(cfg):
    return bool(
        cfg
        and cfg.get("token")
        and cfg.get("owner")
        and cfg.get("repo")
        and cfg.get("path")
        and cfg["owner"] != "YOUR_GITHUB_USERNAME"
        and cfg["repo"] != "YOUR_REPO_NAME"
    )


def get_acr_test_order():
    return [
        "Geometric Accuracy",
        "Slice Thickness Accuracy",
        "Slice Position Accuracy",
        "Image Uniformity T1",
        "Image Uniformity T2",
        "Percentage Signal Ghosting",
        "Low Contrast Detectability",
        "High Contrast Spatial Resolution T1",
        "High Contrast Spatial Resolution T2",
        "Signal to Noise Ratio",
        "Central Frequency",
    ]


def acr_sort_key(test_name):
    order = get_acr_test_order()
    if test_name in order:
        return (order.index(test_name), str(test_name))
    return (999, str(test_name))


def sort_tests_acr(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "test_name" not in df.columns:
        return df
    out = df.copy()
    out["_acr_order"] = out["test_name"].apply(acr_sort_key)
    out = out.sort_values("_acr_order").drop(columns=["_acr_order"])
    return out


def combine_session_results(results):
    """
    Combine parsed results so that:
    - Slice Thickness Accuracy uses the worse of T1/T2
    - Slice Position Accuracy uses the worse of T1/T2
    - other tests remain unchanged
    """
    if not results:
        return []

    df = pd.DataFrame(results).copy()
    if df.empty:
        return []

    if "sequence_label" not in df.columns:
        df["sequence_label"] = ""

    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    combined = []

    for test_name, g in df.groupby("test_name", sort=False, dropna=False):
        g = g.copy()

        if test_name == "Slice Thickness Accuracy":
            valid = g.dropna(subset=["value"]).copy()

            if valid.empty:
                row = g.iloc[0].to_dict()
                row["status"] = "FAIL"
                row["details"] = "Could not determine worst slice thickness from T1/T2."
                combined.append(row)
                continue

            valid["severity"] = (valid["value"] - 5.0).abs()
            worst = valid.sort_values("severity", ascending=False).iloc[0].to_dict()
            overall_pass = (valid["value"].between(4.3, 5.7)).all() and len(valid) == len(g)

            seq_txt = "; ".join(
                f"{(r['sequence_label'] or r['source_file'])}: {r['value']} mm"
                for _, r in valid.iterrows()
            )

            worst["status"] = "PASS" if overall_pass else "FAIL"
            worst["details"] = (
                f"Worst of T1/T2 used for result. {seq_txt}. "
                f"Selected worst value: {worst['value']} mm"
            )
            worst["source_file"] = "COMBINED_T1_T2"
            worst["sequence_label"] = "COMBINED"
            combined.append({k: v for k, v in worst.items() if k != "severity"})
            continue

        if test_name == "Slice Position Accuracy":
            valid = g.dropna(subset=["value"]).copy()

            if valid.empty:
                row = g.iloc[0].to_dict()
                row["status"] = "FAIL"
                row["details"] = "Could not determine worst slice position error from T1/T2."
                combined.append(row)
                continue

            valid["severity"] = valid["value"].abs()
            worst = valid.sort_values("severity", ascending=False).iloc[0].to_dict()
            overall_pass = (valid["value"].abs() <= 5).all() and len(valid) == len(g)

            seq_txt = "; ".join(
                f"{(r['sequence_label'] or r['source_file'])}: {r['value']} mm"
                for _, r in valid.iterrows()
            )

            worst["status"] = "PASS" if overall_pass else "FAIL"
            worst["details"] = (
                f"Worst absolute position error from T1/T2 used for result. {seq_txt}. "
                f"Selected worst value: {worst['value']} mm"
            )
            worst["source_file"] = "COMBINED_T1_T2"
            worst["sequence_label"] = "COMBINED"
            combined.append({k: v for k, v in worst.items() if k != "severity"})
            continue

        if len(g) > 1:
            g = g.sort_values(["sequence_label", "source_file"])
            combined.append(g.iloc[0].to_dict())
        else:
            combined.append(g.iloc[0].to_dict())

    combined_df = pd.DataFrame(combined)
    if combined_df.empty:
        return combined
    combined_df = sort_tests_acr(combined_df)
    return combined_df.to_dict(orient="records")


def build_frontpage_trend_df(history_df, include_current_df=None):
    """
    Build a trend dataframe suitable for plotting.
    Uses one row per session/test/system.
    """
    trend_df = history_df.copy()

    if include_current_df is not None and not include_current_df.empty:
        trend_df = pd.concat([trend_df, include_current_df], ignore_index=True)

    trend_df = normalize_history_df(trend_df)
    if trend_df.empty:
        return trend_df

    trend_df["timestamp_dt"] = pd.to_datetime(trend_df["timestamp"], errors="coerce")
    trend_df = trend_df.dropna(subset=["timestamp_dt"])

    aggregated_rows = []

    group_cols = [
        "timestamp",
        "session_label",
        "site_name",
        "scanner_name",
        "scanner_id",
        "test_name",
    ]

    for _, g in trend_df.groupby(group_cols, dropna=False, sort=False):
        g = g.copy()
        test_name = g["test_name"].iloc[0]

        if test_name == "Slice Thickness Accuracy":
            valid = g.dropna(subset=["value"]).copy()
            if valid.empty:
                continue
            valid["severity"] = (valid["value"] - 5.0).abs()
            worst = valid.sort_values("severity", ascending=False).iloc[0]
            status = "PASS" if (valid["value"].between(4.3, 5.7)).all() else "FAIL"

            aggregated_rows.append(
                {
                    "timestamp": worst["timestamp"],
                    "timestamp_dt": pd.to_datetime(worst["timestamp"], errors="coerce"),
                    "session_label": worst["session_label"],
                    "site_name": worst["site_name"],
                    "scanner_name": worst["scanner_name"],
                    "scanner_id": worst["scanner_id"],
                    "test_name": test_name,
                    "value": float(worst["value"]),
                    "unit": worst["unit"],
                    "criteria": worst["criteria"],
                    "status": status,
                    "details": worst["details"],
                }
            )
            continue

        if test_name == "Slice Position Accuracy":
            valid = g.dropna(subset=["value"]).copy()
            if valid.empty:
                continue
            valid["severity"] = valid["value"].abs()
            worst = valid.sort_values("severity", ascending=False).iloc[0]
            status = "PASS" if (valid["value"].abs() <= 5).all() else "FAIL"

            aggregated_rows.append(
                {
                    "timestamp": worst["timestamp"],
                    "timestamp_dt": pd.to_datetime(worst["timestamp"], errors="coerce"),
                    "session_label": worst["session_label"],
                    "site_name": worst["site_name"],
                    "scanner_name": worst["scanner_name"],
                    "scanner_id": worst["scanner_id"],
                    "test_name": test_name,
                    "value": float(worst["value"]),
                    "unit": worst["unit"],
                    "criteria": worst["criteria"],
                    "status": status,
                    "details": worst["details"],
                }
            )
            continue

        valid = g.dropna(subset=["value"]).copy()
        if valid.empty:
            continue

        chosen = valid.sort_values("timestamp_dt").iloc[-1]
        aggregated_rows.append(
            {
                "timestamp": chosen["timestamp"],
                "timestamp_dt": pd.to_datetime(chosen["timestamp"], errors="coerce"),
                "session_label": chosen["session_label"],
                "site_name": chosen["site_name"],
                "scanner_name": chosen["scanner_name"],
                "scanner_id": chosen["scanner_id"],
                "test_name": test_name,
                "value": float(chosen["value"]),
                "unit": chosen["unit"],
                "criteria": chosen["criteria"],
                "status": chosen["status"],
                "details": chosen["details"],
            }
        )

    out = pd.DataFrame(aggregated_rows)
    if out.empty:
        return out

    out = out.sort_values(["scanner_id", "test_name", "timestamp_dt"]).reset_index(drop=True)
    return out


def build_single_session_df(history_df, scanner_id, timestamp):
    df = normalize_history_df(history_df).copy()
    if df.empty:
        return empty_history_df()

    out = df[
        (df["scanner_id"].astype(str) == str(scanner_id))
        & (df["timestamp"].astype(str) == str(timestamp))
    ].copy()
    return sort_tests_acr(out)


# =========================================================
# PARSERS
# =========================================================
def parse_slice_thickness(text):
    m = re.search(r"Slice thickness:\s*([0-9.\-]+)", text, re.I)
    value = safe_float(m.group(1)) if m else None
    passed = value is not None and 4.3 <= value <= 5.7
    return {
        "test_name": "Slice Thickness Accuracy",
        "value": value,
        "unit": "mm",
        "criteria": "4.3 to 5.7 mm",
        "status": "PASS" if passed else "FAIL",
        "details": f"Measured slice thickness: {value} mm" if value is not None else "Could not parse",
    }


def parse_slice_position(text):
    m1 = re.search(r"Slice\s*1\s*:\s*([-+]?\d+(?:\.\d+)?)", text, re.I)
    m11 = re.search(r"Slice\s*11\s*:\s*([-+]?\d+(?:\.\d+)?)", text, re.I)

    v1 = safe_float(m1.group(1)) if m1 else None
    v11 = safe_float(m11.group(1)) if m11 else None

    candidates = [x for x in [v1, v11] if x is not None]
    worst_abs = max([abs(x) for x in candidates], default=None)

    passed = (
        v1 is not None
        and v11 is not None
        and abs(v1) <= 5
        and abs(v11) <= 5
    )

    return {
        "test_name": "Slice Position Accuracy",
        "value": worst_abs,
        "unit": "mm",
        "criteria": "Absolute value of Slice 1 and Slice 11 <= 5 mm",
        "status": "PASS" if passed else "FAIL",
        "details": f"Slice 1: {v1} mm, Slice 11: {v11} mm, worst absolute error: {worst_abs} mm",
    }


def parse_snr(text):
    m = re.search(r"SNR:\s*([0-9.\-]+)", text, re.I)
    value = safe_float(m.group(1)) if m else None
    passed = value is not None
    return {
        "test_name": "Signal to Noise Ratio",
        "value": value,
        "unit": "",
        "criteria": "Compare to site baseline / trend",
        "status": "PASS" if passed else "FAIL",
        "details": f"SNR: {value}" if value is not None else "Could not parse",
    }


def parse_ghosting(text):
    m = re.search(r"Ghosting Ratio calculated:\s*([0-9.\-]+)\s*%?", text, re.I)
    value = safe_float(m.group(1)) if m else None
    passed = value is not None and value <= 2.5
    return {
        "test_name": "Percentage Signal Ghosting",
        "value": value,
        "unit": "%",
        "criteria": "<= 2.5%",
        "status": "PASS" if passed else "FAIL",
        "details": f"Ghosting ratio: {value}%" if value is not None else "Could not parse",
    }


def parse_lcd(text):
    t1 = re.search(r"Number of complete spokes in T1:\s*([0-9.\-]+)", text, re.I)
    t2 = re.search(r"Number of complete spokes in T2:\s*([0-9.\-]+)", text, re.I)

    v1 = safe_float(t1.group(1)) if t1 else None
    v2 = safe_float(t2.group(1)) if t2 else None

    vals = [v for v in [v1, v2] if v is not None]
    worst_value = min(vals) if vals else None

    passed = (v1 is not None and v1 >= 37) and (v2 is not None and v2 >= 37)

    return {
        "test_name": "Low Contrast Detectability",
        "value": worst_value,
        "unit": "spokes",
        "criteria": "Site/ACR target threshold, commonly >= 37 for strong performance",
        "status": "PASS" if passed else "FAIL",
        "details": f"T1 spokes: {v1}, T2 spokes: {v2}, worst: {worst_value}",
    }


def parse_uniformity(text, modality_name):
    m = re.search(r"Calculated PIU:\s*([0-9.\-]+)", text, re.I)
    value = safe_float(m.group(1)) if m else None
    passed = value is not None and value >= 87.5
    return {
        "test_name": modality_name,
        "value": value,
        "unit": "%",
        "criteria": ">= 87.5%",
        "status": "PASS" if passed else "FAIL",
        "details": f"PIU: {value}%" if value is not None else "Could not parse",
    }


def parse_hcr(text, modality_name):
    upper = re.search(r"Upper hole size \[mm\]:\s*([0-9.\-]+)", text, re.I)
    lower = re.search(r"Lower hole size \[mm\]:\s*([0-9.\-]+)", text, re.I)

    up = safe_float(upper.group(1)) if upper else None
    lo = safe_float(lower.group(1)) if lower else None

    passed = (up is not None and up <= 1.0) and (lo is not None and lo <= 1.0)
    worst_value = max([v for v in [up, lo] if v is not None], default=None)

    return {
        "test_name": modality_name,
        "value": worst_value,
        "unit": "mm",
        "criteria": "Must resolve 1.0 mm holes",
        "status": "PASS" if passed else "FAIL",
        "details": f"Upper: {up} mm, Lower: {lo} mm",
    }


def parse_geometric(text):
    nums = [safe_float(x) for x in re.findall(r"Length:\s*([0-9.\-]+)", text, re.I)]
    nums = [x for x in nums if x is not None and 180 <= x <= 200]

    if len(nums) < 1:
        return {
            "test_name": "Geometric Accuracy",
            "value": None,
            "unit": "mm",
            "criteria": "T1 dimensions should be 190 +/- 2 mm",
            "status": "FAIL",
            "details": "Could not parse geometric measurements.",
        }

    passed = all(188 <= x <= 192 for x in nums)
    value = sum(nums) / len(nums)
    return {
        "test_name": "Geometric Accuracy",
        "value": value,
        "unit": "mm",
        "criteria": "T1 dimensions should be 190 +/- 2 mm",
        "status": "PASS" if passed else "FAIL",
        "details": f"T1 measurements: {nums}",
    }


def parse_central_frequency(text):
    m = re.search(r"Central Frequency .*?:\s*([0-9.\-]+)\s*MHz", text, re.I)
    value = safe_float(m.group(1)) if m else None
    passed = value is not None
    return {
        "test_name": "Central Frequency",
        "value": value,
        "unit": "MHz",
        "criteria": "Trend against system baseline",
        "status": "PASS" if passed else "FAIL",
        "details": f"Central frequency: {value} MHz" if value is not None else "Could not parse",
    }


def infer_parser(filename, text):
    lower_name = filename.lower()

    # Prefer content-based detection first
    if "Slice Thickness Accuracy Test" in text:
        return parse_slice_thickness(text)
    if "Slice Position Accuracy Test" in text:
        return parse_slice_position(text)
    if "Signal to Noise Ratio Test" in text:
        return parse_snr(text)
    if "Percentage Signal Ghosting Test" in text:
        return parse_ghosting(text)
    if "Low Contrast Objective Detectability Test" in text:
        return parse_lcd(text)
    if "Image Intensity Uniformity Test" in text and "T2-weighted" in text:
        return parse_uniformity(text, "Image Uniformity T2")
    if "Image Intensity Uniformity Test" in text and "T1-weighted" in text:
        return parse_uniformity(text, "Image Uniformity T1")
    if "High Contrast Spatial Resolution Test" in text and "T2-weighted" in text:
        return parse_hcr(text, "High Contrast Spatial Resolution T2")
    if "High Contrast Spatial Resolution Test" in text and "T1-weighted" in text:
        return parse_hcr(text, "High Contrast Spatial Resolution T1")
    if "Geometric Accuracy Test" in text:
        return parse_geometric(text)
    if "Central Frequency Test" in text:
        return parse_central_frequency(text)

    # Fallback to filename-based detection
    if "slice thickness" in lower_name:
        return parse_slice_thickness(text)
    if "position" in lower_name:
        return parse_slice_position(text)
    if "snr" in lower_name:
        return parse_snr(text)
    if "ghost" in lower_name:
        return parse_ghosting(text)
    if "lcd" in lower_name:
        return parse_lcd(text)
    if "uniformity" in lower_name and "t2" in lower_name:
        return parse_uniformity(text, "Image Uniformity T2")
    if "uniformity" in lower_name and "t1" in lower_name:
        return parse_uniformity(text, "Image Uniformity T1")
    if "hcr" in lower_name and "t2" in lower_name:
        return parse_hcr(text, "High Contrast Spatial Resolution T2")
    if "hcr" in lower_name and "t1" in lower_name:
        return parse_hcr(text, "High Contrast Spatial Resolution T1")
    if "geometric" in lower_name:
        return parse_geometric(text)
    if "central frequency" in lower_name:
        return parse_central_frequency(text)

    return {
        "test_name": filename,
        "value": None,
        "unit": "",
        "criteria": "Unknown",
        "status": "FAIL",
        "details": "No parser matched this file.",
    }


# =========================================================
# GITHUB HELPERS
# =========================================================
def github_headers(token):
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
    }


def github_get_file(owner, repo, path, token, branch="main"):
    try:
        url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
        resp = requests.get(url, headers=github_headers(token), params={"ref": branch}, timeout=30)
    except requests.RequestException as e:
        return None, None, f"GitHub connection error: {e}"

    if resp.status_code == 200:
        payload = resp.json()
        content = base64.b64decode(payload["content"]).decode("utf-8")
        return content, payload.get("sha"), None

    if resp.status_code == 404:
        return None, None, None

    return None, None, f"GitHub read error {resp.status_code}: {resp.text}"


def github_put_file(owner, repo, path, token, content_text, message, branch="main", sha=None):
    try:
        url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
        payload = {
            "message": message,
            "content": base64.b64encode(content_text.encode("utf-8")).decode("utf-8"),
            "branch": branch,
        }
        if sha:
            payload["sha"] = sha

        resp = requests.put(url, headers=github_headers(token), json=payload, timeout=30)
    except requests.RequestException as e:
        return False, f"GitHub connection error: {e}"

    if resp.status_code in (200, 201):
        return True, None

    return False, f"GitHub write error {resp.status_code}: {resp.text}"


def github_delete_file(owner, repo, path, token, message, branch="main", sha=None):
    try:
        url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
        payload = {
            "message": message,
            "branch": branch,
            "sha": sha,
        }
        resp = requests.delete(url, headers=github_headers(token), json=payload, timeout=30)
    except requests.RequestException as e:
        return False, f"GitHub connection error: {e}"

    if resp.status_code in (200, 204):
        return True, None

    return False, f"GitHub delete error {resp.status_code}: {resp.text}"


def load_history_from_github(owner, repo, path, token, branch="main"):
    content, sha, err = github_get_file(owner, repo, path, token, branch=branch)
    if err:
        return empty_history_df(), None, err

    if content is None:
        return empty_history_df(), None, None

    try:
        df = pd.read_csv(io.StringIO(content))
    except Exception as e:
        return empty_history_df(), None, f"Could not parse GitHub CSV: {e}"

    return normalize_history_df(df), sha, None


def save_history_to_github(df, owner, repo, path, token, branch="main", sha=None):
    csv_text = normalize_history_df(df).to_csv(index=False)
    ok, err = github_put_file(
        owner=owner,
        repo=repo,
        path=path,
        token=token,
        content_text=csv_text,
        message="Update ACR QC history",
        branch=branch,
        sha=sha,
    )
    return ok, err


# =========================================================
# LOCKING
# =========================================================
def acquire_local_lock(lock_path=LOCAL_LOCK_FILE, timeout_seconds=20, stale_lock_seconds=300):
    start = time.time()
    while True:
        if lock_path.exists():
            age = time.time() - lock_path.stat().st_mtime
            if age > stale_lock_seconds:
                try:
                    lock_path.unlink()
                except Exception:
                    pass

        try:
            with open(lock_path, "x", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "lock_id": str(uuid.uuid4()),
                            "created_at": datetime.now().isoformat(timespec="seconds"),
                        }
                    )
                )
            return True
        except FileExistsError:
            pass
        except Exception:
            pass

        if time.time() - start > timeout_seconds:
            return False

        time.sleep(0.5)


def release_local_lock(lock_path=LOCAL_LOCK_FILE):
    try:
        if lock_path.exists():
            lock_path.unlink()
    except Exception:
        pass


def github_lock_path(csv_path):
    csv_path = csv_path.strip("/")
    if "/" in csv_path:
        parent = csv_path.rsplit("/", 1)[0]
        return f"{parent}/acr_qc_history.lock.json"
    return "acr_qc_history.lock.json"


def acquire_github_lock(owner, repo, csv_path, token, branch="main", timeout_seconds=25, stale_lock_seconds=300):
    lock_path = github_lock_path(csv_path)
    start = time.time()

    while True:
        content, sha, err = github_get_file(owner, repo, lock_path, token, branch=branch)
        now = datetime.now()

        if err:
            return False, None, f"GitHub lock read error: {err}"

        if content is not None:
            try:
                payload = json.loads(content)
                created_at = datetime.fromisoformat(payload.get("created_at"))
                if (now - created_at).total_seconds() > stale_lock_seconds:
                    github_delete_file(
                        owner=owner,
                        repo=repo,
                        path=lock_path,
                        token=token,
                        message="Remove stale ACR QC lock",
                        branch=branch,
                        sha=sha,
                    )
                else:
                    if time.time() - start > timeout_seconds:
                        return False, None, "Could not acquire GitHub lock. Another user may be saving right now."
                    time.sleep(1.0)
                    continue
            except Exception:
                if time.time() - start > timeout_seconds:
                    return False, None, "Could not interpret existing GitHub lock file."
                time.sleep(1.0)
                continue

        lock_payload = json.dumps(
            {
                "lock_id": str(uuid.uuid4()),
                "created_at": now.isoformat(timespec="seconds"),
                "owner": owner,
                "repo": repo,
                "path": csv_path,
            },
            indent=2,
        )

        ok, _ = github_put_file(
            owner=owner,
            repo=repo,
            path=lock_path,
            token=token,
            content_text=lock_payload,
            message="Acquire ACR QC lock",
            branch=branch,
            sha=None,
        )

        if ok:
            _, latest_sha, latest_err = github_get_file(owner, repo, lock_path, token, branch=branch)
            if latest_err:
                return False, None, latest_err
            return True, latest_sha, None

        if time.time() - start > timeout_seconds:
            return False, None, "Could not acquire GitHub lock. Another user may be saving right now."

        time.sleep(1.0)


def release_github_lock(owner, repo, csv_path, token, branch="main"):
    lock_path = github_lock_path(csv_path)
    content, sha, err = github_get_file(owner, repo, lock_path, token, branch=branch)
    if err:
        return False, err
    if content is None or not sha:
        return True, None

    return github_delete_file(
        owner=owner,
        repo=repo,
        path=lock_path,
        token=token,
        message="Release ACR QC lock",
        branch=branch,
        sha=sha,
    )


# =========================================================
# HISTORY STORAGE
# =========================================================
def load_history(local_only=True, github_cfg=None):
    if not local_only and github_cfg:
        return load_history_from_github(
            github_cfg["owner"],
            github_cfg["repo"],
            github_cfg["path"],
            github_cfg["token"],
            branch=github_cfg["branch"],
        )

    if LOCAL_HISTORY_CSV.exists():
        df = pd.read_csv(LOCAL_HISTORY_CSV)
    else:
        df = empty_history_df()

    return normalize_history_df(df), None, None


@st.cache_data(ttl=60, show_spinner=False)
def cached_load_history(local_only=True, github_cfg=None):
    return load_history(local_only=local_only, github_cfg=github_cfg)


def save_history_local(df):
    normalize_history_df(df).to_csv(LOCAL_HISTORY_CSV, index=False)


def append_results_to_history(
    results,
    session_label,
    timestamp,
    site_name,
    scanner_name,
    scanner_id,
    local_only=True,
    github_cfg=None,
    sha=None,
):
    history, _, _ = load_history(local_only=local_only, github_cfg=github_cfg)

    rows = []
    for r in results:
        rows.append(
            {
                "timestamp": timestamp,
                "session_label": session_label,
                "site_name": site_name,
                "scanner_name": scanner_name,
                "scanner_id": scanner_id,
                "test_name": r["test_name"],
                "value": r["value"],
                "unit": r["unit"],
                "criteria": r["criteria"],
                "status": r["status"],
                "details": r["details"],
                "source_file": r.get("source_file", ""),
                "sequence_label": r.get("sequence_label", ""),
            }
        )

    updated = pd.concat([history, pd.DataFrame(rows)], ignore_index=True)
    updated = normalize_history_df(updated)

    if local_only:
        save_history_local(updated)
        return updated, None, None

    ok, err = save_history_to_github(
        updated,
        github_cfg["owner"],
        github_cfg["repo"],
        github_cfg["path"],
        github_cfg["token"],
        branch=github_cfg["branch"],
        sha=sha,
    )
    return updated, ok, err


def save_results_with_lock(
    results,
    session_label,
    timestamp,
    site_name,
    scanner_name,
    scanner_id,
    local_only=True,
    github_cfg=None,
):
    if local_only:
        locked = acquire_local_lock()
        if not locked:
            return None, "Could not acquire local file lock. Please try again."
        try:
            updated, _, _ = append_results_to_history(
                results,
                session_label,
                timestamp,
                site_name,
                scanner_name,
                scanner_id,
                local_only=True,
                github_cfg=None,
                sha=None,
            )
            return updated, None
        finally:
            release_local_lock()

    ok, _, lock_err = acquire_github_lock(
        github_cfg["owner"],
        github_cfg["repo"],
        github_cfg["path"],
        github_cfg["token"],
        branch=github_cfg["branch"],
    )
    if not ok:
        return None, lock_err

    try:
        _, existing_sha, load_err = load_history(local_only=False, github_cfg=github_cfg)
        if load_err:
            return None, load_err

        updated, _, save_err = append_results_to_history(
            results,
            session_label,
            timestamp,
            site_name,
            scanner_name,
            scanner_id,
            local_only=False,
            github_cfg=github_cfg,
            sha=existing_sha,
        )
        if save_err:
            return None, save_err

        return updated, None
    finally:
        release_github_lock(
            github_cfg["owner"],
            github_cfg["repo"],
            github_cfg["path"],
            github_cfg["token"],
            branch=github_cfg["branch"],
        )


# =========================================================
# PDF STYLES / HELPERS
# =========================================================
def get_pdf_styles():
    styles = getSampleStyleSheet()

    styles.add(
        ParagraphStyle(
            name="ReportTitleCustom",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#183A63"),
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportSubTitleCustom",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=10,
            leading=13,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#4B5563"),
            spaceAfter=8,
            wordWrap="LTR",
            splitLongWords=1,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SectionHeadingCustom",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=14,
            alignment=TA_LEFT,
            textColor=colors.HexColor("#183A63"),
            spaceBefore=6,
            spaceAfter=6,
            wordWrap="LTR",
            splitLongWords=1,
        )
    )
    styles.add(
        ParagraphStyle(
            name="MetaCustom",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=11,
            alignment=TA_LEFT,
            textColor=colors.black,
            wordWrap="LTR",
            splitLongWords=1,
        )
    )
    styles.add(
        ParagraphStyle(
            name="TableCellCustom",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=8,
            leading=10,
            alignment=TA_LEFT,
            textColor=colors.black,
            wordWrap="LTR",
            splitLongWords=1,
        )
    )
    styles.add(
        ParagraphStyle(
            name="TableHeaderCustom",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=8,
            leading=10,
            alignment=TA_LEFT,
            textColor=colors.white,
            wordWrap="LTR",
            splitLongWords=1,
        )
    )
    styles.add(
        ParagraphStyle(
            name="PassBadge",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=10,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#166534"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="FailBadge",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=10,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#991B1B"),
        )
    )

    return styles


def add_pdf_header(elements, styles, title, subtitle="", site_name="", scanner_name="", include_logo=True):
    if include_logo and LOGO_PATH.exists():
        try:
            logo = RLImage(str(LOGO_PATH), width=110, height=40)
            elements.append(logo)
            elements.append(Spacer(1, 6))
        except Exception:
            pass

    elements.append(Paragraph(title, styles["ReportTitleCustom"]))
    if subtitle:
        elements.append(Paragraph(subtitle, styles["ReportSubTitleCustom"]))
    if site_name or scanner_name:
        meta_line = " | ".join([x for x in [site_name, scanner_name] if x])
        if meta_line:
            elements.append(Paragraph(meta_line, styles["ReportSubTitleCustom"]))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#CBD5E1")))
    elements.append(Spacer(1, 8))


def status_paragraph(status, styles):
    s = str(status).upper().strip()
    if s == "PASS":
        return Paragraph("PASS", styles["PassBadge"])
    return Paragraph("FAIL", styles["FailBadge"])


def format_value_unit(value, unit):
    if pd.isna(value):
        return ""
    try:
        if float(value).is_integer():
            value = int(value)
    except Exception:
        pass
    return f"{value} {unit}".strip()


def format_session_date(ts):
    ts = str(ts)
    return ts.split("T")[0] if ts else ""


def build_results_table(results_df, styles):
    df = normalize_history_df(results_df).copy()
    df = sort_tests_acr(df)

    cell_style = styles["TableCellCustom"]
    header_style = styles["TableHeaderCustom"]

    table_data = [[
        Paragraph("Test", header_style),
        Paragraph("Value", header_style),
        Paragraph("Tolerance", header_style),
        Paragraph("Status", header_style),
    ]]

    for _, row in df.iterrows():
        value_text = format_value_unit(row["value"], row["unit"])
        criteria_text = str(row["criteria"]) if pd.notna(row["criteria"]) else ""
        test_text = str(row["test_name"]) if pd.notna(row["test_name"]) else ""

        table_data.append([
            Paragraph(test_text, cell_style),
            Paragraph(value_text, cell_style),
            Paragraph(criteria_text, cell_style),
            status_paragraph(row["status"], styles),
        ])

    table = Table(
        table_data,
        colWidths=[170, 80, 210, 50],
        repeatRows=1,
        splitByRow=1,
    )

    ts = TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F4E79")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9CA3AF")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.HexColor("#EEF3F8")]),
    ])

    for idx, (_, row) in enumerate(df.iterrows(), start=1):
        if str(row["status"]).upper() == "PASS":
            ts.add("BACKGROUND", (3, idx), (3, idx), colors.HexColor("#DCFCE7"))
        else:
            ts.add("BACKGROUND", (3, idx), (3, idx), colors.HexColor("#FEE2E2"))

    table.setStyle(ts)
    return table


# =========================================================
# CHARTS / PDF
# =========================================================
def fig_to_rl_image(fig, width=500):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=160)
    buf.seek(0)
    img_reader = ImageReader(buf)
    iw, ih = img_reader.getSize()
    aspect = ih / float(iw) if iw else 0.58
    return RLImage(buf, width=width, height=width * aspect)


def add_reference_lines(ax, selected_test):
    if selected_test == "Slice Thickness Accuracy":
        ax.axhline(4.3, linestyle="--", alpha=0.7)
        ax.axhline(5.7, linestyle="--", alpha=0.7)
    elif selected_test == "Slice Position Accuracy":
        ax.axhline(5.0, linestyle="--", alpha=0.7)
        ax.axhline(-5.0, linestyle="--", alpha=0.7)
    elif selected_test == "Percentage Signal Ghosting":
        ax.axhline(2.5, linestyle="--", alpha=0.7)
    elif selected_test in ["Image Uniformity T1", "Image Uniformity T2"]:
        ax.axhline(87.5, linestyle="--", alpha=0.7)
    elif selected_test == "Geometric Accuracy":
        ax.axhline(188, linestyle="--", alpha=0.7)
        ax.axhline(192, linestyle="--", alpha=0.7)


def create_trend_chart(df, test_name):
    sub = build_frontpage_trend_df(df)
    if sub.empty:
        return None, None

    sub = sub[sub["test_name"] == test_name].copy()
    sub = sub.dropna(subset=["timestamp_dt", "value"]).sort_values("timestamp_dt")
    if sub.empty:
        return None, None

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(sub["timestamp_dt"], sub["value"], marker="o")
    ax.set_title(test_name)
    unit = sub["unit"].dropna().iloc[0] if not sub["unit"].dropna().empty else ""
    ax.set_xlabel("Timestamp")
    ax.set_ylabel(f"Value ({unit})")
    ax.grid(True, alpha=0.3)
    add_reference_lines(ax, test_name)
    fig.autofmt_xdate()

    chart_path = CHARTS_DIR / f"{test_name.replace('/', '_').replace(' ', '_')}.png"
    fig.savefig(chart_path, bbox_inches="tight", dpi=160)
    return fig, chart_path


def build_pdf_report(results_df, history_df, site_name, scanner_name, session_label, timestamp_str):
    safe_scanner = sanitize_filename(scanner_name or "scanner")
    safe_date = format_session_date(timestamp_str) or datetime.now().strftime("%Y-%m-%d")
    pdf_path = REPORTS_DIR / f"ACR_QC_Report_{safe_scanner}_{safe_date}.pdf"

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=30,
        bottomMargin=30,
    )
    styles = get_pdf_styles()
    elements = []

    results_df = normalize_history_df(results_df).copy()
    results_df = sort_tests_acr(results_df)

    add_pdf_header(
        elements,
        styles,
        title="ACR MRI Large Phantom QC Compliance Report",
        subtitle="Formal session summary with parsed measurements and trend review",
        site_name=site_name,
        scanner_name=scanner_name,
        include_logo=True,
    )

    elements.append(Paragraph("Session Information", styles["SectionHeadingCustom"]))
    elements.append(Paragraph(f"<b>Session label:</b> {session_label}", styles["MetaCustom"]))
    elements.append(Paragraph(f"<b>Timestamp:</b> {timestamp_str}", styles["MetaCustom"]))
    elements.append(Paragraph(f"<b>Session date:</b> {format_session_date(timestamp_str)}", styles["MetaCustom"]))
    elements.append(Spacer(1, 8))

    overall = "PASS" if (results_df["status"] == "PASS").all() else "FAIL"
    overall_color = "#166534" if overall == "PASS" else "#991B1B"
    elements.append(
        Paragraph(
            f'<font color="{overall_color}"><b>Overall result: {overall}</b></font>',
            styles["SectionHeadingCustom"],
        )
    )
    elements.append(Spacer(1, 4))

    elements.append(Paragraph("Results Summary", styles["SectionHeadingCustom"]))
    elements.append(build_results_table(results_df, styles))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("Parsed Details", styles["SectionHeadingCustom"]))
    for _, row in results_df.iterrows():
        elements.append(Paragraph(f"<b>{row['test_name']}:</b> {row['details']}", styles["MetaCustom"]))
        elements.append(Spacer(1, 4))

    elements.append(Spacer(1, 8))
    elements.append(Paragraph("Trend Charts", styles["SectionHeadingCustom"]))
    added_any_chart = False

    for test_name in results_df["test_name"].tolist():
        fig, _ = create_trend_chart(history_df, test_name)
        if fig is not None:
            added_any_chart = True
            elements.append(Spacer(1, 6))
            elements.append(Paragraph(test_name, styles["MetaCustom"]))
            elements.append(fig_to_rl_image(fig, width=500))
            plt.close(fig)
            elements.append(Spacer(1, 8))

    if not added_any_chart:
        elements.append(Paragraph("No historical numeric data yet for trend charts.", styles["MetaCustom"]))

    doc.build(elements)
    return pdf_path


def build_session_summary_pdf(history_df, site_name=None, scanner_name=None, scanner_id=None):
    scanner_fragment = sanitize_filename(scanner_name or scanner_id or "scanner")
    pdf_path = REPORTS_DIR / f"ACR_QC_Session_Summary_{scanner_fragment}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=30,
        bottomMargin=30,
    )

    styles = get_pdf_styles()
    elements = []

    df = normalize_history_df(history_df).copy()
    if df.empty:
        add_pdf_header(
            elements,
            styles,
            title="ACR MRI Large Phantom QC Session Summary",
            subtitle="Historical report",
            site_name=site_name or "",
            scanner_name=scanner_name or "",
            include_logo=True,
        )
        elements.append(Paragraph("No history data available.", styles["MetaCustom"]))
        doc.build(elements)
        return pdf_path

    if scanner_id:
        df = df[df["scanner_id"] == scanner_id].copy()
    else:
        if site_name:
            df = df[df["site_name"] == site_name].copy()
        if scanner_name:
            df = df[df["scanner_name"] == scanner_name].copy()

    if df.empty:
        add_pdf_header(
            elements,
            styles,
            title="ACR MRI Large Phantom QC Session Summary",
            subtitle="Historical report",
            site_name=site_name or "",
            scanner_name=scanner_name or "",
            include_logo=True,
        )
        elements.append(Paragraph("No matching session history found for the selected scanner.", styles["MetaCustom"]))
        doc.build(elements)
        return pdf_path

    df["timestamp_dt"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp_dt", ascending=False)

    add_pdf_header(
        elements,
        styles,
        title="ACR MRI Large Phantom QC Session Summary",
        subtitle="All recorded sessions for the selected system",
        site_name=site_name or "",
        scanner_name=scanner_name or "",
        include_logo=True,
    )

    group_cols = ["timestamp", "session_label", "site_name", "scanner_name", "scanner_id"]
    grouped_items = list(df.groupby(group_cols, sort=False))

    for idx, ((timestamp, session_label, g_site, g_scanner, g_scanner_id), g) in enumerate(grouped_items):
        g = sort_tests_acr(g.copy())
        overall = "PASS" if (g["status"] == "PASS").all() else "FAIL"
        overall_color = "#166534" if overall == "PASS" else "#991B1B"

        elements.append(Paragraph(f"Session {idx + 1}", styles["SectionHeadingCustom"]))
        elements.append(Paragraph(f"<b>Session date:</b> {format_session_date(timestamp)}", styles["MetaCustom"]))
        elements.append(Paragraph(f"<b>Timestamp:</b> {timestamp}", styles["MetaCustom"]))
        elements.append(Paragraph(f"<b>Session label:</b> {session_label}", styles["MetaCustom"]))
        elements.append(Paragraph(f"<b>Site:</b> {g_site}", styles["MetaCustom"]))
        elements.append(Paragraph(f"<b>Scanner:</b> {g_scanner}", styles["MetaCustom"]))
        elements.append(Paragraph(f"<b>System ID:</b> {g_scanner_id}", styles["MetaCustom"]))
        elements.append(
            Paragraph(
                f'<font color="{overall_color}"><b>Overall result:</b> {overall}</font>',
                styles["MetaCustom"],
            )
        )
        elements.append(Spacer(1, 8))
        elements.append(build_results_table(g, styles))

        if idx < len(grouped_items) - 1:
            elements.append(PageBreak())

    doc.build(elements)
    return pdf_path


def build_single_session_pdf(session_df):
    pdf_path = REPORTS_DIR / f"ACR_QC_Selected_Session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=30,
        bottomMargin=30,
    )

    styles = get_pdf_styles()
    elements = []

    df = normalize_history_df(session_df).copy()
    df = sort_tests_acr(df)

    if df.empty:
        add_pdf_header(
            elements,
            styles,
            title="ACR MRI Large Phantom QC Session Report",
            subtitle="Selected historical session",
            include_logo=True,
        )
        elements.append(Paragraph("No data found for the selected session.", styles["MetaCustom"]))
        doc.build(elements)
        return pdf_path

    first = df.iloc[0]
    overall = "PASS" if (df["status"] == "PASS").all() else "FAIL"
    overall_color = "#166534" if overall == "PASS" else "#991B1B"

    add_pdf_header(
        elements,
        styles,
        title="ACR MRI Large Phantom QC Session Report",
        subtitle="Formal single-session report generated from stored history",
        site_name=str(first["site_name"]),
        scanner_name=str(first["scanner_name"]),
        include_logo=True,
    )

    elements.append(Paragraph("Session Information", styles["SectionHeadingCustom"]))
    elements.append(Paragraph(f"<b>Session date:</b> {format_session_date(first['timestamp'])}", styles["MetaCustom"]))
    elements.append(Paragraph(f"<b>Timestamp:</b> {first['timestamp']}", styles["MetaCustom"]))
    elements.append(Paragraph(f"<b>Session label:</b> {first['session_label']}", styles["MetaCustom"]))
    elements.append(Paragraph(f"<b>System ID:</b> {first['scanner_id']}", styles["MetaCustom"]))
    elements.append(Spacer(1, 8))

    elements.append(
        Paragraph(
            f'<font color="{overall_color}"><b>Overall result: {overall}</b></font>',
            styles["SectionHeadingCustom"],
        )
    )
    elements.append(Spacer(1, 6))

    elements.append(Paragraph("Results Summary", styles["SectionHeadingCustom"]))
    elements.append(build_results_table(df, styles))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("Details", styles["SectionHeadingCustom"]))
    for _, row in df.iterrows():
        elements.append(Paragraph(f"<b>{row['test_name']}:</b> {row['details']}", styles["MetaCustom"]))
        elements.append(Spacer(1, 4))

    doc.build(elements)
    return pdf_path


# =========================================================
# APP
# =========================================================
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption(
    "Upload ACR MRI phantom .txt files, auto-evaluate pass/fail, "
    "save history with timestamp, and generate PDF reports from current or historical sessions."
)

if "session_saved" not in st.session_state:
    st.session_state.session_saved = False

if "parsed_results" not in st.session_state:
    st.session_state.parsed_results = []

if "combined_results" not in st.session_state:
    st.session_state.combined_results = []

if "last_upload_signature" not in st.session_state:
    st.session_state.last_upload_signature = None

if "pdf_report_bytes" not in st.session_state:
    st.session_state.pdf_report_bytes = None
    st.session_state.pdf_report_name = None

if "summary_pdf_bytes" not in st.session_state:
    st.session_state.summary_pdf_bytes = None
    st.session_state.summary_pdf_name = None

if "selected_session_pdf_bytes" not in st.session_state:
    st.session_state.selected_session_pdf_bytes = None
    st.session_state.selected_session_pdf_name = None

try:
    SECRET_GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]
except Exception:
    SECRET_GITHUB_TOKEN = ""

try:
    repo_full = st.secrets["GITHUB_REPO"]
    owner, repo_name = repo_full.split("/", 1)
except Exception:
    owner = DEFAULT_GITHUB_OWNER
    repo_name = DEFAULT_GITHUB_REPO

try:
    github_branch = st.secrets["GITHUB_BRANCH"]
except Exception:
    github_branch = DEFAULT_GITHUB_BRANCH

github_cfg = {
    "owner": owner,
    "repo": repo_name,
    "branch": github_branch,
    "path": DEFAULT_GITHUB_CSV_PATH,
    "token": SECRET_GITHUB_TOKEN,
}
USE_GITHUB = github_is_ready(github_cfg)

history_df, _, preload_err = cached_load_history(
    local_only=not USE_GITHUB,
    github_cfg=github_cfg if USE_GITHUB else None,
)
if preload_err:
    st.error(preload_err)
    history_df = empty_history_df()

history_df = normalize_history_df(history_df)
known_sites = sorted([x for x in history_df["site_name"].unique().tolist() if x])

with st.sidebar:
    st.header("Session info")

    if known_sites:
        site_mode = st.radio("Site entry mode", ["Select existing", "Enter new"], horizontal=False)
        if site_mode == "Select existing":
            site_name = st.selectbox("Site / Hospital", options=known_sites, index=0)
        else:
            site_name = st.text_input("Site / Hospital", value="")
    else:
        site_name = st.text_input("Site / Hospital", value="")

    filtered_scanners = []
    if site_name.strip():
        filtered_scanners = sorted(
            [
                x
                for x in history_df.loc[history_df["site_name"] == site_name.strip(), "scanner_name"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
                if x
            ]
        )

    if filtered_scanners:
        scanner_mode = st.radio("Scanner entry mode", ["Select existing", "Enter new"], horizontal=False)
        if scanner_mode == "Select existing":
            scanner_name = st.selectbox("Scanner / System", options=filtered_scanners, index=0)
        else:
            scanner_name = st.text_input("Scanner / System", value="")
    else:
        scanner_name = st.text_input("Scanner / System", value="")

    scanner_id = build_scanner_id(site_name, scanner_name)
    st.caption(f"System ID: {scanner_id}")

    session_label = st.text_input("Session label", value="Monthly QC")
    custom_timestamp = st.text_input("Timestamp (optional, ISO format)", value="")

    if custom_timestamp.strip():
        if not validate_iso_timestamp(custom_timestamp.strip()):
            st.error("Timestamp must be valid ISO format, e.g. 2026-03-30T14:20:00")
            st.stop()
        timestamp_str = custom_timestamp.strip()
    else:
        timestamp_str = datetime.now().isoformat(timespec="seconds")

    if USE_GITHUB:
        st.success("GitHub history storage is active.")
    else:
        st.warning("GitHub not fully configured. Using local history file.")

uploaded_files = st.file_uploader(
    "Upload the phantom .txt result files",
    type=["txt"],
    accept_multiple_files=True,
)

current_upload_signature = (
    tuple(sorted([f.name for f in uploaded_files])) if uploaded_files else (),
    site_name.strip(),
    scanner_name.strip(),
    session_label.strip(),
    timestamp_str.strip(),
)

if st.session_state.last_upload_signature != current_upload_signature:
    st.session_state.session_saved = False
    st.session_state.last_upload_signature = current_upload_signature

parsed_results = []
combined_results = []
results_df = pd.DataFrame()

if uploaded_files:
    with st.spinner("Parsing files..."):
        for uploaded_file in uploaded_files:
            text = read_text_file(uploaded_file)
            result = infer_parser(uploaded_file.name, text)
            result["source_file"] = uploaded_file.name
            result["sequence_label"] = detect_sequence_label(uploaded_file.name, text)
            parsed_results.append(result)

    combined_results = combine_session_results(parsed_results)

    st.session_state.parsed_results = parsed_results
    st.session_state.combined_results = combined_results

    results_df = pd.DataFrame(combined_results)
    results_df = sort_tests_acr(results_df)

    st.subheader("Current session results")
    display_cols = [
        c for c in
        ["source_file", "sequence_label", "test_name", "value", "unit", "criteria", "status", "details"]
        if c in results_df.columns
    ]
    st.dataframe(
        results_df[display_cols],
        use_container_width=True,
    )

    overall = "PASS" if (results_df["status"] == "PASS").all() else "FAIL"
    st.metric("Overall session result", overall)

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Save session to history", type="primary", key="save_session_to_history"):
            if not site_name.strip() or not scanner_name.strip():
                st.error("Please enter both Site / Hospital and Scanner / System.")
            else:
                history_after_save, save_err = save_results_with_lock(
                    combined_results,
                    session_label,
                    timestamp_str,
                    site_name,
                    scanner_name,
                    scanner_id,
                    local_only=not USE_GITHUB,
                    github_cfg=github_cfg if USE_GITHUB else None,
                )
                if save_err:
                    st.error(save_err)
                else:
                    st.session_state.session_saved = True
                    history_df = normalize_history_df(history_after_save)
                    cached_load_history.clear()
                    st.success(f"Saved {len(combined_results)} results to history for system: {scanner_id}")

    with col2:
        if st.button("Generate PDF report", key="generate_pdf_report"):
            if not site_name.strip() or not scanner_name.strip():
                st.error("Please enter both Site / Hospital and Scanner / System.")
            else:
                temp_history = history_df.copy()
                if uploaded_files and combined_results:
                    current_rows = pd.DataFrame(
                        [
                            {
                                "timestamp": timestamp_str,
                                "session_label": session_label,
                                "site_name": site_name,
                                "scanner_name": scanner_name,
                                "scanner_id": scanner_id,
                                "test_name": r["test_name"],
                                "value": r["value"],
                                "unit": r["unit"],
                                "criteria": r["criteria"],
                                "status": r["status"],
                                "details": r["details"],
                                "source_file": r.get("source_file", ""),
                                "sequence_label": r.get("sequence_label", ""),
                            }
                            for r in combined_results
                        ]
                    )
                    temp_history = pd.concat([temp_history, current_rows], ignore_index=True)
                    temp_history = normalize_history_df(temp_history)

                pdf_path = build_pdf_report(
                    results_df,
                    temp_history,
                    site_name,
                    scanner_name,
                    session_label,
                    timestamp_str,
                )

                with open(pdf_path, "rb") as f:
                    st.session_state.pdf_report_bytes = f.read()
                    st.session_state.pdf_report_name = pdf_path.name

                st.success(f"PDF report created: {pdf_path.name}")

    with col3:
        if st.button("Generate session summary PDF", key="generate_session_summary_pdf"):
            if not site_name.strip() or not scanner_name.strip():
                st.error("Please enter both Site / Hospital and Scanner / System.")
            else:
                summary_history = history_df.copy()

                if uploaded_files and combined_results and not st.session_state.session_saved:
                    current_rows = pd.DataFrame(
                        [
                            {
                                "timestamp": timestamp_str,
                                "session_label": session_label,
                                "site_name": site_name,
                                "scanner_name": scanner_name,
                                "scanner_id": scanner_id,
                                "test_name": r["test_name"],
                                "value": r["value"],
                                "unit": r["unit"],
                                "criteria": r["criteria"],
                                "status": r["status"],
                                "details": r["details"],
                                "source_file": r.get("source_file", ""),
                                "sequence_label": r.get("sequence_label", ""),
                            }
                            for r in combined_results
                        ]
                    )
                    summary_history = pd.concat([summary_history, current_rows], ignore_index=True)
                    summary_history = normalize_history_df(summary_history)

                pdf_path = build_session_summary_pdf(
                    summary_history,
                    site_name=site_name,
                    scanner_name=scanner_name,
                    scanner_id=scanner_id,
                )

                with open(pdf_path, "rb") as f:
                    st.session_state.summary_pdf_bytes = f.read()
                    st.session_state.summary_pdf_name = pdf_path.name

                st.success(f"Session summary PDF created: {pdf_path.name}")

    if st.session_state.pdf_report_bytes:
        st.download_button(
            "Download PDF report",
            data=st.session_state.pdf_report_bytes,
            file_name=st.session_state.pdf_report_name,
            mime="application/pdf",
            key="download_pdf_report",
        )

    if st.session_state.summary_pdf_bytes:
        st.download_button(
            "Download session summary PDF",
            data=st.session_state.summary_pdf_bytes,
            file_name=st.session_state.summary_pdf_name,
            mime="application/pdf",
            key="download_summary_pdf",
        )

else:
    parsed_results = st.session_state.get("parsed_results", [])
    combined_results = st.session_state.get("combined_results", [])
    if combined_results:
        results_df = pd.DataFrame(combined_results)
        results_df = sort_tests_acr(results_df)

# =========================================================
# TREND DATA PREP
# =========================================================
history_df, _, load_err = cached_load_history(
    local_only=not USE_GITHUB,
    github_cfg=github_cfg if USE_GITHUB else None,
)
if load_err:
    st.error(load_err)
    history_df = empty_history_df()

history_df = normalize_history_df(history_df)

current_rows_df = pd.DataFrame()
if uploaded_files and combined_results and not st.session_state.session_saved:
    current_rows_df = pd.DataFrame(
        [
            {
                "timestamp": timestamp_str,
                "session_label": session_label,
                "site_name": site_name,
                "scanner_name": scanner_name,
                "scanner_id": scanner_id,
                "test_name": r["test_name"],
                "value": r["value"],
                "unit": r["unit"],
                "criteria": r["criteria"],
                "status": r["status"],
                "details": r["details"],
                "source_file": r.get("source_file", ""),
                "sequence_label": r.get("sequence_label", ""),
            }
            for r in combined_results
        ]
    )

front_trend_df = build_frontpage_trend_df(history_df, include_current_df=current_rows_df)

# =========================================================
# FRONT PAGE SINGLE TREND PANEL
# =========================================================
st.subheader("Trend preview")

if front_trend_df.empty:
    st.info("No trend data available yet.")
else:
    panel_col1, panel_col2 = st.columns(2)

    system_options = sorted(front_trend_df["scanner_id"].dropna().astype(str).unique().tolist())

    with panel_col1:
        default_idx = 0
        if scanner_id in system_options:
            default_idx = system_options.index(scanner_id)

        selected_system = st.selectbox(
            "Select system",
            system_options,
            index=default_idx,
            key="front_system_select",
        )

    system_df = front_trend_df[front_trend_df["scanner_id"] == selected_system].copy()
    system_df = sort_tests_acr(system_df)

    test_options = system_df["test_name"].dropna().astype(str).unique().tolist()
    test_options = sorted(test_options, key=acr_sort_key)

    with panel_col2:
        selected_test = st.selectbox(
            "Select test",
            test_options,
            key="front_test_select",
        )

    timestamp_options = (
        history_df.loc[history_df["scanner_id"] == selected_system, "timestamp"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    timestamp_options = sorted(timestamp_options, reverse=True)

    if timestamp_options:
        selected_timestamp = st.selectbox(
            "Select session timestamp",
            timestamp_options,
            key="front_timestamp_select",
        )
    else:
        selected_timestamp = None
        st.info("No saved session timestamps available for the selected system.")

    plot_df = system_df[system_df["test_name"] == selected_test].copy().sort_values("timestamp_dt")
    plot_df = plot_df.dropna(subset=["timestamp_dt", "value"])

    if plot_df.empty:
        st.warning("No data available for this selection.")
    else:
        latest = plot_df.iloc[-1]["value"]
        mean_val = plot_df["value"].mean()
        min_val = plot_df["value"].min()
        max_val = plot_df["value"].max()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Latest", f"{latest:.3f}")
        m2.metric("Mean", f"{mean_val:.3f}")
        m3.metric("Min", f"{min_val:.3f}")
        m4.metric("Max", f"{max_val:.3f}")

        fig, ax = plt.subplots(figsize=(9, 4.2))
        ax.plot(plot_df["timestamp_dt"], plot_df["value"], marker="o")
        unit = plot_df["unit"].dropna().iloc[0] if not plot_df["unit"].dropna().empty else ""
        ax.set_title(f"{selected_test} | {selected_system}")
        ax.set_xlabel("Timestamp")
        ax.set_ylabel(f"Value ({unit})")
        ax.grid(True, alpha=0.3)
        add_reference_lines(ax, selected_test)
        fig.autofmt_xdate()
        st.pyplot(fig)
        plt.close(fig)

        with st.expander("Show trend data table"):
            st.dataframe(
                plot_df[
                    [
                        "timestamp",
                        "site_name",
                        "scanner_name",
                        "scanner_id",
                        "session_label",
                        "test_name",
                        "value",
                        "unit",
                        "status",
                        "details",
                    ]
                ],
                use_container_width=True,
            )

        csv_bytes = plot_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download trend CSV",
            data=csv_bytes,
            file_name=f"{selected_test}_{selected_system}_trend.csv".replace(" ", "_").replace("/", "_"),
            mime="text/csv",
            key="download_trend_csv",
        )

    st.subheader("Print selected session")

    if st.button("Generate PDF for selected session", key="front_selected_session_pdf"):
        if not selected_system or not selected_timestamp:
            st.error("Please select system and session timestamp.")
        else:
            session_df = build_single_session_df(history_df, selected_system, selected_timestamp)

            if session_df.empty:
                st.warning("No data found for selected session.")
            else:
                pdf_path = build_single_session_pdf(session_df)

                with open(pdf_path, "rb") as f:
                    st.session_state.selected_session_pdf_bytes = f.read()
                    st.session_state.selected_session_pdf_name = pdf_path.name

                st.success(f"Selected session PDF created: {pdf_path.name}")

    if st.session_state.selected_session_pdf_bytes:
        st.download_button(
            "Download selected session PDF",
            data=st.session_state.selected_session_pdf_bytes,
            file_name=st.session_state.selected_session_pdf_name,
            mime="application/pdf",
            key="front_selected_session_pdf_download",
        )
