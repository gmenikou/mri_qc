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
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    Image as RLImage,
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


def build_scanner_id(site_name: str, scanner_name: str) -> str:
    raw = f"{str(site_name).strip()}__{str(scanner_name).strip()}".lower()
    raw = re.sub(r"[^a-z0-9]+", "_", raw).strip("_")
    return raw or "unknown_scanner"


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

    return combined


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
    m1 = re.search(r"Slice 1:\s*([0-9.\-]+)", text, re.I)
    m11 = re.search(r"Slice 11:\s*([0-9.\-]+)", text, re.I)
    v1 = safe_float(m1.group(1)) if m1 else None
    v11 = safe_float(m11.group(1)) if m11 else None

    candidates = [x for x in [v1, v11] if x is not None]
    worst_abs = max([abs(x) for x in candidates], default=None)

    passed = (v1 is not None and abs(v1) <= 5) and (v11 is not None and abs(v11) <= 5)
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
    passed = (v1 is not None and v1 >= 37) and (v2 is not None and v2 >= 37)
    return {
        "test_name": "Low Contrast Detectability",
        "value": None,
        "unit": "spokes",
        "criteria": "Site/ACR target threshold, commonly >= 37 for strong performance",
        "status": "PASS" if passed else "FAIL",
        "details": f"T1 spokes: {v1}, T2 spokes: {v2}",
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
    best_value = max([v for v in [up, lo] if v is not None], default=None)
    return {
        "test_name": modality_name,
        "value": best_value,
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
# CHARTS / PDF
# =========================================================
def fig_to_rl_image(fig, width=500):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=160)
    buf.seek(0)
    return RLImage(buf, width=width, height=width * 0.58)


def add_reference_lines(ax, selected_test):
    if selected_test == "Slice Thickness Accuracy":
        ax.axhline(4.3, linestyle="--", alpha=0.7)
        ax.axhline(5.7, linestyle="--", alpha=0.7)
    elif selected_test == "Slice Position Accuracy":
        ax.axhline(5.0, linestyle="--", alpha=0.7)
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
    pdf_path = REPORTS_DIR / f"ACR_QC_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("ACR MRI Large Phantom QC Compliance Report", styles["Title"]))
    elements.append(Spacer(1, 8))
    elements.append(Paragraph(f"<b>Site:</b> {site_name}", styles["Normal"]))
    elements.append(Paragraph(f"<b>Scanner:</b> {scanner_name}", styles["Normal"]))
    elements.append(Paragraph(f"<b>Session label:</b> {session_label}", styles["Normal"]))
    elements.append(Paragraph(f"<b>Timestamp:</b> {timestamp_str}", styles["Normal"]))
    elements.append(Spacer(1, 12))

    overall = "PASS" if (results_df["status"] == "PASS").all() else "FAIL"
    elements.append(Paragraph(f"<b>Overall result:</b> {overall}", styles["Heading2"]))
    elements.append(Spacer(1, 8))

    table_data = [["Test", "Value", "Criteria", "Status"]]
    for _, row in results_df.iterrows():
        val = "" if pd.isna(row["value"]) else f"{row['value']} {row['unit']}".strip()
        table_data.append([row["test_name"], val, row["criteria"], row["status"]])

    table = Table(table_data, colWidths=[160, 90, 180, 70])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4F81BD")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.lightgrey]),
            ]
        )
    )
    elements.append(table)
    elements.append(Spacer(1, 14))

    elements.append(Paragraph("Parsed details", styles["Heading2"]))
    for _, row in results_df.iterrows():
        elements.append(Paragraph(f"<b>{row['test_name']}:</b> {row['details']}", styles["Normal"]))
        elements.append(Spacer(1, 4))

    elements.append(Spacer(1, 12))
    elements.append(Paragraph("Trend charts", styles["Heading2"]))
    added_any_chart = False

    for test_name in results_df["test_name"].tolist():
        fig, _ = create_trend_chart(history_df, test_name)
        if fig is not None:
            added_any_chart = True
            elements.append(Spacer(1, 6))
            elements.append(Paragraph(test_name, styles["Heading3"]))
            elements.append(fig_to_rl_image(fig, width=500))
            plt.close(fig)

    if not added_any_chart:
        elements.append(Paragraph("No historical numeric data yet for trend charts.", styles["Normal"]))

    doc.build(elements)
    return pdf_path


# =========================================================
# APP
# =========================================================
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption("Upload ACR MRI phantom .txt files, auto-evaluate pass/fail, save history with timestamp, and generate a PDF report with trendlines.")

if "session_saved" not in st.session_state:
    st.session_state.session_saved = False

if "parsed_results" not in st.session_state:
    st.session_state.parsed_results = []

if "combined_results" not in st.session_state:
    st.session_state.combined_results = []

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
    timestamp_str = custom_timestamp.strip() or datetime.now().isoformat(timespec="seconds")

    st.header("Dashboard behavior")
    default_to_current_scanner = st.checkbox("Default trend dashboard to current scanner", value=True)
    group_history_by_scanner = st.checkbox("Group saved history tables by scanner", value=True)
    show_detailed_preview = st.checkbox("Show detailed preview", value=False)

    if USE_GITHUB:
        st.success("GitHub history storage is active.")
    else:
        st.warning("GitHub not fully configured. Using local history file.")

uploaded_files = st.file_uploader(
    "Upload the phantom .txt result files",
    type=["txt"],
    accept_multiple_files=True,
)

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

    st.subheader("Current session results")
    st.dataframe(
        results_df[["source_file", "sequence_label", "test_name", "value", "unit", "criteria", "status", "details"]],
        use_container_width=True,
    )

    overall = "PASS" if (results_df["status"] == "PASS").all() else "FAIL"
    st.metric("Overall session result", overall)

    with st.expander("Show raw parsed per-file results"):
        raw_df = pd.DataFrame(parsed_results)
        if not raw_df.empty:
            st.dataframe(
                raw_df[["source_file", "sequence_label", "test_name", "value", "unit", "criteria", "status", "details"]],
                use_container_width=True,
            )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Save session to history", type="primary"):
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
        if st.button("Generate PDF report"):
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
                    st.download_button(
                        "Download PDF report",
                        data=f.read(),
                        file_name=pdf_path.name,
                        mime="application/pdf",
                    )
                st.success(f"PDF report created: {pdf_path}")

else:
    parsed_results = st.session_state.get("parsed_results", [])
    combined_results = st.session_state.get("combined_results", [])
    if combined_results:
        results_df = pd.DataFrame(combined_results)

# =========================================================
# OPTIONAL DETAILED PREVIEW DATA PREP
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
# DETAILED PREVIEW
# =========================================================
if show_detailed_preview:
    st.subheader("Integrated trend dashboard")

    trend_source = build_frontpage_trend_df(history_df, include_current_df=current_rows_df)

    if trend_source.empty:
        st.info("No trend data available yet.")
    else:
        numeric_tests = sorted(trend_source["test_name"].dropna().unique().tolist())

        if numeric_tests:
            left, right = st.columns([2, 1])
            with left:
                selected_test = st.selectbox("Choose test for trend analysis", numeric_tests, index=0, key="dashboard_test")
            with right:
                show_only_saved = st.checkbox("Only saved history", value=False)

            display_df = trend_source.copy()
            if show_only_saved:
                display_df = build_frontpage_trend_df(history_df)

            test_df = display_df[display_df["test_name"] == selected_test].sort_values("timestamp_dt")

            metric_cols = st.columns(4)
            if not test_df.empty:
                latest = test_df.iloc[-1]["value"]
                mean_val = test_df["value"].mean()
                min_val = test_df["value"].min()
                max_val = test_df["value"].max()

                metric_cols[0].metric("Latest", f"{latest:.3f}")
                metric_cols[1].metric("Mean", f"{mean_val:.3f}")
                metric_cols[2].metric("Min", f"{min_val:.3f}")
                metric_cols[3].metric("Max", f"{max_val:.3f}")

                systems_for_test = sorted(test_df["scanner_id"].dropna().astype(str).unique().tolist())

                if len(systems_for_test) > 1:
                    tabs = st.tabs(systems_for_test + ["Combined"])

                    for sys_name, tab in zip(systems_for_test, tabs[:-1]):
                        with tab:
                            sys_df = test_df[test_df["scanner_id"] == sys_name].copy().sort_values("timestamp_dt")
                            fig, ax = plt.subplots(figsize=(8, 4.2))
                            ax.plot(sys_df["timestamp_dt"], sys_df["value"], marker="o")
                            ax.set_title(f"{selected_test} | {sys_name}")
                            unit = sys_df["unit"].dropna().iloc[0] if not sys_df["unit"].dropna().empty else ""
                            ax.set_xlabel("Timestamp")
                            ax.set_ylabel(f"Value ({unit})")
                            ax.grid(True, alpha=0.3)
                            add_reference_lines(ax, selected_test)
                            fig.autofmt_xdate()
                            st.pyplot(fig)
                            plt.close(fig)

                            st.dataframe(
                                sys_df[
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

                    with tabs[-1]:
                        fig, ax = plt.subplots(figsize=(8, 4.2))
                        for sys_name in systems_for_test:
                            sys_df = test_df[test_df["scanner_id"] == sys_name].copy().sort_values("timestamp_dt")
                            ax.plot(sys_df["timestamp_dt"], sys_df["value"], marker="o", label=sys_name)
                        ax.set_title(f"{selected_test} | Combined systems")
                        unit = test_df["unit"].dropna().iloc[0] if not test_df["unit"].dropna().empty else ""
                        ax.set_xlabel("Timestamp")
                        ax.set_ylabel(f"Value ({unit})")
                        ax.grid(True, alpha=0.3)
                        ax.legend()
                        add_reference_lines(ax, selected_test)
                        fig.autofmt_xdate()
                        st.pyplot(fig)
                        plt.close(fig)
                else:
                    fig, ax = plt.subplots(figsize=(8, 4.2))
                    ax.plot(test_df["timestamp_dt"], test_df["value"], marker="o")
                    ax.set_title(selected_test)
                    unit = test_df["unit"].dropna().iloc[0] if not test_df["unit"].dropna().empty else ""
                    ax.set_xlabel("Timestamp")
                    ax.set_ylabel(f"Value ({unit})")
                    ax.grid(True, alpha=0.3)
                    add_reference_lines(ax, selected_test)
                    fig.autofmt_xdate()
                    st.pyplot(fig)
                    plt.close(fig)

                st.markdown("**Trend data table**")
                st.dataframe(
                    test_df[
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

                trend_csv = test_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download selected trend CSV",
                    data=trend_csv,
                    file_name=f"{selected_test.replace(' ', '_').replace('/', '_')}_trend.csv",
                    mime="text/csv",
                )
            else:
                st.info("No numeric trend data yet for the selected test.")

        st.markdown("**All saved history**")
        if history_df.empty:
            st.info("No saved history yet. Upload files and click 'Save session to history'.")
        else:
            sorted_history = history_df.copy()
            sorted_history["timestamp_dt"] = pd.to_datetime(sorted_history["timestamp"], errors="coerce")
            sorted_history = sorted_history.sort_values(["scanner_id", "timestamp_dt"], ascending=[True, True]).drop(columns=["timestamp_dt"])

            if group_history_by_scanner:
                for sys_name in sorted(sorted_history["scanner_id"].dropna().astype(str).unique().tolist()):
                    st.markdown(f"**Scanner: {sys_name}**")
                    st.dataframe(sorted_history[sorted_history["scanner_id"] == sys_name], use_container_width=True)
            else:
                st.dataframe(sorted_history, use_container_width=True)

            csv_bytes = history_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download full history CSV",
                data=csv_bytes,
                file_name="acr_qc_history.csv",
                mime="text/csv",
            )

    st.divider()
    st.subheader("History summary")

    summary_source = build_frontpage_trend_df(history_df)

    if summary_source.empty:
        st.info("No saved history yet.")
    else:
        summary_source = summary_source.sort_values(
            ["scanner_id", "site_name", "scanner_name", "test_name", "timestamp_dt"]
        ).copy()

        summary_df = (
            summary_source.groupby(["scanner_id", "site_name", "scanner_name", "test_name"], dropna=False)
            .agg(
                runs=("test_name", "count"),
                pass_count=("status", lambda s: (s == "PASS").sum()),
                fail_count=("status", lambda s: (s == "FAIL").sum()),
                latest_value=("value", "last"),
                latest_timestamp=("timestamp", "last"),
            )
            .reset_index()
        )
        st.dataframe(summary_df, use_container_width=True)
