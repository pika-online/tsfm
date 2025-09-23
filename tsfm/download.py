#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Binance K线下载脚本（现货 /api/v3/klines）

功能：
- 根据参数下载指定交易对、时间级别、起止日期范围内的K线数据
- 并发下载（线程池），失败自动重试（指数退避）
- 进度实时打印
- 导出CSV字段：[timestamps, open, high, low, close, volume, amount]
- 自动对齐期望时间轴，缺失帧或字段以0填充；不允许NaN

用法示例：
python binance_kline_downloader.py \
  --symbol BTCUSDT \
  --interval 1m \
  --start 20200102 \
  --end 20200110 \
  --threads 8 \
  --out ./BTCUSDT-1m-20200102_20200110.csv

注意：
- 该脚本使用现货REST端点 https://api.binance.com/api/v3/klines
- 单次请求最多返回1000根K线；脚本自动切片批量下载
- 若需期货数据，可将 BASE_URL 调整为合约接口（例如USDⓈ-M：https://fapi.binance.com）并保持路径 /fapi/v1/klines
"""

import argparse
import calendar
import concurrent.futures as cf
import datetime as dt
import json
import math
import sys
import threading
import time
from typing import List, Tuple, Dict

import requests

BASE_URL = "https://api.binance.com"
KLINES_PATH = "/api/v3/klines"
MAX_LIMIT = 1000
DEFAULT_TIMEOUT = 20
MAX_RETRIES = 5

# ---- interval 解析 ----
INTERVAL_TO_MS: Dict[str, int] = {
    "1m": 60_000,
    "3m": 3 * 60_000,
    "5m": 5 * 60_000,
    "15m": 15 * 60_000,
    "30m": 30 * 60_000,
    "1h": 60 * 60_000,
    "2h": 2 * 60 * 60_000,
    "4h": 4 * 60 * 60_000,
    "6h": 6 * 60 * 60_000,
    "8h": 8 * 60 * 60_000,
    "12h": 12 * 60 * 60_000,
    "1d": 24 * 60 * 60_000,
    "3d": 3 * 24 * 60 * 60_000,
    "1w": 7 * 24 * 60 * 60_000,
    # 对于1M（月）这种可变长度K线，这里按日历真实月份推进，而不是固定毫秒；
    # 我们在构造期望时间轴时单独处理。
    "1M": -1,
}


def parse_date_yyyymmdd(s: str) -> dt.datetime:
    if len(s) != 8 or not s.isdigit():
        raise ValueError("日期请用YYYYMMDD，例如 20200102")
    # 统一使用 UTC 时区的 00:00:00
    d = dt.datetime.strptime(s, "%Y%m%d")
    return d.replace(tzinfo=dt.timezone.utc)


def month_add(d: dt.datetime, months: int) -> dt.datetime:
    """在日历意义上加减月份，保持日为该月最后一天边界安全。"""
    tz = d.tzinfo
    year = d.year + (d.month - 1 + months) // 12
    month = (d.month - 1 + months) % 12 + 1
    day = min(d.day, calendar.monthrange(year, month)[1])
    return dt.datetime(year, month, day, d.hour, d.minute, d.second, d.microsecond, tzinfo=tz)


def to_ms(ts: dt.datetime) -> int:
    """将 aware datetime 转毫秒。若传入 naive，则视为 UTC。"""
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=dt.timezone.utc)
    return int(ts.timestamp() * 1000)


def build_expected_timestamps(start: dt.datetime, end: dt.datetime, interval: str) -> List[int]:
    """构造期望的K线时间戳序列（毫秒，开盘时间对齐）包含首尾（end为闭区间）。"""
    out: List[int] = []
    if interval == "1M":
        # 月线：从start所在月的对齐点起，每月推进
        cur = start.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        # 如果起始不是月初且你希望严格从start时间开始，则保持start
        if start > cur:
            cur = start
        while cur <= end:
            out.append(to_ms(cur))
            # 下一月的第一天
            cur = month_add(cur.replace(day=1), 1)
    else:
        step = INTERVAL_TO_MS.get(interval)
        if not step or step <= 0:
            raise ValueError(f"不支持的interval: {interval}")
        # 对齐到步长边界（以Unix epoch为锚点）
        start_ms = to_ms(start)
        aligned_start = start_ms - (start_ms % step)
        end_ms = to_ms(end)
        cur = aligned_start
        while cur <= end_ms:
            if cur >= start_ms:
                out.append(cur)
            cur += step
    return out


def chunk_windows(expected_ts: List[int], interval: str) -> List[Tuple[int, int]]:
    """基于期望时间序列切分下载窗口（每窗口最多MAX_LIMIT根）。返回[(start_ms, end_ms_inclusive)]。"""
    windows: List[Tuple[int, int]] = []
    if not expected_ts:
        return windows
    # 每个窗口的起点是第0,1000,2000...根
    for i in range(0, len(expected_ts), MAX_LIMIT):
        s = expected_ts[i]
        last_index = min(i + MAX_LIMIT, len(expected_ts)) - 1
        e = expected_ts[last_index]
        # endTime 传入闭区间末尾的最后毫秒
        if interval == "1M":
            # 粗略：用下一个时间点减1毫秒；最后一窗用当前e加一个大概月长-1ms（不传endTime也行）
            if last_index + 1 < len(expected_ts):
                e_exclusive = expected_ts[last_index + 1]
                e_inclusive = e_exclusive - 1
            else:
                # 不给endTime让服务器按limit返回即可
                e_inclusive = None  # 特殊标记
        else:
            step = INTERVAL_TO_MS[interval]
            e_inclusive = e + (step - 1)
        windows.append((s, e_inclusive if e_inclusive is not None else -1))
    return windows


def http_get_with_retry(session: requests.Session, url: str, params: dict) -> list:
    """GET并带重试，返回JSON数组。"""
    backoff = 1.0
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(url, params=params, timeout=DEFAULT_TIMEOUT)
            if resp.status_code == 200:
                return resp.json()
            else:
                sys.stderr.write(f"HTTP {resp.status_code}: {resp.text}\n")
        except Exception as e:
            sys.stderr.write(f"请求异常（第{attempt}次）：{e}\n")
        if attempt < MAX_RETRIES:
            time.sleep(backoff)
            backoff *= 2
    raise RuntimeError(f"请求失败，已重试{MAX_RETRIES}次: {url} {json.dumps(params, ensure_ascii=False)}")


def fetch_window(session: requests.Session, symbol: str, interval: str, s_ms: int, e_ms_inclusive: int) -> list:
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": MAX_LIMIT,
        "startTime": s_ms,
    }
    if e_ms_inclusive >= 0:
        params["endTime"] = e_ms_inclusive
    return http_get_with_retry(session, BASE_URL + KLINES_PATH, params)


def download_klines(symbol: str, interval: str, start: dt.datetime, end: dt.datetime, threads: int) -> List[list]:
    symbol = symbol.upper()
    expected_ts = build_expected_timestamps(start, end, interval)
    windows = chunk_windows(expected_ts, interval)
    total_windows = len(windows)

    if total_windows == 0:
        return []

    results = []
    results_lock = threading.Lock()
    counter = 0
    counter_lock = threading.Lock()

    def worker(win):
        nonlocal counter
        s_ms, e_ms = win
        with requests.Session() as sess:
            data = fetch_window(sess, symbol, interval, s_ms, e_ms)
        with results_lock:
            results.extend(data)
        with counter_lock:
            counter += 1
            pct = counter * 100.0 / total_windows
            print(f"进度: {counter}/{total_windows} ({pct:.2f}%)", flush=True)

    # 线程池
    max_workers = max(1, int(threads))
    with cf.ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(executor.map(worker, windows))

    return results


def normalize_and_fill(raw_rows: List[list], expected_ts: List[int]) -> List[dict]:
    """将原始返回（乱序/重复可能）规整为去重、按时间对齐的数据；缺失帧补0；字段缺失补0。"""
    # 筛成以开盘时间为键的最后一次出现（或任意一次）
    by_open_time: Dict[int, list] = {}
    for r in raw_rows:
        # Binance返回：
        # 0 open time (ms)
        # 1 open, 2 high, 3 low, 4 close, 5 volume (base)
        # 6 close time, 7 quote asset volume, 8 trade count,
        # 9 taker buy base volume, 10 taker buy quote volume, 11 ignore
        try:
            t = int(r[0])
            by_open_time[t] = r
        except Exception:
            continue

    out_rows: List[dict] = []
    for t in expected_ts:
        r = by_open_time.get(t)
        if r is None:
            # 缺失帧，全部填0
            out_rows.append({
                "timestamps": t,
                "open": 0.0,
                "high": 0.0,
                "low": 0.0,
                "close": 0.0,
                "volume": 0.0,
                "amount": 0.0,
            })
        else:
            def safe_float(x):
                try:
                    return float(x)
                except Exception:
                    return 0.0
            out_rows.append({
                "timestamps": t,
                "open": safe_float(r[1]) if len(r) > 1 else 0.0,
                "high": safe_float(r[2]) if len(r) > 2 else 0.0,
                "low": safe_float(r[3]) if len(r) > 3 else 0.0,
                "close": safe_float(r[4]) if len(r) > 4 else 0.0,
                "volume": safe_float(r[5]) if len(r) > 5 else 0.0,
                "amount": safe_float(r[7]) if len(r) > 7 else 0.0,
            })
    return out_rows


def write_csv(rows: List[dict], out_path: str):
    import csv
    headers = ["timestamps", "open", "high", "low", "close", "volume", "amount"]
    # 若行里包含 time_utc，则一并写出
    if rows and "time_utc" in rows[0]:
        headers = ["time_utc"] + headers
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in rows:
            # 保证无NaN：若出现None/NaN，统一转0
            clean = {}
            for k in headers:
                if k == "time_utc":
                    clean[k] = r.get(k, "")
                    continue
                v = r.get(k, 0)
                if v is None:
                    v = 0
                # 处理NaN
                try:
                    if isinstance(v, float) and math.isnan(v):
                        v = 0
                except Exception:
                    pass
                clean[k] = v
            writer.writerow(clean)


def main():
    parser = argparse.ArgumentParser(description="Binance K线下载脚本（CSV导出）")
    parser.add_argument("--symbol", required=True, help="交易对，例如 BTCUSDT")
    parser.add_argument("--interval", required=True, help="时间级，例如 1m/5m/15m/1h/4h/1d/1w/1M")
    parser.add_argument("--start", required=True, help="开始日，YYYYMMDD，例如 20200102")
    parser.add_argument("--end", required=True, help="结束日，YYYYMMDD（含当日）")
    parser.add_argument("--threads", type=int, default=4, help="下载线程数")
    parser.add_argument("--out", required=True, help="CSV保存路径")
    parser.add_argument("--unit", choices=["ms", "s"], default="ms", help="timestamps 单位：ms（默认）或 s")
    parser.add_argument("--add-iso", action="store_true", help="额外输出 time_utc 列（ISO8601，UTC）用于校验")
    args = parser.parse_args()

    interval = args.interval
    if interval not in INTERVAL_TO_MS:
        raise SystemExit(f"不支持的interval: {interval}")

    start_day = parse_date_yyyymmdd(args.start)
    end_day = parse_date_yyyymmdd(args.end)
    if end_day < start_day:
        raise SystemExit("结束日不得早于开始日")

    # 将日期范围设为 [start 00:00:00, end 23:59:59.999]
    start_dt = start_day.replace(hour=0, minute=0, second=0, microsecond=0)
    end_dt = end_day.replace(hour=23, minute=59, second=59, microsecond=999000)

    print("准备构造期望时间轴...")
    expected_ts = build_expected_timestamps(start_dt, end_dt, interval)
    print(f"期望K线根数：{len(expected_ts)}")

    print("开始下载...")
    raw_rows = download_klines(args.symbol, interval, start_dt, end_dt, args.threads)
    print(f"原始返回记录数：{len(raw_rows)}")

    print("规范化并对齐、补零...")
    rows = normalize_and_fill(raw_rows, expected_ts)

    # 单位转换与可读列
    if args.unit == "s":
        for r in rows:
            try:
                r["timestamps"] = int(r["timestamps"] // 1000)
            except Exception:
                r["timestamps"] = 0
    if args.add_iso:
        for r in rows:
            # 统一按 UTC 输出
            try:
                ts_ms = r["timestamps"] if args.unit == "ms" else r["timestamps"] * 1000
                iso = dt.datetime.utcfromtimestamp(ts_ms / 1000.0).replace(tzinfo=dt.timezone.utc).isoformat()
                r["time_utc"] = iso
            except Exception:
                r["time_utc"] = ""

    # 最终再检视是否存在NaN（防御性）
    for r in rows:
        for k in ["open", "high", "low", "close", "volume", "amount"]:
            v = r.get(k, 0)
            try:
                if v is None or (isinstance(v, float) and math.isnan(v)):
                    r[k] = 0.0
            except Exception:
                r[k] = 0.0

    print(f"写入CSV：{args.out}")
    write_csv(rows, args.out)
    print("完成 ✅")


if __name__ == "__main__":
    main()
