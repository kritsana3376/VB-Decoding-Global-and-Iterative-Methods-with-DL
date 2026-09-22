#!/usr/bin/env python3
"""
สรุป verified / false unverified / false verified symbol ของ decoder แต่ละตัว
แยกตาม input SER (prop_error) และแยก data symbol / parity symbol
 
รูปแบบไฟล์ที่รองรับ (ตามไฟล์ p0_XX_raw_data_all.csv):
  - prop_error            : input SER ของแถวนั้น
  - initial_errors_<DEC>  : dict repr "{5: 3294145190, 12: ...}"  -> ใช้เฉพาะ key
  - unsolved_errors_<DEC> : dict repr เช่นกัน                      -> ใช้เฉพาะ key
  - verified_list_<DEC>   : numpy repr "[1 0 1 1 ...]" ยาว n symbol
 
นิยาม:
  verified         = จำนวนตำแหน่งที่ verified_list == 1
  false verified   = verified_list[i] == 1  และ  i อยู่ใน unsolved_errors
  false unverified = verified_list[i] == 0  และ  i ไม่อยู่ใน initial_errors
  data symbol      = index 0-20,  parity symbol = index 21-30
 
การใช้งาน:
    python decoder_stats.py /path/to/folder
    python decoder_stats.py /path/to/folder -o result.xlsx --pct
    python decoder_stats.py /path/to/folder --limit 5000      # ทดสอบเร็วๆ
"""
 
from __future__ import annotations
 
import argparse
import csv
import os
import re
import time
from collections import defaultdict
 
import numpy as np
import pandas as pd
 
# ===========================================================================
# CONFIG
# ===========================================================================
N_SYMBOLS = 31
DATA_END = 21          # data = index 0..20, parity = index 21..30
 
SER_COLUMN = "prop_error"        # ไม่มีคอลัมน์นี้ -> fallback อ่านจากชื่อไฟล์ p0_11 -> 0.11
SER_FROM_FILENAME = r"p(\d)_(\d+)"
 
# ลำดับแสดงผล: baseline (algebraic) ขึ้นก่อน แล้วตามด้วย CNN / GNN ของตระกูลเดียวกัน
DECODER_ORDER = [
    "EhMP", "E_hMP_guid_CNN", "E_hMP_guid_GNN",
    "MhMP", "M_hMP_guid_CNN", "M_hMP_guid_GNN",
    "VSD_AFV", "VSD_guid_CNN", "VSD_guid_GNN",
]
 
# 6 คอลัมน์แรกที่ไฟล์ต้องมี (ตรงตามลำดับ) ไม่งั้นถือว่าคนละฟอร์แมต -> ข้ามไฟล์นั้น
REQUIRED_HEADER_PREFIX = [
    "n", "k", "symbol_size", "prop_error",
    "Number of Error Symbol", "Number of Error Bits",
]
 
csv.field_size_limit(10 ** 9)
_DICT_KEY_RE = re.compile(r"(\d+)\s*:")
 
 
# ===========================================================================
# Fast parsers (cache เพราะค่าซ้ำเยอะมาก เช่น verified list ที่เป็น 1 ทั้งแถว)
# ===========================================================================
_bits_cache: dict = {}
_keys_cache: dict = {}
_CACHE_MAX = 300_000
 
 
def parse_bits(s):
    """'[1 0 1 ...]' -> ndarray(uint8) ความยาว N_SYMBOLS"""
    v = _bits_cache.get(s)
    if v is not None:
        return v
    b = s.strip().strip("[]").replace(" ", "").replace(",", "").replace("\n", "")
    arr = np.frombuffer(b.encode(), dtype=np.uint8) - 48
    if arr.size != N_SYMBOLS:
        out = np.zeros(N_SYMBOLS, dtype=np.uint8)
        out[: min(N_SYMBOLS, arr.size)] = arr[:N_SYMBOLS]
        arr = out
    if len(_bits_cache) < _CACHE_MAX:
        _bits_cache[s] = arr
    return arr
 
 
def parse_keys(s):
    """'{5: 3294145190, 12: 2339669200}' -> (5, 12)  (เอาเฉพาะ key ไม่เอา magnitude)"""
    if not s or s == "{}":
        return ()
    v = _keys_cache.get(s)
    if v is not None:
        return v
    v = tuple(int(x) for x in _DICT_KEY_RE.findall(s))
    if len(_keys_cache) < _CACHE_MAX:
        _keys_cache[s] = v
    return v
 
 
def ser_from_filename(path):
    m = re.search(SER_FROM_FILENAME, os.path.basename(path))
    return float(f"{m.group(1)}.{m.group(2)}") if m else None
 
 
# ===========================================================================
def resolve_verified(header, name):
    """
    หาคอลัมน์ verified_list ของ decoder ชื่อ name
      1) ชื่อตรงแพตเทิร์น verified_list_<name>   (ปกติจะเจอตรงนี้)
      2) คอลัมน์ verified_list_* ที่ลงท้ายด้วย <name>
         เผื่อไฟล์เก่าที่ตั้งชื่อยาวกว่าปกติ เช่น
         verified_list_VSD_AFV_make_success_VSD_guid_CNN
    """
    exact = f"verified_list_{name}"
    if exact in header:
        return exact
    cands = [c for c in header if c.startswith("verified_list_") and c.endswith(name)]
    return cands[0] if cands else None


def discover_decoders(header):
    """
    หา decoder จากคอลัมน์ unsolved_errors_* แล้วจับคู่กับ initial_errors_* / verified_list_*

    หมายเหตุ: ไล่จาก unsolved_errors_ ไม่ใช่ verified_list_ เพราะชื่อคอลัมน์ verified
    ของ VSD CNN/GNN ไม่ตรงแพตเทิร์น และยังขึ้นต้นซ้ำกับของ VSD_AFV
    """
    hset = set(header)
    found = {}
    for col in header:
        if not col.startswith("unsolved_errors_"):
            continue
        name = col[len("unsolved_errors_"):]
        found[name] = {
            "unsolved": col,
            "initial": (f"initial_errors_{name}"
                        if f"initial_errors_{name}" in hset else None),
            "verified": resolve_verified(header, name),
        }
    return found
 
 
def check_header(header):
    """คืน None ถ้า header ใช้ได้, ไม่งั้นคืนข้อความบอกเหตุผลที่ต้องข้ามไฟล์"""
    need = REQUIRED_HEADER_PREFIX
    got = header[: len(need)]
    if [c.strip() for c in got] != need:
        if len(header) < len(need):
            return f"มีแค่ {len(header)} คอลัมน์ (ต้องการอย่างน้อย {len(need)})"
        bad = [f"col{i + 1}: {g.strip()!r} != {w!r}"
               for i, (g, w) in enumerate(zip(got, need)) if g.strip() != w]
        return "หัวตารางไม่ตรง -> " + "; ".join(bad[:3])
    if not any(c.startswith("verified_list_") for c in header):
        return "ไม่พบคอลัมน์ verified_list_*"
    return None
 
 
def order_decoders(names):
    idx = {n: i for i, n in enumerate(DECODER_ORDER)}
    return sorted(names, key=lambda d: (idx.get(d, len(idx)), d))
 
 
# ===========================================================================
class Acc:
    __slots__ = ("ver_d", "ver_p", "fv_d", "fv_p", "fu_d", "fu_p")
 
    def __init__(self):
        self.ver_d = self.ver_p = 0
        self.fv_d = self.fv_p = 0
        self.fu_d = self.fu_p = 0
 
 
def process_folder(folder, limit=None, verbose=True, strict=True):
    files = sorted(os.path.join(folder, f) for f in os.listdir(folder)
                   if f.lower().endswith(".csv"))
    if not files:
        raise SystemExit(f"ไม่พบไฟล์ .csv ใน {folder}")
 
    n_cw = defaultdict(int)                       # ser -> จำนวน codeword
    acc = defaultdict(lambda: defaultdict(Acc))   # ser -> decoder -> Acc
    all_decoders = []
    used, skipped = [], []
    n_par = N_SYMBOLS - DATA_END
 
    for path in files:
        t0 = time.time()
        fallback_ser = ser_from_filename(path)
        fname = os.path.basename(path)
 
        with open(path, newline="", encoding="utf-8", errors="replace") as fh:
            reader = csv.reader(fh)
            try:
                header = [h.strip().lstrip("\ufeff") for h in next(reader)]
            except StopIteration:
                skipped.append((fname, "ไฟล์ว่าง"))
                print(f"[skip] {fname}   ไฟล์ว่าง")
                continue
 
            reason = check_header(header) if strict else None
            if reason:
                skipped.append((fname, reason))
                print(f"[skip] {fname}   {reason}")
                continue
 
            decoders = discover_decoders(header)
            no_verified = [n for n, m in decoders.items() if not m["verified"]]
            decoders = {n: m for n, m in decoders.items() if m["verified"]}
            for d in decoders:
                if d not in all_decoders:
                    all_decoders.append(d)
 
            pos = {c: i for i, c in enumerate(header)}
            ser_i = pos.get(SER_COLUMN)
            plan = [(name,
                     pos[m["verified"]],
                     pos.get(m["unsolved"]) if m["unsolved"] else None,
                     pos.get(m["initial"]) if m["initial"] else None)
                    for name, m in decoders.items()]
 
            if verbose:
                src = SER_COLUMN if ser_i is not None else f"filename -> {fallback_ser}"
                print(f"[read] {fname}   decoders={len(plan)}   SER from {src}")
                for name in order_decoders(list(decoders)):
                    m = decoders[name]
                    note = "" if m["verified"] == f"verified_list_{name}" else "   <- ชื่อไม่ตรงแพตเทิร์น"
                    print(f"        {name:<18} verified={m['verified']}{note}")
                    if not m["initial"]:
                        print(f"        {'':<18} [เตือน] ไม่พบ initial_errors_{name}")
                if no_verified:
                    print(f"        [เตือน] ข้าม decoder ที่หาคอลัมน์ verified ไม่เจอ: {', '.join(no_verified)}")
 
            nrow = 0
            for row in reader:
                if not row:
                    continue
                if ser_i is not None and row[ser_i]:
                    try:
                        ser = float(row[ser_i])
                    except ValueError:
                        ser = fallback_ser
                else:
                    ser = fallback_ser
                if ser is None:
                    raise SystemExit("หา input SER ไม่เจอ ทั้งในคอลัมน์ prop_error และชื่อไฟล์")
 
                n_cw[ser] += 1
                bucket = acc[ser]
 
                for name, vi, ui, ii in plan:
                    v = parse_bits(row[vi])
                    a = bucket[name]
 
                    vd = int(v[:DATA_END].sum())
                    vp = int(v[DATA_END:].sum())
                    a.ver_d += vd
                    a.ver_p += vp
 
                    # false verified: verified == 1 แต่ index อยู่ใน unsolved
                    if ui is not None:
                        for i in parse_keys(row[ui]):
                            if i < N_SYMBOLS and v[i]:
                                if i < DATA_END:
                                    a.fv_d += 1
                                else:
                                    a.fv_p += 1
 
                    # false unverified: verified == 0 และ index ไม่อยู่ใน initial error
                    zd = DATA_END - vd
                    zp = n_par - vp
                    if ii is not None:
                        for i in parse_keys(row[ii]):
                            if i < N_SYMBOLS and not v[i]:
                                if i < DATA_END:
                                    zd -= 1
                                else:
                                    zp -= 1
                    a.fu_d += zd
                    a.fu_p += zp
 
                nrow += 1
                if limit and nrow >= limit:
                    break
 
        used.append(fname)
        if verbose:
            print(f"        {nrow:,} codewords in {time.time() - t0:.1f}s")
 
    print(f"\nใช้ไฟล์ {len(used)} ไฟล์, ข้าม {len(skipped)} ไฟล์")
    if not used:
        raise SystemExit("ไม่มีไฟล์ที่ฟอร์แมตตรงเลย -> ตรวจ REQUIRED_HEADER_PREFIX "
                         "หรือรันด้วย --loose เพื่อปิดการตรวจหัวตาราง")
    return acc, n_cw, order_decoders(all_decoders)
 
 
# ===========================================================================
def build_table(acc, n_cw, decoders, pct=False):
    n_par = N_SYMBOLS - DATA_END
    metrics = [("Verified symbol", "ver_d", "ver_p"),
               ("False unverified symbol", "fu_d", "fu_p"),
               ("False verified symbol", "fv_d", "fv_p")]
 
    cols = [("Number of decoded symbol", "", "Data"),
            ("Number of decoded symbol", "", "Parity")]
    for d in decoders:
        for label, _, _ in metrics:
            cols.append((d, label, "Data"))
            cols.append((d, label, "Parity"))
 
    rows, index = [], []
    for ser in sorted(acc):
        cw = n_cw[ser]
        tot_d, tot_p = cw * DATA_END, cw * n_par
        r = [tot_d, tot_p]
        for d in decoders:
            a = acc[ser].get(d) or Acc()
            for _, fd, fp in metrics:
                vd, vp = getattr(a, fd), getattr(a, fp)
                if pct:
                    r += [100.0 * vd / tot_d if tot_d else 0.0,
                          100.0 * vp / tot_p if tot_p else 0.0]
                else:
                    r += [vd, vp]
        rows.append(r)
        index.append(ser)
 
    df = pd.DataFrame(
        rows,
        index=pd.Index(index, name="Input SER"),
        columns=pd.MultiIndex.from_tuples(cols, names=["Decoder", "Metric", "Type"]),
    )
    return df.round(4) if pct else df
 
 
# ===========================================================================
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", help="โฟลเดอร์ที่เก็บไฟล์ CSV")
    ap.add_argument("-o", "--output", default="decoder_stats.csv",
                    help="ไฟล์ผลลัพธ์ .csv หรือ .xlsx (default: decoder_stats.csv)")
    ap.add_argument("--pct", action="store_true",
                    help="สร้างตารางเปอร์เซ็นต์เทียบจำนวน symbol ทั้งหมดเพิ่มอีกชุด")
    ap.add_argument("--limit", type=int, default=None,
                    help="อ่านแค่ N แถวแรกของแต่ละไฟล์ (ใช้ทดสอบ)")
    ap.add_argument("--loose", action="store_true",
                    help="ปิดการตรวจหัวตาราง (ปกติจะข้ามไฟล์ที่ 6 คอลัมน์แรกไม่ตรงฟอร์แมต)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
 
    acc, n_cw, decoders = process_folder(args.folder, args.limit,
                                         not args.quiet, strict=not args.loose)
    table = build_table(acc, n_cw, decoders)
 
    pd.set_option("display.width", 100000)
    pd.set_option("display.max_columns", 500)
    print("\n" + table.to_string())
 
    if args.output.lower().endswith(".xlsx"):
        with pd.ExcelWriter(args.output) as xw:
            table.to_excel(xw, sheet_name="counts")
            if args.pct:
                build_table(acc, n_cw, decoders, pct=True).to_excel(xw, sheet_name="percent")
    else:
        table.to_csv(args.output)
        if args.pct:
            p = args.output.rsplit(".", 1)[0] + "_percent.csv"
            build_table(acc, n_cw, decoders, pct=True).to_csv(p)
            print(f"\nบันทึก: {p}")
    print(f"\nบันทึก: {args.output}")



if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:                                    # กด Run เฉยๆ ไม่ได้ใส่ argument
        sys.argv += [r"C:\Users\user\Desktop\ProjectY4\pre_master\new_journal_run\Add_3121",                 # <-- ใส่ path โฟลเดอร์ตรงนี้
                     "-o", 
                     r"C:\Users\user\Desktop\ProjectY4\pre_master\new_journal_run\Add_3121\FV_analysis_result.xlsx",
                     "--pct"]
    main()