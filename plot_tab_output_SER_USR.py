"""
คำนวณ D-SER และ D-USR ของ decoder แต่ละตัว แยกตาม input SER (prop_error)
แล้วบันทึกเป็น CSV (ไม่พล็อตกราฟ)

ไฟล์ที่ 6 คอลัมน์แรกไม่ตรงกับ REQUIRED_HEADER_PREFIX จะถูกข้าม
"""

import re
import csv
import gc
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

# =================================================================
# CONFIGURATION
# =================================================================
CHUNK_SIZE = 50_000
K_FALLBACK = 21               # ใช้เมื่อไฟล์ไม่มีคอลัมน์ k
OUTPUT_NAME = 'SER_plot_summary.csv'
FLOAT_FORMAT = '%.5e'         # รูปแบบตัวเลขในไฟล์ CSV (ยังอ่านกลับเป็นตัวเลขได้)
INCLUDE_RAW_SUMS = False      # True = ใส่คอลัมน์ผลรวมดิบไว้ตรวจย้อนหลังด้วย

# 6 คอลัมน์แรกที่ไฟล์ต้องมี ตรงตามลำดับ ไม่งั้นถือว่าคนละฟอร์แมต -> ข้ามไฟล์
REQUIRED_HEADER_PREFIX = [
    "n", "k", "symbol_size", "prop_error",
    "Number of Error Symbol", "Number of Error Bits",
]

# suffix ในชื่อคอลัมน์ของแต่ละ decoder
DECODER_SUFFIXES = [
    'EhMP', 'E_hMP_guid_CNN', 'E_hMP_guid_GNN',
    'MhMP', 'M_hMP_guid_CNN', 'M_hMP_guid_GNN',
    'VSD_AFV', 'VSD_guid_CNN', 'VSD_guid_GNN',
]

csv.field_size_limit(10 ** 9)
_KEY_RE = re.compile(r'(\d+)\s*:')


# =================================================================
# ตรวจฟอร์แมตไฟล์
# =================================================================
def read_header(path):
    """อ่านเฉพาะบรรทัดหัวตาราง คืน None ถ้าไฟล์ว่าง"""
    with open(path, newline='', encoding='utf-8', errors='replace') as fh:
        try:
            return [h.strip().lstrip('\ufeff') for h in next(csv.reader(fh))]
        except StopIteration:
            return None


def check_header(header):
    """คืน None ถ้าใช้ได้, ไม่งั้นคืนข้อความบอกเหตุผลที่ข้ามไฟล์"""
    if header is None:
        return "ไฟล์ว่าง"

    need = REQUIRED_HEADER_PREFIX
    if len(header) < len(need):
        return f"มีแค่ {len(header)} คอลัมน์ (ต้องการอย่างน้อย {len(need)})"

    bad = [f"col{i + 1}: {g!r} != {w!r}"
           for i, (g, w) in enumerate(zip(header[:len(need)], need)) if g != w]
    if bad:
        return "หัวตารางไม่ตรง -> " + "; ".join(bad[:3])

    if not any(c.startswith('unsolved_errors_') for c in header):
        return "ไม่พบคอลัมน์ unsolved_errors_*"
    return None


# =================================================================
# Helper คำนวณสถิติ (cache ไว้เพราะสตริงซ้ำกันเยอะมาก)
# =================================================================
@lru_cache(maxsize=1_000_000)
def _count_keys_below(dict_str, k):
    """นับ key ใน dict repr ที่ index < k (เฉพาะ data symbol)"""
    if not dict_str or dict_str == '{}':
        return 0
    return sum(1 for key in _KEY_RE.findall(dict_str) if int(key) < k)


@lru_cache(maxsize=1_000_000)
def _count_zeros_prefix(list_str, k):
    """นับเลข 0 ใน k ตำแหน่งแรกของ verified list '[1 0 1 ...]'"""
    if not list_str:
        return 0
    cleaned = list_str.strip().strip('[]')
    if not cleaned:
        return 0
    return cleaned.split()[:k].count('0')


def count_keys_series(series, k):
    return series.fillna('{}').astype(str).map(lambda s: _count_keys_below(s, k))


def count_zeros_series(series, k):
    return series.fillna('').astype(str).map(lambda s: _count_zeros_prefix(s, k))


def resolve_verified(columns, name):
    """
    หาคอลัมน์ verified_list ของ decoder ชื่อ name
      1) verified_list_<name>   (ปกติจะเจอตรงนี้)
      2) คอลัมน์ verified_list_* ที่ลงท้ายด้วย <name>
         เผื่อไฟล์เก่าที่ตั้งชื่อยาวกว่าปกติ เช่น
         verified_list_VSD_AFV_make_success_VSD_guid_CNN
    """
    exact = f'verified_list_{name}'
    if exact in columns:
        return exact
    cands = [c for c in columns
             if c.startswith('verified_list_') and c.endswith(name)]
    return cands[0] if cands else None


def write_csv(df, path, fmt_cols, fmt):
    """เขียน CSV โดยฟอร์แมตเฉพาะคอลัมน์ผลลัพธ์ ไม่ให้ไปโดน prop_error ด้วย"""
    out = df.copy()
    for c in fmt_cols:
        out[c] = out[c].map(lambda x: '' if pd.isna(x) else fmt % x)
    out.to_csv(path, index=False)


# =================================================================
def process_folder(folder_path, output_name=OUTPUT_NAME):
    folder = Path(folder_path)
    if not folder.is_dir():
        print(f"ไม่พบโฟลเดอร์: {folder}")
        return None

    csv_files = sorted(folder.glob('*.csv'))
    if not csv_files:
        print("ไม่พบไฟล์ CSV ในโฟลเดอร์นี้")
        return None

    # ---------- คัดไฟล์ด้วยหัวตาราง ----------
    valid_files, skipped = [], []
    for f in csv_files:
        reason = check_header(read_header(f))
        if reason:
            skipped.append((f.name, reason))
            print(f"[ข้าม] {f.name}   {reason}")
        else:
            valid_files.append(f)

    if not valid_files:
        print("\nไม่มีไฟล์ที่ฟอร์แมตตรงเลย -> ตรวจค่า REQUIRED_HEADER_PREFIX อีกครั้ง")
        return None
    print(f"\nใช้ไฟล์ {len(valid_files)} ไฟล์, ข้าม {len(skipped)} ไฟล์\n")

    agg_result = pd.DataFrame()

    for f in valid_files:
        t0 = time.time()
        print(f"กำลังประมวลผล: {f.name}")

        for chunk in pd.read_csv(f, chunksize=CHUNK_SIZE):
            if 'k' in chunk.columns:
                k_vals = pd.to_numeric(chunk['k'], errors='coerce').dropna().unique()
                k_const = int(k_vals[0]) if len(k_vals) == 1 else None
            else:
                k_const = K_FALLBACK

            temp = pd.DataFrame({'prop_error': chunk['prop_error'].values})

            for suffix in DECODER_SUFFIXES:
                unsolved_col = f'unsolved_errors_{suffix}'
                verified_col = resolve_verified(chunk.columns, suffix)
                if unsolved_col not in chunk.columns:
                    continue

                if k_const is not None:
                    n_unsolved = count_keys_series(chunk[unsolved_col], k_const)
                    denom = k_const
                else:
                    # k ไม่คงที่ในชั้นข้อมูลนี้ -> คำนวณทีละแถว
                    kk = pd.to_numeric(chunk['k'], errors='coerce').fillna(K_FALLBACK).astype(int)
                    n_unsolved = pd.Series(
                        [_count_keys_below(str(s) if pd.notna(s) else '{}', int(k))
                         for s, k in zip(chunk[unsolved_col], kk)], index=chunk.index)
                    denom = kk

                temp[f'ser_ratio_{suffix}'] = (n_unsolved / denom).values

                if verified_col is not None and verified_col in chunk.columns:
                    if k_const is not None:
                        n_zeros = count_zeros_series(chunk[verified_col], k_const)
                    else:
                        n_zeros = pd.Series(
                            [_count_zeros_prefix(str(s) if pd.notna(s) else '', int(k))
                             for s, k in zip(chunk[verified_col], kk)], index=chunk.index)
                    temp[f'usr_ratio_{suffix}'] = (n_zeros / denom).values

            chunk_agg = temp.groupby('prop_error').agg(['sum', 'count'])
            chunk_agg.columns = [f"{c[0]}_{c[1]}" for c in chunk_agg.columns]
            chunk_agg = chunk_agg.reset_index()

            if agg_result.empty:
                agg_result = chunk_agg
            else:
                agg_result = (pd.concat([agg_result, chunk_agg], ignore_index=True)
                                .groupby('prop_error', as_index=False).sum())

            del chunk, temp, chunk_agg
            gc.collect()

        print(f"   เสร็จใน {time.time() - t0:.1f}s")

    if agg_result.empty:
        print("ไม่มีข้อมูลให้ประมวลผล")
        return None

    # ---------- Mean = Sum / Count ----------
    agg_result = agg_result.sort_values('prop_error').reset_index(drop=True)
    summary_df = pd.DataFrame({'prop_error': agg_result['prop_error']})

    count_cols = [c for c in agg_result.columns if c.endswith('_count')]
    if count_cols:
        summary_df['n_codeword'] = agg_result[count_cols].max(axis=1).astype('int64')

    found = []
    for suffix in DECODER_SUFFIXES:
        for kind in ('ser', 'usr'):
            s_col, c_col = f'{kind}_ratio_{suffix}_sum', f'{kind}_ratio_{suffix}_count'
            if s_col in agg_result.columns:
                summary_df[f'{kind}_ratio_{suffix}'] = (
                    agg_result[s_col] / agg_result[c_col].replace(0, np.nan))
                if INCLUDE_RAW_SUMS:
                    summary_df[f'{kind}_sum_{suffix}'] = agg_result[s_col]
                    summary_df[f'{kind}_count_{suffix}'] = agg_result[c_col]
        if f'ser_ratio_{suffix}' in summary_df.columns:
            found.append(suffix)

    if not found:
        print("ไม่พบคอลัมน์ของ decoder ตัวใดเลยในไฟล์ที่อ่านได้")
        return None

    summary_path = folder / output_name
    ratio_cols = [c for c in summary_df.columns
                  if c.startswith(('ser_ratio_', 'usr_ratio_'))]
    write_csv(summary_df, summary_path, ratio_cols, FLOAT_FORMAT)

    print(f"\nพบ decoder {len(found)} ตัว: {', '.join(found)}")
    print(f"บันทึกไฟล์สรุปผลที่: {summary_path}")

    with pd.option_context('display.width', 200, 'display.max_columns', 100):
        print("\n" + summary_df[['prop_error', 'n_codeword']
                                + [f'ser_ratio_{s}' for s in found]].to_string(index=False))

    return summary_df


if __name__ == '__main__':
    import sys

    FOLDER = r"C:\Users\user\Desktop\ProjectY4\pre_master\new_journal_run\Add_3121"
    process_folder(sys.argv[1] if len(sys.argv) > 1 else FOLDER)
