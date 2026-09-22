"""
คำนวณ Probability of Decoding Failure (Pfail) ของ decoder แต่ละตัว
แยกตาม input SER (prop_error) แล้วบันทึกเป็น CSV (ไม่พล็อตกราฟ)

สูตร:  Pfail = (count_all - count_success) / count_all

ไฟล์ที่ 6 คอลัมน์แรกไม่ตรงกับ REQUIRED_HEADER_PREFIX จะถูกข้าม
"""

import csv
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd

# =================================================================
# CONFIGURATION
# =================================================================
CHUNK_SIZE = 200_000
OUTPUT_NAME = 'analysis_summary.csv'
FLOAT_FORMAT = '%.6e'

# None = ตรวจหาคอลัมน์ success_* ทั้งหมดในไฟล์อัตโนมัติ
# หรือระบุเองเป็นลิสต์ เช่น ['success_EhMP', 'success_VSD_AFV']
DECODER_COLUMNS = None

# ลำดับคอลัมน์ในไฟล์ผลลัพธ์ (ชื่อที่ไม่อยู่ในนี้จะต่อท้ายให้เอง)
DECODER_ORDER = [
    'EhMP', 'E_hMP_guid_CNN', 'E_hMP_guid_GNN',
    'MhMP', 'M_hMP_guid_CNN', 'M_hMP_guid_GNN',
    'VSD_AFV', 'VSD_guid_CNN', 'VSD_guid_GNN',
]

# 6 คอลัมน์แรกที่ไฟล์ต้องมี ตรงตามลำดับ ไม่งั้นถือว่าคนละฟอร์แมต -> ข้ามไฟล์
REQUIRED_HEADER_PREFIX = [
    "n", "k", "symbol_size", "prop_error",
    "Number of Error Symbol", "Number of Error Bits",
]

csv.field_size_limit(10 ** 9)


# =================================================================
def read_header(path):
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

    if not any(c.startswith('success_') for c in header):
        return "ไม่พบคอลัมน์ success_*"
    return None


def pick_success_columns(header):
    if DECODER_COLUMNS is None:
        return [c for c in header if c.startswith('success_')]
    return [c for c in DECODER_COLUMNS if c in header]


_TRUE = {'true', 't', 'yes', 'y', '1', '1.0'}
_FALSE = {'false', 'f', 'no', 'n', '0', '0.0', '', 'nan', 'none'}


def to_binary(series):
    """
    แปลงคอลัมน์ success_* เป็น 0/1 รองรับทั้ง bool จริง, 'True'/'False' และ 1/0
    (คอลัมน์เขียนมาจาก bool(...) ถ้า pandas อ่านเป็นสตริงแล้วใช้ to_numeric ตรงๆ
     'True' จะกลายเป็น NaN -> ถูกนับเป็น fail หมด ซึ่งผิด)
    """
    if series.dtype == bool:
        return series.astype('int64')

    num = pd.to_numeric(series, errors='coerce')
    if num.notna().all():
        return (num != 0).astype('int64')

    s = series.astype(str).str.strip().str.lower()
    unknown = set(s.unique()) - _TRUE - _FALSE
    if unknown:
        raise ValueError(f"คอลัมน์ {series.name!r} มีค่าที่แปลงเป็น 0/1 ไม่ได้: {sorted(unknown)[:5]}")
    return s.isin(_TRUE).astype('int64')


def write_csv(df, path, fmt_cols, fmt):
    """เขียน CSV โดยฟอร์แมตเฉพาะคอลัมน์ผลลัพธ์ ไม่ให้ไปโดน prop_error ด้วย"""
    out = df.copy()
    for c in fmt_cols:
        out[c] = out[c].map(lambda x: '' if pd.isna(x) else fmt % x)
    out.to_csv(path, index=False)


# =================================================================
def analyze_decoders(data_folder_path, output_name=OUTPUT_NAME):
    folder = Path(data_folder_path)
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

    agg = pd.DataFrame()
    all_decoders = []

    for f in valid_files:
        t0 = time.time()
        header = read_header(f)
        success_cols = pick_success_columns(header)
        if not success_cols:
            print(f"[ข้าม] {f.name}   ไม่พบคอลัมน์ success_* ที่ต้องการ")
            continue
        for c in success_cols:
            if c not in all_decoders:
                all_decoders.append(c)

        print(f"กำลังประมวลผล: {f.name}   ({len(success_cols)} decoders)")

        for chunk in pd.read_csv(f, usecols=['prop_error'] + success_cols,
                                 chunksize=CHUNK_SIZE):
            for c in success_cols:
                chunk[c] = to_binary(chunk[c])

            g = chunk.groupby('prop_error').agg(['sum', 'count'])
            g.columns = [f"{col}_{stat}" for col, stat in g.columns]
            g = g.reset_index()

            agg = g if agg.empty else (pd.concat([agg, g], ignore_index=True)
                                         .groupby('prop_error', as_index=False).sum())
            del chunk, g
            gc.collect()

        print(f"   เสร็จใน {time.time() - t0:.1f}s")

    if agg.empty:
        print("ไม่มีข้อมูลให้ประมวลผล")
        return None

    # ---------- Pfail = (count_all - count_success) / count_all ----------
    agg = agg.sort_values('prop_error').reset_index(drop=True)
    out = pd.DataFrame({'prop_error': agg['prop_error']})

    count_cols = [c for c in agg.columns if c.endswith('_count')]
    out['count_all'] = agg[count_cols].max(axis=1).astype('int64')

    idx = {f'success_{n}': i for i, n in enumerate(DECODER_ORDER)}
    all_decoders = sorted(all_decoders, key=lambda c: (idx.get(c, len(idx)), c))

    found = []
    for col in all_decoders:
        s_col, c_col = f'{col}_sum', f'{col}_count'
        if s_col not in agg.columns:
            continue
        name = col[len('success_'):]
        n_all = agg[c_col]
        n_ok = agg[s_col]
        out[f'pfail_{name}'] = ((n_all - n_ok) / n_all.replace(0, np.nan))
        found.append((name, col))

    # คอลัมน์จำนวนนับดิบต่อท้าย เผื่อตรวจย้อนหลัง
    for name, col in found:
        out[f'count_success_{name}'] = agg[f'{col}_sum'].astype('int64')
        out[f'count_all_{name}'] = agg[f'{col}_count'].astype('int64')

    summary_path = folder / output_name
    write_csv(out, summary_path, [f'pfail_{n}' for n, _ in found], FLOAT_FORMAT)

    print(f"\nพบ decoder {len(found)} ตัว: {', '.join(n for n, _ in found)}")
    print(f"บันทึกไฟล์สรุปผลที่: {summary_path}")

    with pd.option_context('display.width', 250, 'display.max_columns', 100):
        cols = ['prop_error', 'count_all'] + [f'pfail_{n}' for n, _ in found]
        print("\n" + out[cols].to_string(index=False))

    return out


if __name__ == '__main__':
    import sys

    target_folder = r"C:\Users\user\Desktop\ProjectY4\pre_master\new_journal_run\Add_3121"
    analyze_decoders(sys.argv[1] if len(sys.argv) > 1 else target_folder)
