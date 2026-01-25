"""
24d 사이트 검증 스크립트
- wyu.cif에서 실제 24d Li 좌표 추출
- generate_24d_sites.py의 하드코딩 좌표와 비교
"""

from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import os

# === 1. wyu.cif 로드 ===
cif_path = os.path.join(os.path.dirname(__file__), "wyu.cif")
structure = Structure.from_file(cif_path)

print("=== wyu.cif 원본 Li 사이트 ===")
for i, site in enumerate(structure):
    if "Li" in str(site.species):
        print(f"  {site.label}: {site.frac_coords}")

print()

# === 2. P1으로 확장 ===
sga = SpacegroupAnalyzer(structure)
p1 = sga.get_conventional_standard_structure()

# === 3. Li 좌표 추출 및 정규화 ===
def normalize(coord):
    result = []
    for c in coord:
        c_norm = c % 1.0
        if c_norm > 0.999:
            c_norm = 0.0
        result.append(round(c_norm, 5))
    return tuple(result)

li_coords = []
for site in p1:
    if "Li" in str(site.species):
        li_coords.append(normalize(site.frac_coords))

print(f"=== P1 확장 후 Li 좌표 ({len(li_coords)}개) ===")

# === 4. 24d 패턴 식별 ===
def is_24d_pattern(coord):
    """좌표가 0, 0.25, 0.5, 0.75 조합인지 확인"""
    quarters = [0.0, 0.25, 0.5, 0.75]
    for c in coord:
        if not any(abs(c - q) < 0.02 for q in quarters):
            return False
    return True

li_24d = sorted([c for c in li_coords if is_24d_pattern(c)])
li_96h = sorted([c for c in li_coords if not is_24d_pattern(c)])

print(f"  24d 패턴 Li: {len(li_24d)}개")
print(f"  96h 패턴 Li: {len(li_96h)}개")
print()

print("=== 실제 24d Li 좌표 (P1 확장 결과) ===")
for c in li_24d:
    print(f"  {c}")

# === 5. generate_24d_sites.py의 하드코딩 좌표 ===
print()
print("=== generate_24d_sites.py의 하드코딩 24d 좌표 ===")
hardcoded_24d = [
    # 8개 Li 점유 사이트
    (0.0, 0.0, 0.0), (0.5, 0.0, 0.25), (0.0, 0.5, 0.25), (0.5, 0.5, 0.0),
    (0.0, 0.0, 0.5), (0.5, 0.0, 0.75), (0.0, 0.5, 0.75), (0.5, 0.5, 0.5),
    # 16개 vacancy 사이트
    (0.25, 0.0, 0.125), (0.75, 0.0, 0.125), (0.0, 0.25, 0.125), (0.0, 0.75, 0.125),
    (0.25, 0.5, 0.625), (0.75, 0.5, 0.625), (0.5, 0.25, 0.625), (0.5, 0.75, 0.625),
    (0.25, 0.25, 0.0), (0.75, 0.75, 0.0), (0.25, 0.75, 0.5), (0.75, 0.25, 0.5),
    (0.0, 0.25, 0.375), (0.0, 0.75, 0.375), (0.5, 0.25, 0.875), (0.5, 0.75, 0.875),
]

hardcoded_set = set(normalize(c) for c in hardcoded_24d)
actual_set = set(li_24d)

print(f"  하드코딩 좌표 수: {len(hardcoded_set)}")
print()

# === 6. 비교 ===
print("=== 비교 결과 ===")

# 실제 24d Li 중 하드코딩에 있는 것
matched = actual_set & hardcoded_set
print(f"  일치하는 좌표: {len(matched)}개")

# 실제 24d Li 중 하드코딩에 없는 것
missing_in_hardcode = actual_set - hardcoded_set
print(f"  실제 24d에 있지만 하드코딩에 없음: {len(missing_in_hardcode)}개")
for c in sorted(missing_in_hardcode):
    print(f"    {c}")

# 하드코딩에 있지만 실제 24d Li가 아닌 것
extra_in_hardcode = hardcoded_set - actual_set
print(f"  하드코딩에 있지만 실제 24d Li가 아님: {len(extra_in_hardcode)}개")
for c in sorted(extra_in_hardcode):
    print(f"    {c}")
