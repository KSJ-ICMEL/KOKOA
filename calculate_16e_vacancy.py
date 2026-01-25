"""
16e Tetrahedral Vacancy 좌표 계산
=================================
I4_1/acd (No. 142) 공간군에서 16e Wyckoff 위치 계산

논문 정보:
- 8a (Li 점유): (0, 1/4, 3/8) = tetrahedral
- 16e (vacancy): (x, 0, 1/4) = tetrahedral (x 결정 필요)

Cubic 24d → Tetragonal 8a + 16e 관계에서:
- Cubic 24d 대표 좌표: (3/8, 0, 1/4)
- 따라서 16e tetrahedral vacancy의 x = 3/8 = 0.375
"""

import numpy as np

def normalize(coord):
    """분율 좌표를 [0, 1) 범위로 정규화"""
    result = []
    for c in coord:
        c_norm = float(c) % 1.0
        if c_norm > 0.9999:
            c_norm = 0.0
        if c_norm < 0.0001:
            c_norm = 0.0
        result.append(round(c_norm, 5))
    return tuple(result)

# === I41/acd 16e Wyckoff position ===
# 대표 좌표: (x, 0, 1/4)
# x = 0.375 (3/8) for tetrahedral vacancy

x = 0.375  # 3/8 from cubic 24d correspondence

# 16e의 8개 기본 좌표 (International Tables for Crystallography)
base_16e = [
    (x, 0, 0.25),
    (-x, 0, 0.75),
    (0.5 - x, 0.5, 0.75),
    (0.5 + x, 0.5, 0.25),
    (0, x, 0.25),
    (0, -x, 0.75),
    (0.5, 0.5 - x, 0.75),
    (0.5, 0.5 + x, 0.25),
]

# Body-centered translation (+1/2, +1/2, +1/2) 추가
all_16e = []
for coord in base_16e:
    norm = normalize(coord)
    all_16e.append(norm)
    
    # Body-centered translation
    translated = (coord[0] + 0.5, coord[1] + 0.5, coord[2] + 0.5)
    norm_trans = normalize(translated)
    all_16e.append(norm_trans)

# 중복 제거 및 정렬
unique_16e = sorted(set(all_16e))

print(f"=== 16e Tetrahedral Vacancy 좌표 (x = {x}) ===")
print(f"총 {len(unique_16e)}개 사이트:\n")

for i, coord in enumerate(unique_16e, 1):
    print(f"  {i:2d}. ({coord[0]:.5f}, {coord[1]:.5f}, {coord[2]:.5f})")

print()

# === wyu.cif와 충돌 확인 ===
from pymatgen.core import Structure
import os

cif_path = os.path.join(os.path.dirname(__file__), "wyu.cif")
if os.path.exists(cif_path):
    structure = Structure.from_file(cif_path)
    
    # 기존 원자 좌표 수집
    existing_coords = set()
    for site in structure:
        norm = normalize(site.frac_coords)
        existing_coords.add(norm)
    
    print(f"=== wyu.cif 원자 수: {len(existing_coords)} ===\n")
    
    # 16e vacancy 좌표 중 기존 원자와 겹치는지 확인
    valid_vacancies = []
    overlap_vacancies = []
    
    for coord in unique_16e:
        if coord in existing_coords:
            # 어떤 원자와 겹치는지 확인
            for site in structure:
                if normalize(site.frac_coords) == coord:
                    overlap_vacancies.append((coord, str(site.species)))
                    break
        else:
            valid_vacancies.append(coord)
    
    print(f"=== 충돌 확인 ===")
    print(f"  유효한 vacancy: {len(valid_vacancies)}개")
    print(f"  기존 원자와 겹침: {len(overlap_vacancies)}개")
    
    if overlap_vacancies:
        print("\n  겹치는 좌표:")
        for coord, species in overlap_vacancies:
            print(f"    {coord} → {species}")
    
    if valid_vacancies:
        print(f"\n=== 최종 16e Vacancy 좌표 ({len(valid_vacancies)}개) ===")
        for i, coord in enumerate(valid_vacancies, 1):
            print(f"  ({coord[0]:.5f}, {coord[1]:.5f}, {coord[2]:.5f}),")
