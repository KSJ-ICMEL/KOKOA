"""
LLZO 구조를 P1으로 확장하고 24d vacancy 사이트를 명시적으로 추가
================================================================
중복 좌표를 필터링하고, 24d 사이트 중 16개를 He(vacancy placeholder)로 추가합니다.
"""

import numpy as np
from pymatgen.core import Structure, Element, Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import os

# === 1. wyu.cif 로드 ===
cif_path = os.path.join(os.path.dirname(__file__), "wyu.cif")
structure = Structure.from_file(cif_path)
print(f"Original: {structure.formula}, {len(structure)} atoms")

# === 2. P1으로 확장 ===
sga = SpacegroupAnalyzer(structure)
p1_structure = sga.get_conventional_standard_structure()
print(f"P1 expanded: {p1_structure.formula}, {len(p1_structure)} atoms")

# === 3. 중복 좌표 제거 ===
def normalize_coord(coord, tol=0.001):
    """좌표를 [0, 1) 범위로 정규화하고 반올림"""
    result = []
    for c in coord:
        c_norm = c % 1.0
        # 0.999... -> 0.0 처리
        if c_norm > (1.0 - tol):
            c_norm = 0.0
        result.append(round(c_norm, 5))
    return tuple(result)

lattice = p1_structure.lattice
unique_species = []
unique_coords = []
seen_coords = set()

for site in p1_structure:
    norm = normalize_coord(site.frac_coords)
    if norm not in seen_coords:
        seen_coords.add(norm)
        unique_species.append(site.species)
        unique_coords.append(list(norm))

print(f"After deduplication: {len(unique_species)} sites")

# === 4. 16개 vacancy 좌표 정의 (Li가 없는 24d 위치) ===
# 현재 Li가 점유한 24d 위치 확인
li_24d_occupied = set()
for i, (sp, coord) in enumerate(zip(unique_species, unique_coords)):
    if "Li" in str(sp):
        norm = normalize_coord(coord)
        # 24d 좌표 패턴: (0, 0.25, 0.5, 0.75) 조합
        all_quarter = all(abs(c - round(c * 4) / 4) < 0.02 for c in norm)
        if all_quarter:
            li_24d_occupied.add(norm)

print(f"Li-occupied 24d positions: {len(li_24d_occupied)}")

# 가능한 모든 24d 좌표 (unit cell 내에서 0, 0.25, 0.5, 0.75 조합 중 일부)
# 실제 24d 사이트는 특정 패턴만 해당
all_24d = [
    # 8개 Li 점유 사이트
    (0.0, 0.0, 0.0), (0.5, 0.0, 0.25), (0.0, 0.5, 0.25), (0.5, 0.5, 0.0),
    (0.0, 0.0, 0.5), (0.5, 0.0, 0.75), (0.0, 0.5, 0.75), (0.5, 0.5, 0.5),
    # 16개 vacancy 사이트 (다른 패턴)
    (0.25, 0.0, 0.125), (0.75, 0.0, 0.125), (0.0, 0.25, 0.125), (0.0, 0.75, 0.125),
    (0.25, 0.5, 0.625), (0.75, 0.5, 0.625), (0.5, 0.25, 0.625), (0.5, 0.75, 0.625),
    (0.25, 0.25, 0.0), (0.75, 0.75, 0.0), (0.25, 0.75, 0.5), (0.75, 0.25, 0.5),
    (0.0, 0.25, 0.375), (0.0, 0.75, 0.375), (0.5, 0.25, 0.875), (0.5, 0.75, 0.875),
]

# vacancy 좌표: Li가 점유하지 않은 24d 사이트, 다른 원소와도 겹치지 않는 위치
vacancy_coords = []
for pos in all_24d:
    norm = normalize_coord(pos)
    if norm not in seen_coords:
        vacancy_coords.append(pos)

print(f"Vacancy positions to add: {len(vacancy_coords)}")

# === 5. He(vacancy) 사이트 추가 ===
he_element = Element("He")
for vac_pos in vacancy_coords:
    unique_species.append(he_element)
    unique_coords.append(list(vac_pos))
    seen_coords.add(normalize_coord(vac_pos))

# === 6. 새 구조 생성 ===
new_structure = Structure(lattice, unique_species, unique_coords)
print(f"New structure: {new_structure.formula}, {len(new_structure)} atoms")

# === 7. CIF 저장 ===
output_path = os.path.join(os.path.dirname(__file__), "LLZO_with_vacancy.cif")
new_structure.to(filename=output_path)
print(f"\nSaved: {output_path}")

# === 8. 요약 ===
li_count = sum(1 for site in new_structure if "Li" in str(site.species))
he_count = sum(1 for site in new_structure if "He" in str(site.species))

print("\n" + "="*50)
print("Summary")
print(f"  Total atoms: {len(new_structure)}")
print(f"  Li atoms: {li_count}")
print(f"  He atoms (vacancy placeholder): {he_count}")
print("="*50)
print("\nNote: In KMC simulation, Li can hop to He sites (vacancies).")


