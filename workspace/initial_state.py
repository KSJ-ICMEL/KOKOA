import numpy as np
from pymatgen.core import Structure
import os

# =============================================================================
# 1. 설정 및 초기화 (Configuration & Initialization)
# =============================================================================
print("📂 1. 구조 파일 로드 및 슈퍼셀 생성...")

# CIF 파일 로드 (파일 경로를 환경에 맞게 수정하세요)
cif_path = "./etz.cif" 

if os.path.exists(cif_path):
    cif_string = open(cif_path, "r").read()
    structure = Structure.from_str(cif_string, fmt="cif")
else:
    # 파일이 없을 경우 테스트용 더미 구조 생성 (또는 에러 처리)
    raise FileNotFoundError(f"'{cif_path}' 파일을 찾을 수 없습니다.")

# 슈퍼셀 확장 (Convergence Test: N=2 -> 8배 확장)
N = 8
structure.make_supercell([N, N, N])
print(f"   -> 슈퍼셀 확장 완료 ({N}x{N}x{N}). 총 원자 수: {len(structure)}")

# 초기 리튬 배치 (Occupancy 확률 적용)
initial_sites = []
for site in structure:
    species = site.species
    coords = site.frac_coords
    
    # 리튬 자리인 경우만 처리
    if "Li" in [s.symbol for s in species.elements]:
        prob_li = species.get("Li", 0)
        
        # 몬테카를로 방식으로 초기 상태 결정 (Li or Vacancy)
        state = 1 if np.random.rand() < prob_li else 0
            
        initial_sites.append({
            "coords": coords,
            "state": state,
            "site_name": site.label
        })

print(f"   -> 초기화 완료: 총 {len(initial_sites)}개의 리튬 사이트 설정됨.")

# =============================================================================
# 2. 그래프 구축 (Graph Building with Vectors)
# =============================================================================
print("🕸️ 2. 이동 경로 그래프 구축 (Vectorized Adjacency List)...")

cutoff = 4.0 # 호핑 가능 최대 거리 (Å)
neighbors_data = structure.get_all_neighbors(r=cutoff)
adj_list = {}

for i, site in enumerate(structure):
    # 리튬 사이트만 노드로 등록
    if "Li" not in site.species.elements[0].symbol:
        continue
        
    my_neighbors = []
    for neighbor in neighbors_data[i]:
        target_idx = neighbor.index
        
        # 타겟도 리튬 자리여야 함
        if "Li" in structure[target_idx].species.elements[0].symbol:
            # [핵심] 변위 벡터 계산 (PBC 고려된 Cartesian Vector)
            # 이웃_좌표 - 내_좌표 (이미지 벡터 포함)
            frac_diff = structure[target_idx].frac_coords - site.frac_coords + neighbor.image
            cart_disp = structure.lattice.get_cartesian_coords(frac_diff)
            
            # (도착지 인덱스, 변위 벡터) 저장
            my_neighbors.append((target_idx, cart_disp))
            
    adj_list[i] = my_neighbors

print(f"   -> 그래프 구축 완료. (Cutoff={cutoff}Å)")

# =============================================================================
# 3. 시뮬레이터 클래스 정의 (BKL Engine + MSD Tracking)
# =============================================================================
class KMCSimulator:
    def __init__(self, structure, adj_list, initial_sites, params):
        self.params = params
        self.adj_list = adj_list
        
        # 격자 점유 상태 (0: Vacancy, 1: Li)
        self.occupancy = np.array([s['state'] for s in initial_sites], dtype=int)
        
        # 입자 추적 시스템 (Particle Tracking)
        self.site_to_particle = {}   # {site_idx: particle_id}
        self.particle_positions = {} # {particle_id: {'start': vec, 'current': vec}}
        
        p_id_counter = 0
        for idx, site_info in enumerate(initial_sites):
            if site_info['state'] == 1:
                # 시작 좌표 (Cartesian)
                start_coords = structure.lattice.get_cartesian_coords(site_info['coords'])
                
                self.site_to_particle[idx] = p_id_counter
                self.particle_positions[p_id_counter] = {
                    'start': np.array(start_coords),
                    'current': np.array(start_coords)
                }
                p_id_counter += 1
                
        self.li_indices = set(self.site_to_particle.keys())
        self.num_particles = len(self.li_indices)
        
        # 시간 및 물리 상수
        self.current_time = 0.0
        self.step_count = 0
        
        kb = 8.617e-5 # eV/K
        # Ideal Assumption: 모든 경로의 Rate는 동일함
        self.base_rate = params['nu'] * np.exp(-params['E_a'] / (kb * params['T']))

    def run_step(self):
        # --- (A) 가능한 이벤트 수집 ---
        possible_events = [] 
        cumulative_rates = []
        current_sum = 0.0
        
        for current_site_idx in self.li_indices:
            neighbors = self.adj_list.get(current_site_idx, [])
            
            for neighbor_idx, jump_vector in neighbors:
                # 빈자리(Vacancy)로만 이동 가능
                if self.occupancy[neighbor_idx] == 0:
                    rate = self.base_rate 
                    
                    possible_events.append((current_site_idx, neighbor_idx, jump_vector))
                    current_sum += rate
                    cumulative_rates.append(current_sum)
        
        total_rate = current_sum
        if total_rate == 0: return False # 움직일 곳이 없음 (Deadlock)

        # --- (B) 시간 흐름 (BKL Algorithm) ---
        u1 = np.random.rand()
        dt = -np.log(u1) / total_rate
        self.current_time += dt
        self.step_count += 1
        
        # --- (C) 사건 선택 및 실행 ---
        u2 = np.random.uniform(0, total_rate)
        event_idx = np.searchsorted(cumulative_rates, u2)
        source, target, jump_vector = possible_events[event_idx]
        
        # 1. 입자 ID 식별 및 이동 (Unwrapped Coords Update)
        p_id = self.site_to_particle.pop(source)
        self.particle_positions[p_id]['current'] += jump_vector
        
        # 2. 격자 상태 업데이트
        self.occupancy[source] = 0
        self.occupancy[target] = 1
        self.site_to_particle[target] = p_id
        self.li_indices.remove(source)
        self.li_indices.add(target)
        
        return True

    def calculate_properties(self):
        """ MSD 및 이온 전도도 계산 """
        if self.current_time == 0: return 0, 0
        
        # MSD 계산 (Mean Squared Displacement)
        sq_displacements = []
        for pos_data in self.particle_positions.values():
            delta = pos_data['current'] - pos_data['start']
            sq_displacements.append(np.sum(delta**2))
            
        msd = np.mean(sq_displacements) # Å²
        
        # 확산 계수 D (cm²/s)
        D_sim = msd / (6 * self.current_time) 
        D_cm2s = D_sim * 1e-16
        
        # 전도도 Sigma (S/cm)
        vol_angstrom = self.params['volume']
        n_conc = self.num_particles / (vol_angstrom * 1e-24) # ions/cm³
        q = 1.602e-19
        k_J = 1.38e-23
        
        sigma = (n_conc * (q**2) * D_cm2s) / (k_J * self.params['T'])
        
        return msd, sigma

# =============================================================================
# 4. 시뮬레이션 실행 (Execution)
# =============================================================================
print("🚀 3. 시뮬레이션 시작...")

# 시뮬레이션 파라미터 (Ideal Case)
sim_params = {
    'T': 300,           # 온도 (K)
    'E_a': 0.28,        # 활성화 에너지 (eV)
    'nu': 1e13,         # 시도 빈도 (Hz)
    'volume': structure.volume # 부피 (Å³)
}

sim = KMCSimulator(structure, adj_list, initial_sites, sim_params)

target_time = 50e-9 # 50 ns
log_interval = 2000 # 로그 출력 간격

while sim.current_time < target_time:
    if not sim.run_step():
        print("⚠️ Deadlock 발생으로 중단됨.")
        break
        
    if sim.step_count % log_interval == 0:
        msd, sigma = sim.calculate_properties()
        print(f"[Step {sim.step_count:6d}] Time: {sim.current_time*1e9:6.2f} ns | "
              f"MSD: {msd:6.2f} Å² | σ: {sigma*1000:.4f} mS/cm")

# 최종 결과 리포트
msd, sigma = sim.calculate_properties()
print("\n" + "="*60)
print(f"🏁 시뮬레이션 종료 (목표 시간: {target_time*1e9} ns)")
print(f"   - 온도 (T): {sim_params['T']} K")
print(f"   - 확산 계수 (D): {msd/(6*sim.current_time)*1e-16:.4e} cm²/s")
print(f"   - 이온 전도도 (σ): {sigma:.4e} S/cm ({sigma*1000:.2f} mS/cm)")
print("="*60)