"""
QTE - Quantum Time Evolution
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import pandas as pd
import random
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.utils import algorithm_globals

SEED = 39
np.random.seed(SEED)
random.seed(SEED)
algorithm_globals.random_seed = SEED

CSV_DRAWN = "/data/loto7hh_4582_k22.csv"
CSV_ALL   = "/data/kombinacijeH_39C7.csv"

MIN_VAL = [1, 2, 3, 4, 5, 6, 7]
MAX_VAL = [33, 34, 35, 36, 37, 38, 39]
NUM_QUBITS = 5
WINDOW = 50
TROTTER_STEPS = 3
DT = 0.3
LAMBDA_REG = 0.01


def load_draws():
    df = pd.read_csv(CSV_DRAWN)
    return df.values


def build_empirical(draws, pos):
    n_states = 1 << NUM_QUBITS
    freq = np.zeros(n_states)
    for row in draws:
        v = int(row[pos]) - MIN_VAL[pos]
        if v >= n_states:
            v = v % n_states
        freq[v] += 1
    return freq / freq.sum()


def build_temporal_empirical(draws, pos, window=WINDOW):
    n_states = 1 << NUM_QUBITS
    recent = draws[-window:]
    freq = np.zeros(n_states)
    for row in recent:
        v = int(row[pos]) - MIN_VAL[pos]
        if v >= n_states:
            v = v % n_states
        freq[v] += 1
    return freq / freq.sum()


def build_coupling_matrix(draws, pos, window=WINDOW):
    n_states = 1 << NUM_QUBITS
    recent = draws[-window:]
    J = np.zeros((n_states, n_states))
    for k in range(len(recent) - 1):
        v1 = int(recent[k][pos]) - MIN_VAL[pos]
        v2 = int(recent[k + 1][pos]) - MIN_VAL[pos]
        if v1 >= n_states:
            v1 = v1 % n_states
        if v2 >= n_states:
            v2 = v2 % n_states
        J[v1, v2] += 1
        J[v2, v1] += 1
    if J.max() > 0:
        J /= J.max()
    return J


def trotter_evolve(init_probs, J, dt=DT, steps=TROTTER_STEPS):
    n = len(init_probs)
    H = -J.copy()
    np.fill_diagonal(H, -np.sum(J, axis=1))

    psi = np.sqrt(np.maximum(init_probs, 0)).astype(complex)
    norm = np.linalg.norm(psi)
    if norm > 0:
        psi /= norm

    for step in range(steps):
        psi = psi * np.exp(-1j * dt * np.diag(H))

        vals, vecs = np.linalg.eigh(H)
        psi = vecs @ (np.exp(-1j * dt * vals) * (vecs.T @ psi))

        norm = np.linalg.norm(psi)
        if norm > 0:
            psi /= norm

    return np.abs(psi) ** 2


def quantum_time_features(draws, pos):
    qc = QuantumCircuit(NUM_QUBITS)

    recent = draws[-WINDOW:]
    mean_val = np.mean([int(r[pos]) - MIN_VAL[pos] for r in recent])
    std_val = np.std([int(r[pos]) - MIN_VAL[pos] for r in recent]) + 1e-8
    trend = (np.mean([int(r[pos]) for r in recent[-10:]]) -
             np.mean([int(r[pos]) for r in recent[:10]]))

    theta_mean = mean_val * np.pi / 31.0
    theta_std = std_val * np.pi / 31.0
    theta_trend = np.clip(trend / 10.0, -1, 1) * np.pi

    qc.ry(theta_mean, 0)
    qc.ry(theta_std, 1)
    qc.ry(theta_trend, 2)
    qc.ry((theta_mean + theta_std) / 2, 3)
    qc.ry((theta_mean - theta_trend) / 2, 4)

    for i in range(NUM_QUBITS - 1):
        qc.cx(i, i + 1)
    qc.cx(NUM_QUBITS - 1, 0)

    qc.rz(theta_mean * 0.5, 0)
    qc.rz(theta_std * 0.5, 1)
    qc.rz(theta_trend * 0.5, 2)

    sv = Statevector.from_instruction(qc)
    return sv.probabilities()


def greedy_combo(dists):
    combo = []
    used = set()
    for pos in range(7):
        ranked = sorted(enumerate(dists[pos]),
                        key=lambda x: x[1], reverse=True)
        for mv, score in ranked:
            actual = int(mv) + MIN_VAL[pos]
            if actual > MAX_VAL[pos]:
                continue
            if actual in used:
                continue
            if combo and actual <= combo[-1]:
                continue
            combo.append(actual)
            used.add(actual)
            break
    return combo


def main():
    draws = load_draws()
    print(f"Ucitano izvucenih kombinacija: {len(draws)}")

    df_all_head = pd.read_csv(CSV_ALL, nrows=3)
    print(f"Graf svih kombinacija: {CSV_ALL}")
    print(f"  Primer: {df_all_head.values[0].tolist()} ... "
          f"{df_all_head.values[-1].tolist()}")

    print(f"\n--- Quantum Time Evolution ({NUM_QUBITS}q, "
          f"window={WINDOW}, Trotter steps={TROTTER_STEPS}) ---")

    dists = []
    for pos in range(7):
        print(f"  Poz {pos+1}...", end=" ", flush=True)

        p_all = build_empirical(draws, pos)
        p_recent = build_temporal_empirical(draws, pos)
        J = build_coupling_matrix(draws, pos)
        p_evolved = trotter_evolve(p_recent, J)
        p_quantum = quantum_time_features(draws, pos)

        combined = (0.3 * p_all + 0.3 * p_evolved +
                    0.2 * p_recent + 0.2 * p_quantum)
        combined = combined - combined.min()
        if combined.sum() > 0:
            combined /= combined.sum()
        dists.append(combined)

        top_idx = np.argsort(combined)[::-1][:3]
        info = " | ".join(
            f"{i + MIN_VAL[pos]}:{combined[i]:.3f}" for i in top_idx)
        print(f"top: {info}")

    combo = greedy_combo(dists)

    print(f"\n{'='*50}")
    print(f"Predikcija (QTE, deterministicki, seed={SEED}):")
    print(combo)
    print(f"{'='*50}")


if __name__ == "__main__":
    main()



"""
Ucitano izvucenih kombinacija: 4582
Graf svih kombinacija: /Users/4c/Desktop/GHQ/data/kombinacijeH_39C7.csv
  Primer: [1, 2, 3, 4, 5, 6, 7] ... [1, 2, 3, 4, 5, 6, 9]

--- Quantum Time Evolution (5q, window=50, Trotter steps=3) ---
  Poz 1... top: 1:0.290 | 3:0.282 | 2:0.089
  Poz 2... top: 8:0.230 | 2:0.162 | 5:0.098
  Poz 3... top: 11:0.157 | 3:0.096 | 13:0.068
  Poz 4... top: 14:0.163 | 21:0.071 | 19:0.071
  Poz 5... top: 23:0.160 | 22:0.102 | 35:0.072
  Poz 6... top: 35:0.111 | 34:0.107 | 36:0.092
  Poz 7... top: 38:0.256 | 37:0.168 | 7:0.110

==================================================
Predikcija (QTE, deterministicki, seed=39):
[1, 8, x, y, z, 35, 38]
==================================================
"""



"""
QTE - Quantum Time Evolution

Temporalni pristup: koristi poslednjih 50 izvlacenja kao prozor
Coupling matrica J: iz prelaza izmedju uzastopnih izvlacenja - koje vrednosti slede jednu drugoj
Trotterizovana evolucija: simulira kvantnu vremensku evoluciju H = -J hamiltonijana (3 koraka, dt=0.3)
Kvantni vremenski feature-i: enkodira mean, std i trend poslednjih izvlacenja u 5-qubit kolo
Kombinovani skor: 30% sva izvlacenja + 30% evoluirano + 20% recentno + 20% kvantni feature-i
Prvi model koji eksplicitno koristi vremenske zavisnosti izmedju izvlacenja
Deterministicki, brz, bez iterativnog treniranja
"""
