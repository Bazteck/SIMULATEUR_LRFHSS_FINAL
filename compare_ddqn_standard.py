#!/usr/bin/env python3
"""
compare_simulation_with_ddqn.py
Compare les performances du DDQN avec les DR simulés et les données réelles
sur toutes les distances du fichier de comparaison
AVEC ANALYSE ÉNERGÉTIQUE ET ANALYSE DE DÉBIT AJOUTÉES
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys
import time
import os
import torch
import random
from collections import deque
import logging

# Importer le simulateur et DQN
from simulation import LR_FHSS_Simulation
from integrated_ddqn import IntegratedDDQNAgent, LightweightDDQN
# Ajout: import du modèle énergétique
from energy import EnergyConsumptionModel
from config import LR_FHSS_Config

# ===== FONCTION POUR FIXER TOUTES LES GRAINES ALÉATOIRES =====
def set_all_seeds(seed=42):
    """
    Fixe toutes les graines aléatoires pour garantir la reproductibilité
    des simulations à chaque exécution.
    """
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch (CPU et GPU)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # Pour multi-GPU
    
    # Optionnel: rendre les algorithmes CUDA déterministes
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Pour la reproductibilité sur certains OS
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"Fixation de toutes les graines aléatoires à {seed}")
    print(f"   - random.seed({seed})")
    print(f"   - np.random.seed({seed})")
    print(f"   - torch.manual_seed({seed})")
    if torch.cuda.is_available():
        print(f"   - torch.cuda.manual_seed({seed})")
    print(f"   - cudnn.deterministic = True")

# ===== APPLICATION DE LA GRAINE AU DÉBUT =====
MASTER_SEED = 10  
set_all_seeds(MASTER_SEED)

# Configuration logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

print("\nCOMPARAISON LR-FHSS: DONNÉES RÉELLES vs SIMULATION STANDARD vs DDQN")
print("=" * 90)

# ===== FONCTION DE CALCUL ToA EXACT =====
def calculate_toa_ms(dr: int, payload_bytes: int = 1) -> float:
    """Calcule le Time-on-Air exact pour LR-FHSS selon config.py"""
    return LR_FHSS_Config.calculate_toa_ms(dr, payload_bytes)

# ===== FONCTION DE CALCUL D'ÉNERGIE PAR TRANSMISSION =====
def calculate_transmission_energy(dr: int, payload_bytes: int = 1, tx_power: float = None) -> dict:
    """
    Calcule la consommation énergétique pour une transmission LR-FHSS
    Retourne un dictionnaire avec les métriques énergétiques en Joules
    
    Attention: sleep_duration_s=0.0 pour calculer SEULEMENT l'énergie de transmission
    (pas d'énergie de veille après transmission)
    """
    if tx_power is None:
        tx_power = DR_CONFIG_MAP[dr]['tx']
    
    toa_ms = calculate_toa_ms(dr, payload_bytes)
    
    return EnergyConsumptionModel.calculate_energy_joules(
        tx_power_dbm=tx_power,
        toa_ms=toa_ms,
        pa_type='SX1261_LP',  # SX1261 Low Power PA (14 dBm)
        sleep_duration_s=0.0,  # ✅ IMPORTANT: pas de veille incluse (transmission seule)
        rx_duration_ms=0.0,    # ✅ pas de réception ACK
        voltage_v=3.3          # Tension batterie standard
    )

# ===== DÉFINITION DES DÉBITS PAR DR =====
print("\nDébits théoriques par DR:")
print("-" * 90)
THROUGHPUT_BY_DR = {
    8: 162,   # bps
    9: 325,   # bps
    10: 162,  # bps
    11: 325,  # bps
}
for dr in [8, 9, 10, 11]:
    print(f"  DR{dr}: {THROUGHPUT_BY_DR[dr]} bps ({THROUGHPUT_BY_DR[dr]/1000:.2f} kbps)")

# ===== CHARGER LE MODÈLE DDQN ENTRAÎNÉ =====
print("\nChargement du modèle DDQN...")

# À MODIFIER: Chemin vers votre modèle entraîné
DDQN_MODEL_PATH = "ddqn_checkpoints/ddqn_deploy.pth"

# Vérifier si le modèle existe
if not os.path.exists(DDQN_MODEL_PATH):
    alt_paths = [
        "BEST/dqn_models/dqn_model_best.pth",
        "dqn_models/dqn_model_final.pth",
        "ddqn_sequential_checkpoints/ddqn_final.pth",
        "ddqn_models/ddqn_best.pth"
    ]
    for alt in alt_paths:
        if os.path.exists(alt):
            DDQN_MODEL_PATH = alt
            print(f"Modèle trouvé: {alt}")
            break

print(f"Modèle: {DDQN_MODEL_PATH}")
print(f"Existe: {os.path.exists(DDQN_MODEL_PATH)}")

# ===== CHARGER LES DONNÉES RÉELLES =====
print("\nChargement des données réelles...")
real_data = pd.read_csv('PDR_avgRSSI_distance.csv')
print(f"{len(real_data)} points de mesure réelles")

# Extraire TOUTES les distances et datarates
all_distances = sorted(real_data['Distance(m)'].unique().tolist())
all_datarates = sorted(real_data['DR'].unique().tolist())

# Filtrer pour garder seulement DR8, DR9, DR10, DR11
test_datarates = [dr for dr in all_datarates if dr >= 8]
test_distances = all_distances

print(f"\nConfiguration:")
print(f"   - Distances trouvées: {test_distances}m ({len(test_distances)} distances)")
print(f"   - Datarates LR-FHSS testés: {test_datarates}")
print(f"   - Total simulations standard: {len(test_distances) * len(test_datarates)}")
print(f"   - Total simulations DDQN: {len(test_distances)} (DR variable)")
print(f"   - Locations: {sorted(real_data['Direction'].unique().tolist())}")
print(f"   - Graine maître: {MASTER_SEED} (résultats reproductibles)")

# Configuration de base
BASE_CONFIG = {
    'simulation_duration': 1800,        # 30 minutes
    'num_devices': 10,
    'region': 'EU868',
    'payload_min': 1,
    'payload_max': 1,                    # Payload de 1 byte
    'tx_interval_min': 30,
    'tx_interval_max': 60,
    'shadowing_std_db': 7.0,
    'path_loss_exponent': 3.3,
    'noise_figure_db': 6.0,
    'enable_intelligent_scheduler': False,
    'position_seed': MASTER_SEED,        # Utiliser la graine maître
    'shadowing_seed': MASTER_SEED,       # Utiliser la graine maître
}

# Mapping datarate -> configuration complète
DR_CONFIG_MAP = {
    8: {'bw_khz': 136.71875, 'cr': '1/3', 'tx': 14, 'nominal_rate_bps': 162, 'info_rate_bps': 488},
    9: {'bw_khz': 136.71875, 'cr': '2/3', 'tx': 14, 'nominal_rate_bps': 325, 'info_rate_bps': 488},
    10: {'bw_khz': 335.9375, 'cr': '1/3', 'tx': 14, 'nominal_rate_bps': 162, 'info_rate_bps': 488},
    11: {'bw_khz': 335.9375, 'cr': '2/3', 'tx': 14, 'nominal_rate_bps': 325, 'info_rate_bps': 488},
}

# ===== CALCUL DES ÉNERGIES THÉORIQUES PAR DR =====
print("\nCalcul des énergies théoriques par transmission (payload 1 byte)")
print("-" * 90)

energy_by_dr = {}
for dr in test_datarates:
    energy = calculate_transmission_energy(dr)
    energy_by_dr[dr] = energy
    
    # Conversions pour affichage
    energy_mj = energy['total_energy_j'] * 1000
    energy_uj = energy['total_energy_j'] * 1e6
    
    print(f"  DR{dr}: ToA={calculate_toa_ms(dr):.2f}ms, "
          f"Énergie={energy_mj:.3f}mJ ({energy_uj:.0f}µJ), "
          f"Courant={energy['tx_current_ma']:.1f}mA, "
          f"Batterie={energy['battery_life_years']:.1f}ans, "
          f"Débit={THROUGHPUT_BY_DR[dr]}bps")

# ===== INITIALISER L'AGENT DDQN =====
print("\nInitialisation de l'agent DDQN...")

ddqn_agent = None
ddqn_available = False

try:
    if os.path.exists(DDQN_MODEL_PATH):
        ddqn_agent = IntegratedDDQNAgent(
            checkpoint_path=DDQN_MODEL_PATH,
            deterministic=True
        )
        ddqn_available = True
        print(f"Agent DDQN initialisé avec succès")
        print(f"   - Dimension état: {ddqn_agent.state_dim}")
        print(f"   - Dimension action: {ddqn_agent.action_dim}")
        print(f"   - Mode: Déterministe")
    else:
        print(f"Attention: Modèle non trouvé: {DDQN_MODEL_PATH}")
except Exception as e:
    print(f"Erreur initialisation DDQN: {e}")
    import traceback
    traceback.print_exc()

# ===== LANCER LES SIMULATIONS =====
print("\nLancement des simulations...")
print("=" * 90)

simulation_results = []  # Résultats standard (DR fixe)
ddqn_results = []        # Résultats DDQN (DR adaptatif)

# ── Collecte des événements temporels pour les courbes débit vs temps ──────
# Structure : { (dist_m, dr_or_'ddqn'): [(t_s, success, dr), ...] }
packets_timeline = {}  # clé -> liste de tuples (start_time_s, success, dr)

total_sims_standard = len(test_distances) * len(test_datarates)
total_sims_ddqn = len(test_distances)
current_sim = 0

# ===== 1. SIMULATIONS STANDARD (DR fixe) =====
print("\nPhase 1: Simulations STANDARD (DR fixe)")
print("-" * 90)

for dist_m in test_distances:
    for dr in test_datarates:
        current_sim += 1
        print(f"\n[{current_sim}/{total_sims_standard + total_sims_ddqn}] STANDARD: Distance={dist_m}m, DR={dr}")
        print("-" * 90)
        
        config = BASE_CONFIG.copy()
        config['distance_gtw'] = dist_m
        config['coding_rate'] = DR_CONFIG_MAP[dr]['cr']
        config['tx_power'] = DR_CONFIG_MAP[dr]['tx']
        config['bandwidth_khz'] = DR_CONFIG_MAP[dr]['bw_khz']
        config['enable_dqn'] = False
        
        print(f"  Config: CR={config['coding_rate']}, TX={config['tx_power']} dBm, BW={config['bandwidth_khz']} kHz")
        print(f"  ToA exact: {calculate_toa_ms(dr):.2f} ms")
        
        try:
            start_time = time.time()
            
            sim = LR_FHSS_Simulation(config)
            sim.run()
            
            elapsed = time.time() - start_time
            print(f"  Exécution: {elapsed:.1f}s")
            
            # Extraire les métriques
            total_sent = sim.total_sent
            successful_rx = sim.successful_rx
            pdr = successful_rx / total_sent if total_sent > 0 else 0
            
            # RSSI et SNR
            rssi_mean = sim.detailed_stats.avg_rssi_dbm if hasattr(sim, 'detailed_stats') else -120
            rssi_std = np.std([p.rssi_dbm for p in sim.simulated_packets if hasattr(p, 'rssi_dbm')]) if sim.simulated_packets else 0
            snr_mean = sim.detailed_stats.avg_snr_db if hasattr(sim, 'detailed_stats') else 0
            ber_mean = np.mean([p.ber for p in sim.simulated_packets if hasattr(p, 'ber') and p.ber is not None]) or 0
            
            # ===== CALCUL ÉNERGÉTIQUE POUR CETTE SIMULATION =====
            energy_per_tx = energy_by_dr[dr]['total_energy_j']
            total_energy_j = energy_per_tx * total_sent
            energy_successful_j = energy_per_tx * successful_rx
            energy_failed_j = energy_per_tx * (total_sent - successful_rx)
            efficiency_ratio = (energy_successful_j / total_energy_j * 100) if total_energy_j > 0 else 0
            
            # ===== CALCUL DU DÉBIT EFFECTIF =====
            effective_throughput_bps = (successful_rx * 1 * 8) / config['simulation_duration']  # payload=1 byte
            nominal_throughput_bps = THROUGHPUT_BY_DR[dr]
            efficiency_throughput_pct = (effective_throughput_bps / nominal_throughput_bps * 100) if nominal_throughput_bps > 0 else 0
            
            # Calcul durée vie batterie
            battery_life_years = EnergyConsumptionModel.calculate_battery_life_joules(
                energy_per_transmission_j=energy_per_tx,
                battery_capacity_mah=1000.0,
                voltage_v=3.3,
                transmissions_per_day=24
            )
            
            result = {
                'type': 'standard',
                'distance_m': dist_m,
                'datarate': dr,
                'coding_rate': config['coding_rate'],
                'tx_power': config['tx_power'],
                'num_packets': total_sent,
                'pdr': pdr,
                'successful_packets': successful_rx,
                'failed_packets': total_sent - successful_rx,
                'rssi_mean_dbm': rssi_mean,
                'rssi_std_db': rssi_std,
                'snr_mean_db': snr_mean,
                'ber_mean': ber_mean,
                'simulation_time_s': elapsed,
                
                # Métriques énergétiques (en Joules)
                'toa_ms': calculate_toa_ms(dr),
                'energy_per_tx_j': energy_per_tx,
                'energy_per_tx_mj': energy_per_tx * 1000,
                'energy_per_tx_uj': energy_per_tx * 1e6,
                'total_energy_j': total_energy_j,
                'total_energy_mj': total_energy_j * 1000,
                'energy_successful_j': energy_successful_j,
                'energy_failed_j': energy_failed_j,
                'energy_efficiency_pct': efficiency_ratio,
                'tx_current_ma': energy_by_dr[dr]['tx_current_ma'],
                'battery_life_years': battery_life_years,
                'daily_energy_j': energy_per_tx * 24,  # 24 tx/jour
                
                # Métriques de débit
                'nominal_throughput_bps': nominal_throughput_bps,
                'effective_throughput_bps': effective_throughput_bps,
                'throughput_efficiency_pct': efficiency_throughput_pct,
            }
            
            simulation_results.append(result)

            # ── Collecte timeline temporelle ──────────────────────────────
            timeline_key = (dist_m, dr)
            timeline = []
            if hasattr(sim, 'simulated_packets') and sim.simulated_packets:
                for pkt in sim.simulated_packets:
                    t   = getattr(pkt, 'start_time', None)
                    ok  = getattr(pkt, 'success', False)
                    d   = getattr(pkt, 'dr', dr)
                    if t is not None:
                        timeline.append((float(t), bool(ok), int(d)))
            packets_timeline[timeline_key] = sorted(timeline, key=lambda x: x[0])
            
            print(f"  Résultats - PDR: {pdr*100:.1f}%, RSSI: {rssi_mean:.1f} dBm, SNR: {snr_mean:.1f} dB")
            print(f"  Débit: {effective_throughput_bps:.1f} bps (efficacité: {efficiency_throughput_pct:.1f}%)")
            print(f"  Énergie: {energy_per_tx*1000:.3f} mJ/paquet, Total: {total_energy_j*1000:.3f} mJ")
            print(f"     Efficacité: {efficiency_ratio:.1f}%, Batterie: {battery_life_years:.1f} ans")
            
        except Exception as e:
            print(f"Erreur: {e}")
            import traceback
            traceback.print_exc()
            continue

# ===== 2. SIMULATIONS AVEC DDQN =====
print("\n" + "=" * 90)
print("Phase 2: Simulations avec DDQN (DR adaptatif)")
print("=" * 90)

if ddqn_available:
    for dist_m in test_distances:
        current_sim += 1
        print(f"\n[{current_sim}/{total_sims_standard + total_sims_ddqn}] DDQN: Distance={dist_m}m")
        print("-" * 90)
        
        config = BASE_CONFIG.copy()
        config['distance_gtw'] = dist_m
        config['enable_dqn'] = True
        config['dqn_model_name'] = DDQN_MODEL_PATH
        config['use_dqn_for_dr'] = True
        config['use_dqn_for_power'] = True
        
        config['coding_rate'] = '1/3'
        config['tx_power'] = 14
        config['bandwidth_khz'] = 136.71875
        
        print(f"  Config: DQN activé, modèle={os.path.basename(DDQN_MODEL_PATH)}")
        print(f"  Graine utilisée: {MASTER_SEED}")
        
        try:
            start_time = time.time()
            
            sim = LR_FHSS_Simulation(config)
            sim.run()
            
            elapsed = time.time() - start_time
            
            # Vérifier que DQN est bien initialisé APRÈS sim.run()
            # car _initialize_dqn() est appelée dans run()
            if sim.dqn_manager and sim.dqn_manager.enabled and sim.dqn_manager.agent:
                print(f"  DQN initialisé avec succès")
            else:
                print(f"  Attention: DQN non initialisé")
            
            # Extraire les métriques DQN
            dqn_stats = sim.dqn_manager.get_stats() if sim.dqn_manager else {}
            
            # Analyser les décisions DQN
            dqn_decisions = []
            drs_chosen = []
            powers_chosen = []
            
            if hasattr(sim, 'simulated_packets'):
                for pkt in sim.simulated_packets:
                    if hasattr(pkt, 'dqn_applied') and pkt.dqn_applied:
                        dqn_decisions.append({
                            'time': pkt.start_time,
                            'dr': pkt.dr,
                            'power': pkt.tx_power_dbm,
                            'success': getattr(pkt, 'success', False)
                        })
                        drs_chosen.append(pkt.dr)
                        powers_chosen.append(pkt.tx_power_dbm)
            
            # Métriques globales
            total_sent = sim.total_sent
            successful_rx = sim.successful_rx
            pdr = successful_rx / total_sent if total_sent > 0 else 0
            
            rssi_mean = sim.detailed_stats.avg_rssi_dbm if hasattr(sim, 'detailed_stats') else -120
            rssi_std = np.std([p.rssi_dbm for p in sim.simulated_packets if hasattr(p, 'rssi_dbm')]) if sim.simulated_packets else 0
            snr_mean = sim.detailed_stats.avg_snr_db if hasattr(sim, 'detailed_stats') else 0
            ber_mean = np.mean([p.ber for p in sim.simulated_packets if hasattr(p, 'ber') and p.ber is not None]) or 0
            
            # Statistiques DQN
            dqn_decision_count = len(drs_chosen)
            avg_dr_choice = np.mean(drs_chosen) if drs_chosen else 0
            avg_power_choice = np.mean(powers_chosen) if powers_chosen else 0
            dr_distribution = {8: 0, 9: 0, 10: 0, 11: 0}
            for dr in drs_chosen:
                dr_distribution[int(dr)] = dr_distribution.get(int(dr), 0) + 1
            
            # ===== CALCUL ÉNERGÉTIQUE POUR DDQN (AMÉLIORÉ) =====
            # Calculer l'énergie en fonction DES DR ET PUISSANCES RÉELS choisis
            total_energy_j = 0
            energy_successful_j = 0
            energy_failed_j = 0
            tx_powers_used = []
            
            # Si on a les détails des paquets, on calcule précisément
            if hasattr(sim, 'simulated_packets'):
                for pkt in sim.simulated_packets:
                    dr_val = int(getattr(pkt, 'dr', 8))
                    tx_power = getattr(pkt, 'tx_power_dbm', 14.0)  # Récupérer la puissance réelle
                    success = getattr(pkt, 'success', False)
                    
                    tx_powers_used.append(tx_power)
                    
                    # Calculer l'énergie avec le DR ET la puissance réels
                    energy_result = calculate_transmission_energy(dr=dr_val, tx_power=tx_power)
                    energy_pkt = energy_result['total_energy_j']
                    
                    total_energy_j += energy_pkt
                    if success:
                        energy_successful_j += energy_pkt
                    else:
                        energy_failed_j += energy_pkt
            else:
                # Fallback: utiliser la moyenne des DR et puissances choisis
                if drs_chosen and powers_chosen:
                    avg_energy = np.mean([calculate_transmission_energy(dr=int(dr), tx_power=pwr)['total_energy_j']
                                         for dr, pwr in zip(drs_chosen, powers_chosen)])
                    total_energy_j = avg_energy * total_sent
                    energy_successful_j = avg_energy * successful_rx
                    energy_failed_j = avg_energy * (total_sent - successful_rx)
            
            efficiency_ratio = (energy_successful_j / total_energy_j * 100) if total_energy_j > 0 else 0
            
            # Énergie moyenne par paquet
            avg_energy_per_packet_j = total_energy_j / total_sent if total_sent > 0 else 0
            
            # ===== CALCUL DU DÉBIT EFFECTIF POUR DDQN =====
            effective_throughput_bps = (successful_rx * 1 * 8) / config['simulation_duration']  # payload=1 byte
            
            # Débit nominal pondéré par la distribution des DR choisis
            if drs_chosen:
                weighted_nominal_throughput = np.mean([THROUGHPUT_BY_DR[int(dr)] for dr in drs_chosen])
            else:
                weighted_nominal_throughput = 0
            
            efficiency_throughput_pct = (effective_throughput_bps / weighted_nominal_throughput * 100) if weighted_nominal_throughput > 0 else 0
            
            # Calcul durée vie batterie
            battery_life_years = EnergyConsumptionModel.calculate_battery_life_joules(
                energy_per_transmission_j=avg_energy_per_packet_j,
                battery_capacity_mah=1000.0,
                voltage_v=3.3,
                transmissions_per_day=24
            )
            
            result = {
                'type': 'ddqn',
                'distance_m': dist_m,
                'datarate': 'adaptive',
                'num_packets': total_sent,
                'pdr': pdr,
                'successful_packets': successful_rx,
                'failed_packets': total_sent - successful_rx,
                'rssi_mean_dbm': rssi_mean,
                'rssi_std_db': rssi_std,
                'snr_mean_db': snr_mean,
                'ber_mean': ber_mean,
                'simulation_time_s': elapsed,
                'dqn_decisions': dqn_decision_count,
                'avg_dr_choice': avg_dr_choice,
                'avg_power_choice': avg_power_choice,
                'dr8_count': dr_distribution[8],
                'dr9_count': dr_distribution[9],
                'dr10_count': dr_distribution[10],
                'dr11_count': dr_distribution[11],
                
                # Métriques énergétiques (en Joules)
                'total_energy_j': total_energy_j,
                'total_energy_mj': total_energy_j * 1000,
                'energy_successful_j': energy_successful_j,
                'energy_failed_j': energy_failed_j,
                'energy_efficiency_pct': efficiency_ratio,
                'avg_energy_per_packet_j': avg_energy_per_packet_j,
                'avg_energy_per_packet_mj': avg_energy_per_packet_j * 1000,
                'battery_life_years': battery_life_years,
                'daily_energy_j': avg_energy_per_packet_j * 24,
                
                # Métriques de débit
                'effective_throughput_bps': effective_throughput_bps,
                'weighted_nominal_throughput_bps': weighted_nominal_throughput,
                'throughput_efficiency_pct': efficiency_throughput_pct,
            }
            
            ddqn_results.append(result)

            # ── Collecte timeline temporelle DDQN ─────────────────────────
            timeline_key = (dist_m, 'ddqn')
            timeline = []
            if hasattr(sim, 'simulated_packets') and sim.simulated_packets:
                for pkt in sim.simulated_packets:
                    t  = getattr(pkt, 'start_time', None)
                    ok = getattr(pkt, 'success', False)
                    d  = getattr(pkt, 'dr', 8)
                    if t is not None:
                        timeline.append((float(t), bool(ok), int(d)))
            packets_timeline[timeline_key] = sorted(timeline, key=lambda x: x[0])

            print(f"  Résultats - PDR: {pdr*100:.1f}%, RSSI: {rssi_mean:.1f} dBm, SNR: {snr_mean:.1f} dB")
            print(f"  Décisions DQN: {dqn_decision_count} (DR moy: {avg_dr_choice:.1f}, Puiss moy: {avg_power_choice:.1f} dBm)")
            print(f"     Distribution DR: 8:{dr_distribution[8]}, 9:{dr_distribution[9]}, 10:{dr_distribution[10]}, 11:{dr_distribution[11]}")
            print(f"  Débit effectif: {effective_throughput_bps:.1f} bps (efficacité: {efficiency_throughput_pct:.1f}%)")
            print(f"  Énergie: {avg_energy_per_packet_j*1000:.3f} mJ/paquet, Total: {total_energy_j*1000:.3f} mJ")
            print(f"     Efficacité: {efficiency_ratio:.1f}%, Batterie: {battery_life_years:.1f} ans")
            
        except Exception as e:
            print(f"Erreur DDQN: {e}")
            import traceback
            traceback.print_exc()
            continue
else:
    print("Attention: DDQN non disponible - saut des simulations avec DQN")

print("\n" + "=" * 90)
print("Toutes les simulations terminées")
print("=" * 90)

# ===== CONVERTIR EN DATAFRAMES =====
df_standard = pd.DataFrame(simulation_results)
df_ddqn = pd.DataFrame(ddqn_results)

# Sauvegarder les résultats
csv_standard = 'simulation_standard_results.csv'
df_standard.to_csv(csv_standard, index=False)
print(f"Résultats standard sauvegardés: {csv_standard}")

if len(df_ddqn) > 0:
    csv_ddqn = 'simulation_ddqn_results.csv'
    df_ddqn.to_csv(csv_ddqn, index=False)
    print(f"Résultats DDQN sauvegardés: {csv_ddqn}")

# ===== COMPARAISON ET STATISTIQUES =====
print("\nCOMPARAISON GLOBALE: RÉEL vs STANDARD vs DDQN")
print("=" * 90)

comparison_rows = []

for distance_m in test_distances:
    print(f"\nDistance: {distance_m}m")
    print("-" * 90)
    
    # Données réelles (moyenne sur tous les DR)
    real_subset = real_data[(real_data['Distance(m)'] == distance_m) & (real_data['DR'] >= 8)]
    
    if len(real_subset) == 0:
        print(f"  Aucune donnée réelle pour cette distance")
        continue
    
    pdr_real_avg = real_subset['PDR'].mean() * 100
    rssi_real_avg = real_subset['Avg_RSSI(dBm)'].mean()
    pdr_std_real = real_subset['PDR'].std() * 100
    
    # Données réelles par DR
    pdr_real_by_dr = {}
    rssi_real_by_dr = {}
    for dr in test_datarates:
        dr_subset = real_subset[real_subset['DR'] == dr]
        if len(dr_subset) > 0:
            pdr_real_by_dr[dr] = dr_subset['PDR'].mean() * 100
            rssi_real_by_dr[dr] = dr_subset['Avg_RSSI(dBm)'].mean()
    
    # Simulation standard - Grouper par Coding Rate
    std_subset = df_standard[df_standard['distance_m'] == distance_m]
    if len(std_subset) > 0:
        pdr_std_avg = std_subset['pdr'].mean() * 100
        rssi_std_avg = std_subset['rssi_mean_dbm'].mean()
        pdr_std_std = std_subset['pdr'].std() * 100
        
        # Métriques de débit standard
        if 'effective_throughput_bps' in std_subset.columns:
            throughput_std_avg = std_subset['effective_throughput_bps'].mean()
            throughput_std_by_dr = {}
            for dr in test_datarates:
                dr_data = std_subset[std_subset['datarate'] == dr]
                if len(dr_data) > 0:
                    throughput_std_by_dr[dr] = dr_data['effective_throughput_bps'].iloc[0]
        else:
            throughput_std_avg = 0
            throughput_std_by_dr = {}
        
        # AJOUT: Métriques énergétiques standard PAR GROUPE DE CR
        # Groupe 1: DR8 + DR10 (CR 1/3)
        std_cr13 = std_subset[std_subset['datarate'].isin([8, 10])]
        if len(std_cr13) > 0:
            energy_std_cr13 = (std_cr13['total_energy_mj'].sum() / std_cr13['num_packets'].sum() 
                              if std_cr13['num_packets'].sum() > 0 else 0)
        else:
            energy_std_cr13 = 0
        
        # Groupe 2: DR9 + DR11 (CR 2/3)
        std_cr23 = std_subset[std_subset['datarate'].isin([9, 11])]
        if len(std_cr23) > 0:
            energy_std_cr23 = (std_cr23['total_energy_mj'].sum() / std_cr23['num_packets'].sum() 
                              if std_cr23['num_packets'].sum() > 0 else 0)
        else:
            energy_std_cr23 = 0
        
        # Moyenne générale (pour compatibilité avec le code existant)
        if 'total_energy_mj' in std_subset.columns and 'num_packets' in std_subset.columns:
            energy_std_avg = (std_subset['total_energy_mj'].sum() / std_subset['num_packets'].sum() 
                            if std_subset['num_packets'].sum() > 0 else 0)
        else:
            energy_std_avg = 0
        
        battery_std_avg = std_subset['battery_life_years'].mean() if 'battery_life_years' in std_subset.columns else 18.5
        efficiency_std_avg = std_subset['energy_efficiency_pct'].mean() if 'energy_efficiency_pct' in std_subset.columns else 0
    else:
        pdr_std_avg = rssi_std_avg = pdr_std_std = energy_std_avg = battery_std_avg = efficiency_std_avg = 0
        energy_std_cr13 = energy_std_cr23 = 0
        throughput_std_avg = 0
        throughput_std_by_dr = {}
    
    # Simulation DDQN
    ddqn_subset = df_ddqn[df_ddqn['distance_m'] == distance_m]
    if len(ddqn_subset) > 0:
        pdr_ddqn = ddqn_subset['pdr'].iloc[0] * 100
        rssi_ddqn = ddqn_subset['rssi_mean_dbm'].iloc[0]
        avg_dr = ddqn_subset['avg_dr_choice'].iloc[0]
        avg_power = ddqn_subset['avg_power_choice'].iloc[0]
        dr8_count = ddqn_subset['dr8_count'].iloc[0]
        dr9_count = ddqn_subset['dr9_count'].iloc[0]
        dr10_count = ddqn_subset['dr10_count'].iloc[0]
        dr11_count = ddqn_subset['dr11_count'].iloc[0]
        total_ddqn_decisions = dr8_count + dr9_count + dr10_count + dr11_count
        
        # Métriques de débit DDQN
        throughput_ddqn = ddqn_subset['effective_throughput_bps'].iloc[0] if 'effective_throughput_bps' in ddqn_subset.columns else 0
        throughput_efficiency_ddqn = ddqn_subset['throughput_efficiency_pct'].iloc[0] if 'throughput_efficiency_pct' in ddqn_subset.columns else 0
        
        # AJOUT: Métriques énergétiques DDQN
        energy_ddqn = ddqn_subset['avg_energy_per_packet_mj'].iloc[0] if 'avg_energy_per_packet_mj' in ddqn_subset.columns else 0
        battery_ddqn = ddqn_subset['battery_life_years'].iloc[0] if 'battery_life_years' in ddqn_subset.columns else 0
        efficiency_ddqn = ddqn_subset['energy_efficiency_pct'].iloc[0] if 'energy_efficiency_pct' in ddqn_subset.columns else 0
        
        # Calculer les pourcentages de choix
        dr8_pct = (dr8_count / total_ddqn_decisions * 100) if total_ddqn_decisions > 0 else 0
        dr9_pct = (dr9_count / total_ddqn_decisions * 100) if total_ddqn_decisions > 0 else 0
        dr10_pct = (dr10_count / total_ddqn_decisions * 100) if total_ddqn_decisions > 0 else 0
        dr11_pct = (dr11_count / total_ddqn_decisions * 100) if total_ddqn_decisions > 0 else 0
    else:
        pdr_ddqn = rssi_ddqn = avg_dr = avg_power = 0
        energy_ddqn = battery_ddqn = efficiency_ddqn = 0
        throughput_ddqn = throughput_efficiency_ddqn = 0
        dr8_pct = dr9_pct = dr10_pct = dr11_pct = 0
    
    # Calculer les écarts PDR
    if pdr_real_avg > 0 and pdr_std_avg > 0:
        std_error_abs = abs(pdr_real_avg - pdr_std_avg)
        std_error_pct = (std_error_abs / pdr_real_avg * 100) if pdr_real_avg > 0 else 0
        
        if pdr_ddqn > 0:
            ddqn_error_abs = abs(pdr_real_avg - pdr_ddqn)
            ddqn_error_pct = (ddqn_error_abs / pdr_real_avg * 100) if pdr_real_avg > 0 else 0
            
            # Comparaison DDQN vs Standard: amélioration ou dégradation
            pdr_improvement = pdr_ddqn - pdr_std_avg  # Positif = amélioration
            improvement_pct = (pdr_improvement / pdr_std_avg * 100) if pdr_std_avg > 0 else 0
            
            if pdr_improvement > 0:
                improvement_status = f"+{improvement_pct:.1f}%"
            else:
                improvement_status = f"{improvement_pct:.1f}%"
        else:
            ddqn_error_abs = ddqn_error_pct = 0
            pdr_improvement = improvement_pct = 0
            improvement_status = "N/A"
        
        print(f"\n  PDR Moyen:")
        print(f"     Réel:       {pdr_real_avg:.1f}% (±{pdr_std_real:.1f}%)")
        print(f"     Standard:   {pdr_std_avg:.1f}% (±{pdr_std_std:.1f}%)")
        print(f"     Écart:      {std_error_abs:.1f}%")
        
        if pdr_ddqn > 0:
            print(f"     DDQN:       {pdr_ddqn:.1f}%")
            print(f"     Amélioration: {improvement_status}  (vs Standard)")
        
        print(f"\n  RSSI Moyen:")
        print(f"     Réel:       {rssi_real_avg:.1f} dBm")
        print(f"     Standard:   {rssi_std_avg:.1f} dBm")
        
        if rssi_ddqn != 0:
            print(f"     DDQN:       {rssi_ddqn:.1f} dBm")
        
        # Affichage des métriques de débit
        print(f"\n  Débit effectif (bps):")
        print(f"     Standard (par DR):")
        for dr in test_datarates:
            if dr in throughput_std_by_dr:
                print(f"        DR{dr}: {throughput_std_by_dr[dr]:.1f} bps")
        print(f"     Standard (moyen):  {throughput_std_avg:.1f} bps")
        if throughput_ddqn > 0:
            print(f"     DDQN:               {throughput_ddqn:.1f} bps")
            throughput_diff = ((throughput_ddqn - throughput_std_avg) / throughput_std_avg * 100) if throughput_std_avg > 0 else 0
            print(f"     vs Standard: {throughput_diff:+.1f}%")
            print(f"     Efficacité débit DDQN: {throughput_efficiency_ddqn:.1f}%")
        
        # Affichage des métriques énergétiques
        print(f"\n  Énergie par paquet (mJ) - Groupée par Coding Rate:")
        print(f"     Standard (DR8+DR10, CR 1/3): {energy_std_cr13:.3f} mJ")
        print(f"     Standard (DR9+DR11, CR 2/3): {energy_std_cr23:.3f} mJ")
        print(f"     Standard (MOYENNE):           {energy_std_avg:.3f} mJ")
        if energy_ddqn > 0:
            print(f"     DDQN (adaptatif):            {energy_ddqn:.3f} mJ")
            energy_diff_cr13 = ((energy_std_cr13 - energy_ddqn) / energy_std_cr13 * 100) if energy_std_cr13 > 0 else 0
            energy_diff_cr23 = ((energy_std_cr23 - energy_ddqn) / energy_std_cr23 * 100) if energy_std_cr23 > 0 else 0
            print(f"     vs CR 1/3: {energy_diff_cr13:+.1f}% | vs CR 2/3: {energy_diff_cr23:+.1f}%")
        
        print(f"\n  Durée vie batterie (ans):")
        print(f"     Standard:   {battery_std_avg:.1f} ans")
        if battery_ddqn > 0:
            print(f"     DDQN:       {battery_ddqn:.1f} ans")
    
    # Détail des choix DQN
    if total_ddqn_decisions > 0:
        print(f"\n  Choix du DDQN:")
        print(f"     DR moyen:   {avg_dr:.1f}")
        print(f"     Puissance:  {avg_power:.1f} dBm")
        print(f"     Distribution DR:")
        print(f"        DR8:  {dr8_pct:.1f}% ({dr8_count})")
        print(f"        DR9:  {dr9_pct:.1f}% ({dr9_count})")
        print(f"        DR10: {dr10_pct:.1f}% ({dr10_count})")
        print(f"        DR11: {dr11_pct:.1f}% ({dr11_count})")
    
    # Sauvegarder pour export
    comparison_rows.append({
        'distance_m': distance_m,
        'pdr_real': pdr_real_avg,
        'pdr_real_std': pdr_std_real,
        'pdr_standard': pdr_std_avg,
        'pdr_standard_std': pdr_std_std,
        'pdr_ddqn': pdr_ddqn if pdr_ddqn > 0 else None,
        'rssi_real': rssi_real_avg,
        'rssi_standard': rssi_std_avg,
        'rssi_ddqn': rssi_ddqn if rssi_ddqn != 0 else None,
        'ddqn_avg_dr': avg_dr if avg_dr > 0 else None,
        'ddqn_avg_power': avg_power if avg_power > 0 else None,
        'dr8_pct': dr8_pct,
        'dr9_pct': dr9_pct,
        'dr10_pct': dr10_pct,
        'dr11_pct': dr11_pct,
        
        # Métriques de débit
        'throughput_standard_dr8': throughput_std_by_dr.get(8, None),
        'throughput_standard_dr9': throughput_std_by_dr.get(9, None),
        'throughput_standard_dr10': throughput_std_by_dr.get(10, None),
        'throughput_standard_dr11': throughput_std_by_dr.get(11, None),
        'throughput_standard_avg': throughput_std_avg if throughput_std_avg > 0 else None,
        'throughput_ddqn': throughput_ddqn if throughput_ddqn > 0 else None,
        'throughput_efficiency_ddqn': throughput_efficiency_ddqn if throughput_efficiency_ddqn > 0 else None,
        
        # Métriques énergétiques GROUPÉES PAR CR
        'energy_standard_cr13_mj': energy_std_cr13 if energy_std_cr13 > 0 else None,  # DR8+DR10
        'energy_standard_cr23_mj': energy_std_cr23 if energy_std_cr23 > 0 else None,  # DR9+DR11
        'energy_standard_avg_mj': energy_std_avg if energy_std_avg > 0 else None,     # Moyenne générale
        'energy_ddqn_mj': energy_ddqn if energy_ddqn > 0 else None,
        'battery_standard_years': battery_std_avg if battery_std_avg > 0 else None,
        'battery_ddqn_years': battery_ddqn if battery_ddqn > 0 else None,
        'efficiency_standard_pct': efficiency_std_avg if efficiency_std_avg > 0 else None,
        'efficiency_ddqn_pct': efficiency_ddqn if efficiency_ddqn > 0 else None,
    })

# ===== VISUALISATION =====
print("\n\nGénération de la visualisation comparative...")

df_comp = pd.DataFrame(comparison_rows)

# Créer un dossier pour les figures par DR
os.makedirs('figures_dr', exist_ok=True)

# ===== FIGURES SÉPARÉES POUR CHAQUE DR (PDR et RSSI séparés) =====
print("\n" + "=" * 90)
print("GÉNÉRATION DES FIGURES SÉPARÉES PAR DR (COURBES SIMPLES)")
print("=" * 90)

# ===== PALETTE PROFESSIONNELLE ET STYLE GLOBAL =====
import matplotlib as mpl

# Style de base épuré
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 15,
    'axes.titleweight': 'bold',
    'axes.labelsize': 13,
    'axes.labelweight': 'bold',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 1.2,
    'axes.grid': True,
    'grid.alpha': 0.22,
    'grid.linewidth': 0.8,
    'grid.linestyle': '--',
    'grid.color': '#9CA3AF',
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'legend.fontsize': 11,
    'legend.framealpha': 0.95,
    'legend.edgecolor': '#D1D5DB',
    'legend.borderpad': 0.7,
    'figure.dpi': 150,
    'savefig.dpi': 180,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
    'figure.facecolor': 'white',
    'axes.facecolor': '#FAFAFA',
})

# Palette de couleurs professionnelle
C_REAL     = '#1E293B'   # Ardoise très sombre
C_STD      = '#2563EB'   # Bleu cobalt
C_DDQN     = '#DC2626'   # Rouge cardinal

C_DR8      = '#4F46E5'   # Indigo
C_DR9      = '#0891B2'   # Cyan profond
C_DR10     = '#059669'   # Émeraude
C_DR11     = '#D97706'   # Ambre

# Fonction utilitaire : style de fond professionnel pour chaque axe
def style_ax(ax, title=None, xlabel=None, ylabel=None):
    """Applique un style propre et professionnel à un axe."""
    ax.set_facecolor('#F8FAFC')
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color('#CBD5E1')
        ax.spines[spine].set_linewidth(1.1)
    ax.tick_params(colors='#475569', which='both', length=4)
    ax.yaxis.label.set_color('#1E293B')
    ax.xaxis.label.set_color('#1E293B')
    ax.title.set_color('#0F172A')
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    ax.grid(True, which='major', alpha=0.22, linestyle='--', linewidth=0.8, color='#9CA3AF')
    ax.set_axisbelow(True)

# Annotations soignées
def add_annotation(ax, x, y, text, offset=(0, 10), color='#1E293B', bg='white', ec='#CBD5E1', fs=7.5):
    ax.annotate(text, (x, y), xytext=offset, textcoords='offset points',
                ha='center', va='bottom', fontsize=fs, color=color, fontweight='semibold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=bg, edgecolor=ec,
                          linewidth=0.8, alpha=0.92))

LINE_STYLES = {
    'real':     {'color': C_REAL, 'marker': 'o', 'linestyle': '-',  'linewidth': 2.8, 'markersize': 8,
                 'markerfacecolor': C_REAL, 'markeredgecolor': 'white', 'markeredgewidth': 1.5,
                 'label': 'Données Réelles', 'zorder': 4},
    'standard': {'color': C_STD,  'marker': 's', 'linestyle': '--', 'linewidth': 2.4, 'markersize': 8,
                 'markerfacecolor': C_STD,  'markeredgecolor': 'white', 'markeredgewidth': 1.5,
                 'label': 'Simulation Standard', 'zorder': 3},
    'ddqn':     {'color': C_DDQN, 'marker': 'D', 'linestyle': '-.', 'linewidth': 2.4, 'markersize': 8,
                 'markerfacecolor': C_DDQN, 'markeredgecolor': 'white', 'markeredgewidth': 1.5,
                 'label': 'DDQN Adaptatif', 'zorder': 5},
}

for dr in test_datarates:
    print(f"\n  Création des figures pour DR{dr}...")
    print("-" * 50)
    
    # Filtrer les données
    real_dr = real_data[real_data['DR'] == dr].sort_values('Distance(m)')
    std_dr = df_standard[df_standard['datarate'] == dr].sort_values('distance_m')
    
    if len(real_dr) == 0 and len(std_dr) == 0:
        print(f"    ⚠ Pas de données pour DR{dr}")
        continue
    
    # ===== FIGURE PDR =====
    fig_pdr, ax_pdr = plt.subplots(figsize=(16, 7))
    
    cr_info = DR_CONFIG_MAP[dr]['cr']
    bw_info = DR_CONFIG_MAP[dr]['bw_khz']
    style_ax(ax_pdr, xlabel='Distance (m)', ylabel='PDR (%)')

    # Courbe Simulation Standard
    if len(std_dr) > 0:
        ls = LINE_STYLES['standard']
        ax_pdr.plot(std_dr['distance_m'].values, std_dr['pdr'].values * 100, **{k: v for k, v in ls.items() if k != 'label'}, label=ls['label'])
        for x, y in zip(std_dr['distance_m'].values, std_dr['pdr'].values * 100):
            add_annotation(ax_pdr, x, y, f'{y:.0f}%', offset=(0, 11), color=C_STD, bg='#EFF6FF', ec='#BFDBFE')

    # Courbe DDQN
    if len(df_ddqn) > 0:
        ls = LINE_STYLES['ddqn']
        ax_pdr.plot(df_ddqn['distance_m'].values, df_ddqn['pdr'].values * 100, **{k: v for k, v in ls.items() if k != 'label'}, label=ls['label'])
        for x, y in zip(df_ddqn['distance_m'].values, df_ddqn['pdr'].values * 100):
            add_annotation(ax_pdr, x, y, f'{y:.0f}%', offset=(22, 0), color=C_DDQN, bg='#FEF2F2', ec='#FECACA')

    ax_pdr.set_xlim(0, 4200)
    ax_pdr.set_ylim(-2, 112)
    ax_pdr.legend(loc='upper right', frameon=True, shadow=False)

    plt.tight_layout()
    output_file = f'figures_dr/dr{dr}_pdr_courbes.png'
    plt.savefig(output_file)
    print(f"    ✓ FIGURE PDR: {output_file}")
    plt.close(fig_pdr)
    
    # ===== FIGURE RSSI =====
    fig_rssi, ax_rssi = plt.subplots(figsize=(16, 7))
    style_ax(ax_rssi, xlabel='Distance (m)', ylabel='RSSI (dBm)')

    # Courbe Simulation Standard RSSI
    if len(std_dr) > 0:
        ls = LINE_STYLES['standard']
        ax_rssi.plot(std_dr['distance_m'].values, std_dr['rssi_mean_dbm'].values,
                    **{k: v for k, v in ls.items() if k != 'label'}, label=ls['label'])
        for x, y in zip(std_dr['distance_m'].values, std_dr['rssi_mean_dbm'].values):
            add_annotation(ax_rssi, x, y, f'{y:.0f}', offset=(0, 11), color=C_STD, bg='#EFF6FF', ec='#BFDBFE')

    # Courbe DDQN RSSI
    if len(df_ddqn) > 0:
        ls = LINE_STYLES['ddqn']
        ax_rssi.plot(df_ddqn['distance_m'].values, df_ddqn['rssi_mean_dbm'].values,
                    **{k: v for k, v in ls.items() if k != 'label'}, label=ls['label'])
        for x, y in zip(df_ddqn['distance_m'].values, df_ddqn['rssi_mean_dbm'].values):
            add_annotation(ax_rssi, x, y, f'{y:.0f}', offset=(22, 0), color=C_DDQN, bg='#FEF2F2', ec='#FECACA')

    ax_rssi.set_xlim(0, 4200)
    ax_rssi.set_ylim(-150, 0)  # 0 en haut, valeurs négatives en bas
    ax_rssi.legend(loc='lower right', frameon=True)

    plt.tight_layout()
    output_file = f'figures_dr/dr{dr}_rssi_courbes.png'
    plt.savefig(output_file)
    print(f"    ✓ FIGURE RSSI: {output_file}")
    plt.close(fig_rssi)

# ===== FIGURES COMBINÉES =====
print("\n" + "=" * 90)
print("GÉNÉRATION DES FIGURES COMBINÉES")
print("=" * 90)

# Figure PDR combinée
fig_pdr_all, axes = plt.subplots(2, 2, figsize=(20, 12))
axes = axes.flatten()

for idx, dr in enumerate(test_datarates):
    ax = axes[idx]
    std_dr = df_standard[df_standard['datarate'] == dr].sort_values('distance_m')

    if len(std_dr) > 0:
        ls = LINE_STYLES['standard']
        ax.plot(std_dr['distance_m'].values, std_dr['pdr'].values * 100,
                color=ls['color'], marker=ls['marker'], linestyle=ls['linestyle'],
                linewidth=2.0, markersize=6, markerfacecolor=ls['markerfacecolor'],
                markeredgecolor='white', markeredgewidth=1.3,
                label='Standard', zorder=3)

    if len(df_ddqn) > 0:
        ls = LINE_STYLES['ddqn']
        ax.plot(df_ddqn['distance_m'].values, df_ddqn['pdr'].values * 100,
                color=ls['color'], marker=ls['marker'], linestyle=ls['linestyle'],
                linewidth=2.0, markersize=6, markerfacecolor=ls['markerfacecolor'],
                markeredgecolor='white', markeredgewidth=1.3,
                label='DDQN', zorder=5)

    style_ax(ax, xlabel='Distance (m)', ylabel='PDR (%)')
    ax.set_xlim(0, 4200)
    ax.set_ylim(-2, 112)
    ax.legend(loc='upper right', fontsize=9.5)

plt.tight_layout()
output_file = 'figures_dr/all_drs_pdr_courbes.png'
plt.savefig(output_file)
print(f"  ✓ FIGURE PDR COMBINÉE: {output_file}")
plt.close(fig_pdr_all)

print("\n" + "=" * 90)
print("RÉSUMÉ")
print("=" * 90)
print(f"\nFigures créées dans 'figures_dr/':")
print(f"  • {len(test_datarates)} figures PDR individuelles")
print(f"  • {len(test_datarates)} figures RSSI individuelles")
print(f"  • 1 figure PDR combinée")
print(f"\nCaractéristiques des courbes :")
print(f"  • Données Réelles : ligne continue noire avec ronds")
print(f"  • Standard : tirets bleus avec carrés")
print(f"  • DDQN : tirets-points rouges avec losanges")

# ===== TABLEAU RÉCAPITULATIF PAR DR (NOUVEAU) =====
print("\n" + "=" * 90)
print("CRÉATION DU TABLEAU RÉCAPITULATIF PAR DR")
print("=" * 90)

with open('figures_dr/comparison_table_by_dr.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 140 + "\n")
    f.write("TABLEAU COMPARATIF PAR DR: DONNÉES RÉELLES vs SIMULATION STANDARD vs DDQN\n")
    f.write("=" * 140 + "\n\n")
    
    for dr in test_datarates:
        real_dr = real_data[real_data['DR'] == dr].sort_values('Distance(m)')
        std_dr = df_standard[df_standard['datarate'] == dr].sort_values('distance_m')
        
        if len(real_dr) == 0 and len(std_dr) == 0:
            continue
        
        f.write(f"\n{'='*70}\n")
        f.write(f"DR{dr}\n")
        f.write(f"{'='*70}\n")
        f.write(f"\n{'Distance':<12} {'PDR Réel':<15} {'PDR Std':<15} {'PDR DDQN':<15} "
                f"{'RSSI Réel':<15} {'RSSI Std':<15} {'RSSI DDQN':<15}\n")
        f.write(f"{'(m)':<12} {'(%)':<15} {'(%)':<15} {'(%)':<15} "
                f"{'(dBm)':<15} {'(dBm)':<15} {'(dBm)':<15}\n")
        f.write("-" * 117 + "\n")
        
        # Prendre toutes les distances uniques
        all_distances = sorted(set(real_dr['Distance(m)'].values) | 
                              set(std_dr['distance_m'].values) | 
                              set(df_ddqn['distance_m'].values))
        
        
        for dist in all_distances:
            # PDR Réel
            real_row = real_dr[real_dr['Distance(m)'] == dist]
            real_pdr = real_row['PDR'].values[0] * 100 if len(real_row) > 0 else None
            real_rssi = real_row['Avg_RSSI(dBm)'].values[0] if len(real_row) > 0 else None
            
            # PDR Standard
            std_row = std_dr[std_dr['distance_m'] == dist]
            std_pdr = std_row['pdr'].values[0] * 100 if len(std_row) > 0 else None
            std_rssi = std_row['rssi_mean_dbm'].values[0] if len(std_row) > 0 else None
            
            # PDR DDQN
            ddqn_row = df_ddqn[df_ddqn['distance_m'] == dist]
            ddqn_pdr = ddqn_row['pdr'].values[0] * 100 if len(ddqn_row) > 0 else None
            ddqn_rssi = ddqn_row['rssi_mean_dbm'].values[0] if len(ddqn_row) > 0 else None
            
            # Formatage
            real_pdr_str = f"{real_pdr:.1f}" if real_pdr is not None else "N/A"
            std_pdr_str = f"{std_pdr:.1f}" if std_pdr is not None else "N/A"
            ddqn_pdr_str = f"{ddqn_pdr:.1f}" if ddqn_pdr is not None else "N/A"
            
            real_rssi_str = f"{real_rssi:.1f}" if real_rssi is not None else "N/A"
            std_rssi_str = f"{std_rssi:.1f}" if std_rssi is not None else "N/A"
            ddqn_rssi_str = f"{ddqn_rssi:.1f}" if ddqn_rssi is not None else "N/A"
            
            f.write(f"{dist:<12} {real_pdr_str:>8}%      {std_pdr_str:>8}%      "
                    f"{ddqn_pdr_str:>8}%      {real_rssi_str:>8}      "
                    f"{std_rssi_str:>8}      {ddqn_rssi_str:>8}\n")
        
        f.write("-" * 117 + "\n")
        
        # Calculer les écarts moyens
        valid_dists = []
        for dist in all_distances:
            real_row = real_dr[real_dr['Distance(m)'] == dist]
            std_row = std_dr[std_dr['distance_m'] == dist]
            ddqn_row = df_ddqn[df_ddqn['distance_m'] == dist]
            
            if len(real_row) > 0 and len(std_row) > 0 and len(ddqn_row) > 0:
                valid_dists.append(dist)
        
        if len(valid_dists) > 0:
            f.write(f"\nÉcarts moyens (sur {len(valid_dists)} distances):\n")
            
            # Calculer les écarts
            pdr_std_ecarts = []
            pdr_ddqn_ecarts = []
            rssi_std_ecarts = []
            rssi_ddqn_ecarts = []
            
            for dist in valid_dists:
                real_pdr = real_dr[real_dr['Distance(m)'] == dist]['PDR'].values[0] * 100
                std_pdr = std_dr[std_dr['distance_m'] == dist]['pdr'].values[0] * 100
                ddqn_pdr = df_ddqn[df_ddqn['distance_m'] == dist]['pdr'].values[0] * 100
                
                real_rssi = real_dr[real_dr['Distance(m)'] == dist]['Avg_RSSI(dBm)'].values[0]
                std_rssi = std_dr[std_dr['distance_m'] == dist]['rssi_mean_dbm'].values[0]
                ddqn_rssi = df_ddqn[df_ddqn['distance_m'] == dist]['rssi_mean_dbm'].values[0]
                
                pdr_std_ecarts.append(abs(real_pdr - std_pdr))
                pdr_ddqn_ecarts.append(abs(real_pdr - ddqn_pdr))
                rssi_std_ecarts.append(abs(real_rssi - std_rssi))
                rssi_ddqn_ecarts.append(abs(real_rssi - ddqn_rssi))
            
            f.write(f"  • Écart PDR Standard vs Réel: {np.mean(pdr_std_ecarts):.2f}%\n")
            f.write(f"  • Écart PDR DDQN vs Réel: {np.mean(pdr_ddqn_ecarts):.2f}%\n")
            f.write(f"  • Écart RSSI Standard vs Réel: {np.mean(rssi_std_ecarts):.2f} dB\n")
            f.write(f"  • Écart RSSI DDQN vs Réel: {np.mean(rssi_ddqn_ecarts):.2f} dB\n")
        
        f.write("-" * 117 + "\n")

print(f"\n  ✓ TABLEAU CRÉÉ: figures_dr/comparison_table_by_dr.txt")

# ===== RÉSUMÉ DES FIGURES CRÉÉES =====
print("\n" + "=" * 90)
print("RÉSUMÉ DES FIGURES PAR DR CRÉÉES")
print("=" * 90)

for dr in test_datarates:
    real_dr = real_data[real_data['DR'] == dr]
    std_dr = df_standard[df_standard['datarate'] == dr]
    
    if len(real_dr) > 0 or len(std_dr) > 0:
        file_path = f'figures_dr/dr{dr}_comparison.png'
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path) / 1024  # Taille en KB
            print(f"  ✓ DR{dr}: {file_path} ({file_size:.1f} KB)")
            print(f"     - Données réelles: {len(real_dr)} points")
            print(f"     - Simulation Standard: {len(std_dr)} points")
            print(f"     - DDQN: {len(df_ddqn)} points")
        else:
            print(f"  ⚠ DR{dr}: Fichier non trouvé")

# ===== TABLEAU RÉCAPITULATIF PAR DR (NOUVEAU) =====
print("\n" + "=" * 90)
print("CRÉATION DU TABLEAU RÉCAPITULATIF PAR DR")
print("=" * 90)

with open('figures_dr/comparison_table_by_dr.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 120 + "\n")
    f.write("TABLEAU COMPARATIF DONNÉES RÉELLES vs DDQN PAR DR\n")
    f.write("=" * 120 + "\n\n")
    
    for dr in test_datarates:
        real_dr = real_data[real_data['DR'] == dr].sort_values('Distance(m)')
        
        if len(real_dr) == 0:
            continue
        
        f.write(f"\n{'='*60}\n")
        f.write(f"DR{dr}\n")
        f.write(f"{'='*60}\n")
        f.write(f"\n{'Distance':<12} {'PDR Réel':<15} {'PDR DDQN':<15} {'Écart PDR':<15} "
                f"{'RSSI Réel':<15} {'RSSI DDQN':<15} {'Écart RSSI':<15}\n")
        f.write(f"{'(m)':<12} {'(%)':<15} {'(%)':<15} {'(%)':<15} "
                f"{'(dBm)':<15} {'(dBm)':<15} {'(dB)':<15}\n")
        f.write("-" * 102 + "\n")
        
        total_pdr_ecart = 0
        total_rssi_ecart = 0
        count = 0
        
        for _, real_row in real_dr.iterrows():
            dist = real_row['Distance(m)']
            real_pdr = real_row['PDR'] * 100
            real_rssi = real_row['Avg_RSSI(dBm)']
            
            # Trouver la ligne DDQN correspondante
            ddqn_row = df_ddqn[df_ddqn['distance_m'] == dist]
            
            if len(ddqn_row) > 0:
                ddqn_pdr = ddqn_row['pdr'].iloc[0] * 100
                ddqn_rssi = ddqn_row['rssi_mean_dbm'].iloc[0]
                
                pdr_ecart = ddqn_pdr - real_pdr
                rssi_ecart = ddqn_rssi - real_rssi
                
                total_pdr_ecart += abs(pdr_ecart)
                total_rssi_ecart += abs(rssi_ecart)
                count += 1
                
                # Formater avec couleur dans le texte
                pdr_str = f"{pdr_ecart:>+6.1f}%"
                rssi_str = f"{rssi_ecart:>+6.1f}"
                
                # Ajouter un indicateur visuel
                pdr_indicator = "✓" if pdr_ecart > 0 else "✗" if pdr_ecart < 0 else "="
                rssi_indicator = "✓" if rssi_ecart > 0 else "✗" if rssi_ecart < 0 else "="
                
                f.write(f"{dist:<12} {real_pdr:>8.1f}%      {ddqn_pdr:>8.1f}%      "
                        f"{pdr_str:>8}   {pdr_indicator}   "
                        f"{real_rssi:>8.1f}      {ddqn_rssi:>8.1f}      "
                        f"{rssi_str:>8}   {rssi_indicator}\n")
        
        if count > 0:
            f.write("-" * 102 + "\n")
            f.write(f"Moyenne écarts absolus: PDR: {total_pdr_ecart/count:.2f}% | "
                    f"RSSI: {total_rssi_ecart/count:.2f} dB\n")
        
        f.write("-" * 102 + "\n")

print(f"\n  ✓ TABLEAU CRÉÉ: figures_dr/comparison_table_by_dr.txt")

# ===== RÉSUMÉ DES FIGURES CRÉÉES =====
print("\n" + "=" * 90)
print("RÉSUMÉ DES FIGURES PAR DR CRÉÉES")
print("=" * 90)

for dr in test_datarates:
    real_dr = real_data[real_data['DR'] == dr]
    if len(real_dr) > 0:
        file_path = f'figures_dr/dr{dr}_comparison.png'
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path) / 1024  # Taille en KB
            print(f"  ✓ DR{dr}: {file_path} ({file_size:.1f} KB)")
        else:
            print(f"  ⚠ DR{dr}: Fichier non trouvé")

# ===== VISUALISATION ORIGINALE (conservée) =====
print("\n" + "=" * 90)
print("GÉNÉRATION DES VISUALISATIONS GLOBALES (originales)")
print("=" * 90)

# 1. COMPARAISON PDR: RÉEL vs STANDARD vs DDQN
print("\n[1] Génération de l'image PDR comparative...")

fig, ax = plt.subplots(figsize=(18, 8))

# Zone d'écart-type réel (ombre douce)
ax.fill_between([], [], alpha=0)  # pas de zone d'écart-type

# Simulation Standard
ls = LINE_STYLES['standard']
ax.plot(df_comp['distance_m'].values, df_comp['pdr_standard'].values,
        color=ls['color'], marker=ls['marker'], linestyle=ls['linestyle'],
        linewidth=ls['linewidth'], markersize=ls['markersize'],
        markerfacecolor=ls['markerfacecolor'], markeredgecolor='white', markeredgewidth=1.5,
        label='Simulation Standard', zorder=3)

# DDQN
if not df_comp['pdr_ddqn'].isna().all():
    ls = LINE_STYLES['ddqn']
    ax.plot(df_comp['distance_m'].values, df_comp['pdr_ddqn'].values,
            color=ls['color'], marker=ls['marker'], linestyle=ls['linestyle'],
            linewidth=ls['linewidth'], markersize=ls['markersize'],
            markerfacecolor=ls['markerfacecolor'], markeredgecolor='white', markeredgewidth=1.5,
            label='DDQN Adaptatif', zorder=5)

style_ax(ax, xlabel='Distance (m)', ylabel='PDR (%)')
ax.set_xlim(0, 4200)
ax.set_ylim(-2, 112)
ax.legend(loc='upper right', frameon=True)

plt.tight_layout()
output_file = 'comparison_pdr_real_standard_ddqn.png'
fig.savefig(output_file)
print(f"   Sauvegardé: {output_file}")
plt.close(fig)

# 2. CHOIX DU DR PAR DDQN
print("\n[2] Génération de l'image des choix DQN...")

if not df_comp['dr8_pct'].isna().all():
    fig, ax = plt.subplots(figsize=(20, 8))

    x_offset = np.arange(len(df_comp))
    bar_width_separate = 0.20
    BAR_COLORS = [C_DR8, C_DR9, C_DR10, C_DR11]
    DR_LABELS  = ['DR8  (CR 1/3, BW 137 kHz)', 'DR9  (CR 2/3, BW 137 kHz)',
                  'DR10 (CR 1/3, BW 336 kHz)', 'DR11 (CR 2/3, BW 336 kHz)']
    DR_COLS    = ['dr8_pct', 'dr9_pct', 'dr10_pct', 'dr11_pct']
    offsets    = [-1.5, -0.5, 0.5, 1.5]

    for col, lbl, col_c, off in zip(DR_COLS, DR_LABELS, BAR_COLORS, offsets):
        bars = ax.bar(x_offset + off * bar_width_separate,
                      df_comp[col].values,
                      width=bar_width_separate,
                      label=lbl, color=col_c, alpha=0.88,
                      edgecolor='white', linewidth=0.7)
        # Valeurs en vertical DANS la barre (blanc si assez haute, couleur sinon)
        for bar in bars:
            h = bar.get_height()
            if h > 8:   # valeur à l'intérieur en blanc
                ax.text(bar.get_x() + bar.get_width() / 2., h / 2,
                        f'{h:.0f}%', ha='center', va='center',
                        fontsize=7.5, fontweight='bold', color='white', rotation=90)
            elif h > 2: # valeur au-dessus en couleur, petite
                ax.text(bar.get_x() + bar.get_width() / 2., h + 0.5,
                        f'{h:.0f}%', ha='center', va='bottom',
                        fontsize=6.5, fontweight='bold', color=col_c)

    style_ax(ax, xlabel='Distance (m)', ylabel='Proportion des choix DDQN (%)')
    ax.set_xticks(x_offset)
    ax.set_xticklabels([f'{int(d)}' for d in df_comp['distance_m'].values],
                       fontsize=10, rotation=45, ha='right')
    ax.legend(loc='upper right', fontsize=10, frameon=True)
    ymax = max(df_comp[c].max() for c in DR_COLS)
    ax.set_ylim(0, ymax + 14)
    plt.tight_layout()
    output_file = 'ddqn_dr_choices.png'
    fig.savefig(output_file)
    print(f"   Sauvegardé: {output_file}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# DÉBIT RÉEL — basé sur la taille des paquets LR-FHSS et le ToA par DR
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 100)
print("DÉBIT RÉEL — basé sur la taille des paquets et le ToA dépendant du DR")
print("=" * 100)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES DE TRAME LR-FHSS
# Structure : | OCW (2o) | Header saut (3o) | Payload (No) | MIC (4o) |
# ─────────────────────────────────────────────────────────────────────────────
LRFHSS_OCW_BYTES     = 2   # Owner Channel Word
LRFHSS_HEADER_BYTES  = 3   # Header de saut (par fragment)
LRFHSS_MIC_BYTES     = 4   # Message Integrity Code (CRC)
LRFHSS_OVERHEAD_BYTES = LRFHSS_OCW_BYTES + LRFHSS_HEADER_BYTES + LRFHSS_MIC_BYTES  # 9 octets

PAYLOAD_BYTES_REAL    = BASE_CONFIG.get('payload_min', 1)  # payload applicatif (1 octet)
FRAME_TOTAL_BYTES     = PAYLOAD_BYTES_REAL + LRFHSS_OVERHEAD_BYTES  # 10 octets
FRAME_TOTAL_BITS      = FRAME_TOTAL_BYTES * 8                        # 80 bits

SIM_DURATION_S        = BASE_CONFIG['simulation_duration']           # 1800 s

print(f"\n  Structure trame LR-FHSS :")
print(f"    OCW       : {LRFHSS_OCW_BYTES} octets")
print(f"    Header hop: {LRFHSS_HEADER_BYTES} octets")
print(f"    MIC (CRC) : {LRFHSS_MIC_BYTES} octets")
print(f"    Payload   : {PAYLOAD_BYTES_REAL} octet(s) applicatif(s)")
print(f"    ─────────────────────────────")
print(f"    Trame totale : {FRAME_TOTAL_BYTES} octets = {FRAME_TOTAL_BITS} bits")
print(f"    Durée simulation : {SIM_DURATION_S} s")


# ─────────────────────────────────────────────────────────────────────────────
# FONCTIONS DE CALCUL
# ─────────────────────────────────────────────────────────────────────────────

def calc_raw_channel_bps(dr: int) -> float:
    """
    Débit brut canal [bps] = taille_trame [bits] / ToA [s]
    Mesure la vitesse instantanée des bits sur l'air (utiles + overhead).
    Constant pour un DR donné.
    """
    toa_s = calculate_toa_ms(dr, PAYLOAD_BYTES_REAL) / 1000.0
    return FRAME_TOTAL_BITS / toa_s if toa_s > 0 else 0.0


def calc_useful_bps(n_rx: int) -> float:
    """
    Débit utile effectif [bps] = paquets_reçus × payload [bits] / durée_sim [s]
    Ce que l'application reçoit réellement.
    """
    return (n_rx * PAYLOAD_BYTES_REAL * 8) / SIM_DURATION_S if SIM_DURATION_S > 0 else 0.0


def calc_spectral_eff(useful_bps: float, raw_bps: float) -> float:
    """Efficacité spectrale [%] = débit_utile / débit_brut_canal × 100"""
    return useful_bps / raw_bps * 100.0 if raw_bps > 0 else 0.0


def calc_channel_occ(n_sent: int, dr: int) -> float:
    """Taux d'occupation canal [%] = ToA_total / durée_sim × 100"""
    toa_s = calculate_toa_ms(dr, PAYLOAD_BYTES_REAL) / 1000.0
    return min(n_sent * toa_s / SIM_DURATION_S * 100.0, 100.0) if SIM_DURATION_S > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# CALCUL DES MÉTRIQUES PAR DR
# ─────────────────────────────────────────────────────────────────────────────
print("\n  ToA et débit brut canal par DR :")
print(f"  {'DR':<6} {'ToA (ms)':<12} {'Débit brut canal (bps)':<25} {'CR':<8} {'BW (kHz)'}")
print("  " + "-" * 65)

toa_by_dr     = {}
raw_bps_by_dr = {}
for dr in test_datarates:
    toa_ms  = calculate_toa_ms(dr, PAYLOAD_BYTES_REAL)
    raw_bps = calc_raw_channel_bps(dr)
    toa_by_dr[dr]     = toa_ms
    raw_bps_by_dr[dr] = raw_bps
    cr = DR_CONFIG_MAP[dr]['cr']
    bw = DR_CONFIG_MAP[dr]['bw_khz']
    print(f"  DR{dr:<4} {toa_ms:<12.2f} {raw_bps:<25.2f} {cr:<8} {bw:.3f}")

print(f"\n  Formule : débit_brut_canal = {FRAME_TOTAL_BITS} bits / ToA[s]")


# ─────────────────────────────────────────────────────────────────────────────
# COLLECTE DES DONNÉES POUR LES GRAPHIQUES
# ─────────────────────────────────────────────────────────────────────────────
plot_data_std  = []
plot_data_ddqn = []

print("\n  Métriques de débit réel par configuration :")
print("  " + "-" * 110)
print(f"  {'Dist(m)':<10} {'DR':<8} {'N_sent':<8} {'N_rx':<8} {'PDR%':<7} "
      f"{'ToA(ms)':<10} {'Brut canal(bps)':<18} {'Utile eff.(bps)':<18} {'Eff.spec%':<11} {'Occ.can%'}")
print("  " + "-" * 110)

for dist in test_distances:
    # ── Standard (DR fixe) ──────────────────────────────────────────────────
    for dr in test_datarates:
        row = df_standard[(df_standard['distance_m'] == dist) &
                          (df_standard['datarate']   == dr)]
        if len(row) == 0:
            continue
        r = row.iloc[0]
        n_sent  = int(r['num_packets'])
        n_rx    = int(r['successful_packets'])
        pdr_pct = r['pdr'] * 100

        toa_ms  = toa_by_dr[dr]
        raw_bps = raw_bps_by_dr[dr]
        eff_bps = calc_useful_bps(n_rx)
        spec    = calc_spectral_eff(eff_bps, raw_bps)
        occ     = calc_channel_occ(n_sent, dr)

        print(f"  {int(dist):<10} DR{dr:<6} {n_sent:<8} {n_rx:<8} {pdr_pct:<7.1f} "
              f"{toa_ms:<10.2f} {raw_bps:<18.2f} {eff_bps:<18.4f} {spec:<11.2f} {occ:.2f}")

        plot_data_std.append({
            'dist': dist, 'dr': dr,
            'toa_ms': toa_ms, 'raw_bps': raw_bps,
            'eff_bps': eff_bps, 'spec_eff_pct': spec,
            'canal_occ_pct': occ, 'pdr_pct': pdr_pct,
            'n_sent': n_sent, 'n_rx': n_rx,
        })

    # ── DDQN (DR adaptatif) ─────────────────────────────────────────────────
    row = df_ddqn[df_ddqn['distance_m'] == dist]
    if len(row) == 0:
        continue
    r = row.iloc[0]
    n_sent  = int(r['num_packets'])
    n_rx    = int(r['successful_packets'])
    pdr_pct = r['pdr'] * 100

    dr_counts = {8: r['dr8_count'], 9: r['dr9_count'],
                 10: r['dr10_count'], 11: r['dr11_count']}
    total_dec = sum(dr_counts.values())

    if total_dec > 0:
        # Moyennes pondérées par la distribution réelle des DR choisis
        toa_ms_avg  = sum(toa_by_dr[dr] * cnt     for dr, cnt in dr_counts.items()) / total_dec
        raw_bps_avg = sum(raw_bps_by_dr[dr] * cnt for dr, cnt in dr_counts.items()) / total_dec
        # Occupation canal : chaque paquet a son propre ToA selon le DR choisi
        total_air_s = sum((cnt / total_dec) * n_sent * (toa_by_dr[dr] / 1000.0)
                          for dr, cnt in dr_counts.items())
    else:
        toa_ms_avg  = float(np.mean(list(toa_by_dr.values())))
        raw_bps_avg = float(np.mean(list(raw_bps_by_dr.values())))
        total_air_s = n_sent * (toa_ms_avg / 1000.0)

    eff_bps = calc_useful_bps(n_rx)
    spec    = calc_spectral_eff(eff_bps, raw_bps_avg)
    occ     = min(total_air_s / SIM_DURATION_S * 100.0, 100.0)

    print(f"  {int(dist):<10} {'DDQN':<8} {n_sent:<8} {n_rx:<8} {pdr_pct:<7.1f} "
          f"{toa_ms_avg:<10.2f} {raw_bps_avg:<18.2f} {eff_bps:<18.4f} {spec:<11.2f} {occ:.2f}")
    print("  " + "-" * 110)

    plot_data_ddqn.append({
        'dist': dist,
        'toa_ms_avg': toa_ms_avg, 'raw_bps_avg': raw_bps_avg,
        'eff_bps': eff_bps, 'spec_eff_pct': spec,
        'canal_occ_pct': occ, 'pdr_pct': pdr_pct,
        'n_sent': n_sent, 'n_rx': n_rx,
        'dr_counts': dr_counts, 'total_dec': total_dec,
    })

df_plot_std  = pd.DataFrame(plot_data_std)
df_plot_ddqn = pd.DataFrame(plot_data_ddqn)

DR_COLORS = {8: C_DR8, 9: C_DR9, 10: C_DR10, 11: C_DR11}
DR_LABELS  = {
    8:  'DR8  (CR 1/3, 137 kHz)',
    9:  'DR9  (CR 2/3, 137 kHz)',
    10: 'DR10 (CR 1/3, 336 kHz)',
    11: 'DR11 (CR 2/3, 336 kHz)',
}

os.makedirs('figures_dr', exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE A : ToA et débit brut canal par DR (vue synthétique)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[A] Débit réel — ToA et débit brut canal par DR...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle(
    f"Time-on-Air et Débit Brut Canal par DR LR-FHSS\n"
    f"Trame = {PAYLOAD_BYTES_REAL}o payload + {LRFHSS_OVERHEAD_BYTES}o overhead = {FRAME_TOTAL_BYTES}o = {FRAME_TOTAL_BITS} bits",
    fontsize=13, fontweight='bold'
)

dr_list  = list(test_datarates)
toa_vals = [toa_by_dr[dr]     for dr in dr_list]
raw_vals = [raw_bps_by_dr[dr] for dr in dr_list]
colors   = [DR_COLORS[dr]     for dr in dr_list]

# ToA
style_ax(ax1, xlabel='Data Rate', ylabel='ToA (ms)')
bars = ax1.bar([f'DR{dr}' for dr in dr_list], toa_vals,
               color=colors, alpha=0.85, edgecolor='white', linewidth=0.7)
for bar, v, dr in zip(bars, toa_vals, dr_list):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'{v:.2f} ms\n(CR {DR_CONFIG_MAP[dr]["cr"]})',
             ha='center', va='bottom', fontsize=10.5, fontweight='bold')
ax1.set_title("Time-on-Air par DR", fontsize=12, fontweight='bold')
ax1.set_ylim(0, max(toa_vals) * 1.25)

# Débit brut canal
style_ax(ax2, xlabel='Data Rate', ylabel='Débit brut canal (bps)')
bars = ax2.bar([f'DR{dr}' for dr in dr_list], raw_vals,
               color=colors, alpha=0.85, edgecolor='white', linewidth=0.7)
for bar, v, dr in zip(bars, raw_vals, dr_list):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
             f'{v:.2f} bps\n({FRAME_TOTAL_BITS}b / {toa_by_dr[dr]:.2f}ms)',
             ha='center', va='bottom', fontsize=10.5, fontweight='bold')
ax2.set_title(f"Débit Brut Canal = {FRAME_TOTAL_BITS} bits / ToA", fontsize=12, fontweight='bold')
ax2.set_ylim(0, max(raw_vals) * 1.25)

plt.tight_layout()
out = 'figures_dr/debit_reel_A_toa_et_brut_par_dr.png'
fig.savefig(out, dpi=160, bbox_inches='tight')
print(f"   ✓ Sauvegardé: {out}")
plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE C : Débit utile effectif — toutes courbes (DR fixe + DDQN) vs distance
# ═══════════════════════════════════════════════════════════════════════════════
print("[C] Débit réel — débit utile effectif toutes courbes vs distance...")

fig, ax = plt.subplots(figsize=(20, 8))
style_ax(ax, xlabel='Distance (m)', ylabel='Débit utile effectif (bps)')

for dr in test_datarates:
    sub = df_plot_std[df_plot_std['dr'] == dr].sort_values('dist')
    if len(sub) == 0:
        continue
    ax.plot(sub['dist'].values, sub['eff_bps'].values,
            color=DR_COLORS[dr], marker='o', linestyle='--', linewidth=2.0,
            markersize=7, markerfacecolor=DR_COLORS[dr],
            markeredgecolor='white', markeredgewidth=1.2,
            label=DR_LABELS[dr], zorder=3)
    for x, y in zip(sub['dist'].values, sub['eff_bps'].values):
        add_annotation(ax, x, y, f'{y:.4f}', offset=(0, 9),
                       color=DR_COLORS[dr], bg='white', ec='#D1D5DB', fs=6.5)

if len(df_plot_ddqn) > 0:
    ax.plot(df_plot_ddqn['dist'].values, df_plot_ddqn['eff_bps'].values,
            color=C_DDQN, marker='D', linestyle='-', linewidth=2.8,
            markersize=9, markerfacecolor=C_DDQN,
            markeredgecolor='white', markeredgewidth=1.5,
            label='DDQN Adaptatif', zorder=6)
    for x, y in zip(df_plot_ddqn['dist'].values, df_plot_ddqn['eff_bps'].values):
        add_annotation(ax, x, y, f'{y:.4f}', offset=(22, 0),
                       color=C_DDQN, bg='#FEF2F2', ec='#FECACA', fs=7)

ax.set_title(
    f"Débit Utile Effectif = paquets reçus × {PAYLOAD_BYTES_REAL*8} bits / {SIM_DURATION_S}s\n"
    "(ce que l'application reçoit réellement)",
    fontsize=13, fontweight='bold'
)
ax.set_xlim(0, max(test_distances) * 1.08)
ax.legend(loc='lower left', fontsize=10, frameon=True)


plt.tight_layout()
out = 'figures_dr/debit_reel_C_utile_effectif_vs_distance.png'
fig.savefig(out, dpi=160, bbox_inches='tight')
print(f"   ✓ Sauvegardé: {out}")
plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE F : Débit réel de transmission — barres groupées par distance
#
#  Standard DR fixe : débit = FRAME_BITS / ToA(DR)  → constant par DR
#  DDQN adaptatif   : débit moyen réel calculé paquet par paquet depuis
#                     packets_timeline[(dist,'ddqn')]
#                     Pour chaque paquet : débit_i = FRAME_BITS / ToA(dr_i)
#                     Débit DDQN(dist) = moyenne de tous les débit_i
# ═══════════════════════════════════════════════════════════════════════════════
print("[F] Débit réel de transmission — barres groupées, DDQN calculé paquet par paquet...")

# ── Calcul du débit DDQN paquet par paquet ───────────────────────────────────
ddqn_real_bps = {}   # dist -> débit réel moyen DDQN [bps]
ddqn_detail   = {}   # dist -> {'mean', 'std', 'n', 'by_dr': {dr: (count, bps)}}

for dist in test_distances:
    tl = packets_timeline.get((dist, 'ddqn'), [])
    if not tl:
        # Fallback : utiliser la distribution dr_counts depuis df_ddqn
        row = df_ddqn[df_ddqn['distance_m'] == dist]
        if len(row) == 0:
            ddqn_real_bps[dist] = 0
            continue
        r = row.iloc[0]
        dr_counts = {8: int(r['dr8_count']), 9: int(r['dr9_count']),
                     10: int(r['dr10_count']), 11: int(r['dr11_count'])}
        total = sum(dr_counts.values())
        if total == 0:
            ddqn_real_bps[dist] = 0
            continue
        # Moyenne pondérée par les counts
        bps_per_pkt = [raw_bps_by_dr[dr]
                       for dr, cnt in dr_counts.items()
                       for _ in range(cnt)]
        ddqn_real_bps[dist] = float(np.mean(bps_per_pkt))
        ddqn_detail[dist] = {
            'mean': ddqn_real_bps[dist],
            'std':  float(np.std(bps_per_pkt)),
            'n':    total,
            'by_dr': {dr: (cnt, raw_bps_by_dr[dr])
                       for dr, cnt in dr_counts.items()},
            'source': 'dr_counts'
        }
    else:
        # Calcul exact paquet par paquet depuis la timeline
        # Chaque tuple = (start_time, success, dr)
        bps_per_pkt = [raw_bps_by_dr[int(pkt[2])] for pkt in tl
                       if int(pkt[2]) in raw_bps_by_dr]
        if not bps_per_pkt:
            ddqn_real_bps[dist] = 0
            continue
        by_dr = {}
        for pkt in tl:
            dr_pkt = int(pkt[2])
            if dr_pkt not in by_dr:
                by_dr[dr_pkt] = [0, raw_bps_by_dr.get(dr_pkt, 0)]
            by_dr[dr_pkt][0] += 1
        ddqn_real_bps[dist] = float(np.mean(bps_per_pkt))
        ddqn_detail[dist] = {
            'mean': ddqn_real_bps[dist],
            'std':  float(np.std(bps_per_pkt)),
            'n':    len(bps_per_pkt),
            'by_dr': {dr: (cnt, raw_bps_by_dr.get(dr, 0))
                       for dr, (cnt, _) in by_dr.items()},
            'source': 'paquet_par_paquet'
        }

# ── Affichage tableau détaillé ────────────────────────────────────────────────
# ── Affichage tableau détaillé ────────────────────────────────────────────────
print(f"\n  Débit DDQN réel calculé paquet par paquet :")
print("  " + "-" * 90)
print(f"  {'Dist(m)':<10} {'N paquets':<12} {'Débit moy(bps)':<18} "
      f"{'±std':<12} {'Distribution DR':<30} {'Source'}")
print("  " + "-" * 90)
for dist in test_distances:
    d = ddqn_detail.get(dist)
    if not d:
        print(f"  {int(dist):<10} {'N/A'}")
        continue
    # Simplifier DR : DR8/10 et DR9/11
    dr_simpl = {}
    for dr, (cnt, _) in d['by_dr'].items():
        if dr in [8,10]:
            dr_simpl[8] = dr_simpl.get(8,(0,0))
            dr_simpl[8] = (dr_simpl[8][0]+cnt, 0)
        elif dr in [9,11]:
            dr_simpl[9] = dr_simpl.get(9,(0,0))
            dr_simpl[9] = (dr_simpl[9][0]+cnt, 0)
    dr_str = "  ".join([f"DR{dr}:{cnt}({cnt/d['n']*100:.0f}%)"
                         for dr, (cnt, _) in sorted(dr_simpl.items())])
    print(f"  {int(dist):<10} {d['n']:<12} {d['mean']:<18.4f} "
          f"{d['std']:<12.4f} {dr_str:<30} [{d['source']}]")
print("  " + "-" * 90)


# ── Tracé barres groupées simplifié ───────────────────────────────────────────
simplified_drs = [8,9]  # DR8/10 et DR9/11
n_dr = len(simplified_drs)
n_series = n_dr + 1  # DR fixes + DDQN

bw_g = 0.13
spacing_g = 0.50
x_centers = np.arange(len(test_distances)) * (n_series * bw_g + spacing_g + 0.2)
x_by_dr = [x_centers + (i - n_series/2 + 0.5) * bw_g * 1.25 for i in range(n_dr)]
x_ddqn_g = x_centers + (n_dr - n_series/2 + 0.5) * bw_g * 1.25

fig, ax = plt.subplots(figsize=(20, 8))
style_ax(ax, xlabel='Distance (m)', ylabel='Débit de transmission (bps)')

# Barres DR fixes simplifiées
for idx, dr in enumerate(simplified_drs):
    vals = [raw_bps_by_dr[dr] for dist in test_distances]
    bars = ax.bar(x_by_dr[idx], vals, width=bw_g,
                  label=f'DR{dr}/{dr+2}', color=DR_COLORS[dr],
                  alpha=0.82, edgecolor='white', linewidth=0.6)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v * 1.005,
                f'{v:.2f}', ha='center', va='bottom',
                fontsize=6.5, fontweight='bold',
                color=DR_COLORS[dr], rotation=90)

# Barres DDQN
ddqn_vals_f = [ddqn_real_bps.get(dist, 0) for dist in test_distances]
bars_dqn = ax.bar(x_ddqn_g, ddqn_vals_f, width=bw_g,
                  label='DDQN', color=C_DDQN,
                  alpha=0.88, edgecolor='white', linewidth=0.6, zorder=4)

# Texte au-dessus des barres DDQN
for bar, dist, v in zip(bars_dqn, test_distances, ddqn_vals_f):
    if v > 0:
        ax.text(bar.get_x() + bar.get_width()/2, v * 1.005,
                f'{v:.2f}', ha='center', va='bottom',
                fontsize=6.5, fontweight='bold', color=C_DDQN, rotation=90)

# Lignes horizontales de référence
for dr in simplified_drs:
    ax.axhline(raw_bps_by_dr[dr], color=DR_COLORS[dr],
               linestyle=':', linewidth=0.9, alpha=0.35)

# Titres et axes
ax.set_xticks(x_centers)
ax.set_xticklabels([f'{int(d)} m' for d in test_distances],
                   fontsize=11, rotation=30, ha='right')
ax.set_title(
    f"Débit Réel de Transmission — Standard (DR8/10, DR9/11) vs DDQN",
    fontsize=12, fontweight='bold'
)
ax.set_ylim(bottom=80)  # si tu veux commencer l'axe Y à 80

# Légende
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, fontsize=9, frameon=True)

plt.tight_layout()
out = 'figures_dr/debit_reel_brut.png'
fig.savefig(out, dpi=300, bbox_inches='tight')
print(f"   ✓ Sauvegardé: {out}")
plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# RÉSUMÉ STATISTIQUE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("RÉSUMÉ STATISTIQUE — DÉBIT RÉEL")
print("=" * 80)

print(f"\n  Débit brut canal (fixe par DR) :")
for dr in test_datarates:
    print(f"    DR{dr}: {raw_bps_by_dr[dr]:.2f} bps  (ToA={toa_by_dr[dr]:.2f}ms  CR={DR_CONFIG_MAP[dr]['cr']})")

if len(df_plot_std) > 0:
    print(f"\n  Débit utile effectif — Standard (min / moy / max) :")
    for dr in test_datarates:
        sub = df_plot_std[df_plot_std['dr'] == dr]['eff_bps']
        if len(sub) > 0:
            print(f"    DR{dr}: {sub.min():.4f} / {sub.mean():.4f} / {sub.max():.4f} bps")

if len(df_plot_ddqn) > 0:
    vals = df_plot_ddqn['eff_bps']
    print(f"\n  Débit utile effectif — DDQN : {vals.min():.4f} / {vals.mean():.4f} / {vals.max():.4f} bps")

    mean_std  = df_plot_std['eff_bps'].mean() if len(df_plot_std) > 0 else 0
    mean_dqn  = df_plot_ddqn['eff_bps'].mean()
    gain_eff  = (mean_dqn - mean_std) / mean_std * 100 if mean_std > 0 else 0
    print(f"\n  Gain débit utile DDQN vs Standard : {gain_eff:+.1f}%")

print(f"\n  Figures générées (préfixe 'debit_reel_') dans figures_dr/ :")
for lbl in ['A', 'C', 'F']:
    print(f"    [{lbl}] debit_reel_{lbl}_*.png")




# 4. COMPARAISON ÉNERGÉTIQUE
print("\n[4] Génération de l'image de comparaison énergétique...")

if not df_comp['energy_standard_avg_mj'].isna().all() and not df_comp['energy_ddqn_mj'].isna().all():
    fig, ax = plt.subplots(figsize=(20, 8))

    x_pos = np.arange(len(df_comp))
    bar_width = 0.36

    bars1 = ax.bar(x_pos - bar_width/2, df_comp['energy_standard_avg_mj'].values,
                   width=bar_width, label='Standard (moyenne)', color=C_STD,
                   alpha=0.85, edgecolor='white', linewidth=0.7)
    bars2 = ax.bar(x_pos + bar_width/2, df_comp['energy_ddqn_mj'].values,
                   width=bar_width, label='DDQN Adaptatif', color=C_DDQN,
                   alpha=0.85, edgecolor='white', linewidth=0.7)

    ymax_val = max(df_comp['energy_standard_avg_mj'].max(), df_comp['energy_ddqn_mj'].max())
    thresh = ymax_val * 0.12  # seuil minimal pour écrire dans la barre

    for bars, col in [(bars1, C_STD), (bars2, C_DDQN)]:
        for bar in bars:
            h = bar.get_height()
            if h > thresh:  # valeur à l'intérieur en blanc vertical
                ax.text(bar.get_x() + bar.get_width()/2., h / 2,
                        f'{h:.3f}', ha='center', va='center',
                        fontsize=7.5, fontweight='bold', color='white', rotation=90)
            elif h > 0:     # valeur au-dessus en couleur
                ax.text(bar.get_x() + bar.get_width()/2., h + ymax_val * 0.01,
                        f'{h:.3f}', ha='center', va='bottom',
                        fontsize=7, fontweight='bold', color=col)

    # Indicateur d'économie UNE seule ligne au-dessus du groupe
    for i, (e_std, e_ddqn) in enumerate(zip(df_comp['energy_standard_avg_mj'].values,
                                             df_comp['energy_ddqn_mj'].values)):
        if e_std > 0 and e_ddqn > 0:
            saving = (e_std - e_ddqn) / e_std * 100
            color  = '#16A34A' if saving > 0 else '#DC2626'
            symbol = '▼' if saving > 0 else '▲'
            ax.text(i, max(e_std, e_ddqn) + ymax_val * 0.04,
                    f'{symbol}{abs(saving):.0f}%', ha='center',
                    fontsize=8, fontweight='bold', color=color)

    style_ax(ax, xlabel='Distance (m)', ylabel='Énergie par paquet (mJ)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{int(d)}' for d in df_comp['distance_m'].values],
                       fontsize=10, rotation=45, ha='right')
    ax.legend(loc='upper left', fontsize=11, frameon=True)
    ax.set_ylim(0, ymax_val * 1.22)
    ax.grid(True, axis='y', alpha=0.22, linestyle='--')
    plt.tight_layout()
    output_file = 'energy_comparison_standard_vs_ddqn.png'
    fig.savefig(output_file)
    print(f"   Sauvegardé: {output_file}")
    plt.close(fig)

# 5. TABLEAU DE PUISSANCE PAR DISTANCE
print("\n[5] Génération de l'image de puissance par distance...")

if not df_comp['ddqn_avg_power'].isna().all():
    fig, ax = plt.subplots(figsize=(20, 7))

    x_pos = np.arange(len(df_comp))
    bar_width = 0.36

    std_subset_list = []
    for dist in df_comp['distance_m'].values:
        std_data = df_standard[df_standard['distance_m'] == dist]
        std_subset_list.append(std_data['tx_power'].mean() if len(std_data) > 0 else 0)

    bars1 = ax.bar(x_pos - bar_width/2, std_subset_list,
                   width=bar_width, label='Standard', color=C_STD,
                   alpha=0.85, edgecolor='white', linewidth=0.7)
    bars2 = ax.bar(x_pos + bar_width/2, df_comp['ddqn_avg_power'].values,
                   width=bar_width, label='DDQN Adaptatif', color=C_DDQN,
                   alpha=0.85, edgecolor='white', linewidth=0.7)

    for bars, col in [(bars1, C_STD), (bars2, C_DDQN)]:
        for bar in bars:
            h = bar.get_height()
            if h >= 2:   # valeur à l'intérieur en blanc vertical
                ax.text(bar.get_x() + bar.get_width()/2., h / 2,
                        f'{h:.1f}', ha='center', va='center',
                        fontsize=8, fontweight='bold', color='white', rotation=90)
            elif h > 0:
                ax.text(bar.get_x() + bar.get_width()/2., h + 0.1,
                        f'{h:.1f}', ha='center', va='bottom',
                        fontsize=7.5, fontweight='bold', color=col)

    style_ax(ax, xlabel='Distance (m)', ylabel='Puissance TX (dBm)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{int(d)}' for d in df_comp['distance_m'].values],
                       fontsize=10, rotation=45, ha='right')
    ax.legend(loc='upper right', fontsize=11, frameon=True)
    ax.set_ylim(0, 17)
    ax.grid(True, axis='y', alpha=0.22, linestyle='--')
    plt.tight_layout()
    output_file = 'power_comparison_by_distance.png'
    fig.savefig(output_file)
    print(f"   Sauvegardé: {output_file}")
    plt.close(fig)

# 6. TABLEAU DE PDR MOYEN PAR DISTANCE
print("\n[6] Génération de l'image PDR moyen par distance...")

fig, ax = plt.subplots(figsize=(20, 8))

x_pos = np.arange(len(df_comp))
bar_width = 0.36

bars1 = ax.bar(x_pos - bar_width/2, df_comp['pdr_standard'].values,
               width=bar_width, label='Standard', color=C_STD,
               alpha=0.85, edgecolor='white', linewidth=0.7)
bars2 = ax.bar(x_pos + bar_width/2, df_comp['pdr_ddqn'].values,
               width=bar_width, label='DDQN Adaptatif', color=C_DDQN,
               alpha=0.85, edgecolor='white', linewidth=0.7)

for bars, col in [(bars1, C_STD), (bars2, C_DDQN)]:
    for bar in bars:
        h = bar.get_height()
        if h > 15:  # valeur à l'intérieur en blanc vertical
            ax.text(bar.get_x() + bar.get_width()/2., h / 2,
                    f'{h:.1f}%', ha='center', va='center',
                    fontsize=8, fontweight='bold', color='white', rotation=90)
        elif h > 0:
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.8,
                    f'{h:.1f}%', ha='center', va='bottom',
                    fontsize=7.5, fontweight='bold', color=col)

# Indicateurs d'amélioration (compacts, sans espace)
for i, (std_pdr, ddqn_pdr) in enumerate(zip(df_comp['pdr_standard'].values, df_comp['pdr_ddqn'].values)):
    if std_pdr > 0:
        improvement = (ddqn_pdr - std_pdr) / std_pdr * 100
        color  = '#16A34A' if improvement > 0 else '#DC2626' if improvement < 0 else '#6B7280'
        symbol = '▲' if improvement > 0 else '▼' if improvement < 0 else '▶'
        ax.text(i, max(std_pdr, ddqn_pdr) + 5,
                f'{symbol}{improvement:+.0f}%', ha='center',
                fontsize=8, fontweight='bold', color=color)

style_ax(ax, xlabel='Distance (m)', ylabel='PDR (%)')
ax.set_xticks(x_pos)
ax.set_xticklabels([f'{int(d)}' for d in df_comp['distance_m'].values],
                   fontsize=10, rotation=45, ha='right')
ax.legend(loc='upper right', fontsize=11, frameon=True)
ax.set_ylim(0, 130)
ax.grid(True, axis='y', alpha=0.22, linestyle='--')

plt.tight_layout()
output_file = 'pdr_by_distance.png'
fig.savefig(output_file)
print(f"   Sauvegardé: {output_file}")
plt.close(fig)

# 7. COMPARAISON PDR PAR DR
print("\n[7] Génération de l'image PDR par DR avec DDQN...")

fig, axes = plt.subplots(2, 2, figsize=(20, 12))
axes = axes.flatten()

for idx, dr in enumerate(test_datarates):
    ax = axes[idx]
    std_dr  = df_standard[(df_standard['datarate'] == dr)].sort_values('distance_m')

    if len(std_dr) > 0:
        ls = LINE_STYLES['standard']
        ax.plot(std_dr['distance_m'].values, std_dr['pdr'].values * 100,
                color=ls['color'], marker=ls['marker'], linestyle=ls['linestyle'],
                linewidth=2.0, markersize=6, markerfacecolor=ls['markerfacecolor'],
                markeredgecolor='white', markeredgewidth=1.3,
                label='Standard', zorder=3)

    if not df_comp['pdr_ddqn'].isna().all():
        ls = LINE_STYLES['ddqn']
        ax.plot(df_comp['distance_m'].values, df_comp['pdr_ddqn'].values,
                color=ls['color'], marker=ls['marker'], linestyle=ls['linestyle'],
                linewidth=2.0, markersize=6, markerfacecolor=ls['markerfacecolor'],
                markeredgecolor='white', markeredgewidth=1.3,
                label='DDQN', zorder=5)

    style_ax(ax, xlabel='Distance (m)', ylabel='PDR (%)')
    ax.set_xlim(0, 4200)
    ax.set_ylim(-2, 112)
    ax.legend(loc='upper right', fontsize=9.5, frameon=True)

plt.tight_layout()
output_file = 'comparison_by_dr_with_ddqn.png'
fig.savefig(output_file)
print(f"   Sauvegardé: {output_file}")
plt.close(fig)

# ===== [8] GOODPUT — BITS UTILES REÇUS / DURÉE TOTALE =====
# Goodput = payload reçu avec succès sur le canal
# Formule : goodput [bps] = N_rx_succès × payload_bits / durée_sim
#
# Standard DR fixe  : calculé par DR (8,9,10,11) pour chaque distance
# DDQN adaptatif    : calculé directement depuis successful_packets
#
# Différence vs throughput effectif du code existant :
#   - effective_throughput_bps utilise payload=1 byte (8 bits) → identique ici
#   - Le goodput est tracé ici proprement par DR sur toutes les distances
#     avec le DDQN en overlay, et un panneau de gain DDQN vs meilleur DR
# ══════════════════════════════════════════════════════════════════════════════
print("\n[8] Génération du graphique Goodput...")

PAYLOAD_BITS   = 8          # 1 octet de payload applicatif = 8 bits
SIM_DUR_S      = BASE_CONFIG['simulation_duration']   # 1800 s

# ── Couleurs et styles ────────────────────────────────────────────────────────
GP_COLORS = {8: C_DR8, 9: C_DR9, 10: C_DR10, 11: C_DR11}
GP_MARKERS = {8: 'o', 9: 's', 10: '^', 11: 'D'}
GP_LINES   = {8: '--', 9: '--', 10: '--', 11: '--'}
GP_LABELS  = {
    8:  'DR8  (CR 1/3, 137 kHz)',
    9:  'DR9  (CR 2/3, 137 kHz)',
    10: 'DR10 (CR 1/3, 336 kHz)',
    11: 'DR11 (CR 2/3, 336 kHz)',
}

dists_sorted = sorted(test_distances)

# ── Calcul goodput par (distance, DR) ────────────────────────────────────────
goodput_std  = {}   # (dist, dr) -> bps
goodput_ddqn = {}   # dist       -> bps

for dist in dists_sorted:
    # Standard
    for dr in test_datarates:
        row = df_standard[(df_standard['distance_m'] == dist) &
                          (df_standard['datarate']   == dr)]
        if len(row) == 0:
            goodput_std[(dist, dr)] = np.nan
            continue
        n_rx = int(row.iloc[0]['successful_packets'])
        goodput_std[(dist, dr)] = (n_rx * PAYLOAD_BITS) / SIM_DUR_S

    # DDQN
    row = df_ddqn[df_ddqn['distance_m'] == dist]
    if len(row) == 0:
        goodput_ddqn[dist] = np.nan
        continue
    n_rx = int(row.iloc[0]['successful_packets'])
    goodput_ddqn[dist] = (n_rx * PAYLOAD_BITS) / SIM_DUR_S

# ── Affichage console ─────────────────────────────────────────────────────────
print(f"\n  Goodput = N_rx_succès × {PAYLOAD_BITS} bits / {SIM_DUR_S}s")
print("  " + "-" * 90)
print(f"  {'Dist(m)':<10} {'DR8':<12} {'DR9':<12} {'DR10':<12} {'DR11':<12} {'DDQN':<12}")
print("  " + "-" * 90)
for dist in dists_sorted:
    vals = [f"{goodput_std.get((dist,dr), float('nan')):.4f}" for dr in test_datarates]
    ddqn_val = f"{goodput_ddqn.get(dist, float('nan')):.4f}"
    print(f"  {int(dist):<10} " + " ".join(f"{v:<12}" for v in vals) + f" {ddqn_val:<12}")
print("  " + "-" * 90)

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 8a : Goodput — courbes DR fixe + DDQN vs distance
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(20, 8))
style_ax(ax, xlabel='Distance (m)', ylabel='Goodput (bps)')
ax.set_title(
    f"Goodput  =  paquets reçus × {PAYLOAD_BITS} bits / {SIM_DUR_S}s\n"
    f"(bits de payload applicatif reçus avec succès par seconde)",
    fontsize=13, fontweight='bold'
)

# Courbes Standard par DR
for dr in test_datarates:
    vals = [goodput_std.get((d, dr), np.nan) for d in dists_sorted]
    ax.plot(dists_sorted, vals,
            color=GP_COLORS[dr], marker=GP_MARKERS[dr],
            linestyle=GP_LINES[dr], linewidth=2.0,
            markersize=7, markerfacecolor=GP_COLORS[dr],
            markeredgecolor='white', markeredgewidth=1.2,
            label=GP_LABELS[dr], zorder=3)
    for x, y in zip(dists_sorted, vals):
        if not np.isnan(y):
            add_annotation(ax, x, y, f'{y:.4f}', offset=(0, 9),
                           color=GP_COLORS[dr], bg='white', ec='#D1D5DB', fs=6.5)

# Courbe DDQN
ddqn_vals = [goodput_ddqn.get(d, np.nan) for d in dists_sorted]
ax.plot(dists_sorted, ddqn_vals,
        color=C_DDQN, marker='D', linestyle='-', linewidth=2.8,
        markersize=9, markerfacecolor=C_DDQN,
        markeredgecolor='white', markeredgewidth=1.5,
        label='DDQN Adaptatif', zorder=6)
for x, y in zip(dists_sorted, ddqn_vals):
    if not np.isnan(y):
        add_annotation(ax, x, y, f'{y:.4f}', offset=(0, -18),
                       color=C_DDQN, bg='#FEF2F2', ec='#FECACA', fs=7)

ax.set_xlim(0, max(dists_sorted) * 1.08)
all_gp = [v for v in list(goodput_std.values()) + list(goodput_ddqn.values())
          if not np.isnan(v) and v > 0]
if all_gp:
    ax.set_ylim(min(all_gp) * 0.88, max(all_gp) * 1.15)
ax.legend(loc='upper right', fontsize=10, frameon=True)
ax.text(0.01, 0.97,
        f"Goodput = N_rx × {PAYLOAD_BITS} bits / {SIM_DUR_S}s\n"
        f"Payload applicatif = 1 octet = {PAYLOAD_BITS} bits",
        transform=ax.transAxes, fontsize=8.5, va='top',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#F0FDF4',
                  edgecolor='#86EFAC', alpha=0.9))

plt.tight_layout()
output_file = 'goodput_vs_distance.png'
fig.savefig(output_file, dpi=180, bbox_inches='tight')
print(f"   ✓ Sauvegardé: {output_file}")
plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 8b : Goodput — barres groupées par distance, DR fixe + DDQN
# ══════════════════════════════════════════════════════════════════════════════
print("[8b] Goodput — barres groupées par distance...")

n_series   = len(test_datarates) + 1
bw_gp      = 0.13
spacing_gp = 0.45
x_centers  = np.arange(len(dists_sorted)) * (n_series * bw_gp + spacing_gp + 0.15)

fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(24, 14))
fig.suptitle(
    f"Goodput LR-FHSS  —  Standard (DR fixe) vs DDQN Adaptatif\n"
    f"Goodput = N_rx_succès × {PAYLOAD_BITS} bits / {SIM_DUR_S}s  |  payload applicatif = 1 octet",
    fontsize=14, fontweight='bold'
)

# ── Panneau haut : barres groupées goodput par DR + DDQN ─────────────────────
style_ax(ax_top, xlabel='', ylabel='Goodput (bps)')
ax_top.set_title("Goodput par DR et DDQN pour chaque distance", fontsize=12, fontweight='bold')

for idx, dr in enumerate(test_datarates):
    offset = (idx - n_series/2 + 0.5) * bw_gp * 1.25
    vals   = [goodput_std.get((d, dr), 0) for d in dists_sorted]
    bars   = ax_top.bar(x_centers + offset, vals, width=bw_gp,
                        label=GP_LABELS[dr], color=GP_COLORS[dr],
                        alpha=0.82, edgecolor='white', linewidth=0.6)
    for bar, v in zip(bars, vals):
        if v > 0:
            ax_top.text(bar.get_x() + bar.get_width()/2, v * 1.005,
                        f'{v:.4f}', ha='center', va='bottom',
                        fontsize=6, fontweight='bold',
                        color=GP_COLORS[dr], rotation=90)

# Barres DDQN
offset_ddqn = (len(test_datarates) - n_series/2 + 0.5) * bw_gp * 1.25
ddqn_bar_vals = [goodput_ddqn.get(d, 0) for d in dists_sorted]
bars_ddqn = ax_top.bar(x_centers + offset_ddqn, ddqn_bar_vals, width=bw_gp,
                       label='DDQN Adaptatif', color=C_DDQN,
                       alpha=0.88, edgecolor='white', linewidth=0.6, zorder=4)
for bar, v in zip(bars_ddqn, ddqn_bar_vals):
    if v > 0:
        ax_top.text(bar.get_x() + bar.get_width()/2, v * 1.005,
                    f'{v:.4f}', ha='center', va='bottom',
                    fontsize=6, fontweight='bold', color=C_DDQN, rotation=90)

ax_top.set_xticks(x_centers)
ax_top.set_xticklabels([f'{int(d)} m' for d in dists_sorted],
                       fontsize=10, rotation=30, ha='right')
ax_top.legend(loc='upper right', fontsize=10, frameon=True, ncol=3)
if all_gp:
    ax_top.set_ylim(min(all_gp) * 0.88, max(all_gp) * 1.18)

# ── Panneau bas : gain DDQN vs meilleur DR standard ──────────────────────────
style_ax(ax_bot, xlabel='Distance (m)', ylabel='Gain Goodput DDQN (bps)')
ax_bot.set_title(
    "Gain Goodput DDQN  vs  meilleur DR Standard à chaque distance\n"
    "(vert = DDQN meilleur, rouge = DDQN moins bon)",
    fontsize=12, fontweight='bold'
)

gain_vals  = []
gain_pcts  = []
best_dr_lbls = []

for dist in dists_sorted:
    gp_ddqn = goodput_ddqn.get(dist, np.nan)
    # Meilleur goodput standard à cette distance
    std_gps = {dr: goodput_std.get((dist, dr), np.nan) for dr in test_datarates}
    valid   = {dr: v for dr, v in std_gps.items() if not np.isnan(v)}
    if not valid or np.isnan(gp_ddqn):
        gain_vals.append(0); gain_pcts.append(0); best_dr_lbls.append('')
        continue
    best_dr  = max(valid, key=valid.get)
    best_gp  = valid[best_dr]
    gain     = gp_ddqn - best_gp
    gain_pct = (gain / best_gp * 100) if best_gp > 0 else 0
    gain_vals.append(gain)
    gain_pcts.append(gain_pct)
    best_dr_lbls.append(f'DR{best_dr}')

colors_gain = ['#16A34A' if g >= 0 else '#DC2626' for g in gain_vals]
bars_gain = ax_bot.bar(x_centers, gain_vals, width=bw_gp * 3.5,
                       color=colors_gain, alpha=0.85,
                       edgecolor='white', linewidth=0.6)

for bar, g, pct, lbl in zip(bars_gain, gain_vals, gain_pcts, best_dr_lbls):
    h = bar.get_height()
    y_text = h + max(abs(v) for v in gain_vals if v != 0) * 0.04 if h >= 0              else h - max(abs(v) for v in gain_vals if v != 0) * 0.04
    va = 'bottom' if h >= 0 else 'top'
    if abs(pct) > 0.01:
        ax_bot.text(bar.get_x() + bar.get_width()/2, y_text,
                    f'{pct:+.2f}%\nvs {lbl}',
                    ha='center', va=va,
                    fontsize=7.5, fontweight='bold',
                    color='#16A34A' if g >= 0 else '#DC2626')

ax_bot.axhline(0, color='#374151', linestyle='-', linewidth=1.2, alpha=0.7)
ax_bot.set_xticks(x_centers)
ax_bot.set_xticklabels([f'{int(d)} m' for d in dists_sorted],
                       fontsize=10, rotation=30, ha='right')

from matplotlib.patches import Patch
legend_gain = [Patch(facecolor='#16A34A', alpha=0.85, label='DDQN meilleur que le best DR'),
               Patch(facecolor='#DC2626', alpha=0.85, label='DDQN moins bon que le best DR')]
ax_bot.legend(handles=legend_gain, loc='upper right', fontsize=10, frameon=True)

if gain_vals and any(v != 0 for v in gain_vals):
    max_abs = max(abs(v) for v in gain_vals)
    ax_bot.set_ylim(-max_abs * 1.4, max_abs * 1.4)

plt.tight_layout()
output_file = 'goodput_barres_et_gain.png'
fig.savefig(output_file, dpi=180, bbox_inches='tight')
print(f"   ✓ Sauvegardé: {output_file}")
plt.close(fig)

print(f"\n  Résumé Goodput:")
print(f"  {'Distance':<10} {'Best DR':<10} {'Best Goodput':<16} {'DDQN Goodput':<16} {'Gain'}")
print("  " + "-" * 65)
for dist, g, pct, lbl in zip(dists_sorted, gain_vals, gain_pcts, best_dr_lbls):
    best_gp_val = goodput_ddqn.get(dist, 0) - g
    ddqn_gp_val = goodput_ddqn.get(dist, 0)
    print(f"  {int(dist):<10} {lbl:<10} {best_gp_val:<16.4f} {ddqn_gp_val:<16.4f} {pct:+.2f}%")


# ===== RÉSUMÉ STATISTIQUE =====
print("\nRÉSUMÉ STATISTIQUE PAR DISTANCE")
print("=" * 130)

if len(df_comp) > 0 and not df_comp['pdr_ddqn'].isna().all():
    # Tableau: Résultats par distance
    print("\nTableau comparatif: Standard vs DDQN (Énergie groupée par Coding Rate)")
    print("-" * 160)
    print(f"{'Distance':<12} {'Type':<10} {'Puissance':<12} {'PDR':<10} {'Débit':<12} {'Énergie':<20} {'Batterie':<15}")
    print(f"{'(m)':<12} {'':<10} {'(dBm)':<12} {'(%)':<10} {'(bps)':<12} {'(mJ) CR1/3|CR2/3|Moy':<20} {'(années)':<15}")
    print("-" * 160)
    
    results_by_distance = []
    
    for _, row in df_comp.iterrows():
        dist = int(row['distance_m'])
        
        # Données Standard
        std_data = df_standard[df_standard['distance_m'] == dist]
        if len(std_data) > 0:
            std_power = std_data['tx_power'].mean()
            std_pdr = row['pdr_standard']
            std_throughput = std_data['effective_throughput_bps'].mean() if 'effective_throughput_bps' in std_data.columns else 0
            std_energy_cr13 = row['energy_standard_cr13_mj'] if row['energy_standard_cr13_mj'] is not None else 0
            std_energy_cr23 = row['energy_standard_cr23_mj'] if row['energy_standard_cr23_mj'] is not None else 0
            std_energy_avg = row['energy_standard_avg_mj'] if row['energy_standard_avg_mj'] is not None else 0
            std_battery = row['battery_standard_years'] if row['battery_standard_years'] is not None else 0
            
            energy_str = f"{std_energy_cr13:.3f}|{std_energy_cr23:.3f}|{std_energy_avg:.3f}"
            print(f"{dist:<12} {'Standard':<10} {std_power:<12.1f} {std_pdr:<10.1f} {std_throughput:<12.1f} {energy_str:<20} {std_battery:<15.1f}")
        
        # Données DDQN
        ddqn_data = df_ddqn[df_ddqn['distance_m'] == dist]
        if len(ddqn_data) > 0:
            ddqn_power = row['ddqn_avg_power'] if row['ddqn_avg_power'] is not None else 0
            ddqn_pdr = row['pdr_ddqn']
            ddqn_throughput = ddqn_data['effective_throughput_bps'].iloc[0] if 'effective_throughput_bps' in ddqn_data.columns else 0
            ddqn_energy = row['energy_ddqn_mj'] if row['energy_ddqn_mj'] is not None else 0
            ddqn_battery = row['battery_ddqn_years'] if row['battery_ddqn_years'] is not None else 0
            
            pdr_improvement = ddqn_pdr - std_pdr
            throughput_improvement = ddqn_throughput - std_throughput
            energy_diff_cr13 = std_energy_cr13 - ddqn_energy if std_energy_cr13 > 0 else 0
            energy_diff_cr23 = std_energy_cr23 - ddqn_energy if std_energy_cr23 > 0 else 0
            
            status = "[+]" if pdr_improvement > 0 else "[-]" if pdr_improvement < 0 else "[=]"
            
            print(f"{dist:<12} {'DDQN':<10} {ddqn_power:<12.1f} {ddqn_pdr:<10.1f} {ddqn_throughput:<12.1f} {ddqn_energy:<20.3f} {ddqn_battery:<15.1f} {status}")
            
            # Impact de l'amélioration
            if pdr_improvement != 0 or throughput_improvement != 0 or energy_diff_cr13 != 0 or energy_diff_cr23 != 0:
                pct_pdr = (pdr_improvement / std_pdr * 100) if std_pdr > 0 else 0
                pct_throughput = (throughput_improvement / std_throughput * 100) if std_throughput > 0 else 0
                pct_energy_cr13 = (energy_diff_cr13 / std_energy_cr13 * 100) if std_energy_cr13 > 0 else 0
                pct_energy_cr23 = (energy_diff_cr23 / std_energy_cr23 * 100) if std_energy_cr23 > 0 else 0
                impact_energy = f"vs CR1/3: {pct_energy_cr13:+.1f}% | vs CR2/3: {pct_energy_cr23:+.1f}%"
                print(f"{'→ Impact':<12} {'DDQN':<10} {'':<12} {pct_pdr:+.1f}%       {pct_throughput:+.1f}%      {impact_energy:<20} {'':<15}")
        
        print("-" * 160)
        
        results_by_distance.append({
            'distance_m': dist,
            'std_power': std_data['tx_power'].mean() if len(std_data) > 0 else 0,
            'std_pdr': row['pdr_standard'],
            'std_throughput': std_throughput,
            'std_energy': row['energy_standard_avg_mj'] if row['energy_standard_avg_mj'] is not None else 0,
            'std_energy_cr13': row['energy_standard_cr13_mj'] if row['energy_standard_cr13_mj'] is not None else 0,
            'std_energy_cr23': row['energy_standard_cr23_mj'] if row['energy_standard_cr23_mj'] is not None else 0,
            'ddqn_power': row['ddqn_avg_power'] if row['ddqn_avg_power'] is not None else 0,
            'ddqn_pdr': row['pdr_ddqn'],
            'ddqn_throughput': ddqn_throughput,
            'ddqn_energy': row['energy_ddqn_mj'] if row['energy_ddqn_mj'] is not None else 0,
        })
    
    # Résumé global
    print("\nRÉSUMÉ GLOBAL")
    print("=" * 130)
    
    pdr_improvements = df_comp['pdr_ddqn'] - df_comp['pdr_standard']
    mean_pdr_improvement = pdr_improvements.mean()
    improvement_pct = (mean_pdr_improvement / df_comp['pdr_standard'].mean() * 100) if df_comp['pdr_standard'].mean() > 0 else 0
    
    better_count = sum(pdr_improvements > 0)
    worse_count = sum(pdr_improvements < 0)
    equal_count = sum(pdr_improvements == 0)
    
    print(f"\nAmélioration PDR du DDQN vs Standard:")
    print(f"   - Amélioration moyenne: {mean_pdr_improvement:+.1f}% (soit {improvement_pct:+.1f}%)")
    print(f"   - Cas où DDQN est meilleur: {better_count}/{len(df_comp)}")
    if worse_count > 0:
        print(f"   - Cas où DDQN est moins bon: {worse_count}/{len(df_comp)}")
    if equal_count > 0:
        print(f"   - Cas égaux: {equal_count}/{len(df_comp)}")
    
    # Métriques de débit
    valid_throughput = df_comp[df_comp['distance_m'].isin(df_ddqn['distance_m'].values)]
    if len(valid_throughput) > 0:
        std_throughputs = []
        ddqn_throughputs = []
        for dist in valid_throughput['distance_m']:
            std_at_dist = df_standard[df_standard['distance_m'] == dist]
            if len(std_at_dist) > 0:
                std_throughputs.append(std_at_dist['effective_throughput_bps'].mean())
            ddqn_at_dist = df_ddqn[df_ddqn['distance_m'] == dist]
            if len(ddqn_at_dist) > 0:
                ddqn_throughputs.append(ddqn_at_dist['effective_throughput_bps'].iloc[0])
        
        if std_throughputs and ddqn_throughputs:
            avg_std_throughput = np.mean(std_throughputs)
            avg_ddqn_throughput = np.mean(ddqn_throughputs)
            throughput_improvement = ((avg_ddqn_throughput - avg_std_throughput) / avg_std_throughput * 100)
            
            print(f"\nDébit effectif moyen:")
            print(f"   - Standard: {avg_std_throughput:.1f} bps")
            print(f"   - DDQN: {avg_ddqn_throughput:.1f} bps")
            print(f"   - Amélioration: {throughput_improvement:+.1f}%")
    
    # Métriques énergétiques
    valid_energy = df_comp[df_comp['energy_standard_avg_mj'].notna() & df_comp['energy_ddqn_mj'].notna()]
    if len(valid_energy) > 0:
        avg_energy_std = valid_energy['energy_standard_avg_mj'].mean()
        avg_energy_ddqn = valid_energy['energy_ddqn_mj'].mean()
        energy_saving = ((avg_energy_std - avg_energy_ddqn) / avg_energy_std * 100)
        
        print(f"\nÉnergie moyenne par paquet:")
        print(f"   - Standard: {avg_energy_std:.3f} mJ")
        print(f"   - DDQN: {avg_energy_ddqn:.3f} mJ")
        if energy_saving > 0:
            print(f"   - Économie: {energy_saving:+.1f}%")
        else:
            print(f"   - Consommation supplémentaire: {energy_saving:.1f}%")
        
        # DR préféré par distance
        print(f"\nStratégie du DDQN par distance:")
        for _, row in df_comp.iterrows():
            dist = int(row['distance_m'])
            dr_dist = {
                8: row['dr8_pct'],
                9: row['dr9_pct'],
                10: row['dr10_pct'],
                11: row['dr11_pct']
            }
            preferred_dr = max(dr_dist, key=dr_dist.get)
            print(f"   {dist}m: DR{preferred_dr} principalement ({dr_dist[preferred_dr]:.0f}%)")
    
    # Sauvegarder le résumé complet
    with open('ddqn_comparison_summary.txt', 'w') as f:
        f.write("=" * 130 + "\n")
        f.write("RÉSUMÉ COMPARAISON DDQN vs SIMULATION STANDARD - PAR DISTANCE\n")
        f.write("=" * 130 + "\n\n")
        f.write(f"Graine maître utilisée: {MASTER_SEED}\n")
        f.write("-" * 130 + "\n\n")
        
        f.write(f"{'Distance':<12} {'Type':<10} {'Puissance':<12} {'PDR':<10} {'Débit':<12} {'Énergie':<15} {'Batterie':<15}\n")
        f.write(f"{'(m)':<12} {'':<10} {'(dBm)':<12} {'(%)':<10} {'(bps)':<12} {'(mJ/paquet)':<15} {'(années)':<15}\n")
        f.write("-" * 130 + "\n")
        
        for result in results_by_distance:
            dist = result['distance_m']
            
            f.write(f"{dist:<12} {'Standard':<10} {result['std_power']:<12.1f} {result['std_pdr']:<10.1f} {result['std_throughput']:<12.1f} {result['std_energy']:<15.3f} {'N/A':<15}\n")
            f.write(f"{dist:<12} {'DDQN':<10} {result['ddqn_power']:<12.1f} {result['ddqn_pdr']:<10.1f} {result['ddqn_throughput']:<12.1f} {result['ddqn_energy']:<15.3f} {'N/A':<15}\n")
            
            pdr_improvement = result['ddqn_pdr'] - result['std_pdr']
            throughput_improvement = result['ddqn_throughput'] - result['std_throughput']
            energy_diff = result['std_energy'] - result['ddqn_energy']
            pct_pdr = (pdr_improvement / result['std_pdr'] * 100) if result['std_pdr'] > 0 else 0
            pct_throughput = (throughput_improvement / result['std_throughput'] * 100) if result['std_throughput'] > 0 else 0
            pct_energy = (energy_diff / result['std_energy'] * 100) if result['std_energy'] > 0 else 0
            
            f.write(f"{'→ Impact':<12} {'DDQN':<10} {'':<12} {pct_pdr:+.1f}%       {pct_throughput:+.1f}%      {pct_energy:+.1f}%\n")
            f.write("-" * 130 + "\n")
        
        f.write("\n" + "=" * 130 + "\n")
        f.write("RÉSUMÉ GLOBAL\n")
        f.write("=" * 130 + "\n")
        f.write(f"Amélioration PDR moyenne: {mean_pdr_improvement:+.1f}% ({improvement_pct:+.1f}%)\n")
        f.write(f"Cas où DDQN est meilleur: {better_count}/{len(df_comp)}\n")
        f.write(f"Cas où DDQN est moins bon: {worse_count}/{len(df_comp)}\n")
        
        # Ajouter les métriques de débit
        if std_throughputs and ddqn_throughputs:
            f.write(f"\nMÉTRIQUES DE DÉBIT:\n")
            f.write(f"Débit moyen standard: {avg_std_throughput:.1f} bps\n")
            f.write(f"Débit moyen DDQN: {avg_ddqn_throughput:.1f} bps\n")
            f.write(f"Amélioration débit: {throughput_improvement:+.1f}%\n")
        
        # Ajouter les métriques énergétiques
        if len(valid_energy) > 0:
            f.write(f"\nMÉTRIQUES ÉNERGÉTIQUES:\n")
            f.write(f"Énergie moyenne standard: {avg_energy_std:.3f} mJ\n")
            f.write(f"Énergie moyenne DDQN: {avg_energy_ddqn:.3f} mJ\n")
            f.write(f"Économie d'énergie: {energy_saving:+.1f}%\n")
    
    print(f"\nRésumé sauvegardé: ddqn_comparison_summary.txt")
    
    # Sauvegarder aussi les résultats détaillés en CSV
    df_results_detailed = pd.DataFrame(results_by_distance)
    df_results_detailed.to_csv('ddqn_comparison_detailed_by_distance.csv', index=False)
    print(f"Résultats détaillés: ddqn_comparison_detailed_by_distance.csv")

print("\n" + "=" * 130)
print("Comparaison DDQN avec analyse énergétique et de débit terminée")
print(f"Graine maître utilisée: {MASTER_SEED} (résultats reproductibles)")
print("=" * 130)