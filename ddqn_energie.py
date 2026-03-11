#!/usr/bin/env python3
"""
compare_ddqn_energy_detailed.py
Comparaison énergétique détaillée LR-FHSS avec vrai calcul ToA
Réel vs Simulation Standard (DR fixe) vs DDQN (DR adaptatif)
Toutes les mesures en Joules conformément à energy.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys
import time
import os
import torch
from typing import Dict
from collections import deque
import logging

# Importer les modules nécessaires
from simulation import LR_FHSS_Simulation
from integrated_ddqn import IntegratedDDQNAgent
from energy import EnergyConsumptionModel, LR_FHSS_EnergyAnalyzer
from config import LR_FHSS_Config

# Configuration logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

print("=" * 100)
print("⚡ COMPARAISON ÉNERGÉTIQUE DÉTAILLÉE LR-FHSS")
print("   Réel vs Simulation Standard (DR fixe) vs DDQN (DR adaptatif)")
print("   AVEC VRAI CALCUL ToA PAR DR")
print("=" * 100)

# ===== CHARGER LE MODÈLE DDQN ENTRAÎNÉ =====
print("\n📥 Chargement du modèle DDQN...")

DDQN_MODEL_PATH = "ddqn_checkpoints/ddqn_final.pth"

# Vérifier si le modèle existe
if not os.path.exists(DDQN_MODEL_PATH):
    alt_paths = [
        "BEST/dqn_models/dqn_model_best.pth",
        "dqn_models/dqn_model_final.pth",
        "ddqn_sequential_checkpoints/ddqn_final.pth"
    ]
    for alt in alt_paths:
        if os.path.exists(alt):
            DDQN_MODEL_PATH = alt
            break

print(f"✓ Modèle: {DDQN_MODEL_PATH}")
print(f"✓ Existe: {os.path.exists(DDQN_MODEL_PATH)}")

# ===== CHARGER LES DONNÉES RÉELLES =====
print("\n📥 Chargement des données réelles...")
try:
    real_data = pd.read_csv('PDR_avgRSSI_distance.csv')
    print(f"✓ {len(real_data)} points de mesure réelles")
except FileNotFoundError:
    print("⚠️ Fichier PDR_avgRSSI_distance.csv non trouvé. Création de données simulées...")
    # Créer des données de test si le fichier n'existe pas
    distances = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500]
    dr_list = [8, 9, 10, 11]
    directions = ['N', 'S', 'E', 'O']
    
    rows = []
    for d in distances:
        for dr in dr_list:
            for dir in directions:
                # PDR qui diminue avec la distance
                if d < 300:
                    pdr = 0.95 - (d/1000)
                elif d < 800:
                    pdr = 0.80 - (d/2000)
                else:
                    pdr = 0.50 - (d/3000)
                
                pdr = max(0.1, min(0.99, pdr))
                rssi = -80 - (d/20) + np.random.normal(0, 2)
                
                rows.append({
                    'Distance(m)': d,
                    'DR': dr,
                    'Direction': dir,
                    'PDR': pdr,
                    'Avg_RSSI(dBm)': rssi
                })
    
    real_data = pd.DataFrame(rows)
    print(f"✓ {len(real_data)} points de mesure générés")

# Extraire toutes les distances et datarates
all_distances = sorted(real_data['Distance(m)'].unique().tolist())
all_datarates = sorted(real_data['DR'].unique().tolist())
test_datarates = [dr for dr in all_datarates if dr >= 8]
test_distances = all_distances

print(f"\n📍 Configuration:")
print(f"   • Distances: {len(test_distances)} valeurs: {test_distances}")
print(f"   • Datarates LR-FHSS: {test_datarates}")
print(f"   • Payload: 1 byte (fixe)")
print(f"   • Total simulations standard: {len(test_distances) * len(test_datarates)}")
print(f"   • Total simulations DDQN: {len(test_distances)}")

# Configuration de base
BASE_CONFIG = {
    'simulation_duration': 1800,
    'num_devices': 10,
    'region': 'EU868',
    'payload_min': 1,
    'payload_max': 1,
    'tx_interval_min': 30,
    'tx_interval_max': 60,
    'shadowing_std_db': 7.0,
    'path_loss_exponent': 3.3,
    'noise_figure_db': 6.0,
    'enable_intelligent_scheduler': False,
    'position_seed': 42,
    'shadowing_seed': 42,
}

# Configuration DR
DR_CONFIG_MAP = {
    8: {'bw_khz': 136.71875, 'cr': '1/3', 'tx': 14, 'nominal_rate_bps': 162, 'info_rate_bps': 488},
    9: {'bw_khz': 136.71875, 'cr': '2/3', 'tx': 14, 'nominal_rate_bps': 325, 'info_rate_bps': 488},
    10: {'bw_khz': 335.9375, 'cr': '1/3', 'tx': 14, 'nominal_rate_bps': 162, 'info_rate_bps': 488},
    11: {'bw_khz': 335.9375, 'cr': '2/3', 'tx': 14, 'nominal_rate_bps': 325, 'info_rate_bps': 488},
}

# ===== FONCTION DE CALCUL ToA EXACT =====
def calculate_toa_ms(dr: int, payload_bytes: int = 1) -> float:
    """
    Calcule le Time-on-Air exact pour LR-FHSS selon config.py
    
    Args:
        dr: Data Rate (8-11)
        payload_bytes: Taille du payload en bytes (1 par défaut)
    
    Returns:
        ToA en millisecondes
    """
    return LR_FHSS_Config.calculate_toa_ms(dr, payload_bytes)

# ===== FONCTION DE CALCUL D'ÉNERGIE POUR DR FIXE =====
def calculate_energy_for_dr(dr: int, payload_bytes: int = 1) -> Dict:
    """
    Calcule la consommation énergétique pour un DR spécifique
    
    Args:
        dr: Data Rate (8-11)
        payload_bytes: Taille du payload
    
    Returns:
        Dictionnaire avec métriques énergétiques
    """
    toa_ms = calculate_toa_ms(dr, payload_bytes)
    tx_power = DR_CONFIG_MAP[dr]['tx']
    
    return EnergyConsumptionModel.calculate_energy_joules(
        tx_power_dbm=tx_power,
        toa_ms=toa_ms,
        pa_type='SX1261_LP',
        voltage_v=3.3
    )

# ===== INITIALISER L'AGENT DDQN =====
print("\n🤖 Initialisation de l'agent DDQN...")

ddqn_agent = None
ddqn_available = False

try:
    if os.path.exists(DDQN_MODEL_PATH):
        ddqn_agent = IntegratedDDQNAgent(
            checkpoint_path=DDQN_MODEL_PATH,
            deterministic=True
        )
        ddqn_available = True
        print(f"✅ Agent DDQN initialisé avec succès!")
    else:
        print(f"⚠️ Modèle non trouvé: {DDQN_MODEL_PATH}")
except Exception as e:
    print(f"❌ Erreur initialisation DDQN: {e}")

# ===== CALCUL DES ÉNERGIES THÉORIQUES POUR DR FIXES =====
print("\n📊 Calcul des énergies théoriques par DR (payload 1 byte)")
print("-" * 70)

energy_by_dr = {}
for dr in test_datarates:
    energy = calculate_energy_for_dr(dr)
    toa_ms = calculate_toa_ms(dr)
    
    energy_by_dr[dr] = energy
    
    print(f"DR{dr}:")
    print(f"  • ToA: {toa_ms:.2f} ms")
    print(f"  • Puissance TX: {energy['tx_power_dbm']} dBm")
    print(f"  • Courant TX: {energy['tx_current_ma']:.1f} mA")
    print(f"  • Énergie TX: {energy['tx_energy_j']*1000:.3f} mJ")
    print(f"  • Énergie totale: {energy['total_energy_j']*1000:.3f} mJ")
    print(f"  • Énergie par bit: {energy['energy_per_bit_j']*1e9:.1f} nJ/bit")
    print(f"  • Durée vie batterie: {energy['battery_life_years']:.1f} ans")

# ===== CALCUL DE L'ÉNERGIE RÉELLE BASÉE SUR PDR =====
print("\n📈 Calcul de l'énergie réelle (basée sur PDR et retransmissions)")
print("-" * 70)

real_energy_results = []

for distance_m in test_distances:
    for dr in test_datarates:
        # Filtrer les données réelles pour cette distance et DR
        real_subset = real_data[
            (real_data['Distance(m)'] == distance_m) & 
            (real_data['DR'] == dr)
        ]
        
        if len(real_subset) == 0:
            continue
        
        # PDR réel moyen pour cette distance/DR
        pdr_real = real_subset['PDR'].mean()
        
        # Énergie théorique pour ce DR
        energy_per_tx = energy_by_dr[dr]['total_energy_j']
        
        # ÉNERGIE RÉELLE: Prend en compte les retransmissions nécessaires
        # Si PDR = 0.5, il faut en moyenne 2 transmissions pour qu'un paquet réussisse
        if pdr_real > 0:
            # Nombre moyen de transmissions = 1/PDR
            avg_transmissions = 1.0 / pdr_real
            real_energy_j = energy_per_tx * avg_transmissions
        else:
            # Si PDR = 0, énergie très élevée (10 transmissions)
            real_energy_j = energy_per_tx * 10
        
        # RSSI réel moyen
        rssi_real = real_subset['Avg_RSSI(dBm)'].mean()
        
        real_energy_results.append({
            'distance_m': distance_m,
            'datarate': dr,
            'pdr_real': pdr_real * 100,  # en %
            'energy_per_tx_j': energy_per_tx,
            'energy_per_tx_mj': energy_per_tx * 1000,
            'avg_transmissions': 1.0/pdr_real if pdr_real > 0 else 10,
            'total_energy_j': real_energy_j,
            'total_energy_mj': real_energy_j * 1000,
            'rssi_mean_dbm': rssi_real,
            'toa_ms': calculate_toa_ms(dr),
            'tx_power_dbm': DR_CONFIG_MAP[dr]['tx'],
            'tx_current_ma': energy_by_dr[dr]['tx_current_ma'],
            'battery_life_years': EnergyConsumptionModel.calculate_battery_life_joules(
                energy_per_transmission_j=real_energy_j,
                battery_capacity_mah=1000.0,
                voltage_v=3.3,
                transmissions_per_day=24
            )
        })

df_real_energy = pd.DataFrame(real_energy_results)

# Moyenne par distance pour l'énergie réelle
real_energy_by_distance = df_real_energy.groupby('distance_m').agg({
    'pdr_real': 'mean',
    'total_energy_j': 'mean',
    'total_energy_mj': 'mean',
    'rssi_mean_dbm': 'mean',
    'battery_life_years': 'mean'
}).reset_index()

print("\n📊 RÉSUMÉ ÉNERGIE RÉELLE PAR DISTANCE (moyenne sur tous DR):")
for _, row in real_energy_by_distance.iterrows():
    dist = int(row['distance_m'])
    print(f"  {dist}m: PDR={row['pdr_real']:.1f}%, "
          f"Énergie={row['total_energy_mj']:.3f} mJ, "
          f"Batterie={row['battery_life_years']:.1f} ans")

# ===== LANCER LES SIMULATIONS =====
print("\n🎬 Lancement des simulations avec analyse énergétique...")
print("=" * 100)

simulation_results = []  # Résultats standard (DR fixe)
ddqn_results = []        # Résultats DDQN (DR adaptatif)

total_sims_standard = len(test_distances) * len(test_datarates)
total_sims_ddqn = len(test_distances)
current_sim = 0

# Facteurs de conversion
J_TO_mJ = 1000.0
J_TO_uJ = 1e6

# ===== 1. SIMULATIONS STANDARD (DR fixe) =====
print("\n📊 PHASE 1: Simulations STANDARD (DR fixe)")
print("-" * 100)

for dist_m in test_distances:
    for dr in test_datarates:
        current_sim += 1
        print(f"\n[{current_sim}/{total_sims_standard + total_sims_ddqn}] STANDARD: Distance={dist_m}m, DR={dr}")
        print("-" * 100)
        
        # Configuration pour cette simulation
        config = BASE_CONFIG.copy()
        config['distance_gtw'] = dist_m
        config['coding_rate'] = DR_CONFIG_MAP[dr]['cr']
        config['tx_power'] = DR_CONFIG_MAP[dr]['tx']
        config['bandwidth_khz'] = DR_CONFIG_MAP[dr]['bw_khz']
        config['enable_dqn'] = False
        
        # Calculer ToA exact pour ce DR
        toa_ms = calculate_toa_ms(dr)
        
        print(f"  Config: CR={config['coding_rate']}, TX={config['tx_power']} dBm, BW={config['bandwidth_khz']} kHz")
        print(f"  ToA exact: {toa_ms:.2f} ms")
        
        try:
            start_time = time.time()
            
            sim = LR_FHSS_Simulation(config)
            
            # Ajouter l'analyseur énergétique
            energy_analyzer = LR_FHSS_EnergyAnalyzer(sim.dashboard)
            
            # Modifier l'évaluation des paquets
            original_evaluate = sim.dashboard._evaluate_packet_end
            
            def energy_evaluate_packet(packet):
                original_evaluate(packet)
                # Forcer le ToA correct pour le calcul d'énergie
                packet.toa_ms = toa_ms
                if hasattr(packet, 'success'):
                    energy_analyzer.analyze_packet_energy(packet)
            
            sim.dashboard._evaluate_packet_end = energy_evaluate_packet
            
            sim.run()
            
            elapsed = time.time() - start_time
            
            # Récupérer les métriques
            energy_stats = energy_analyzer.energy_stats
            
            total_sent = sim.total_sent
            successful_rx = sim.successful_rx
            pdr = successful_rx / total_sent if total_sent > 0 else 0
            
            rssi_mean = sim.detailed_stats.avg_rssi_dbm if hasattr(sim, 'detailed_stats') else -120
            snr_mean = sim.detailed_stats.avg_snr_db if hasattr(sim, 'detailed_stats') else 0
            
            # Métriques énergétiques
            total_energy_j = energy_stats['total_energy_j']
            energy_successful_j = energy_stats['energy_successful_j']
            energy_failed_j = energy_stats['energy_failed_j']
            efficiency_ratio = energy_stats['efficiency_ratio']
            avg_energy_per_packet_j = energy_stats['avg_energy_per_packet_j']
            
            # Calculer la durée de vie batterie
            battery_life_years = EnergyConsumptionModel.calculate_battery_life_joules(
                energy_per_transmission_j=avg_energy_per_packet_j,
                battery_capacity_mah=1000.0,
                voltage_v=3.3,
                transmissions_per_day=24
            )
            
            result = {
                'type': 'standard',
                'distance_m': dist_m,
                'datarate': dr,
                'toa_ms': toa_ms,
                'num_packets': total_sent,
                'pdr': pdr,
                'successful_packets': successful_rx,
                'failed_packets': total_sent - successful_rx,
                'rssi_mean_dbm': rssi_mean,
                'snr_mean_db': snr_mean,
                
                # Énergie (en Joules)
                'total_energy_j': total_energy_j,
                'total_energy_mj': total_energy_j * J_TO_mJ,
                'energy_successful_j': energy_successful_j,
                'energy_failed_j': energy_failed_j,
                'energy_efficiency_pct': efficiency_ratio,
                'avg_energy_per_packet_j': avg_energy_per_packet_j,
                'avg_energy_per_packet_mj': avg_energy_per_packet_j * J_TO_mJ,
                'battery_life_years': battery_life_years,
                
                # Paramètres électriques
                'tx_current_ma': energy_by_dr[dr]['tx_current_ma'],
                'tx_power_dbm': DR_CONFIG_MAP[dr]['tx'],
                
                'simulation_time_s': elapsed,
            }
            
            simulation_results.append(result)
            
            print(f"  📊 PDR: {pdr*100:.1f}%")
            print(f"  ⚡ Énergie/paquet: {avg_energy_per_packet_j*1000:.3f} mJ")
            print(f"  ⚡ Énergie totale: {total_energy_j*1000:.3f} mJ")
            print(f"  ⚡ Efficacité: {efficiency_ratio:.1f}%")
            print(f"  🔋 Durée vie batterie: {battery_life_years:.1f} ans")
            
        except Exception as e:
            print(f"❌ ERREUR: {e}")
            import traceback
            traceback.print_exc()
            continue

# ===== 2. SIMULATIONS AVEC DDQN =====
print("\n" + "=" * 100)
print("🤖 PHASE 2: Simulations avec DDQN (DR adaptatif)")
print("=" * 100)

if ddqn_available:
    for dist_m in test_distances:
        current_sim += 1
        print(f"\n[{current_sim}/{total_sims_standard + total_sims_ddqn}] DDQN: Distance={dist_m}m")
        print("-" * 100)
        
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
        
        try:
            start_time = time.time()
            
            sim = LR_FHSS_Simulation(config)
            
            # Ajouter l'analyseur énergétique
            energy_analyzer = LR_FHSS_EnergyAnalyzer(sim.dashboard)
            
            # Modifier l'évaluation des paquets
            original_evaluate = sim.dashboard._evaluate_packet_end
            
            def energy_evaluate_packet(packet):
                original_evaluate(packet)
                # Calculer ToA basé sur le DR choisi par DQN
                if hasattr(packet, 'dr'):
                    packet.toa_ms = calculate_toa_ms(int(packet.dr))
                if hasattr(packet, 'success'):
                    energy_analyzer.analyze_packet_energy(packet)
            
            sim.dashboard._evaluate_packet_end = energy_evaluate_packet
            
            if sim.dqn_manager and sim.dqn_manager.agent:
                sim.dqn_manager.agent.deterministic = True
            
            sim.run()
            
            elapsed = time.time() - start_time
            
            # Récupérer les métriques
            energy_stats = energy_analyzer.energy_stats
            
            total_sent = sim.total_sent
            successful_rx = sim.successful_rx
            pdr = successful_rx / total_sent if total_sent > 0 else 0
            
            rssi_mean = sim.detailed_stats.avg_rssi_dbm if hasattr(sim, 'detailed_stats') else -120
            snr_mean = sim.detailed_stats.avg_snr_db if hasattr(sim, 'detailed_stats') else 0
            
            # Analyser les décisions DQN
            drs_chosen = []
            powers_chosen = []
            toas_chosen = []
            
            if hasattr(sim, 'simulated_packets'):
                for pkt in sim.simulated_packets:
                    if hasattr(pkt, 'dqn_applied') and pkt.dqn_applied:
                        dr_val = int(pkt.dr)
                        drs_chosen.append(dr_val)
                        powers_chosen.append(pkt.tx_power_dbm)
                        toas_chosen.append(calculate_toa_ms(dr_val))
            
            # Métriques énergétiques
            total_energy_j = energy_stats['total_energy_j']
            energy_successful_j = energy_stats['energy_successful_j']
            energy_failed_j = energy_stats['energy_failed_j']
            efficiency_ratio = energy_stats['efficiency_ratio']
            avg_energy_per_packet_j = energy_stats['avg_energy_per_packet_j']
            
            # Calculer la durée de vie batterie
            battery_life_years = EnergyConsumptionModel.calculate_battery_life_joules(
                energy_per_transmission_j=avg_energy_per_packet_j,
                battery_capacity_mah=1000.0,
                voltage_v=3.3,
                transmissions_per_day=24
            )
            
            # Distribution des DR
            dr_distribution = {8: 0, 9: 0, 10: 0, 11: 0}
            for dr in drs_chosen:
                dr_distribution[int(dr)] = dr_distribution.get(int(dr), 0) + 1
            
            # Calcul de l'énergie théorique moyenne basée sur les choix DQN
            theoretical_energy_per_dr = {dr: energy_by_dr[dr]['total_energy_j'] for dr in test_datarates}
            
            result = {
                'type': 'ddqn',
                'distance_m': dist_m,
                'datarate': 'adaptive',
                'num_packets': total_sent,
                'pdr': pdr,
                'successful_packets': successful_rx,
                'failed_packets': total_sent - successful_rx,
                'rssi_mean_dbm': rssi_mean,
                'snr_mean_db': snr_mean,
                
                # Métriques DQN
                'dqn_decisions': len(drs_chosen),
                'avg_dr_choice': np.mean(drs_chosen) if drs_chosen else 0,
                'avg_power_choice': np.mean(powers_chosen) if powers_chosen else 0,
                'avg_toa_ms': np.mean(toas_chosen) if toas_chosen else 0,
                'dr8_count': dr_distribution[8],
                'dr9_count': dr_distribution[9],
                'dr10_count': dr_distribution[10],
                'dr11_count': dr_distribution[11],
                
                # Énergie (en Joules)
                'total_energy_j': total_energy_j,
                'total_energy_mj': total_energy_j * J_TO_mJ,
                'energy_successful_j': energy_successful_j,
                'energy_failed_j': energy_failed_j,
                'energy_efficiency_pct': efficiency_ratio,
                'avg_energy_per_packet_j': avg_energy_per_packet_j,
                'avg_energy_per_packet_mj': avg_energy_per_packet_j * J_TO_mJ,
                'battery_life_years': battery_life_years,
                
                # Énergie théorique moyenne
                'theoretical_energy_per_packet_j': np.mean([theoretical_energy_per_dr.get(int(dr), 0) for dr in drs_chosen]) if drs_chosen else 0,
                
                'simulation_time_s': elapsed,
            }
            
            ddqn_results.append(result)
            
            # Afficher les résultats
            dr8_pct = (dr_distribution[8] / len(drs_chosen) * 100) if drs_chosen else 0
            dr9_pct = (dr_distribution[9] / len(drs_chosen) * 100) if drs_chosen else 0
            dr10_pct = (dr_distribution[10] / len(drs_chosen) * 100) if drs_chosen else 0
            dr11_pct = (dr_distribution[11] / len(drs_chosen) * 100) if drs_chosen else 0
            
            print(f"  📊 PDR: {pdr*100:.1f}%")
            print(f"  🤖 DR moyen: {np.mean(drs_chosen) if drs_chosen else 0:.1f}")
            print(f"     Distribution: DR8={dr8_pct:.0f}%, DR9={dr9_pct:.0f}%, DR10={dr10_pct:.0f}%, DR11={dr11_pct:.0f}%")
            print(f"  ⚡ Énergie/paquet: {avg_energy_per_packet_j*1000:.3f} mJ")
            print(f"  ⚡ Énergie totale: {total_energy_j*1000:.3f} mJ")
            print(f"  ⚡ Efficacité: {efficiency_ratio:.1f}%")
            print(f"  🔋 Durée vie batterie: {battery_life_years:.1f} ans")
            
        except Exception as e:
            print(f"❌ ERREUR DDQN: {e}")
            import traceback
            traceback.print_exc()
            continue
else:
    print("⚠️ DDQN non disponible - saut des simulations")

print("\n" + "=" * 100)
print("✅ Toutes les simulations terminées!")
print("=" * 100)

# ===== CONVERTIR EN DATAFRAMES =====
df_standard = pd.DataFrame(simulation_results)
df_ddqn = pd.DataFrame(ddqn_results)

# Sauvegarder les résultats
csv_standard = 'simulation_standard_energy_detailed.csv'
df_standard.to_csv(csv_standard, index=False)
print(f"✅ Résultats standard: {csv_standard}")

if len(df_ddqn) > 0:
    csv_ddqn = 'simulation_ddqn_energy_detailed.csv'
    df_ddqn.to_csv(csv_ddqn, index=False)
    print(f"✅ Résultats DDQN: {csv_ddqn}")

csv_real = 'real_energy_detailed.csv'
df_real_energy.to_csv(csv_real, index=False)
print(f"✅ Résultats réels: {csv_real}")

# ===== COMPARAISON ÉNERGÉTIQUE DÉTAILLÉE =====
print("\n📊 COMPARAISON ÉNERGÉTIQUE DÉTAILLÉE")
print("=" * 100)
print("   RÉEL vs STANDARD vs DDQN")
print("   (Toutes les énergies en mJ)")
print("-" * 100)

comparison_rows = []

for distance_m in test_distances:
    # Données réelles
    real_row = real_energy_by_distance[real_energy_by_distance['distance_m'] == distance_m]
    
    if len(real_row) == 0:
        continue
    
    pdr_real = real_row['pdr_real'].iloc[0]
    energy_real_mj = real_row['total_energy_mj'].iloc[0]
    battery_real = real_row['battery_life_years'].iloc[0]
    rssi_real = real_row['rssi_mean_dbm'].iloc[0]
    
    # Simulation standard
    std_subset = df_standard[df_standard['distance_m'] == distance_m]
    
    if len(std_subset) > 0:
        pdr_std = std_subset['pdr'].mean() * 100
        energy_std_mj = std_subset['avg_energy_per_packet_mj'].mean()
        battery_std = std_subset['battery_life_years'].mean()
        rssi_std = std_subset['rssi_mean_dbm'].mean()
        efficiency_std = std_subset['energy_efficiency_pct'].mean()
    else:
        pdr_std = energy_std_mj = battery_std = rssi_std = efficiency_std = 0
    
    # Simulation DDQN
    ddqn_subset = df_ddqn[df_ddqn['distance_m'] == distance_m]
    
    if len(ddqn_subset) > 0:
        pdr_ddqn = ddqn_subset['pdr'].iloc[0] * 100
        energy_ddqn_mj = ddqn_subset['avg_energy_per_packet_mj'].iloc[0]
        battery_ddqn = ddqn_subset['battery_life_years'].iloc[0]
        rssi_ddqn = ddqn_subset['rssi_mean_dbm'].iloc[0]
        efficiency_ddqn = ddqn_subset['energy_efficiency_pct'].iloc[0]
        avg_dr = ddqn_subset['avg_dr_choice'].iloc[0]
    else:
        pdr_ddqn = energy_ddqn_mj = battery_ddqn = rssi_ddqn = efficiency_ddqn = avg_dr = 0
    
    # Calculer les économies d'énergie
    if energy_real_mj > 0 and energy_std_mj > 0:
        std_vs_real_pct = ((energy_std_mj - energy_real_mj) / energy_real_mj) * 100
    else:
        std_vs_real_pct = 0
    
    if energy_real_mj > 0 and energy_ddqn_mj > 0:
        ddqn_vs_real_pct = ((energy_ddqn_mj - energy_real_mj) / energy_real_mj) * 100
        ddqn_vs_std_pct = ((energy_std_mj - energy_ddqn_mj) / energy_std_mj) * 100 if energy_std_mj > 0 else 0
    else:
        ddqn_vs_real_pct = ddqn_vs_std_pct = 0
    
    # Afficher la comparaison
    print(f"\n🏘️  Distance: {distance_m}m")
    print("-" * 70)
    print(f"  📈 PDR:")
    print(f"     Réel:    {pdr_real:.1f}%")
    print(f"     Standard: {pdr_std:.1f}%")
    print(f"     DDQN:    {pdr_ddqn:.1f}%")
    
    print(f"\n  ⚡ Énergie par paquet (mJ):")
    print(f"     Réel:    {energy_real_mj:.3f} mJ")
    print(f"     Standard: {energy_std_mj:.3f} mJ  ({std_vs_real_pct:+.1f}% vs réel)")
    print(f"     DDQN:    {energy_ddqn_mj:.3f} mJ  ({ddqn_vs_real_pct:+.1f}% vs réel, {ddqn_vs_std_pct:+.1f}% vs std)")
    
    print(f"\n  🔋 Durée vie batterie (ans):")
    print(f"     Réel:    {battery_real:.1f} ans")
    print(f"     Standard: {battery_std:.1f} ans")
    print(f"     DDQN:    {battery_ddqn:.1f} ans")
    
    if avg_dr > 0:
        print(f"\n  🤖 DDQN: DR moyen={avg_dr:.1f}")
    
    comparison_rows.append({
        'distance_m': distance_m,
        'pdr_real': pdr_real,
        'pdr_standard': pdr_std,
        'pdr_ddqn': pdr_ddqn,
        'energy_real_mj': energy_real_mj,
        'energy_standard_mj': energy_std_mj,
        'energy_ddqn_mj': energy_ddqn_mj,
        'battery_real_years': battery_real,
        'battery_standard_years': battery_std,
        'battery_ddqn_years': battery_ddqn,
        'rssi_real_dbm': rssi_real,
        'rssi_standard_dbm': rssi_std,
        'rssi_ddqn_dbm': rssi_ddqn,
        'efficiency_standard_pct': efficiency_std,
        'efficiency_ddqn_pct': efficiency_ddqn,
        'std_vs_real_energy_pct': std_vs_real_pct,
        'ddqn_vs_real_energy_pct': ddqn_vs_real_pct,
        'ddqn_vs_std_energy_pct': ddqn_vs_std_pct,
        'avg_dr_choice': avg_dr if avg_dr > 0 else None
    })

df_comp = pd.DataFrame(comparison_rows)

# Sauvegarder la comparaison
csv_comp = 'energy_comparison_detailed.csv'
df_comp.to_csv(csv_comp, index=False)
print(f"\n✅ Comparaison sauvegardée: {csv_comp}")

# ===== VISUALISATIONS =====
print("\n📊 Génération des visualisations...")

# 1. Comparaison de l'énergie par paquet
plt.figure(figsize=(14, 8))

x = df_comp['distance_m'].values
width = 25

plt.bar(x - width, df_comp['energy_real_mj'].values, width=width, 
        label='Réel (basé sur PDR + retransmissions)', color='black', alpha=0.7)
plt.bar(x, df_comp['energy_standard_mj'].values, width=width, 
        label='Simulation Standard (DR fixe)', color='blue', alpha=0.7)
plt.bar(x + width, df_comp['energy_ddqn_mj'].values, width=width, 
        label='DDQN (DR adaptatif)', color='red', alpha=0.7)

plt.xlabel('Distance (m)', fontsize=12, fontweight='bold')
plt.ylabel('Énergie par paquet (mJ)', fontsize=12, fontweight='bold')
plt.title('Comparaison de la consommation énergétique par paquet\nRéel vs Simulation Standard vs DDQN', 
          fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3, linestyle=':', linewidth=1)
plt.xticks(x, [f'{int(d)}m' for d in x])

plt.tight_layout()
plt.savefig('energy_comparison_packet.png', dpi=150, bbox_inches='tight')
print("✅ energy_comparison_packet.png")

# 2. Comparaison de la durée de vie batterie
plt.figure(figsize=(14, 8))

plt.plot(df_comp['distance_m'].values, df_comp['battery_real_years'].values, 
         'o-', linewidth=3, markersize=10, label='Réel', color='black')
plt.plot(df_comp['distance_m'].values, df_comp['battery_standard_years'].values, 
         's-', linewidth=3, markersize=10, label='Simulation Standard', color='blue')
plt.plot(df_comp['distance_m'].values, df_comp['battery_ddqn_years'].values, 
         'D-', linewidth=3, markersize=10, label='DDQN', color='red')

plt.xlabel('Distance (m)', fontsize=12, fontweight='bold')
plt.ylabel('Durée de vie batterie (années)', fontsize=12, fontweight='bold')
plt.title('Comparaison de la durée de vie batterie\n(1000mAh, 24 transmissions/jour)', 
          fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3, linestyle=':', linewidth=1)
plt.xticks(df_comp['distance_m'].values, [f'{int(d)}m' for d in df_comp['distance_m'].values])

plt.tight_layout()
plt.savefig('energy_comparison_battery.png', dpi=150, bbox_inches='tight')
print("✅ energy_comparison_battery.png")

# 3. Économie d'énergie du DDQN par rapport au standard
plt.figure(figsize=(14, 6))

colors = ['green' if x > 0 else 'red' for x in df_comp['ddqn_vs_std_energy_pct'].values]
bars = plt.bar(df_comp['distance_m'].values, df_comp['ddqn_vs_std_energy_pct'].values, 
               width=30, color=colors, alpha=0.7)

plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
plt.xlabel('Distance (m)', fontsize=12, fontweight='bold')
plt.ylabel('Économie d\'énergie du DDQN vs Standard (%)', fontsize=12, fontweight='bold')
plt.title('Économie d\'énergie apportée par le DDQN\n(valeurs positives = DDQN plus économique)', 
          fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, linestyle=':', linewidth=1, axis='y')
plt.xticks(df_comp['distance_m'].values, [f'{int(d)}m' for d in df_comp['distance_m'].values])

# Ajouter les valeurs sur les barres
for i, (bar, val) in enumerate(zip(bars, df_comp['ddqn_vs_std_energy_pct'].values)):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., 
             height + (2 if height > 0 else -4),
             f'{val:.1f}%', ha='center', va='bottom' if height > 0 else 'top', 
             fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('energy_savings_ddqn.png', dpi=150, bbox_inches='tight')
print("✅ energy_savings_ddqn.png")

# 4. Graphique combiné avec deux axes (si DDQN disponible)
if len(df_ddqn) > 0:
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    x = df_comp['distance_m'].values
    
    # Axe 1: Énergie
    ax1.plot(x, df_comp['energy_real_mj'].values, 'o-', linewidth=3, 
             markersize=10, label='Énergie réelle', color='black')
    ax1.plot(x, df_comp['energy_standard_mj'].values, 's-', linewidth=3, 
             markersize=10, label='Énergie standard', color='blue')
    ax1.plot(x, df_comp['energy_ddqn_mj'].values, 'D-', linewidth=3, 
             markersize=10, label='Énergie DDQN', color='red')
    
    ax1.set_xlabel('Distance (m)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Énergie par paquet (mJ)', fontsize=12, fontweight='bold', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True, alpha=0.3, linestyle=':', linewidth=1)
    
    # Axe 2: DR moyen choisi par DDQN
    ax2 = ax1.twinx()
    ax2.plot(x, df_comp['avg_dr_choice'].values, 'd--', linewidth=2, 
             markersize=8, label='DR moyen DDQN', color='purple', alpha=0.7)
    ax2.set_ylabel('DR moyen choisi', fontsize=12, fontweight='bold', color='purple')
    ax2.tick_params(axis='y', labelcolor='purple')
    ax2.set_ylim(7.5, 11.5)
    
    # Légende combinée
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)
    
    plt.title('Consommation énergétique et choix de DR du DDQN', fontsize=14, fontweight='bold')
    plt.xticks(x, [f'{int(d)}m' for d in x])
    
    plt.tight_layout()
    plt.savefig('energy_with_dr_choices.png', dpi=150, bbox_inches='tight')
    print("✅ energy_with_dr_choices.png")

# ===== RÉSUMÉ STATISTIQUE =====
print("\n📈 RÉSUMÉ STATISTIQUE GLOBAL")
print("=" * 70)

if len(df_comp) > 0:
    # Calculer les moyennes
    avg_energy_real = df_comp['energy_real_mj'].mean()
    avg_energy_std = df_comp['energy_standard_mj'].mean()
    avg_energy_ddqn = df_comp['energy_ddqn_mj'].mean()
    
    avg_battery_real = df_comp['battery_real_years'].mean()
    avg_battery_std = df_comp['battery_standard_years'].mean()
    avg_battery_ddqn = df_comp['battery_ddqn_years'].mean()
    
    # Économies globales
    total_energy_saving_std_vs_real = ((avg_energy_std - avg_energy_real) / avg_energy_real * 100) if avg_energy_real > 0 else 0
    total_energy_saving_ddqn_vs_real = ((avg_energy_ddqn - avg_energy_real) / avg_energy_real * 100) if avg_energy_real > 0 else 0
    total_energy_saving_ddqn_vs_std = ((avg_energy_std - avg_energy_ddqn) / avg_energy_std * 100) if avg_energy_std > 0 else 0
    
    # Compter où DDQN est meilleur
    better_count = sum((df_comp['energy_ddqn_mj'] < df_comp['energy_standard_mj']).astype(int))
    
    print(f"\n📊 MOYENNES SUR TOUTES LES DISTANCES:")
    print(f"   Énergie par paquet:")
    print(f"     • Réel:    {avg_energy_real:.3f} mJ")
    print(f"     • Standard: {avg_energy_std:.3f} mJ ({total_energy_saving_std_vs_real:+.1f}% vs réel)")
    print(f"     • DDQN:    {avg_energy_ddqn:.3f} mJ ({total_energy_saving_ddqn_vs_real:+.1f}% vs réel)")
    print(f"     • Économie DDQN vs Standard: {total_energy_saving_ddqn_vs_std:+.1f}%")
    
    print(f"\n   Durée de vie batterie:")
    print(f"     • Réel:    {avg_battery_real:.1f} ans")
    print(f"     • Standard: {avg_battery_std:.1f} ans")
    print(f"     • DDQN:    {avg_battery_ddqn:.1f} ans")
    print(f"     • Gain DDQN vs Standard: {avg_battery_ddqn - avg_battery_std:.1f} ans")
    
    print(f"\n🏆 Le DDQN est plus économe en énergie que la simulation standard")
    print(f"   dans {better_count}/{len(df_comp)} cas ({better_count/len(df_comp)*100:.0f}%)")
    
    # Sauvegarder le résumé
    with open('energy_comparison_summary.txt', 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("RÉSUMÉ COMPARAISON ÉNERGÉTIQUE LR-FHSS\n")
        f.write("RÉEL vs STANDARD vs DDQN\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Nombre de distances analysées: {len(df_comp)}\n")
        f.write(f"DR testés: {test_datarates}\n")
        f.write(f"Payload: 1 byte\n\n")
        
        f.write("MOYENNES SUR TOUTES LES DISTANCES:\n")
        f.write(f"  Énergie réelle:          {avg_energy_real:.3f} mJ\n")
        f.write(f"  Énergie standard:        {avg_energy_std:.3f} mJ\n")
        f.write(f"  Énergie DDQN:            {avg_energy_ddqn:.3f} mJ\n\n")
        
        f.write(f"  Écart Standard vs Réel:  {total_energy_saving_std_vs_real:+.1f}%\n")
        f.write(f"  Écart DDQN vs Réel:       {total_energy_saving_ddqn_vs_real:+.1f}%\n")
        f.write(f"  Économie DDQN vs Standard: {total_energy_saving_ddqn_vs_std:+.1f}%\n\n")
        
        f.write(f"  Durée vie batterie réelle:   {avg_battery_real:.1f} ans\n")
        f.write(f"  Durée vie batterie standard: {avg_battery_std:.1f} ans\n")
        f.write(f"  Durée vie batterie DDQN:     {avg_battery_ddqn:.1f} ans\n\n")
        
        f.write(f"  Le DDQN est plus économe dans {better_count}/{len(df_comp)} cas\n")
        
        f.write("\n" + "=" * 70 + "\n")
    
    print(f"\n✅ Résumé sauvegardé: energy_comparison_summary.txt")

print("\n" + "=" * 100)
print("✨ COMPARAISON ÉNERGÉTIQUE TERMINÉE!")
print("=" * 100)