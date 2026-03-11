#!/usr/bin/env python3
"""
Comparaison COMPLÈTE: LR-FHSS (DR8-DR11) sur TOUTES les distances réelles
vs Données Réelles (toutes locations)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys
import time

# Importer le simulateur
from simulation import LR_FHSS_Simulation

print("🚀 COMPARAISON LR-FHSS (DR8-DR11): TOUTES DISTANCES - Simulation DIRECTE vs Données Réelles")
print("=" * 90)

# ===== CHARGER LES DONNÉES RÉELLES =====
print("\n📥 Chargement des données réelles...")
real_data = pd.read_csv('PDR_avgRSSI_distance.csv')
print(f"✓ {len(real_data)} points de mesure réelles")

# Extraire TOUTES les distances et datarates des données réelles
all_distances = sorted(real_data['Distance(m)'].unique().tolist())
all_datarates = sorted(real_data['DR'].unique().tolist())

# Filtrer pour garder seulement DR8, DR9, DR10, DR11 (pas DR0, DR5)
test_datarates = [dr for dr in all_datarates if dr >= 8]

# Distances à tester
test_distances = all_distances

# Agréger les données réelles par (DR, Distance) en moyennant sur les directions
real_data_agg = real_data.groupby(['DR', 'Distance(m)'], as_index=False)[['PDR', 'Avg_RSSI(dBm)']].mean()

print(f"\n📍 Configuration:")
print(f"   • Distances trouvées: {test_distances}m ({len(test_distances)} distances)")
print(f"   • Datarates LR-FHSS testés: {test_datarates} (excluant DR0, DR5)")
print(f"   • Total simulations: {len(test_distances)} × {len(test_datarates)} = {len(test_distances) * len(test_datarates)}")
print(f"   • Locations couverts: {sorted(real_data['Direction'].unique().tolist())}")

# Configuration de base (du simulateur)
BASE_CONFIG = {
    'simulation_duration': 1800,        # 30 minutes
    'num_devices': 10,
    'region': 'EU868',
    'payload_min': 1,
    'payload_max': 1,
    'tx_interval_min': 20,
    'tx_interval_max': 60,
    'shadowing_std_db': 7.0,
    'path_loss_exponent': 3.3,
    'noise_figure_db': 6.0,
    'enable_dqn': False,
    'enable_intelligent_scheduler': False,
    'position_seed': 42,
    'shadowing_seed': 42,
}

print(f"\n✓ Configuration de base:")
print(f"  - Durée: {BASE_CONFIG['simulation_duration']}s")
print(f"  - Devices: {BASE_CONFIG['num_devices']}")

# Mapping datarate -> configuration complète - LR-FHSS seulement
DR_CONFIG_MAP = {
    8: {'bw_khz': 136.71875, 'cr': '1/3', 'tx': 14, 'nominal_rate_bps': 162, 'info_rate_bps': 488},
    9: {'bw_khz': 136.71875, 'cr': '2/3', 'tx': 14, 'nominal_rate_bps': 325, 'info_rate_bps': 488},
    10: {'bw_khz': 335.9375, 'cr': '1/3', 'tx': 14, 'nominal_rate_bps': 162, 'info_rate_bps': 488},
    11: {'bw_khz': 335.9375, 'cr': '2/3', 'tx': 14, 'nominal_rate_bps': 325, 'info_rate_bps': 488},
}

# ===== LANCER LES SIMULATIONS =====
print("\n🎬 Lancement des simulations...")
print("=" * 90)

simulation_results = []
total_sims = len(test_distances) * len(test_datarates)
current_sim = 0

for dist_m in test_distances:
    for dr in test_datarates:
        current_sim += 1
        print(f"\n[{current_sim}/{total_sims}] Distance={dist_m}m, DR={dr}")
        print("-" * 90)
        
        # Configuration pour cette simulation
        config = BASE_CONFIG.copy()
        config['distance_gtw'] = dist_m
        config['coding_rate'] = DR_CONFIG_MAP[dr]['cr']
        config['tx_power'] = DR_CONFIG_MAP[dr]['tx']
        config['bandwidth_khz'] = DR_CONFIG_MAP[dr]['bw_khz']
        
        print(f"  Config: CR={config['coding_rate']}, TX={config['tx_power']} dBm, BW={config['bandwidth_khz']} kHz")
        
        try:
            # LANCER LA SIMULATION 🎯
            print(f"  ▶️  Exécution...", end=" ", flush=True)
            start_time = time.time()
            
            sim = LR_FHSS_Simulation(config)
            sim.run()
            
            elapsed = time.time() - start_time
            print(f"✓ ({elapsed:.1f}s)")
            
            # Extraire les métriques
            total_sent = sim.total_sent
            successful_rx = sim.successful_rx
            pdr = successful_rx / total_sent if total_sent > 0 else 0
            
            # RSSI et SNR
            rssi_mean = sim.detailed_stats.avg_rssi_dbm if hasattr(sim, 'detailed_stats') else -120
            rssi_std = np.std([p.rssi_dbm for p in sim.simulated_packets if hasattr(p, 'rssi_dbm')]) if sim.simulated_packets else 0
            snr_mean = sim.detailed_stats.avg_snr_db if hasattr(sim, 'detailed_stats') else 0
            
            # BER moyen
            ber_values = [p.ber for p in sim.simulated_packets if hasattr(p, 'ber') and p.ber is not None]
            ber_mean = np.mean(ber_values) if ber_values else 0
            
            result = {
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
            }
            
            simulation_results.append(result)
            
            # Afficher les résultats
            print(f"\n  📊 Résultats:")
            print(f"     • Paquets: {total_sent} (Succès: {successful_rx}, Échecs: {total_sent - successful_rx})")
            print(f"     • PDR: {pdr*100:.1f}%")
            print(f"     • RSSI: {rssi_mean:.1f} dBm (σ={rssi_std:.1f})")
            print(f"     • SNR: {snr_mean:.1f} dB")
            print(f"     • BER: {ber_mean:.2e}")
            
        except Exception as e:
            print(f"❌ ERREUR: {e}")
            import traceback
            traceback.print_exc()
            continue

print("\n" + "=" * 90)
print("✅ Toutes les simulations terminées!")
print("=" * 90)

# ===== COMPARAISON ET STATISTIQUES =====
df_sim = pd.DataFrame(simulation_results)

print("\n📊 COMPARAISON GLOBALE PAR DISTANCE")
print("=" * 90)

for distance_m in test_distances:
    print(f"\n🏘️  Distance: {distance_m}m")
    print("-" * 90)
    
    # Données réelles
    real_subset = real_data_agg[(real_data_agg['Distance(m)'] == distance_m) & (real_data_agg['DR'] >= 8)]
    
    if len(real_subset) == 0:
        print(f"  ⚠️  Aucune donnée réelle pour cette distance")
        continue
    
    pdr_real = real_subset['PDR'].mean() * 100
    rssi_real = real_subset['Avg_RSSI(dBm)'].mean()
    pdr_std_real = real_subset['PDR'].std() * 100
    rssi_std_real = real_subset['Avg_RSSI(dBm)'].std()
    
    # Simulation
    sim_subset = df_sim[df_sim['distance_m'] == distance_m]
    if len(sim_subset) > 0:
        pdr_sim = sim_subset['pdr'].mean() * 100
        rssi_sim = sim_subset['rssi_mean_dbm'].mean()
        pdr_std_sim = sim_subset['pdr'].std() * 100
        rssi_std_sim = sim_subset['rssi_mean_dbm'].std()
        
        pdr_error = abs(pdr_real - pdr_sim)
        rssi_error = abs(rssi_real - rssi_sim)
        
        pdr_status = '✅ BON' if pdr_error < 15 else '⚠️  MOYEN' if pdr_error < 25 else '❌ MAUVAIS'
        rssi_status = '✅ EXCELLENT' if rssi_error < 2 else '✅ BON' if rssi_error < 5 else '⚠️  MOYEN' if rssi_error < 10 else '❌ MAUVAIS'
        
        print(f"\n  📈 PDR:")
        print(f"     Réel:       {pdr_real:.1f}% (±{pdr_std_real:.1f}%)")
        print(f"     Simulation: {pdr_sim:.1f}% (±{pdr_std_sim:.1f}%)")
        print(f"     Erreur:     {pdr_error:.1f}% {pdr_status}")
        
        print(f"\n  📡 RSSI:")
        print(f"     Réel:       {rssi_real:.1f} dBm (±{rssi_std_real:.1f} dB)")
        print(f"     Simulation: {rssi_sim:.1f} dBm (±{rssi_std_sim:.1f} dB)")
        print(f"     Erreur:     {rssi_error:.1f} dB {rssi_status}")

# ===== VISUALISATION =====
print("\n\n📊 Génération de la visualisation...")

# Dictionnaire de couleurs pour chaque DR
dr_colors = {
    8: '#1f77b4',   # Bleu
    9: '#ff7f0e',   # Orange
    10: '#2ca02c',  # Vert
    11: '#d62728',  # Rouge
}

# Fonction pour alléger une couleur (pour simulation)
def lighten_color(color_hex, factor=0.5):
    import matplotlib.colors as mcolors
    rgb = mcolors.hex2color(color_hex)
    return tuple(min(1, c + (1-c)*factor) for c in rgb)

# ===== 1. IMAGES SÉPARÉES PAR DR (une par DR) =====
print("\n1️⃣  Génération des images séparées par DR...")

for dr in test_datarates:
    fig, ax = plt.subplots(figsize=(14, 8))
    
    color_dark = dr_colors[dr]
    color_light = lighten_color(color_dark, 0.4)
    
    # Données réelles
    real_dr = real_data_agg[real_data_agg['DR'] == dr].sort_values('Distance(m)')
    sim_dr = df_sim[df_sim['datarate'] == dr].sort_values('distance_m')
    
    # Créer un axe secondaire pour RSSI
    ax_rssi = ax.twinx()
    
    # PDR (axe principal)
    if len(real_dr) > 0:
        ax.plot(real_dr['Distance(m)'].values, real_dr['PDR'].values * 100, 'o-',
               linewidth=1.8, markersize=7, label='PDR Réel', 
               color=color_dark, alpha=0.85)
    
    if len(sim_dr) > 0:
        ax.plot(sim_dr['distance_m'].values, sim_dr['pdr'].values * 100, 's--',
               linewidth=1.5, markersize=6, label='PDR Simulation', 
               color=color_light, alpha=0.75)
    
    # RSSI (axe secondaire)
    if len(real_dr) > 0:
        ax_rssi.plot(real_dr['Distance(m)'].values, real_dr['Avg_RSSI(dBm)'].values, '^-',
                    linewidth=1.8, markersize=7, label='RSSI Réel', 
                    color=color_dark, alpha=0.6)
    
    if len(sim_dr) > 0:
        ax_rssi.plot(sim_dr['distance_m'].values, sim_dr['rssi_mean_dbm'].values, 'D--',
                    linewidth=1.5, markersize=6, label='RSSI Simulation', 
                    color=color_light, alpha=0.55)
    
    # Configuration des axes
    ax.set_xlabel('Distance (m)', fontsize=12, fontweight='normal')
    ax.set_ylabel('PDR (%)', fontsize=12, fontweight='normal', color=color_dark)
    ax_rssi.set_ylabel('RSSI moyen (dBm)', fontsize=12, fontweight='normal', color=color_dark)
    
    # Titres et couleurs
    ax.set_title(f'DR{dr}: PDR et RSSI vs Distance', fontsize=13, fontweight='normal', pad=15)
    ax.tick_params(axis='y', labelcolor=color_dark, labelsize=9)
    ax_rssi.tick_params(axis='y', labelcolor=color_dark, labelsize=9)
    ax.tick_params(axis='x', labelsize=9)
    
    # Grille style cahier millimétré
    ax.grid(True, which='major', alpha=0.4, linestyle='-', linewidth=0.9, color='gray')
    ax.grid(True, which='minor', alpha=0.15, linestyle='-', linewidth=0.5, color='gray')
    ax.minorticks_on()
    ax.set_axisbelow(True)
    ax.set_ylim(0, 105)
    
    # Légendes
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_rssi.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    output_file = f'dr{dr}_pdr_rssi_vs_distance.png'
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"   ✅ {output_file}")
    plt.close(fig)

# ===== 2. IMAGE GLOBALE: PDR vs Distance (tous les DR) =====
print("\n2️⃣  Génération de l'image PDR globale...")

fig, ax = plt.subplots(figsize=(16, 9))

for dr in test_datarates:
    color_dark = dr_colors[dr]
    color_light = lighten_color(color_dark, 0.4)
    
    # Données réelles
    real_dr = real_data_agg[real_data_agg['DR'] == dr].sort_values('Distance(m)')
    if len(real_dr) > 0:
        ax.plot(real_dr['Distance(m)'].values, real_dr['PDR'].values * 100, 'o-',
               linewidth=1.8, markersize=7, label=f'DR{dr} (Réel)', 
               color=color_dark, alpha=0.85)
    
    # Données simulation
    sim_dr = df_sim[df_sim['datarate'] == dr].sort_values('distance_m')
    if len(sim_dr) > 0:
        ax.plot(sim_dr['distance_m'].values, sim_dr['pdr'].values * 100, 's--',
               linewidth=1.5, markersize=6, label=f'DR{dr} (Sim)', 
               color=color_light, alpha=0.75)

ax.set_xlabel('Distance (m)', fontsize=12, fontweight='normal')
ax.set_ylabel('PDR (%)', fontsize=12, fontweight='normal')
ax.set_title('PDR vs Distance - Tous les DR (DR8-11)', fontsize=13, fontweight='normal', pad=15)
ax.legend(fontsize=9, ncol=4, loc='best', framealpha=0.9)
# Grille style cahier millimétré
ax.grid(True, which='major', alpha=0.4, linestyle='-', linewidth=0.9, color='gray')
ax.grid(True, which='minor', alpha=0.15, linestyle='-', linewidth=0.5, color='gray')
ax.minorticks_on()
ax.set_axisbelow(True)
ax.set_ylim(0, 105)
ax.tick_params(axis='both', labelsize=9)

plt.tight_layout()
output_file = 'all_dr_pdr_vs_distance.png'
fig.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"   ✅ {output_file}")
plt.close(fig)

# ===== 3. IMAGE GLOBALE: RSSI vs Distance (tous les DR) =====
print("\n3️⃣  Génération de l'image RSSI globale...")

fig, ax = plt.subplots(figsize=(16, 9))

for dr in test_datarates:
    color_dark = dr_colors[dr]
    color_light = lighten_color(color_dark, 0.4)
    
    # Données réelles
    real_dr = real_data_agg[real_data_agg['DR'] == dr].sort_values('Distance(m)')
    if len(real_dr) > 0:
        ax.plot(real_dr['Distance(m)'].values, real_dr['Avg_RSSI(dBm)'].values, 'o-',
               linewidth=1.8, markersize=7, label=f'DR{dr} (Réel)', 
               color=color_dark, alpha=0.85)
    
    # Données simulation
    sim_dr = df_sim[df_sim['datarate'] == dr].sort_values('distance_m')
    if len(sim_dr) > 0:
        ax.plot(sim_dr['distance_m'].values, sim_dr['rssi_mean_dbm'].values, 's--',
               linewidth=1.5, markersize=6, label=f'DR{dr} (Sim)', 
               color=color_light, alpha=0.75)

ax.set_xlabel('Distance (m)', fontsize=12, fontweight='normal')
ax.set_ylabel('RSSI moyen (dBm)', fontsize=12, fontweight='normal')
ax.set_title('RSSI vs Distance - Tous les DR (DR8-11)', fontsize=13, fontweight='normal', pad=15)
ax.legend(fontsize=9, ncol=4, loc='best', framealpha=0.9)
# Grille style cahier millimétré
ax.grid(True, which='major', alpha=0.4, linestyle='-', linewidth=0.9, color='gray')
ax.grid(True, which='minor', alpha=0.15, linestyle='-', linewidth=0.5, color='gray')
ax.minorticks_on()
ax.set_axisbelow(True)
ax.tick_params(axis='both', labelsize=9)

plt.tight_layout()
output_file = 'all_dr_rssi_vs_distance.png'
fig.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"   ✅ {output_file}")
plt.close(fig)

# Sauvegarder les résultats
csv_output = 'simulation_lrfhss_all_distances.csv'
df_sim.to_csv(csv_output, index=False)
print(f"✅ Résultats CSV: {csv_output}")

print("\n" + "=" * 90)
print("✨ Comparaison LR-FHSS complète terminée!")
print("=" * 90)