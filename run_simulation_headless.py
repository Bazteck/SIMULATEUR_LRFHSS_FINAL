#!/usr/bin/env python3
"""
run_simulation_headless.py
Simulation LR-FHSS en mode headless (sans interface graphique)
Export direct des résultats en CSV
"""

import numpy as np
import pandas as pd
import time
import json
import os
import sys
import argparse
from datetime import datetime
import logging

# Configuration logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Importer la simulation
from simulation import LR_FHSS_Simulation, DQN_AVAILABLE, INTELLIGENT_SCHEDULER_AVAILABLE, ENERGY_MODULE_AVAILABLE
from channel import ShadowingManager, shadowing_manager


class HeadlessSimulation:
    """
    Simulation LR-FHSS sans interface graphique
    Export direct des résultats en CSV
    """
    
    def __init__(self, config: dict = None):
        """
        Initialise la simulation headless
        
        Args:
            config: Configuration de simulation (optionnel)
        """
        self.config = config or self._get_default_config()
        self.simulation = None
        self.results = {
            'packets': [],
            'metrics': {},
            'shadowing_stats': {},
            'energy_stats': {}
        }
        
        logger.info("=" * 70)
        logger.info("🚀 SIMULATION LR-FHSS HEADLESS INITIALISÉE")
        logger.info("=" * 70)
        logger.info(f"📊 Configuration:")
        for key, value in self.config.items():
            logger.info(f"   • {key}: {value}")
        logger.info("=" * 70)
    
    def _get_default_config(self) -> dict:
        """Configuration par défaut"""
        return {
            # Paramètres de base
            'simulation_duration': 3600,  # 1h en secondes
            'num_devices': 100,
            'distance_gtw': 5000,  # mètres - 5 km pour tester le modèle BER
            'tx_power': 14.0,  # dBm
            
            # Paramètres LR-FHSS
            'region': 'EU868',
            'coding_rate': '1/3',
            'bandwidth_khz': 136.71875,
            'payload_min': 25,
            'payload_max': 25,
            'tx_interval_min': 3600,  # 1h
            'tx_interval_max': 3600,
            
            # Canal RF
            'shadowing_std_db': 7.0,
            'path_loss_exponent': 2.7,
            'doppler_hz': 0.0,
            'multipath_enabled': True,
            'noise_figure_db': 6.0,
            
            # Énergie
            'pa_type': 'SX1261_LP',
            'battery_capacity_mah': 1000.0,
            'transmissions_per_day': 24,
            
            # Optimisations
            'enable_intelligent_scheduler': False,
            'enable_dqn': False,
            'dqn_model_name': None,
            'dqn_exploration': 0.0,
            
            # Seed pour reproductibilité
            'position_seed': 42,
            'shadowing_seed': 42
        }
    
    def generate_device_positions(self, num_devices: int, max_distance_m: float, seed: int = 42) -> list:
        """
        Génère les positions des devices de manière déterministe
        
        Returns:
            Liste de tuples (x, y) en mètres
        """
        np.random.seed(seed)
        
        positions = []
        angles = np.random.uniform(0, 2 * np.pi, num_devices)
        radii = max_distance_m * np.sqrt(np.random.uniform(0, 1, num_devices))
        
        for i in range(num_devices):
            x = radii[i] * np.cos(angles[i])
            y = radii[i] * np.sin(angles[i])
            positions.append((float(x), float(y)))
        
        # Statistiques
        distances = [np.sqrt(x**2 + y**2) for x, y in positions]
        logger.info(f"📐 Positions générées (seed={seed}):")
        logger.info(f"   • min: {min(distances):.1f}m")
        logger.info(f"   • max: {max(distances):.1f}m")
        logger.info(f"   • moy: {np.mean(distances):.1f}m")
        logger.info(f"   • σ: {np.std(distances):.1f}m")
        
        return positions
    
    def run(self) -> dict:
        """
        Exécute la simulation et retourne les résultats
        """
        logger.info("▶️ Démarrage de la simulation...")
        
        # Générer les positions
        positions = self.generate_device_positions(
            self.config['num_devices'],
            self.config['distance_gtw'],
            self.config.get('position_seed', 42)
        )
        
        # Ajouter les positions à la config
        self.config['device_positions'] = positions
        
        # Créer la simulation
        self.simulation = LR_FHSS_Simulation(self.config)
        
        # Démarrer la simulation (run() au lieu de start() pour exécution bloquante)
        start_time = time.time()
        self.simulation.run()
        
        # Fois d'exécution
        elapsed = time.time() - start_time
        logger.info(f"✅ Simulation terminée en {elapsed:.1f}s")
        
        # Récupérer les résultats
        self._collect_results()
        
        return self.results
    
    def _collect_results(self):
        """Collecte tous les résultats de la simulation"""
        
        # Métriques générales
        self.results['metrics'] = {
            'total_packets': self.simulation.total_sent,
            'successful_packets': self.simulation.successful_rx,
            'failed_packets': self.simulation.total_sent - self.simulation.successful_rx,
            'success_rate': self.simulation.success_rate,
            'collisions': self.simulation.collisions,
            'collision_rate': self.simulation.collision_rate,
            'simulated_time_s': self.simulation.simulated_time,
            'toa_brut_total_s': self.simulation.toa_brut_total,
            'toa_net_total_s': self.simulation.toa_net_total,
            'spectral_efficiency': self.simulation.spectral_efficiency,
            'occupation_rate': self.simulation.occupation_rate,
            'avg_rssi_dbm': self.simulation.avg_rssi_dbm,
            'avg_snr_db': self.simulation.avg_snr_db,
            'avg_ber': self.simulation.avg_ber,
        }
        
        # Statistiques DQN
        if self.config.get('enable_dqn', False):
            self.results['metrics'].update({
                'dqn_decisions': self.simulation.dqn_decisions,
                'dqn_successes': self.simulation.dqn_successes,
                'dqn_success_rate': self.simulation.dqn_success_rate,
                'dqn_avg_power_saved': self.simulation.dqn_avg_power_saved,
            })
        
        # Statistiques scheduler
        if self.config.get('enable_intelligent_scheduler', False):
            self.results['metrics'].update({
                'scheduler_decisions': self.simulation.detailed_stats.scheduler_decisions,
                'scheduler_delays': self.simulation.detailed_stats.scheduler_delays,
                'scheduler_delays_sum': self.simulation.detailed_stats.scheduler_delays_sum,
                'scheduler_freq_shifts': self.simulation.detailed_stats.scheduler_freq_shifts,
                'scheduler_power_boosts': self.simulation.detailed_stats.scheduler_power_boosts,
            })
        
        # Statistiques énergie
        if ENERGY_MODULE_AVAILABLE and hasattr(self.simulation, 'energy_analyzer'):
            energy_stats = self.simulation.energy_analyzer.energy_stats
            self.results['energy_stats'] = {
                'total_energy_j': energy_stats.get('total_energy_j', 0.0),
                'energy_successful_j': energy_stats.get('energy_successful_j', 0.0),
                'energy_failed_j': energy_stats.get('energy_failed_j', 0.0),
                'avg_energy_per_packet_j': energy_stats.get('avg_energy_per_packet_j', 0.0),
                'avg_current_ma': energy_stats.get('avg_current_ma', 0.0),
                'battery_life_years': energy_stats.get('battery_life_years', 0.0),
                'energy_per_bit_j': energy_stats.get('energy_per_bit_j', 0.0),
                'efficiency_ratio': energy_stats.get('efficiency_ratio', 0.0),
                'packets_analyzed': energy_stats.get('packets_analyzed', 0),
            }
        
        # Paquets avec shadowing
        self.results['packets'] = self._extract_packet_data()
        
        # Statistiques shadowing
        self.results['shadowing_stats'] = self._calculate_shadowing_stats()
    
    def _extract_packet_data(self) -> list:
        """Extrait les données des paquets pour export CSV"""
        packet_data = []
        
        for packet in self.simulation.simulated_packets:
            # Récupérer position
            position = getattr(packet, 'position', (0, 0))
            if position == (0, 0) and packet.device_id in self.simulation.devices_state:
                position = self.simulation.devices_state[packet.device_id].get('position', (0, 0))
            
            # Distance
            distance_km = getattr(packet, 'distance_km', 0)
            if distance_km == 0 and packet.device_id in self.simulation.devices_state:
                distance_km = self.simulation.devices_state[packet.device_id].get('distance_km', 0)
            
            # Shadowing
            shadowing_db = getattr(packet, 'shadowing_db', 0)
            if shadowing_db == 0 and hasattr(packet, 'rssi_dbm') and hasattr(packet, 'tx_power_dbm'):
                path_loss = getattr(packet, 'path_loss_db', 0)
                shadowing_db = packet.rssi_dbm - packet.tx_power_dbm + path_loss
            
            row = {
                'packet_id': getattr(packet, 'packet_id', ''),
                'device_id': getattr(packet, 'device_id', ''),
                'start_time': getattr(packet, 'start_time', 0),
                'end_time': getattr(packet, 'end_time', 0),
                'toa_ms': getattr(packet, 'toa_ms', 0),
                'frequency_mhz': getattr(packet, 'frequency_mhz', 0),
                'tx_power_dbm': getattr(packet, 'tx_power_dbm', 0),
                'dr': getattr(packet, 'dr', 0),
                'cr': getattr(packet, 'cr', ''),
                'bw_khz': getattr(packet, 'bw_khz', 0),
                'payload_bytes': getattr(packet, 'payload_bytes', 0),
                
                # Métriques RF
                'rssi_dbm': getattr(packet, 'rssi_dbm', -120),
                'path_loss_db': getattr(packet, 'path_loss_db', 0),
                'shadowing_db': shadowing_db,
                'snr_db': getattr(packet, 'snr_db', 0),
                'ber': getattr(packet, 'ber', 0),
                
                # Position
                'position_x_m': position[0],
                'position_y_m': position[1],
                'distance_km': distance_km,
                'distance_m': distance_km * 1000,
                
                # Résultat
                'success': getattr(packet, 'success', False),
                'collision': getattr(packet, 'collision', False),
                'fec_recovered': getattr(packet, 'fec_recovered', False),
                'failure_reason': getattr(packet, 'failure_reason', ''),
                
                # DQN
                'dqn_applied': getattr(packet, 'dqn_applied', False),
                'dqn_dr': getattr(packet, 'dqn_dr', None),
                'dqn_power_dbm': getattr(packet, 'dqn_power', None),
                
                # Scheduler
                'scheduler_applied': getattr(packet, 'scheduler_applied', False),
                'scheduler_strategy': getattr(packet, 'scheduler_strategy', ''),
                'scheduler_delay_s': getattr(packet, 'scheduler_delay', 0),
                
                # Énergie
                'pa_type': getattr(packet, 'pa_type_used', self.config.get('pa_type', 'SX1261_LP')),
                'energy_j': getattr(packet, 'energy_metrics', {}).get('total_energy_j', 0) if hasattr(packet, 'energy_metrics') else 0,
            }
            
            # Collisions
            collision_details = getattr(packet, 'collision_details', [])
            row['collision_count'] = len(collision_details)
            row['header_collisions'] = sum(1 for c in collision_details if c.fragment1.fragment_type == 'header')
            row['payload_collisions'] = sum(1 for c in collision_details if c.fragment1.fragment_type == 'payload')
            row['capture_effects'] = sum(1 for c in collision_details if getattr(c, 'capture_effect', False))
            
            packet_data.append(row)
        
        logger.info(f"📦 {len(packet_data)} paquets extraits")
        return packet_data
    
    def _calculate_shadowing_stats(self) -> dict:
        """Calcule les statistiques de shadowing"""
        if not self.results['packets']:
            return {}
        
        shadowings = [p['shadowing_db'] for p in self.results['packets'] if p['shadowing_db'] != 0]
        if not shadowings:
            return {'error': 'Aucune donnée de shadowing'}
        
        stats = {
            'count': len(shadowings),
            'mean_db': float(np.mean(shadowings)),
            'std_db': float(np.std(shadowings)),
            'min_db': float(np.min(shadowings)),
            'max_db': float(np.max(shadowings)),
            'q1_db': float(np.percentile(shadowings, 25)),
            'median_db': float(np.percentile(shadowings, 50)),
            'q3_db': float(np.percentile(shadowings, 75)),
            'config_std_db': self.config.get('shadowing_std_db', 7.0),
        }
        
        # Test de normalité (si assez d'échantillons)
        if len(shadowings) >= 30:
            from scipy import stats as scipy_stats
            statistic, p_value = scipy_stats.shapiro(shadowings[:5000])  # Limiter à 5000
            stats['normality_test'] = {
                'shapiro_statistic': float(statistic),
                'shapiro_p_value': float(p_value),
                'is_normal': p_value > 0.05
            }
        
        return stats
    
    def export_csv(self, filename: str = None) -> str:
        """
        Exporte les résultats en CSV
        
        Args:
            filename: Nom du fichier (optionnel)
        
        Returns:
            Chemin du fichier créé
        """
        if not self.results['packets']:
            logger.warning("⚠️ Aucun paquet à exporter")
            return None
        
        # Générer nom de fichier
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"lrfhss_headless_{timestamp}.csv"
        
        # Créer DataFrame
        df = pd.DataFrame(self.results['packets'])
        
        # Trier par temps
        df = df.sort_values('start_time')
        
        # Sauvegarder
        df.to_csv(filename, index=False, encoding='utf-8')
        
        logger.info(f"💾 CSV exporté: {filename}")
        logger.info(f"   • {len(df)} lignes")
        logger.info(f"   • {len(df.columns)} colonnes")
        
        # Afficher les premières lignes
        print("\n📋 Aperçu des données:")
        print(df[['device_id', 'distance_km', 'shadowing_db', 'rssi_dbm', 'success']].head(10).to_string())
        
        return filename
    
    def export_json(self, filename: str = None) -> str:
        """
        Exporte les métriques en JSON
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"lrfhss_metrics_{timestamp}.json"
        
        # Préparer les données
        export_data = {
            'config': self.config,
            'metrics': self.results['metrics'],
            'shadowing_stats': self.results['shadowing_stats'],
            'energy_stats': self.results['energy_stats'],
            'timestamp': datetime.now().isoformat(),
            'packet_count': len(self.results['packets'])
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"💾 JSON exporté: {filename}")
        return filename
    
    def print_summary(self):
        """Affiche un résumé des résultats"""
        print("\n" + "=" * 80)
        print("📊 RÉSUMÉ DE LA SIMULATION")
        print("=" * 80)
        
        m = self.results['metrics']
        print(f"\n📈 MÉTRIQUES GÉNÉRALES:")
        print(f"   • Paquets totaux: {m.get('total_packets', 0):,}")
        print(f"   • Succès: {m.get('successful_packets', 0):,} ({m.get('success_rate', 0):.1f}%)")
        print(f"   • Échecs: {m.get('failed_packets', 0):,}")
        print(f"   • Collisions: {m.get('collisions', 0):,} ({m.get('collision_rate', 0):.1f}%)")
        
        print(f"\n⏱️  TEMPS:")
        print(f"   • Simulé: {m.get('simulated_time_s', 0):.1f}s")
        print(f"   • ToA Brut: {m.get('toa_brut_total_s', 0):.2f}s")
        print(f"   • ToA Net: {m.get('toa_net_total_s', 0):.2f}s")
        print(f"   • Efficacité spectrale: {m.get('spectral_efficiency', 0):.1f}%")
        
        print(f"\n📡 CANAL RF:")
        print(f"   • RSSI moyen: {m.get('avg_rssi_dbm', 0):.1f} dBm")
        print(f"   • SNR moyen: {m.get('avg_snr_db', 0):.1f} dB")
        print(f"   • BER moyen: {m.get('avg_ber', 0):.2e}")
        
        if self.results['shadowing_stats']:
            s = self.results['shadowing_stats']
            print(f"\n🌫️  SHADOWING:")
            print(f"   • Moyen: {s.get('mean_db', 0):.2f} dB")
            print(f"   • Écart-type: {s.get('std_db', 0):.2f} dB (config: {s.get('config_std_db', 0):.1f} dB)")
            print(f"   • Min/Max: [{s.get('min_db', 0):.2f}, {s.get('max_db', 0):.2f}] dB")
            
            if 'normality_test' in s:
                nt = s['normality_test']
                status = "✅" if nt.get('is_normal') else "⚠️"
                print(f"   • Test normalité: {status} (p={nt.get('shapiro_p_value', 0):.4f})")
        
        if self.results['energy_stats']:
            e = self.results['energy_stats']
            print(f"\n🔋 ÉNERGIE:")
            print(f"   • Énergie totale: {e.get('total_energy_j', 0):.3f} J")
            print(f"   • Moyenne/paquet: {e.get('avg_energy_per_packet_j', 0)*1000:.2f} mJ")
            print(f"   • Efficacité: {e.get('efficiency_ratio', 0):.1f}%")
            print(f"   • Durée vie batterie: {e.get('battery_life_years', 0):.1f} ans")
        
        print("\n" + "=" * 80)


def run_multiple_simulations(configs: list, base_filename: str = "simulation"):
    """
    Exécute plusieurs simulations avec différentes configurations
    Utile pour des études paramétriques
    """
    all_results = []
    
    for i, config in enumerate(configs):
        print(f"\n{'='*80}")
        print(f"Simulation {i+1}/{len(configs)}")
        print(f"{'='*80}")
        
        sim = HeadlessSimulation(config)
        sim.run()
        
        # Export avec index
        csv_file = sim.export_csv(f"{base_filename}_{i+1}.csv")
        json_file = sim.export_json(f"{base_filename}_{i+1}.json")
        
        all_results.append({
            'config': config,
            'metrics': sim.results['metrics'],
            'csv_file': csv_file,
            'json_file': json_file
        })
    
    # Résumé comparatif
    print("\n" + "="*80)
    print("📊 RÉSUMÉ COMPARATIF")
    print("="*80)
    
    df = pd.DataFrame([r['metrics'] for r in all_results])
    print(df[['total_packets', 'success_rate', 'collision_rate', 'avg_rssi_dbm']].to_string())
    
    return all_results


def main():
    """Fonction principale avec arguments en ligne de commande"""
    parser = argparse.ArgumentParser(description='Simulation LR-FHSS Headless')
    
    # Paramètres principaux
    parser.add_argument('--duration', type=int, default=86400, help='Durée simulation (s)')
    parser.add_argument('--devices', type=int, default=100, help='Nombre de devices')
    parser.add_argument('--distance', type=float, default=15000, help='Distance max (m)')
    parser.add_argument('--tx-power', type=float, default=14.0, help='Puissance TX (dBm)')
    
    # LR-FHSS
    parser.add_argument('--dr', type=int, choices=[8,9,10,11], default=8, help='Data Rate')
    parser.add_argument('--payload', type=int, default=25, help='Payload size (bytes)')
    parser.add_argument('--interval', type=float, default=3600, help='Interval TX (s)')
    
    # Canal
    parser.add_argument('--shadowing-std', type=float, default=7.0, help='Shadowing std (dB)')
    parser.add_argument('--path-loss-exp', type=float, default=2.7, help='Path loss exponent')
    
    # Sortie
    parser.add_argument('--output', type=str, default='simulation_results.csv', help='Fichier sortie CSV')
    parser.add_argument('--no-summary', action='store_true', help='Ne pas afficher le résumé')
    
    args = parser.parse_args()
    
    # Convertir DR en CR et BW
    dr_to_params = {
        8: {'cr': '1/3', 'bw': 136.71875},
        9: {'cr': '2/3', 'bw': 136.71875},
        10: {'cr': '1/3', 'bw': 335.9375},
        11: {'cr': '2/3', 'bw': 335.9375},
    }
    params = dr_to_params[args.dr]
    
    config = {
        'simulation_duration': args.duration,
        'num_devices': args.devices,
        'distance_gtw': args.distance,
        'tx_power': args.tx_power,
        'coding_rate': params['cr'],
        'bandwidth_khz': params['bw'],
        'payload_min': args.payload,
        'payload_max': args.payload,
        'tx_interval_min': args.interval,
        'tx_interval_max': args.interval,
        'shadowing_std_db': args.shadowing_std,
        'path_loss_exponent': args.path_loss_exp,
        'position_seed': 42,
        'shadowing_seed': 42,
    }
    
    # Exécuter
    sim = HeadlessSimulation(config)
    sim.run()
    
    if not args.no_summary:
        sim.print_summary()
    
    sim.export_csv(args.output)


if __name__ == "__main__":
    # Exemple d'utilisation
    print("🧪 SIMULATION LR-FHSS HEADLESS")
    print("=" * 70)
    print(f"📦 Modules disponibles:")
    print(f"   • DQN: {'✅' if DQN_AVAILABLE else '❌'}")
    print(f"   • Scheduler: {'✅' if INTELLIGENT_SCHEDULER_AVAILABLE else '❌'}")
    print(f"   • Énergie: {'✅' if ENERGY_MODULE_AVAILABLE else '❌'}")
    print("=" * 70)
    
    # Mode ligne de commande
    if len(sys.argv) > 1:
        main()
    else:
        # Mode démo avec configuration simple
        config = {
            'simulation_duration': 3600,  # 1h
            'num_devices': 50,
            'distance_gtw': 5000,  # 5km pour voir variation BER
            'tx_power': 14.0,
            'coding_rate': '1/3',
            'bandwidth_khz': 136.71875,
            'payload_min': 25,
            'payload_max': 25,
            'tx_interval_min': 60,  # Transmission chaque minute
            'tx_interval_max': 120,  # 1-2 minutes
            'shadowing_std_db': 7.0,
            'path_loss_exponent': 2,  # EU868
            'enable_dqn': False,
            'enable_intelligent_scheduler': False,
        }
        
        sim = HeadlessSimulation(config)
        sim.run()
        sim.print_summary()
        sim.export_csv("demo_results.csv")