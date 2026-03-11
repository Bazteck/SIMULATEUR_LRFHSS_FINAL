#!/usr/bin/env python3
"""
simulation.py
Logique de simulation LR-FHSS avec DQN, Scheduler Intelligent et Énergie
"""

import numpy as np
import pandas as pd
import threading
import math
import heapq
from queue import Queue, Empty
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import time
import logging
import random

from lr_fhss import (
    calculate_noise_power,
    TransmissionFragment, SimulatedPacket, FragmentCollisionResult,
    generate_lrfhss_fragments,
    check_collision,
    evaluate_transmission,
    is_significant_collision
)

# Importer les functions correctes du channel (NEW CHANNEL MODEL)
from channel import calculate_rssi, calculate_rssi_with_details, calculate_snr

# Importer DQN
try:
    from integrated_ddqn import DQNManager
    DQN_AVAILABLE = True
except ImportError as e:
    logging.warning(f"❌ Module DQN non disponible: {e}")
    DQN_AVAILABLE = False
    
    class DQNManager:
        def __init__(self, simulation):
            self.simulation = simulation
            self.enabled = False
            self.stats = {}
            
        def initialize(self, *args, **kwargs):
            return False
        
        def get_recommendation(self, *args, **kwargs):
            return None
        
        def record_feedback(self, *args, **kwargs):
            pass
        
        def get_stats(self):
            return {}
        
        def save_model(self, *args, **kwargs):
            return None
        
        def reset_stats(self):
            pass

# Importer scheduler intelligent
try:
    from smart_scheduler import IntelligentScheduler
    INTELLIGENT_SCHEDULER_AVAILABLE = True
except ImportError as e:
    INTELLIGENT_SCHEDULER_AVAILABLE = False
    
    class IntelligentScheduler:
        def __init__(self, **kwargs):
            self.is_loaded = False
            
        def schedule_transmission(self, *args, **kwargs):
            return {'start_time': 0.0, 'strategy': 'IMMEDIATE', 'adjusted_fragments': []}
        
        def cleanup_old_fragments(self, *args):
            pass
        
        def get_network_metrics(self):
            return {}

# Importer module énergie
try:
    from energy import EnergyConsumptionModel, LR_FHSS_EnergyAnalyzer
    ENERGY_MODULE_AVAILABLE = True
except ImportError as e:
    ENERGY_MODULE_AVAILABLE = False
    
    class EnergyConsumptionModel:
        @classmethod
        def calculate_energy_joules(cls, *args, **kwargs):
            return {
                'tx_current_ma': 0.0,
                'total_energy_j': 0.0,
                'energy_per_bit_j': 0.0,
                'battery_life_years': 0.0,
                'pa_type': 'SX1261_LP'
            }
        
        @classmethod
        def calculate_battery_life_joules(cls, *args, **kwargs):
            return 0.0
    
    class LR_FHSS_EnergyAnalyzer:
        def __init__(self, simulation):
            self.simulation = simulation
            self.energy_stats = {
                'total_energy_j': 0.0,
                'avg_energy_per_packet_j': 0.0,
                'avg_current_ma': 0.0,
                'battery_life_years': 0.0,
                'energy_per_bit_j': 0.0,
                'packets_analyzed': 0,
                'pa_type': 'SX1261_LP',
                'daily_energy_j': 0.0
            }
        
        def analyze_packet_energy(self, packet):
            return None
        
        def get_energy_report(self):
            return "⚠️ Module énergie non disponible"
        
        def reset_stats(self):
            pass

# Importer les modules conformes
from config import LR_FHSS_Config, Region

# Importer le module de gestion des centres de fréquence
from frequency_center import (
    get_base_frequency_for_transmission,
    get_frequency_centers,
    LR_FHSS_FREQUENCY_CENTERS
)

# Configuration logging
logger = logging.getLogger(__name__)


class DetailedStatistics:
    """Classe pour stocker des statistiques détaillées"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.total_packets = 0
        self.successful_packets = 0
        self.failed_packets = 0
        
        # Détails des échecs
        self.failures_rssi = 0
        self.failures_snr = 0
        self.failures_demod = 0
        self.failures_crc = 0
        self.failures_fec = 0
        self.failures_collision = 0
        
        # Détails collisions
        self.collisions_detected = 0
        self.collisions_header = 0
        self.collisions_payload = 0
        self.collisions_mixed = 0
        self.capture_effects = 0
        self.collisions_causing_failure = 0
        
        # Métriques RF
        self.avg_rssi_dbm = 0.0
        self.avg_snr_db = 0.0
        self.avg_ber = 0.0
        self.min_rssi_dbm = 0.0
        self.max_rssi_dbm = 0.0
        self.min_snr_db = 0.0
        self.max_snr_db = 0.0
        
        # Timing
        self.total_toa_ms = 0.0
        self.successful_toa_ms = 0.0
        self.spectral_efficiency = 0.0
        self.airtime_percentage = 0.0
        
        # Statistiques scheduler
        self.scheduler_decisions = 0
        self.scheduler_delays = 0
        self.scheduler_delays_sum = 0.0
        self.scheduler_freq_shifts = 0
        self.scheduler_power_boosts = 0
        self.scheduler_collisions_prevented = 0
        self.scheduler_strategies = defaultdict(int)
        
        # Statistiques DQN
        self.dqn_decisions = 0
        self.dqn_successes = 0
        self.dqn_power_saved = 0.0
        self.dqn_recommendations = {
            'dr8': 0, 'dr9': 0, 'dr10': 0, 'dr11': 0,
            'power_low': 0, 'power_medium': 0, 'power_high': 0
        }
    
    def add_scheduler_decision(self, strategy: str, delay_applied: float, 
                              freq_shift_applied: float, power_boost_db: float = 0):
        """Ajoute une décision du scheduler"""
        self.scheduler_decisions += 1
        self.scheduler_strategies[strategy] += 1
        
        if delay_applied > 0.01:
            self.scheduler_delays += 1
            self.scheduler_delays_sum += delay_applied
            
        if abs(freq_shift_applied) > 0.1:
            self.scheduler_freq_shifts += 1
            
        if power_boost_db > 0.5:
            self.scheduler_power_boosts += 1
    
    def add_dqn_decision(self, dr: int, power: float, success: bool):
        """Ajoute une décision DQN"""
        self.dqn_decisions += 1
        
        if success:
            self.dqn_successes += 1
        
        # Catégoriser DR
        dr_key = f'dr{dr}'
        if dr_key in self.dqn_recommendations:
            self.dqn_recommendations[dr_key] += 1
        
        # Catégoriser puissance
        if power < 8.0:
            self.dqn_recommendations['power_low'] += 1
        elif power < 12.0:
            self.dqn_recommendations['power_medium'] += 1
        else:
            self.dqn_recommendations['power_high'] += 1
    
    def add_packet_result(self, packet, success: bool, failure_reason: str = None):
        """Ajoute le résultat d'un paquet"""
        self.total_packets += 1
        
        if success:
            self.successful_packets += 1
        else:
            self.failed_packets += 1
            
            if failure_reason:
                if 'rssi' in failure_reason.lower():
                    self.failures_rssi += 1
                elif 'snr' in failure_reason.lower():
                    self.failures_snr += 1
                elif 'demod' in failure_reason.lower():
                    self.failures_demod += 1
                elif 'crc' in failure_reason.lower():
                    self.failures_crc += 1
                elif 'fec' in failure_reason.lower():
                    self.failures_fec += 1
                elif 'collision' in failure_reason.lower():
                    self.failures_collision += 1
            
            if getattr(packet, 'collision', False):
                self.collisions_causing_failure += 1
        
        # Mettre à jour les métriques RF
        if success:
            self._update_rf_metrics(packet)
    
    def _update_rf_metrics(self, packet):
        """Met à jour les métriques RF"""
        if self.successful_packets == 1:
            self.min_rssi_dbm = packet.rssi_dbm
            self.max_rssi_dbm = packet.rssi_dbm
            self.min_snr_db = packet.snr_db
            self.max_snr_db = packet.snr_db
            self.avg_rssi_dbm = packet.rssi_dbm
            self.avg_snr_db = packet.snr_db
            self.avg_ber = packet.ber
        else:
            self.min_rssi_dbm = min(self.min_rssi_dbm, packet.rssi_dbm)
            self.max_rssi_dbm = max(self.max_rssi_dbm, packet.rssi_dbm)
            self.min_snr_db = min(self.min_snr_db, packet.snr_db)
            self.max_snr_db = max(self.max_snr_db, packet.snr_db)
            
            prev_count = self.successful_packets - 1
            self.avg_rssi_dbm = (self.avg_rssi_dbm * prev_count + packet.rssi_dbm) / self.successful_packets
            self.avg_snr_db = (self.avg_snr_db * prev_count + packet.snr_db) / self.successful_packets
            self.avg_ber = (self.avg_ber * prev_count + packet.ber) / self.successful_packets


class LR_FHSS_Simulation:
    """Classe principale de simulation LR-FHSS avec DQN, Scheduler et Énergie"""
    
    def __init__(self, config: Dict):
        """
        Initialise la simulation avec les paramètres de configuration.
        
        Args:
            config: Dictionnaire de configuration
        """
        self.config = config
        
        # État simulation
        self.is_running = False
        self.simulation_thread = None
        self.metric_queue = Queue()
        self.log_queue = Queue()
        
        # Initialiser les composants
        self._initialize_components()
        self._initialize_state()
        
        logger.info("Simulation LR-FHSS initialisée")
    
    def _initialize_components(self):
        """Initialise tous les composants de simulation"""
        # Configuration régionale
        region = Region.EU868 if self.config.get('region', 'EU868') == 'EU868' else Region.US915
        self.region_config = LR_FHSS_Config.REGIONAL_CONFIGS[region]
        
        # Channel model calculations now use calculate_rssi_with_shadowing()
        # and calculate_path_loss_db() directly (see lr_fhss._evaluate_lrfhss_without_collisions)
        
        # Initialiser le DQN Manager
        self.dqn_manager = DQNManager(self)
        
        # Initialiser le Scheduler Intelligent
        self.scheduler = None
        if self.config.get('enable_intelligent_scheduler', False) and INTELLIGENT_SCHEDULER_AVAILABLE:
            self._initialize_intelligent_scheduler()
        # Gestion des centres de fréquence LR-FHSS
        self.frequency_center_usage = {}  # Tracker pour load balancing
        self.frequency_selection_method = 'deterministic'  # Méthode de sélection
        # Initialiser l'analyseur énergétique
        if ENERGY_MODULE_AVAILABLE:
            self.energy_analyzer = LR_FHSS_EnergyAnalyzer(self)
        else:
            self.energy_analyzer = LR_FHSS_EnergyAnalyzer(self)
        
        # Initialiser les statistiques
        self.detailed_stats = DetailedStatistics()
        
        # État simulation
        self.devices_state = {}
        self.active_fragments = []
        self.collision_events = []
        self.simulated_packets = []
        self.simulated_time = 0.0
        self.transmission_counters = {}
        
        # Structures de collision
        self._initialize_collision_detection()
        self._reset_statistics()
    
    def _initialize_intelligent_scheduler(self):
        """Initialise le scheduler intelligent"""
        try:
            self.scheduler = IntelligentScheduler(
                time_resolution_ms=1.0,
                prediction_horizon_s=5.0
            )
            logger.info("✅ Scheduler intelligent initialisé")
        except Exception as e:
            logger.error(f"❌ Erreur initialisation scheduler: {e}")
            self.scheduler = None
    
    def _initialize_collision_detection(self):
        """Initialise les structures pour la détection de collisions"""
        self.active_fragments_lock = threading.RLock()
        self.stats_lock = threading.RLock()
        self.event_lock = threading.RLock()
        
        self.active_fragments = []
        self.collision_events = []
        self.event_queue = []
        
        self.collision_pairs = set()
        self.collisions_by_type = {
            'header': 0,
            'payload': 0,
            'mixed': 0
        }
        
        self.header_collisions = 0
        self.payload_collisions = 0
        self.mixed_collisions = 0
        self.capture_effect_count = 0
    
    def _reset_statistics(self):
        """Réinitialise toutes les statistiques"""
        with self.stats_lock:
            self.total_sent = 0
            self.successful_rx = 0
            self.collisions = 0
            self.demod_failed = 0
            self.fec_failed = 0
            self.fec_recovered = 0
            self.failures_rssi = 0
            self.success_rate = 0.0
            self.collision_rate = 0.0
            
            # Métriques RF
            self.avg_rssi_dbm = -120.0
            self.avg_snr_db = 0.0
            self.avg_ber = 0.0
            
            # Métriques ToA
            self._last_toa_count = 0
            self.toa_brut_total = 0.0
            self.toa_net_total = 0.0
            self.spectral_efficiency = 0.0
            self.occupation_rate = 0.0
            
            # Statistiques DQN
            self.dqn_decisions = 0
            self.dqn_successes = 0
            self.dqn_power_saved = 0.0
            self.dqn_success_rate = 0.0
            self.dqn_avg_power_saved = 0.0
            self.dqn_decision_history = []
            
            # Réinitialiser les statistiques détaillées
            self.detailed_stats.reset()

    def _generate_default_positions(self, num_devices, seed=42):
        """Génère des positions par défaut avec un seed spécifique - TOUS À LA MÊME DISTANCE"""
        original_state = np.random.get_state()
        np.random.seed(seed)
        
        positions = []
        distance_gtw = self.config.get('distance_gtw', 10000)
        
        for i in range(num_devices):
            angle = np.random.uniform(0, 2 * np.pi)
            radius = distance_gtw  # ✅ Distance EXACTE pour tous les appareils
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            positions.append((x, y))
        
        np.random.set_state(original_state)
        
        # Logger les statistiques
        if positions:
            distances = [np.sqrt(x**2 + y**2) for x, y in positions]
            logger.info(f"[SIMULATION] Positions générées (seed={seed}): TOUS À {distance_gtw:.1f}m (min={min(distances):.1f}m, max={max(distances):.1f}m, moy={np.mean(distances):.1f}m)")
        
        return positions

    def _initialize_state(self):
        """Initialise l'état pour la simulation avec distances COHÉRENTES"""
        self.devices_state = {}
        self._global_event_counter = 0
        
        num_devices = self.config.get('num_devices', 1000)
        tx_interval_min = self.config.get('tx_interval_min', 3600)
        tx_interval_max = self.config.get('tx_interval_max', 3600)
        
        # PRIORITÉ 1: Utiliser les positions fournies dans la configuration
        device_positions = self.config.get('device_positions', [])
        position_seed = self.config.get('position_seed', 42)
        
        # Log détaillé pour débogage
        logger.info(f"[SIMULATION] Configuration reçue - Num devices: {num_devices}")
        logger.info(f"[SIMULATION] Position seed dans config: {position_seed}")
        
        if device_positions:
            # VÉRIFIER QUE LE NOMBRE DE POSITIONS CORRESPOND AU NOMBRE DE DEVICES
            if len(device_positions) == num_devices:
                logger.info(f"[SIMULATION] Positions fournies dans config: {len(device_positions)} positions")
                
                # Calculer et logger les distances
                distances = []
                for x, y in device_positions:
                    dist_m = np.sqrt(x**2 + y**2)
                    distances.append(dist_m / 1000.0)  # en km
                
                if distances:
                    logger.info(f"[SIMULATION] Distances des positions config: min={min(distances):.3f}km, max={max(distances):.3f}km, moy={np.mean(distances):.3f}km")
            else:
                logger.warning(f"[SIMULATION] Incohérence: {len(device_positions)} positions pour {num_devices} devices")
                # Générer des positions par défaut
                device_positions = self._generate_default_positions(num_devices, position_seed)
        else:
            # Générer des positions par défaut avec le seed fourni
            logger.info(f"[SIMULATION] Aucune position fournie, génération avec seed: {position_seed}")
            device_positions = self._generate_default_positions(num_devices, position_seed)
        
        # CALCULER ET LOGGER LES DISTANCES FINALES
        final_distances = []
        if device_positions:
            for x, y in device_positions:
                dist_m = np.sqrt(x**2 + y**2)
                final_distances.append(dist_m / 1000.0)
            
            if final_distances:
                logger.info(f"[SIMULATION] Distances finales: min={min(final_distances):.3f}km, max={max(final_distances):.3f}km, moy={np.mean(final_distances):.3f}km")
        
        # Créer l'état des devices
        for device_id in range(num_devices):
            device_id_str = f"Dev-{device_id:04d}"
            
            # Fréquence aléatoire
            frequency = np.random.choice([868.1, 868.3, 868.5])  # Plage de fréquence LR-FHSS
            tx_interval = np.random.uniform(tx_interval_min, tx_interval_max)
            next_tx_time = np.random.uniform(0, min(tx_interval, 
                                                    self.config.get('simulation_duration', 7200)/10))
            
            # Calculer distance (si positions disponibles)
            distance_km = 0.0
            device_position = (0, 0)
            if device_id < len(device_positions):
                x, y = device_positions[device_id]
                device_position = (x, y)
                distance_m = np.sqrt(x**2 + y**2)
                distance_km = distance_m / 1000.0  # Convertir en km
            
            # DEBUG: Vérifier la distance pour chaque device (premiers seulement)
            if device_id < 5:  # Logger seulement les 5 premiers pour éviter le spam
                logger.info(f"[SIMULATION] Device {device_id_str}: position=({x:.1f}m, {y:.1f}m), distance={distance_m:.1f}m = {distance_km:.3f}km")
            
            self.devices_state[device_id_str] = {
                'device_id': device_id_str,
                'next_tx_time': next_tx_time,
                'tx_interval': tx_interval,
                'tx_count': 0,
                'success_count': 0,
                'collision_count': 0,
                'frequency_mhz': frequency,
                'last_frequency': frequency,
                'success_history': [],
                'success_rate': 0.5,
                'total_tx': 0,
                'successful_tx': 0,
                'distance_km': distance_km,  # ← DISTANCE EN KM (IMPORTANT POUR DQN)
                'position': device_position,  # Position (x, y) en mètres
                'battery': np.random.beta(2, 1.5),
                'last_rssi': np.random.uniform(-120, -80),
                'last_snr': np.random.uniform(-10, 10),
                'last_dr': 8,
                'last_tx_power': self.config.get('tx_power', 14.0),
                'last_tx_time': 0,
                'consecutive_successes': 0
            }
    
    def _generate_fragments(self, start_time: float, cr: str, bw_khz: float,
                      frequency_mhz: float, payload_bytes: int, dr: int,
                      device_id: str = "", transmission_id: int = None):
        """Génère des fragments LR-FHSS avec la fréquence centrale spécifiée"""
        params = {
            'cr': cr,
            'bw_khz': bw_khz,
            'frequency_mhz': frequency_mhz,  # ← FRÉQUENCE CENTRALE PASSÉE EN PARAMÈTRE
            'payload_bytes': payload_bytes,
            'dr': dr
        }
        
        return generate_lrfhss_fragments(
            start_time=start_time,
            params=params,
            device_id=device_id,
            transmission_id=transmission_id
        )
    
    def _generate_packets_for_device(self, device_id: str, current_time: float):
        """Génère un paquet avec optimisation DQN et scheduler"""
        state = self.devices_state[device_id]
        
        # 1. CONSULTER DQN POUR RECOMMANDATION
        dqn_recommendation = None
        if self.config.get('enable_dqn', False) and self.dqn_manager.enabled:
            dqn_recommendation = self.dqn_manager.get_recommendation(device_id)
        
        # 2. DÉTERMINER LES PARAMÈTRES
        region = self.config.get('region', 'EU868')
        if region == 'EU868':
            max_power = 14.0
        else:
            max_power = 30.0
        
        use_dqn_for_dr = self.config.get('use_dqn_for_dr', True)
        use_dqn_for_power = self.config.get('use_dqn_for_power', True)
        use_dqn_for_frequency = self.config.get('use_dqn_for_frequency', False)
        
        # 3. INITIALISER BANDWIDTH (nécessaire pour la sélection de fréquence)
        bandwidth = self.config.get('bandwidth_khz', 136.71875)
        coding_rate = self.config.get('coding_rate', '1/3')
        
        # ============ LOGIQUE DE SÉLECTION DE FRÉQUENCE ============
        
        # Sélectionner la fréquence centrale selon la bande passante
        from config import LR_FHSS_Config
        
        # Obtenir les centres de fréquence disponibles
        bw_config = LR_FHSS_Config.PHYSICAL_LAYER['channel_bandwidths'].get(bandwidth)
        
        if bw_config and 'centers' in bw_config and bw_config['centers']:
            available_centers_mhz = [freq / 1e6 for freq in bw_config['centers']]
            
            # Sélection déterministe basée sur device_id et compteur de transmission
            transmission_count = state.get('tx_count', 0)
            seed_str = f"{device_id}_{transmission_count}_center"
            import hashlib
            seed_hash = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16)
            center_index = seed_hash % len(available_centers_mhz)
            default_frequency = available_centers_mhz[center_index]
        else:
            # Fallback si pas de centres définis
            if region == 'EU868':
                if bandwidth == 136.71875:
                    # Pour BW=136.71875: 868.1, 868.3, 868.5 MHz
                    seed_str = f"{device_id}_{state.get('tx_count', 0)}"
                    import hashlib
                    seed_hash = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16)
                    center_index = seed_hash % 3
                    default_frequency = [868.1, 868.3, 868.5][center_index]
                elif bandwidth == 335.9375:
                    # Pour BW=335.9375: 868.13 ou 868.53 MHz
                    seed_str = f"{device_id}_{state.get('tx_count', 0)}"
                    import hashlib
                    seed_hash = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16)
                    default_frequency = 868.13 if seed_hash % 2 == 0 else 868.53
                else:
                    default_frequency = 868.1
            else:  # US915
                # Gamme 902-928 MHz pour US915
                seed_str = f"{device_id}_{state.get('tx_count', 0)}"
                import hashlib
                seed_hash = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16)
                default_frequency = 902.0 + (seed_hash % 2600) / 100.0  # 902.0 à 928.0 MHz
        # ============ FIN LOGIQUE FRÉQUENCE ============
        
        if dqn_recommendation and (use_dqn_for_dr or use_dqn_for_power or use_dqn_for_frequency):
            # Utiliser DQN
            if use_dqn_for_dr:
                dr = dqn_recommendation['dr']
                # Calculer coding_rate et bandwidth basés sur le DR
                if dr in [8, 9]:
                    bandwidth = 136.71875
                    coding_rate = '1/3' if dr == 8 else '2/3'
                elif dr in [10, 11]:
                    bandwidth = 335.9375
                    coding_rate = '1/3' if dr == 10 else '2/3'
                else:
                    # Fallback pour d'autres DR
                    bandwidth = 136.71875
                    coding_rate = '1/3'
            else:
                # Garder valeurs UI pour DR
                bandwidth = self.config.get('bandwidth_khz', 136.71875)
                coding_rate = self.config.get('coding_rate', '1/3')
                if bandwidth == 136.71875:
                    dr_for_cr = {'1/3': 8, '2/3': 9}
                elif bandwidth == 335.9375:
                    dr_for_cr = {'1/3': 10, '2/3': 11}
                else:
                    dr_for_cr = {'1/3': 12, '2/3': 13}
                dr = dr_for_cr.get(coding_rate, 8)
            
            if use_dqn_for_power:
                tx_power = dqn_recommendation['tx_power_dbm']
            else:
                tx_power = min(self.config.get('tx_power', 14.0), max_power)
            
            if use_dqn_for_frequency:
                frequency = dqn_recommendation['frequency_mhz']
            else:
                frequency = default_frequency  # Utiliser fréquence calculée
            
            dqn_used = True
            
        else:
            # Valeurs par défaut (pas de DQN)
            bandwidth = self.config.get('bandwidth_khz', 136.71875)
            coding_rate = self.config.get('coding_rate', '1/3')
            if bandwidth == 136.71875:
                dr_for_cr = {'1/3': 8, '2/3': 9}
            elif bandwidth == 335.9375:
                dr_for_cr = {'1/3': 10, '2/3': 11}
            else:
                dr_for_cr = {'1/3': 12, '2/3': 13}
            
            dr = dr_for_cr.get(coding_rate, 8)
            tx_power = min(self.config.get('tx_power', 14.0), max_power)
            frequency = default_frequency  # Utiliser fréquence calculée
            dqn_used = False
        
        # 4. UTILISER LE SCHEDULER INTELLIGENT
        actual_start_time = current_time
        scheduler_strategy = 'IMMEDIATE'
        scheduler_delay = 0.0
        scheduler_freq_shift = 0.0
        scheduler_power_boost = 0.0
        adjusted_fragments = None
        
        if self.config.get('enable_intelligent_scheduler', False) and self.scheduler:
            try:
                # Convertir en DR string
                if bandwidth == 136.71875:
                    dr_str = "DR8" if coding_rate == '1/3' else "DR9"
                elif bandwidth == 335.9375:
                    dr_str = "DR10" if coding_rate == '1/3' else "DR11"
                elif bandwidth == 1523:
                    dr_str = "DR12" if coding_rate == '1/3' else "DR13"
                else:
                    dr_str = f"DR{dr}"

                scheduling_info = {
                    'device_id': device_id,
                    'dr': dr_str,
                    'frequency_mhz': frequency,
                    'payload_bytes': np.random.randint(
                        self.config.get('payload_min', 25),
                        self.config.get('payload_max', 25) + 1
                    ),
                    'tx_power_dbm': tx_power,
                }

                slot_result = self.scheduler.schedule_transmission(**scheduling_info)
                
                actual_start_time = current_time + slot_result['start_time']
                scheduler_strategy = slot_result['strategy']
                scheduler_delay = slot_result['delay_applied']
                scheduler_freq_shift = slot_result.get('freq_shift_applied', 0.0)
                scheduler_power_boost = slot_result.get('power_boost_db', 0.0)
                
                # Appliquer ajustements
                if scheduler_strategy == 'FREQ_SHIFT' and self.config.get('scheduler_allow_freq_shift', True):
                    frequency += (scheduler_freq_shift / 1000.0)
                    # Limiter la fréquence aux canaux valides
                    if region == 'EU868':
                        if bandwidth == 136.71875:
                            # Arrondir à 868.1, 868.3, ou 868.5 MHz
                            frequencies = [868.1, 868.3, 868.5]
                            closest = min(frequencies, key=lambda x: abs(x - frequency))
                            frequency = closest
                        elif bandwidth == 335.9375:
                            # Arrondir à 868.13 ou 868.53 MHz
                            frequencies = [868.13, 868.53]
                            closest = min(frequencies, key=lambda x: abs(x - frequency))
                            frequency = closest
                if scheduler_strategy == 'POWER_BOOST' and self.config.get('scheduler_allow_power_boost', False):
                    tx_power = min(max_power, tx_power + scheduler_power_boost)
                
                if 'adjusted_fragments' in slot_result and slot_result['adjusted_fragments']:
                    adjusted_fragments = self._convert_scheduler_fragments(
                        slot_result['adjusted_fragments'], 
                        device_id, 
                        slot_result['payload_bytes']
                    )
                
                # Statistiques
                with self.stats_lock:
                    self.detailed_stats.add_scheduler_decision(
                        scheduler_strategy, scheduler_delay, scheduler_freq_shift, scheduler_power_boost
                    )
                    
            except Exception as e:
                logger.error(f"Erreur scheduler {device_id}: {e}")
        
        # 5. VÉRIFIER LIMITES SIMULATION
        simulation_duration = self.config.get('simulation_duration', 7200)
        tx_interval_min = float(self.config.get('tx_interval_min', 3600))
        tx_interval_max = float(self.config.get('tx_interval_max', 3600))
        if tx_interval_min > tx_interval_max:
            tx_interval_min, tx_interval_max = tx_interval_max, tx_interval_min
        
        tx_interval = np.random.uniform(tx_interval_min, tx_interval_max)
        
        # Taille du payload
        p_min = int(self.config.get('payload_min', 25))
        p_max = int(self.config.get('payload_max', 25))
        if p_min > p_max:
            p_min, p_max = p_max, p_min
        
        if p_min == p_max:
            payload = p_min
        else:
            payload = np.random.randint(p_min, p_max + 1)
        
        toa_ms = LR_FHSS_Config.calculate_toa_ms(dr, payload)
        toa_s = toa_ms / 1000.0
        
        if actual_start_time + toa_s > simulation_duration:
            # Planifier prochain réveil seulement
            next_wakeup = current_time + tx_interval
            if next_wakeup < simulation_duration:
                heapq.heappush(self.event_queue, (
                    next_wakeup,
                    self._global_event_counter,
                    'DEVICE_WAKEUP',
                    {'device_id': device_id}
                ))
                self._global_event_counter += 1
            return
        
        # 6. COMPTEUR DE TRANSMISSION POUR UNICITÉ
        if device_id not in self.transmission_counters:
            self.transmission_counters[device_id] = 0
        else:
            self.transmission_counters[device_id] += 1
        
        transmission_id = self.transmission_counters[device_id]
        
        # 7. GÉNÉRER FRAGMENTS LR-FHSS
        if adjusted_fragments:
            fragments = adjusted_fragments
        else:
            fragments = self._generate_fragments(
                start_time=actual_start_time,
                cr=coding_rate,
                bw_khz=bandwidth,
                frequency_mhz=frequency,  # ← PASSER LA FRÉQUENCE CENTRALE
                payload_bytes=payload,
                dr=dr,
                device_id=device_id,
                transmission_id=transmission_id
            )
        
        # 8. CRÉER PAQUET COMPLET
        packet = SimulatedPacket(
            packet_id=f"{device_id}_{actual_start_time:.3f}_{transmission_id}",
            device_id=device_id,
            start_time=actual_start_time,
            end_time=actual_start_time + toa_s,
            toa_ms=toa_ms,
            frequency_mhz=frequency,
            fragments=fragments,
            tx_power_dbm=tx_power,
            dr=dr,
            cr=coding_rate,
            bw_khz=bandwidth,
            payload_bytes=payload,
            distance_km=state['distance_km'],
            scheduler_applied=(scheduler_strategy != 'IMMEDIATE'),
            scheduler_strategy=scheduler_strategy,
            scheduler_delay=scheduler_delay,
            scheduler_freq_shift=scheduler_freq_shift,
            scheduler_power_boost=scheduler_power_boost,
            
            # Info DQN
            dqn_applied=dqn_used,
            dqn_dr=dr if dqn_used else None,
            dqn_power=tx_power if dqn_used else None,
            dqn_frequency=frequency if dqn_used else None,
            dqn_recommendation=dqn_recommendation if dqn_used else None,
            
            # Info Énergie
            pa_type=self.config.get('pa_type', 'SX1261_LP')
        )
        
        # 8B. CALCULER RSSI ET SHADOWING AVEC LE NOUVEAU MODÈLE DE CHANNEL
        device_position = state.get('position', (0, 0))
        device_id_str = state.get('device_id', device_id)
        
        try:
            rssi_dbm, path_loss_db, shadowing_db = calculate_rssi_with_details(
                tx_power_dbm=tx_power,
                distance_km=state['distance_km'],
                frequency_mhz=frequency,
                device_id=device_id_str,
                position=device_position,
                seed_global=self.config.get('seed_global', 42),
                path_loss_exponent=self.config.get('path_loss_exponent', 2.7),
                shadowing_std_db=self.config.get('shadowing_std_db', 7.0),
                reference_loss_db=125.0,  # EU868
                apply_shadowing=True
            )
            
            # Stocker RSSI et shadowing sur le paquet
            packet.rssi_dbm = rssi_dbm
            packet.path_loss_db = path_loss_db
            packet.shadowing_db = shadowing_db
            
            # Calculer SNR
            snr_db = calculate_snr(
                rssi_dbm=rssi_dbm,
                bandwidth_khz=bandwidth,
                noise_figure_db=self.config.get('noise_figure_db', 6.0)
            )
            packet.snr_db = snr_db
            
        except Exception as e:
            logger.error(f"Erreur calcul RSSI pour {device_id}: {e}")
            packet.rssi_dbm = -120.0
            packet.path_loss_db = 0.0
            packet.shadowing_db = 0.0
            packet.snr_db = -30.0
        
        # 9. MISE À JOUR STATISTIQUES DQN
        if dqn_used:
            with self.stats_lock:
                self.dqn_decisions += 1
                
                # Comparer puissance avec défaut
                default_power = min(self.config.get('tx_power', 14.0), max_power)
                power_saving = default_power - tx_power
                if power_saving > 0:
                    self.dqn_power_saved += power_saving
                
                # Enregistrer la décision détaillée
                decision = {
                    'time': float(current_time),
                    'device_id': device_id,
                    'distance_km': state.get('distance_km', None),
                    'dqn_dr': dr,
                    'dqn_power_dbm': tx_power,
                    'frequency_mhz': frequency,
                    'payload_bytes': payload,
                }
                self.dqn_decision_history.append(decision)
        
        # 10. AJOUTER AUX STATISTIQUES
        with self.stats_lock:
            self.total_sent += 1
            self.simulated_packets.append(packet)
        
        # 11. PLANIFIER ÉVÉNEMENTS FRAGMENTS
        for fragment in fragments:
            heapq.heappush(self.event_queue, (
                fragment.start_time,
                self._global_event_counter,
                'FRAGMENT_START',
                {'fragment': fragment, 'packet': packet}
            ))
            self._global_event_counter += 1
            
            heapq.heappush(self.event_queue, (
                fragment.end_time,
                self._global_event_counter,
                'FRAGMENT_END',
                {'fragment': fragment, 'packet': packet}
            ))
            self._global_event_counter += 1
        
        # 12. ÉVÉNEMENT FIN TRANSMISSION
        heapq.heappush(self.event_queue, (
            packet.end_time,
            self._global_event_counter,
            'TRANSMISSION_END',
            {'packet': packet}
        ))
        self._global_event_counter += 1
        
        # 13. PLANIFIER PROCHAINE TRANSMISSION
        next_tx_time = actual_start_time + tx_interval
        if next_tx_time < simulation_duration:
            heapq.heappush(self.event_queue, (
                next_tx_time,
                self._global_event_counter,
                'DEVICE_WAKEUP',
                {'device_id': device_id}
            ))
            self._global_event_counter += 1
        
        # 14. METTRE À JOUR ÉTAT DEVICE
        state['next_tx_time'] = next_tx_time
        state['tx_interval'] = tx_interval
        state['tx_count'] += 1
        state['last_tx_time'] = actual_start_time
        state['last_dr'] = dr
        state['last_tx_power'] = tx_power
        state['last_frequency'] = frequency
        
        # Ajouter au log avec la fréquence et bande passante
        if self.total_sent % 20 == 0:
            self.log_queue.put(f"📤 {device_id}: Paquet #{self.total_sent} (DR{dr}, {tx_power:.1f}dBm, {frequency:.3f}MHz, BW={bandwidth:.1f}kHz)")
        
    
    def _convert_scheduler_fragments(self, scheduler_fragments: List, device_id: str, payload_bytes: int) -> List[TransmissionFragment]:
        """Convertit les fragments du scheduler"""
        fragments = []
        
        for i, sched_frag in enumerate(scheduler_fragments):
            fragment = TransmissionFragment(
                start_time=sched_frag.start_time,
                end_time=sched_frag.end_time,
                frequency_mhz=sched_frag.frequency_mhz,
                fragment_type=sched_frag.fragment_type,
                fragment_index=i,
                bw_khz=sched_frag.bw_khz,
                channel_offset=sched_frag.channel_offset,  # ✅ CORRIGÉ
                absolute_channel=sched_frag.absolute_channel,
                grid_id=sched_frag.grid_id,
                cr=sched_frag.cr,
                hop_number=sched_frag.hop_number,
                instantaneous_bw_hz=sched_frag.instantaneous_bw_hz,
                grid_spacing_khz=sched_frag.grid_spacing_khz,
                bits_in_fragment=sched_frag.bits_in_fragment,
                is_last_hop=sched_frag.is_last_hop,
                tx_power_dbm=sched_frag.tx_power_dbm
            )
            fragments.append(fragment)
        
        return fragments
    
    def _process_event(self, event_type: str, event_data: Dict):
        """Traite un événement de simulation"""
        if event_type == 'DEVICE_WAKEUP':
            device_id = event_data['device_id']
            self._generate_packets_for_device(device_id, self.simulated_time)
            
        elif event_type == 'FRAGMENT_START':
            fragment = event_data['fragment']
            packet = event_data['packet']
            
            # Ajouter aux fragments actifs
            with self.active_fragments_lock:
                self.active_fragments.append({
                    'fragment': fragment,
                    'packet': packet,
                    'start_time': self.simulated_time
                })
            
            # Vérifier collisions
            self._check_collisions_for_fragment(fragment, packet)
            
        elif event_type == 'FRAGMENT_END':
            fragment = event_data['fragment']
            
            # Retirer des fragments actifs
            with self.active_fragments_lock:
                self.active_fragments = [
                    f for f in self.active_fragments
                    if id(f['fragment']) != id(fragment)
                ]
            
        elif event_type == 'TRANSMISSION_END':
            packet = event_data['packet']
            self._evaluate_packet_end(packet)
    
    def _check_collisions_for_fragment(self, new_fragment, new_packet):
        """Vérifie les collisions pour un nouveau fragment - OPTIMISÉE POUR 488 Hz"""
        with self.active_fragments_lock:
            if len(self.active_fragments) < 2:
                return
            
            # Bande passante instantanée des fragments LR-FHSS
            fragment_bw_hz = new_fragment.instantaneous_bw_hz if hasattr(new_fragment, 'instantaneous_bw_hz') else 488.28125
            min_separation_hz = fragment_bw_hz * 1.5  # ~732 Hz
            
            for active_info in self.active_fragments:
                active_frag = active_info['fragment']
                active_pkt = active_info['packet']
                
                if id(active_frag) == id(new_fragment) or active_pkt.packet_id == new_packet.packet_id:
                    continue
                
                # OPTIMISATION: Vérification rapide de fréquence à 488 Hz
                freq_diff_hz = abs(new_fragment.frequency_mhz - active_frag.frequency_mhz) * 1e6
                
                # Si les fragments sont sur des canaux 488 Hz différents, pas de collision
                if freq_diff_hz > min_separation_hz:
                    continue  # Canaux différents, pas besoin de vérifier plus
                
                # DEBUG log
                if freq_diff_hz > fragment_bw_hz * 3:  # > 1.5 kHz
                    self.log_queue.put(f"[DEBUG 488Hz] Canaux éloignés: {new_fragment.frequency_mhz:.3f} vs {active_frag.frequency_mhz:.3f} MHz (diff={freq_diff_hz:.1f} Hz)")
                    continue
                
                # Vérifier collision avec la fonction corrigée
                collision = check_collision(
                    new_fragment, active_frag,
                    new_packet.packet_id, active_pkt.packet_id
                )
                
                if collision and is_significant_collision(collision):
                    # Vérifier si cette paire a déjà été comptée
                    frag1_id = id(collision.fragment1)
                    frag2_id = id(collision.fragment2)
                    pair_key = tuple(sorted([frag1_id, frag2_id]))
                    
                    if pair_key not in self.collision_pairs:
                        self.collision_pairs.add(pair_key)
                        
                        # Log détaillé de la collision
                        if hasattr(collision, 'fragment_bw_hz'):
                            self.log_queue.put(f"[COLLISION 488Hz] {new_packet.packet_id} vs {active_pkt.packet_id}")
                            self.log_queue.put(f"  Fréquence: {collision.fragment1.frequency_mhz:.6f} vs {collision.fragment2.frequency_mhz:.6f} MHz")
                            self.log_queue.put(f"  Offset: {collision.frequency_offset_hz:.1f} Hz (seuil: {collision.fragment_bw_hz:.1f} Hz)")
                            self.log_queue.put(f"  Type: {collision.collision_type}")
                        
                        self._process_collision(collision)

    def _process_collision(self, collision):
        """Traite une collision détectée"""
        # Trouver les paquets impliqués
        packet1 = None
        packet2 = None
        
        for frag_info in self.active_fragments:
            frag = frag_info['fragment']
            if id(frag) == id(collision.fragment1):
                packet1 = frag_info['packet']
            elif id(frag) == id(collision.fragment2):
                packet2 = frag_info['packet']
            
            if packet1 and packet2:
                break
        
        if not packet1 or not packet2:
            return
        
        # Marquer les paquets
        if not packet1.collision:
            packet1.collision = True
            packet1.collision_details = []
        
        if not packet2.collision:
            packet2.collision = True
            packet2.collision_details = []
        
        if packet1.collision_details is not None:
            packet1.collision_details.append(collision)
        
        if packet2.collision_details is not None:
            packet2.collision_details.append(collision)
        
        # Mettre à jour les compteurs
        if hasattr(collision, 'collision_type'):
            coll_type = collision.collision_type
            if 'header' in coll_type and 'payload' not in coll_type:
                self.collisions_by_type['header'] += 1
                self.header_collisions += 1
            elif 'payload' in coll_type and 'header' not in coll_type:
                self.collisions_by_type['payload'] += 1
                self.payload_collisions += 1
            else:
                self.collisions_by_type['mixed'] += 1
                self.mixed_collisions += 1
        
        if hasattr(collision, 'capture_effect') and collision.capture_effect:
            self.capture_effect_count += 1
    
    def _evaluate_packet_end(self, packet):
        """Évalue une transmission terminée avec shadowing"""
        if hasattr(packet, '_evaluated') and packet._evaluated:
            return
            
        packet._evaluated = True
        
        # Copier les paquets actifs
        active_packets = []
        try:
            with self.active_fragments_lock:
                seen_packet_ids = set()
                seen_packet_ids.add(packet.packet_id)
                
                for frag_info in self.active_fragments:
                    p = frag_info['packet']
                    if p.packet_id not in seen_packet_ids:
                        active_packets.append(p)
                        seen_packet_ids.add(p.packet_id)
        except Exception as e:
            logger.error(f"Erreur lors de la copie des fragments actifs: {e}")
            active_packets = []
        
        # Configuration pour l'évaluation
        config = {
            'distance_gtw': self.config.get('distance_gtw', 10000),
            'path_loss_exponent': self.config.get('path_loss_exponent', 2.7),
            'shadowing_std_db': self.config.get('shadowing_std_db', 7.0),
            'noise_figure_db': self.config.get('noise_figure_db', 6.0),
            'bandwidth_khz': packet.bw_khz if hasattr(packet, 'bw_khz') else self.config.get('bandwidth_khz', 136.71875),
            'coding_rate': packet.cr if hasattr(packet, 'cr') else self.config.get('coding_rate', '1/3'),
            'seed_global': self.config.get('seed_global', 42)  # Important pour shadowing déterministe
        }
        
        # Récupérer la position du device
        position = (0, 0)
        if packet.device_id in self.devices_state:
            position = self.devices_state[packet.device_id].get('position', (0, 0))
        
        # Évaluation avec shadowing
        try:
            success, failure_reason = evaluate_transmission(
                packet, 
                config, 
                active_packets,
                device_position=position  # Passer la position réelle
            )
            
            # VÉRIFICATION FINALE
            if hasattr(packet, 'shadowing_db'):
                shadowing = packet.shadowing_db
                # Le shadowing ne devrait PAS être 0 si shadowing_std_db > 0
                if shadowing == 0 and self.config.get('shadowing_std_db', 7.0) > 1.0:
                    logger.warning(f"Shadowing=0 pour {packet.device_id} @ {position}")
                    
        except Exception as e:
            logger.error(f"Crash evaluation packet {packet.packet_id}: {e}")
            success = False
            failure_reason = "INTERNAL_ERROR"
        
    # ... reste du code (analyse énergétique, DQN, statistiques) ...
        # ANALYSE ÉNERGÉTIQUE POUR TOUS LES PAQUETS (succès ET échecs)
        if ENERGY_MODULE_AVAILABLE:
            packet.success = success  # Ajouter le statut de succès au paquet
            self._analyze_packet_energy(packet)

        # 1. FEEDBACK POUR DQN
        if (self.config.get('enable_dqn', False) and self.dqn_manager.enabled and 
            hasattr(packet, 'dqn_applied') and packet.dqn_applied and
            hasattr(packet, 'dqn_recommendation') and packet.dqn_recommendation):
            
            try:
                # Utiliser RSSI et SNR calculés avec le nouveau modèle de channel
                rssi_dbm = packet.rssi_dbm if hasattr(packet, 'rssi_dbm') else -120.0
                snr_db = packet.snr_db if hasattr(packet, 'snr_db') else -30.0
                
                # Enregistrer expérience DQN avec les vraies valeurs
                self.dqn_manager.record_feedback(
                    device_id=packet.device_id,
                    success=success,
                    rssi_dbm=rssi_dbm,
                    snr_db=snr_db,
                    failure_reason=failure_reason
                )
                
                # Mettre à jour statistiques détaillées
                with self.stats_lock:
                    self.detailed_stats.add_dqn_decision(
                        packet.dr, packet.tx_power_dbm, success
                    )
                
                # Mettre à jour état device pour DQN
                if packet.device_id in self.devices_state:
                    dev_state = self.devices_state[packet.device_id]
                    dev_state['last_rssi'] = rssi_dbm
                    dev_state['last_snr'] = snr_db
                    dev_state['last_dr'] = packet.dr
                    dev_state['last_tx_power'] = packet.tx_power_dbm
                    
                    # Mettre à jour taux succès
                    dev_state['success_history'].append(1.0 if success else 0.0)
                    if len(dev_state['success_history']) > 20:
                        dev_state['success_history'] = dev_state['success_history'][-20:]
                    
                    if dev_state['success_history']:
                        dev_state['success_rate'] = np.mean(dev_state['success_history'])
                
            except Exception as e:
                logger.error(f"Erreur feedback DQN: {e}")
        
        # 2. MISE À JOUR DES STATISTIQUES GÉNÉRALES
        with self.stats_lock:
            if success:
                packet.success = True
                self.successful_rx += 1
            else:
                packet.success = False
                
                if hasattr(packet, 'collision') and packet.collision and "COLLISION" in str(failure_reason).upper():
                    self.collisions += 1
                    self.fec_failed += 1
                else:
                    if failure_reason == "RSSI_INSUFFISANT":
                        self.failures_rssi += 1
                    elif failure_reason in ["DEMODULATION", "DEMODULATION_FAILED"]:
                        self.failures_demod += 1
                    elif failure_reason == "FEC":
                        self.failures_fec += 1
                    elif "COLLISION" in str(failure_reason).upper():
                        self.collisions += 1
                    else:
                        self.demod_failed += 1
        
        # 3. MISE À JOUR ÉTAT DEVICE (si pas déjà fait par DQN)
        if not (hasattr(packet, 'dqn_applied') and packet.dqn_applied):
            if packet.device_id in self.devices_state:
                dev_state = self.devices_state[packet.device_id]
                if success:
                    dev_state['success_count'] += 1
                    dev_state['success_history'].append(1.0)
                else:
                    dev_state['collision_count'] += (1 if hasattr(packet, 'collision') and packet.collision else 0)
                    dev_state['success_history'].append(0.0)
                
                # Mettre à jour RSSI et SNR depuis le paquet (nouveau modèle channel)
                if hasattr(packet, 'rssi_dbm'):
                    dev_state['last_rssi'] = packet.rssi_dbm
                if hasattr(packet, 'snr_db'):
                    dev_state['last_snr'] = packet.snr_db
                
                # Limiter historique
                if len(dev_state['success_history']) > 20:
                    dev_state['success_history'] = dev_state['success_history'][-20:]
                
                # Mise à jour taux de succès
                if dev_state['success_history']:
                    dev_state['success_rate'] = np.mean(dev_state['success_history'])
        
        # 4. AJOUTER AUX STATS DÉTAILLÉES
        self.detailed_stats.add_packet_result(packet, success, failure_reason)
    
    def _analyze_packet_energy(self, packet):
        """Analyse la consommation énergétique d'un paquet"""
        try:
            if not ENERGY_MODULE_AVAILABLE:
                return None
            
            if not hasattr(packet, 'toa_ms') or not hasattr(packet, 'tx_power_dbm'):
                return None
            
            # Déterminer PA type
            pa_type = self.config.get('pa_type', 'SX1261_LP')
            if packet.tx_power_dbm > 14.0 and pa_type == 'SX1261_LP':
                pa_type = 'SX1262_HP'  # Auto-upgrade si nécessaire
            
            # Calculer consommation
            energy_metrics = EnergyConsumptionModel.calculate_energy_joules(
                tx_power_dbm=packet.tx_power_dbm,
                toa_ms=packet.toa_ms,
                pa_type=pa_type,
                rx_duration_ms=100.0,  # Réception ACK
                voltage_v=3.3
            )
            
            # Ajouter au paquet
            packet.energy_metrics = energy_metrics
            packet.pa_type_used = pa_type
            
            # DEBUG
            pkt_success = getattr(packet, 'success', 'UNDEFINED')
            
            # Mettre à jour l'analyseur
            if hasattr(self, 'energy_analyzer'):
                self.energy_analyzer.analyze_packet_energy(packet)
            
            return energy_metrics
            
        except Exception as e:
            logger.error(f"Erreur analyse énergie paquet: {e}")
            return None
    
    def _calculate_intermediate_metrics(self):
        """Calcule les métriques intermédiaires"""
        with self.stats_lock:
            # 1. Calculer taux de succès de base
            if self.total_sent > 0:
                self.success_rate = (self.successful_rx / self.total_sent) * 100
                self.collision_rate = (self.collisions / self.total_sent) * 100 if self.collisions > 0 else 0.0
            else:
                self.success_rate = 0.0
                self.collision_rate = 0.0
            
            # 2. Mettre à jour ToA brut
            if self.simulated_packets and len(self.simulated_packets) > self._last_toa_count:
                new_packets = self.simulated_packets[self._last_toa_count:]
                new_toa_ms = sum(p.toa_ms for p in new_packets)
                self.toa_brut_total += (new_toa_ms / 1000.0)
                self._last_toa_count = len(self.simulated_packets)
                
                # Calculer l'efficacité spectrale
                if self.toa_brut_total > 0.001:
                    if self.toa_net_total > 0:
                        self.spectral_efficiency = (self.toa_net_total / self.toa_brut_total) * 100
                        self.spectral_efficiency = min(100.0, max(0.0, self.spectral_efficiency))
                    else:
                        self.spectral_efficiency = 100.0
                else:
                    self.spectral_efficiency = 0.0
                
                # Calculer le taux d'occupation
                max_time = max(self.simulated_time, 0.001)
                self.occupation_rate = (self.toa_brut_total / max_time) * 100
                self.occupation_rate = min(100.0, max(0.0, self.occupation_rate))
            
            # 3. Métriques RF moyennes
            successful_packets = [p for p in self.simulated_packets if hasattr(p, 'success') and p.success]
            
            if successful_packets:
                # RSSI
                if all(hasattr(p, 'rssi_dbm') for p in successful_packets):
                    rssi_values = [p.rssi_dbm for p in successful_packets]
                    self.avg_rssi_dbm = np.mean(rssi_values)
                    self.min_rssi_dbm = min(rssi_values)
                    self.max_rssi_dbm = max(rssi_values)
                
                # SNR
                if all(hasattr(p, 'snr_db') for p in successful_packets):
                    snr_values = [p.snr_db for p in successful_packets]
                    self.avg_snr_db = np.mean(snr_values)
                    self.min_snr_db = min(snr_values)
                    self.max_snr_db = max(snr_values)
                
                # BER
                if all(hasattr(p, 'ber') for p in successful_packets):
                    ber_values = [p.ber for p in successful_packets]
                    self.avg_ber = np.mean(ber_values)
            
            else:
                # Valeurs par défaut si aucun paquet réussi
                self.avg_rssi_dbm = -120.0
                self.avg_snr_db = 0.0
                self.avg_ber = 1e-3
                self.min_rssi_dbm = -120.0
                self.max_rssi_dbm = -120.0
                self.min_snr_db = 0.0
                self.max_snr_db = 0.0
            
            # 4. Métriques collisions détaillées
            if self.simulated_packets:
                collisions_causing_failure = 0
                for p in self.simulated_packets:
                    if hasattr(p, 'success') and not p.success:
                        if hasattr(p, 'collision') and p.collision:
                            collisions_causing_failure += 1
                
                self.detailed_stats.collisions_causing_failure = collisions_causing_failure
            
            # 5. Métriques DQN
            if self.config.get('enable_dqn', False) and self.dqn_decisions > 0:
                self.dqn_success_rate = (self.dqn_successes / self.dqn_decisions) * 100
                self.dqn_avg_power_saved = self.dqn_power_saved / self.dqn_decisions
            else:
                self.dqn_success_rate = 0.0
                self.dqn_avg_power_saved = 0.0
    
    def _calculate_net_toa(self, packets):
        """
        Calcule le ToA net = occupation spectrale réelle en fusionnant les chevauchements
        
        Le ToA net fusionne TOUS les intervalles de transmission qui se chevauchent au niveau PAQUET,
        en GARDANT les positions temporelles réelles (AVEC délais scheduler).
        """
        if not packets:
            return 0.0
        
        # Créer la liste de tous les PAQUETS avec leurs positions RÉELLES
        # On utilise les intervalles des paquets complets (pas les fragments individuels)
        packet_intervals = []
        for packet in packets:
            packet_intervals.append({
                'start': packet.start_time,
                'end': packet.end_time,
                'packet_id': packet.packet_id
            })
        
        if not packet_intervals:
            return 0.0
        
        # Trier par temps de début
        sorted_packets = sorted(packet_intervals, key=lambda x: x['start'])
        
        # Fusionner les intervalles qui se chevauchent
        merged_intervals = []
        current_start = sorted_packets[0]['start']
        current_end = sorted_packets[0]['end']
        
        for pkt in sorted_packets[1:]:
            pkt_start = pkt['start']
            pkt_end = pkt['end']
            
            # Si l'intervalle actuel chevauche le suivant
            if pkt_start <= current_end:
                # Fusionner : étendre la fin si nécessaire
                current_end = max(current_end, pkt_end)
            else:
                # Pas de chevauchement, sauvegarder l'intervalle actuel
                merged_intervals.append((current_start, current_end))
                # Commencer un nouvel intervalle
                current_start = pkt_start
                current_end = pkt_end
        
        # Ajouter le dernier intervalle
        merged_intervals.append((current_start, current_end))
        
        # Calculer la durée totale des intervalles fusionnés
        net_toa = sum(end - start for start, end in merged_intervals)
        
        return max(0.0, net_toa)
    
    def run(self):
        """Exécute la simulation - Version Événementielle"""
        try:
            # CORRIGER: Mettre is_running à True pour que la boucle événementielle s'exécute
            self.is_running = True
            
            # Réinitialiser
            self._reset_statistics()
            self.simulated_packets = []
            self.event_queue = []
            self._global_event_counter = 0
            
            # Initialiser DQN
            if self.config.get('enable_dqn', False):
                self._initialize_dqn()
            
            # Initialiser scheduler
            if self.config.get('enable_intelligent_scheduler', False):
                self._initialize_intelligent_scheduler()
            
            # Initialiser la simulation temps réel
            self._initialize_state()
            
            # Log des paramètres de fréquence
            bandwidth = self.config.get('bandwidth_khz', 136.71875)
            region = self.config.get('region', 'EU868')
            coding_rate = self.config.get('coding_rate', '1/3')
            
            from config import LR_FHSS_Config
            
            # Déterminer le DR
            if bandwidth == 136.71875:
                dr = 8 if coding_rate == '1/3' else 9
            elif bandwidth == 335.9375:
                dr = 10 if coding_rate == '1/3' else 11
            else:
                dr = 12 if coding_rate == '1/3' else 13
            
            bw_config = LR_FHSS_Config.PHYSICAL_LAYER['channel_bandwidths'].get(bandwidth)
            
            self.log_queue.put("=" * 60)
            self.log_queue.put("🚀 DÉMARRAGE SIMULATION LR-FHSS")
            self.log_queue.put("=" * 60)
            self.log_queue.put(f"📡 Région: {region}")
            self.log_queue.put(f"📡 DR: {dr} | CR: {coding_rate} | BW: {bandwidth:.1f} kHz")
            
            if bw_config and 'centers' in bw_config and bw_config['centers']:
                centers_mhz = [freq / 1e6 for freq in bw_config['centers']]
                self.log_queue.put(f"📡 Centres de fréquence disponibles: {', '.join([f'{f:.3f} MHz' for f in centers_mhz])}")
            else:
                self.log_queue.put(f"📡 Pas de centres définis pour BW={bandwidth:.1f} kHz")
            
            if self.config.get('enable_dqn', False):
                self.log_queue.put(f"🤖 DQN activé - Modèle: {self.config.get('dqn_model_name', 'Nouveau')}")
            
            if self.config.get('enable_intelligent_scheduler', False):
                self.log_queue.put(f"⏱️ Scheduler intelligent activé")
            
            if ENERGY_MODULE_AVAILABLE:
                self.log_queue.put(f"🔋 Module énergie activé - PA: {self.config.get('pa_type', 'SX1261_LP')}, Batterie: {self.config.get('battery_capacity_mah', 1000.0):.0f} mAh")
            
            self.log_queue.put("=" * 60)
            
            # 1. PLANIFIER PREMIERS RÉVEILS
            for device_id, state in self.devices_state.items():
                first_wakeup = state['next_tx_time']
                heapq.heappush(self.event_queue, (
                    first_wakeup,
                    self._global_event_counter,
                    'DEVICE_WAKEUP',
                    {'device_id': device_id}
                ))
                self._global_event_counter += 1
            
            last_ui_update = 0.0
            last_scheduler_cleanup = 0.0
            
            simulation_duration = self.config.get('simulation_duration', 7200)
            
            # 2. BOUCLE ÉVÉNEMENTIELLE
            while self.event_queue and self.is_running and self.simulated_time < simulation_duration:
                # Récupérer prochain événement
                if not self.event_queue:
                    break
                    
                event_time, _, event_type, event_data = heapq.heappop(self.event_queue)
                
                # Avancer le temps
                self.simulated_time = event_time
                progress = max(0, min(100, (self.simulated_time / simulation_duration) * 100))
                
                # Traiter l'événement
                self._process_event(event_type, event_data)
                
                # Calculer et envoyer les métriques périodiquement
                current_real_time = time.time()
                if current_real_time - last_ui_update > 0.2:  # 5 FPS
                    self._calculate_intermediate_metrics()
                    self._send_metrics_to_queue(progress)
                    last_ui_update = current_real_time
                
                # Nettoyer le scheduler périodiquement
                if (self.config.get('enable_intelligent_scheduler', False) and 
                    self.scheduler and self.simulated_time - last_scheduler_cleanup > 2.0):
                    self.scheduler.cleanup_old_fragments(self.simulated_time)
                    last_scheduler_cleanup = self.simulated_time
            
            # Force final metric calculation
            self._calculate_intermediate_metrics()
            
            # Calculer ToA Net final
            if self.simulated_packets:
                self.toa_net_total = self._calculate_net_toa(self.simulated_packets)
                
                # Recalculer l'efficacité spectrale avec les vraies valeurs
                if self.toa_brut_total > 0.001:
                    self.spectral_efficiency = (self.toa_net_total / self.toa_brut_total) * 100
                    self.spectral_efficiency = min(100.0, max(0.0, self.spectral_efficiency))
                else:
                    self.spectral_efficiency = 0.0
                
                # Recalculer le taux d'occupation avec ToA Net
                max_time = max(self.simulated_time, 0.001)
                self.occupation_rate = (self.toa_net_total / max_time) * 100
                self.occupation_rate = min(100.0, max(0.0, self.occupation_rate))
            
            # Finaliser avec log détaillé
            self.log_queue.put("=" * 60)
            self.log_queue.put("✅ SIMULATION TERMINÉE")
            self.log_queue.put("=" * 60)
            self.log_queue.put(f"📊 Résultats finaux:")
            self.log_queue.put(f"   • Temps simulé: {self.simulated_time:.1f}s")
            self.log_queue.put(f"   • Paquets envoyés: {self.total_sent:,}")
            self.log_queue.put(f"   • Paquets réussis: {self.successful_rx:,} ({self.success_rate:.1f}%)")
            self.log_queue.put(f"   • Collisions: {self.collisions:,}")
            self.log_queue.put(f"   • ToA Brut total: {self.toa_brut_total:.2f}s")
            self.log_queue.put(f"   • ToA Net total: {self.toa_net_total:.2f}s")
            self.log_queue.put(f"   • Efficacité spectrale: {self.spectral_efficiency:.1f}%")
            self.log_queue.put(f"   • Taux d'occupation: {self.occupation_rate:.1f}%")
            
            if self.config.get('enable_dqn', False) and hasattr(self, 'dqn_decisions') and self.dqn_decisions > 0:
                dqn_success_rate = (self.dqn_successes / self.dqn_decisions * 100) if self.dqn_decisions > 0 else 0
                self.log_queue.put(f"🤖 DQN: {self.dqn_decisions} décisions, {dqn_success_rate:.1f}% de succès")
            
            if self.config.get('enable_intelligent_scheduler', False) and hasattr(self, 'detailed_stats'):
                scheduler_stats = self.detailed_stats
                if scheduler_stats.scheduler_decisions > 0:
                    self.log_queue.put(f"⏱️ Scheduler: {scheduler_stats.scheduler_decisions} décisions")
                    for strategy, count in scheduler_stats.scheduler_strategies.items():
                        percentage = (count / scheduler_stats.scheduler_decisions * 100)
                        self.log_queue.put(f"   • {strategy}: {count} ({percentage:.1f}%)")
            
            # Finaliser
            self._finalize_simulation()
            
        except Exception as e:
            import traceback
            error_msg = f"❌ ERREUR DANS LA SIMULATION: {str(e)}"
            self.log_queue.put(error_msg)
            self.log_queue.put("Traceback:")
            for line in traceback.format_exc().split('\n'):
                if line.strip():
                    self.log_queue.put(f"  {line}")
            
            self.is_running = False
            
            # Envoyer un message d'erreur final
            error_metrics = {
                'is_final': True,
                'error': str(e),
                'progress': 100,
                'total_sent': getattr(self, 'total_sent', 0),
                'successful_rx': getattr(self, 'successful_rx', 0),
                'simulated_time': getattr(self, 'simulated_time', 0.0),
            }
            self.metric_queue.put(('FINAL_METRICS', error_metrics))
    
    def _send_metrics_to_queue(self, progress):
        """Envoie les métriques à la queue"""
        try:
            metrics = {}
            
            with self.stats_lock:
                # 1. Métriques de base
                total_sent = self.total_sent
                successful = self.successful_rx
                total_failed = total_sent - successful if total_sent > 0 else 0
                
                collision_failures = self.collisions
                
                if total_sent > 0:
                    success_rate = (successful / total_sent) * 100
                    collision_rate = (collision_failures / total_sent) * 100
                else:
                    success_rate = 0.0
                    collision_rate = 0.0
                
                # 2. Métriques principales
                metrics = {
                    'total_sent': total_sent,
                    'successful_rx': successful,
                    'failed_packets': total_failed,
                    'failures_rssi': getattr(self, 'failures_rssi', 0),
                    'collisions': collision_failures,
                    'demod_failed_packets': self.demod_failed,
                    'fec_recovered': self.fec_recovered,
                    'fec_failed': self.fec_failed,
                    'success_rate': success_rate,
                    'collision_rate': collision_rate,
                    'progress': progress,
                    
                    # Métriques RF
                    'avg_rssi': self.avg_rssi_dbm,
                    'avg_snr': self.avg_snr_db,
                    'avg_ber': self.avg_ber,
                    'min_rssi': getattr(self, 'min_rssi_dbm', -120.0),
                    'max_rssi': getattr(self, 'max_rssi_dbm', -120.0),
                    'min_snr': getattr(self, 'min_snr_db', 0.0),
                    'max_snr': getattr(self, 'max_snr_db', 0.0),
                    
                    # Métriques collisions détaillées
                    'header_collisions': self.header_collisions,
                    'payload_collisions': self.payload_collisions,
                    'mixed_collisions': self.mixed_collisions,
                    'capture_effect_count': self.capture_effect_count,
                    
                    # Métriques temporelles
                    'simulated_time': self.simulated_time,
                    
                    # MÉTRIQUES TOA
                    'toa_brut_total': getattr(self, 'toa_brut_total', 0.0),
                    'toa_net_total': getattr(self, 'toa_net_total', 0.0),
                    'spectral_efficiency': getattr(self, 'spectral_efficiency', 0.0),
                    'occupation_rate': getattr(self, 'occupation_rate', 0.0),
                    
                    'is_final': False
                }
                
                # 3. Métriques scheduler
                if self.config.get('enable_intelligent_scheduler', False):
                    metrics['scheduler_decisions'] = self.detailed_stats.scheduler_decisions
                    metrics['scheduler_delays'] = self.detailed_stats.scheduler_delays
                    metrics['scheduler_delays_sum'] = self.detailed_stats.scheduler_delays_sum
                    metrics['scheduler_freq_shifts'] = self.detailed_stats.scheduler_freq_shifts
                    metrics['scheduler_strategies'] = dict(self.detailed_stats.scheduler_strategies)
                
                # 4. Métriques DQN
                if self.config.get('enable_dqn', False) and hasattr(self, 'dqn_manager'):
                    dqn_stats = self.dqn_manager.get_stats()
                    
                    if dqn_stats:
                        metrics.update({
                            'dqn_decisions': dqn_stats.get('total_decisions', 0),
                            'dqn_successes': dqn_stats.get('successful_decisions', 0),
                            'dqn_success_rate': dqn_stats.get('success_rate_pct', 0.0),
                            'dqn_avg_power_saved': dqn_stats.get('avg_power_saved', 0.0),
                            'dqn_epsilon': dqn_stats.get('epsilon', 0.1)
                        })
                    else:
                        # Fallback sur les variables locales
                        dqn_decisions = getattr(self, 'dqn_decisions', 0)
                        dqn_successes = getattr(self, 'dqn_successes', 0)
                        dqn_power_saved = getattr(self, 'dqn_power_saved', 0.0)
                        
                        if dqn_decisions > 0:
                            dqn_success_rate = (dqn_successes / dqn_decisions) * 100
                            dqn_avg_power_saved = dqn_power_saved / dqn_decisions
                        else:
                            dqn_success_rate = 0.0
                            dqn_avg_power_saved = 0.0
                        
                        metrics.update({
                            'dqn_decisions': dqn_decisions,
                            'dqn_successes': dqn_successes,
                            'dqn_success_rate': dqn_success_rate,
                            'dqn_avg_power_saved': dqn_avg_power_saved,
                            'dqn_epsilon': self.config.get('dqn_exploration', 0.0)
                        })
                
                # 6. Métriques Énergie
                if ENERGY_MODULE_AVAILABLE and hasattr(self, 'energy_analyzer'):
                    energy_stats = self.energy_analyzer.energy_stats
                    
                    metrics.update({
                        'total_energy_j': energy_stats.get('total_energy_j', 0.0),
                        'avg_energy_per_packet_j': energy_stats.get('avg_energy_per_packet_j', 0.0),
                        'avg_current_ma': energy_stats.get('avg_current_ma', 0.0),
                        'battery_life_years': energy_stats.get('battery_life_years', 0.0),
                        'energy_per_bit_j': energy_stats.get('energy_per_bit_j', 0.0),
                        'packets_analyzed': energy_stats.get('packets_analyzed', 0),
                        'pa_type': energy_stats.get('pa_type', self.config.get('pa_type', 'SX1261_LP'))
                    })
            
            # Envoyer les métriques à la queue
            self.metric_queue.put(('METRICS', metrics))
            
        except Exception as e:
            logger.error(f"Erreur dans _send_metrics_to_queue: {e}")
    
    def _finalize_simulation(self):
        """Finalise la simulation"""
        self.is_running = False
        
        # Calculer les métriques finales
        collision_failures = self.collisions
        success_rate = (self.successful_rx / self.total_sent * 100) if self.total_sent > 0 else 0
        collision_rate = (collision_failures / self.total_sent * 100) if self.total_sent > 0 else 0
        
        # Récupérer les statistiques
        scheduler_decisions = self.detailed_stats.scheduler_decisions
        scheduler_delays = self.detailed_stats.scheduler_delays
        scheduler_delays_sum = self.detailed_stats.scheduler_delays_sum
        scheduler_freq_shifts = self.detailed_stats.scheduler_freq_shifts
        scheduler_strategies = dict(self.detailed_stats.scheduler_strategies)
        
        dqn_decisions = getattr(self, 'dqn_decisions', 0)
        dqn_successes = getattr(self, 'dqn_successes', 0)
        dqn_power_saved = getattr(self, 'dqn_power_saved', 0.0)
        
        # Calcul des taux de succès
        dqn_success_rate = (dqn_successes / dqn_decisions * 100) if dqn_decisions > 0 else 0
        dqn_avg_power_saved = (dqn_power_saved / dqn_decisions) if dqn_decisions > 0 else 0
        
        # Récupérer les métriques d'énergie
        energy_stats = {}
        if ENERGY_MODULE_AVAILABLE and hasattr(self, 'energy_analyzer'):
            energy_stats = self.energy_analyzer.energy_stats
        
        # Envoyer métriques finales
        final_metrics = {
            'is_final': True,
            'total_sent': self.total_sent,
            'successful_rx': self.successful_rx,
            'failed_packets': self.total_sent - self.successful_rx,
            'failures_rssi': getattr(self, 'failures_rssi', 0),
            'collisions': collision_failures,
            'demod_failed_packets': self.demod_failed,
            'fec_recovered': self.fec_recovered,
            'fec_failed': self.fec_failed,
            'success_rate': success_rate,
            'collision_rate': collision_rate,
            'progress': 100,
            'avg_rssi': self.avg_rssi_dbm,
            'avg_snr': self.avg_snr_db,
            'avg_ber': self.avg_ber,
            'header_collisions': self.header_collisions,
            'payload_collisions': self.payload_collisions,
            'mixed_collisions': self.mixed_collisions,
            'capture_effect_count': self.capture_effect_count,
            'simulated_time': self.simulated_time,
            'spectral_efficiency': self.spectral_efficiency,
            'occupation_rate': self.occupation_rate,
            'toa_brut_total': self.toa_brut_total,
            'toa_net_total': self.toa_net_total,
            'scheduler_decisions': scheduler_decisions,
            'scheduler_delays': scheduler_delays,
            'scheduler_delays_sum': scheduler_delays_sum,
            'scheduler_freq_shifts': scheduler_freq_shifts,
            'scheduler_strategies': scheduler_strategies,
            'dqn_decisions': dqn_decisions,
            'dqn_successes': dqn_successes,
            'dqn_power_saved': dqn_power_saved,
            'dqn_success_rate': dqn_success_rate,
            'dqn_avg_power_saved': dqn_avg_power_saved,
            # Métriques d'énergie
            'total_energy_j': energy_stats.get('total_energy_j', 0.0),
            'avg_energy_per_packet_j': energy_stats.get('avg_energy_per_packet_j', 0.0),
            'avg_current_ma': energy_stats.get('avg_current_ma', 0.0),
            'battery_life_years': energy_stats.get('battery_life_years', 0.0),
            'energy_per_bit_j': energy_stats.get('energy_per_bit_j', 0.0),
            'pa_type': self.config.get('pa_type', 'SX1261_LP')
        }
        
        self.metric_queue.put(('FINAL_METRICS', final_metrics))
        
        # Sauvegarder modèles finaux
        if self.config.get('enable_dqn', False) and self.dqn_manager.agent:
            self._save_dqn_model(suffix="_final")
    
    def _initialize_dqn(self):
        """Initialise l'agent DQN"""
        if self.config.get('enable_dqn', False) and DQN_AVAILABLE:
            try:
                model_name = self.config.get('dqn_model_name', None)
                model_path = None
                
                logger.info("\n🤖 INITIALISATION DQN")
                logger.info("=" * 60)
                
                if model_name and model_name != 'Nouveau modèle':
                    import os
                    # Chercher dans plusieurs répertoires
                    model_dirs = ["BEST/dqn_models", "ddqn_models", "ddqn_checkpoints"]
                    base = model_name
                    
                    # Si model_name already contains an extension or path, use it as-is
                    if os.path.isabs(base) or os.path.sep in base:
                        model_path = base
                    else:
                        # Chercher dans les directories candidate
                        for model_dir in model_dirs:
                            for ext in [".pth", ".pt", ""]:
                                candidate = os.path.join(model_dir, base + ext) if ext else os.path.join(model_dir, base)
                                if os.path.exists(candidate):
                                    model_path = candidate
                                    break
                            if model_path:
                                break
                    
                    logger.info(f"📥 Modèle fourni: {base}")
                    logger.info(f"📁 Chemin complet: {model_path}")
                    logger.info(f"✓ Fichier existe: {os.path.exists(model_path) if model_path else False}")
                else:
                    logger.warning("⚠️  Aucun modèle spécifié!")
                
                if model_path:
                    logger.info(f"\n⏳ Tentative d'initialisation...")
                    success = self.dqn_manager.initialize(model_path)
                    
                    if success:
                        # Configurer l'exploration
                        if self.dqn_manager.agent:
                            self.dqn_manager.agent.epsilon = self.config.get('dqn_exploration', 0.0)
                            logger.info(f"✅ DQN initialisé avec succès!")
                            logger.info(f"   • Agent: {self.dqn_manager.agent.__class__.__name__}")
                            logger.info(f"   • État dimensions: {self.dqn_manager.agent.state_dim}")
                            logger.info(f"   • Actions possibles: {self.dqn_manager.agent.action_dim}")
                            logger.info("=" * 60)
                        return True
                    else:
                        logger.error(f"❌ Échec initialisation DQN!")
                        logger.error("=" * 60)
                        return False
                else:
                    logger.error("❌ Aucun chemin de modèle valide!")
                    logger.error("=" * 60)
                    return False
                    
            except Exception as e:
                logger.error(f"❌ Erreur initialisation DQN: {e}")
                import traceback
                logger.error(traceback.format_exc())
                logger.error("=" * 60)
                return False
        return True
    
    def _save_dqn_model(self, suffix=""):
        """Sauvegarde le modèle DQN (si auto_save activée)"""
        try:
            # Ne sauvegarder que si auto_save est activé
            if (self.config.get('enable_dqn', False) and 
                self.dqn_manager.agent and 
                self.dqn_manager.auto_save_enabled):
                saved_path = self.dqn_manager.save_model(suffix=suffix)
                if saved_path:
                    self.log_queue.put(f"💾 Modèle DQN sauvegardé: {saved_path}")
                return saved_path
        except Exception as e:
            logger.error(f"Erreur sauvegarde DQN: {e}")
        return None
    
    def start(self):
        """Démarre la simulation"""
        if self.is_running:
            return False
        
        self.is_running = True
        
        # Réinitialiser composants
        self._initialize_components()
        
        # Initialiser DQN
        if self.config.get('enable_dqn', False):
            self._initialize_dqn()
        
        # Initialiser scheduler
        if self.config.get('enable_intelligent_scheduler', False) and INTELLIGENT_SCHEDULER_AVAILABLE:
            self._initialize_intelligent_scheduler()
        
        # Initialiser analyseur énergie
        if ENERGY_MODULE_AVAILABLE and hasattr(self, 'energy_analyzer'):
            self.energy_analyzer.reset_stats()
        
        self._initialize_collision_detection()
        
        self._reset_statistics()
        self.simulated_packets = []
        
        # Vider queues
        while not self.metric_queue.empty():
            try:
                self.metric_queue.get_nowait()
            except Empty:
                break
        
        while not self.log_queue.empty():
            try:
                self.log_queue.get_nowait()
            except Empty:
                break
        
        # Démarrer thread
        self.simulation_thread = threading.Thread(
            target=self.run,
            daemon=True
        )
        self.simulation_thread.start()
        
        return True
    
    def stop(self):
        """Arrête la simulation"""
        self.is_running = False
        return True
    
    def reset(self):
        """Réinitialise la simulation"""
        self.is_running = False
        
        self._initialize_collision_detection()
        self._reset_statistics()
        
        # Réinitialiser énergie
        if hasattr(self, 'energy_analyzer'):
            self.energy_analyzer.reset_stats()
        
        self.simulated_packets = []
        self.devices_state = {}
        self.active_fragments = []
        self.event_queue = []
        self.collision_events = []
        self.simulated_time = 0.0
        
        return True
    
    def get_metrics(self):
        """Récupère les dernières métriques depuis la queue"""
        metrics_list = []
        while not self.metric_queue.empty():
            try:
                msg_type, data = self.metric_queue.get_nowait()
                metrics_list.append((msg_type, data))
            except Empty:
                break
        return metrics_list
    
    def get_logs(self):
        """Récupère les logs depuis la queue"""
        logs = []
        while not self.log_queue.empty():
            try:
                log = self.log_queue.get_nowait()
                logs.append(log)
            except Empty:
                break
        return logs
    
    def generate_report(self):
        """Génère un rapport détaillé de la simulation"""
        stats = self.detailed_stats
        
        def safe_divide(num, denom, default=0.0):
            return num / denom if denom != 0 else default
        
        report = []
        
        # En-tête
        report.append("=" * 40)
        report.append("📊 RAPPORT DÉTAILLÉ DE SIMULATION LR-FHSS AVEC DQN, SCHEDULER ET ÉNERGIE")
        report.append("=" * 40)
        
        if stats.total_packets == 0:
            report.append("\n⚠️  AUCUNE DONNÉE DISPONIBLE")
            report.append(f"   Temps simulé: {self.simulated_time:.1f}s")
            report.append("   La simulation n'a généré aucun paquet.")
            report.append("=" * 80)
            return "\n".join(report)
        
        # Résumé général
        success_rate_pct = safe_divide(stats.successful_packets, stats.total_packets) * 100
        failure_rate_pct = safe_divide(stats.failed_packets, stats.total_packets) * 100
        collision_failure_pct = safe_divide(stats.collisions_causing_failure, stats.total_packets) * 100
        
        report.append("\n📈 RÉSUMÉ GÉNÉRAL:")
        report.append(f"   • Temps simulé: {self.simulated_time:.1f}s")
        report.append(f"   • Nombre total de paquets: {stats.total_packets:,}")
        report.append(f"   • ✅ Paquets réussis: {stats.successful_packets:,} ({success_rate_pct:.1f}%)")
        report.append(f"   • ❌ Paquets échoués: {stats.failed_packets:,} ({failure_rate_pct:.1f}%)")
        report.append(f"   • 💥 Collisions (échecs): {stats.collisions_causing_failure:,} ({collision_failure_pct:.1f}%)")
        
        # Performance Énergie
        if ENERGY_MODULE_AVAILABLE and hasattr(self, 'energy_analyzer'):
            energy_stats = self.energy_analyzer.energy_stats
            packets_analyzed = energy_stats.get('packets_analyzed', 0)
            
            if packets_analyzed > 0:
                total_energy_j = energy_stats.get('total_energy_j', 0.0)
                avg_energy_per_packet_j = energy_stats.get('avg_energy_per_packet_j', 0.0)
                battery_life_years = energy_stats.get('battery_life_years', 0.0)
                avg_current_ma = energy_stats.get('avg_current_ma', 0.0)
                
                report.append("\n🔋 PERFORMANCE ÉNERGÉTIQUE:")
                report.append(f"   • Paquets analysés: {packets_analyzed:,}")
                report.append(f"   • Énergie totale consommée: {total_energy_j:.3f} J")
                report.append(f"   • Énergie moyenne par paquet: {avg_energy_per_packet_j*1000:.1f} mJ")
                report.append(f"   • Courant Tx moyen: {avg_current_ma:.1f} mA")
                report.append(f"   • Durée vie batterie estimée: {battery_life_years:.1f} ans")
                report.append(f"   • Type PA utilisé: {self.config.get('pa_type', 'SX1261_LP')}")
        
        # Performance DQN
        if self.config.get('enable_dqn', False) and stats.dqn_decisions > 0:
            dqn_success_rate = safe_divide(stats.dqn_successes, stats.dqn_decisions) * 100
            dqn_avg_power_saved = safe_divide(stats.dqn_power_saved, stats.dqn_decisions)
            
            report.append("\n🤖 PERFORMANCE DQN (DÉCISIONS TEMPS RÉEL):")
            report.append(f"   • Décisions DQN: {stats.dqn_decisions:,}")
            report.append(f"   • Succès DQN: {stats.dqn_successes:,} ({dqn_success_rate:.1f}%)")
            report.append(f"   • Puissance économisée moyenne: {dqn_avg_power_saved:.2f} dBm")
            
            # Distribution DR DQN
            report.append("   • Distribution DR DQN:")
            total_dqn_dr = sum(stats.dqn_recommendations[f'dr{dr}'] for dr in [8, 9, 10, 11])
            if total_dqn_dr > 0:
                for dr in [8, 9, 10, 11]:
                    count = stats.dqn_recommendations[f'dr{dr}']
                    percentage = safe_divide(count, total_dqn_dr) * 100
                    report.append(f"     - DR{dr}: {count:,} ({percentage:.1f}%)")
        
        # Performance du scheduler
        if self.config.get('enable_intelligent_scheduler', False) and stats.scheduler_decisions > 0:
            report.append("\n⏱️ PERFORMANCE DU SCHEDULER INTELLIGENT:")
            report.append(f"   • Décisions scheduler: {stats.scheduler_decisions:,}")
            report.append(f"   • Délais appliqués: {stats.scheduler_delays:,} ({safe_divide(stats.scheduler_delays, stats.scheduler_decisions)*100:.1f}%)")
            report.append(f"   • Somme des délais: {stats.scheduler_delays_sum:.2f}s")
            report.append(f"   • Shifts fréquentiels: {stats.scheduler_freq_shifts:,} ({safe_divide(stats.scheduler_freq_shifts, stats.scheduler_decisions)*100:.1f}%)")
            report.append(f"   • Boosts puissance: {stats.scheduler_power_boosts:,} ({safe_divide(stats.scheduler_power_boosts, stats.scheduler_decisions)*100:.1f}%)")
            
            # Stratégies utilisées
            report.append("   • Stratégies utilisées:")
            for strategy, count in sorted(stats.scheduler_strategies.items(), key=lambda x: -x[1]):
                percentage = safe_divide(count, stats.scheduler_decisions) * 100
                report.append(f"     - {strategy}: {count:,} ({percentage:.1f}%)")
        
        # ... (rest of report generation similar to original)
        
        report.append("=" * 80)
        return "\n".join(report)
    
    def export_report(self):
        """Exporte le rapport complet AVEC shadowing"""
        import json
        import pandas as pd
        import time as time_module
        
        timestamp = time_module.strftime("%Y%m%d_%H%M%S")
        
        # Générer rapport structuré
        structured_report = {
            'metadata': {
                'export_timestamp': time_module.strftime("%Y-%m-%d %H:%M:%S"),
                'simulation_duration_s': self.config.get('simulation_duration', 7200),
                'num_devices': self.config.get('num_devices', 50),
                'region': self.config.get('region', 'EU868'),
                'coding_rate': self.config.get('coding_rate', '1/3'),
                'bandwidth_khz': self.config.get('bandwidth_khz', 136.71875),
                'tx_power_dbm': self.config.get('tx_power', 14.0),
                'distance_max_m': self.config.get('distance_gtw', 10000),
                'shadowing_std_db': self.config.get('shadowing_std_db', 7.0),
                'enable_dqn': self.config.get('enable_dqn', False),
                'enable_scheduler': self.config.get('enable_intelligent_scheduler', False),
                'dqn_model': self.config.get('dqn_model_name', None),
                'energy_module_available': ENERGY_MODULE_AVAILABLE,
                'pa_type': self.config.get('pa_type', 'SX1261_LP'),
                'battery_capacity_mah': self.config.get('battery_capacity_mah', 1000.0)
            },
            'summary_metrics': {
                'total_packets': self.total_sent,
                'successful_packets': self.successful_rx,
                'failed_packets': self.total_sent - self.successful_rx if self.total_sent > 0 else 0,
                'success_rate_pct': self.success_rate,
                'collision_rate_pct': self.collision_rate,
                'simulated_time_s': self.simulated_time
            },
            'shadowing_stats': self._get_shadowing_statistics(),  # NOUVELLE SECTION
            'energy_metrics': {
                'total_energy_j': self.energy_analyzer.energy_stats.get('total_energy_j', 0.0) if hasattr(self, 'energy_analyzer') else 0.0,
                'avg_energy_per_packet_j': self.energy_analyzer.energy_stats.get('avg_energy_per_packet_j', 0.0) if hasattr(self, 'energy_analyzer') else 0.0,
                'avg_current_ma': self.energy_analyzer.energy_stats.get('avg_current_ma', 0.0) if hasattr(self, 'energy_analyzer') else 0.0,
                'battery_life_years': self.energy_analyzer.energy_stats.get('battery_life_years', 0.0) if hasattr(self, 'energy_analyzer') else 0.0,
                'energy_per_bit_j': self.energy_analyzer.energy_stats.get('energy_per_bit_j', 0.0) if hasattr(self, 'energy_analyzer') else 0.0
            } if ENERGY_MODULE_AVAILABLE else {},
            'dqn_performance': {
                'decisions': self.dqn_decisions,
                'successes': self.dqn_successes,
                'avg_power_saved_dbm': self.dqn_avg_power_saved
            },
            'scheduler_performance': {
                'decisions': self.detailed_stats.scheduler_decisions,
                'delays_applied': self.detailed_stats.scheduler_delays,
                'freq_shifts_applied': self.detailed_stats.scheduler_freq_shifts,
                'strategy_distribution': dict(self.detailed_stats.scheduler_strategies)
            }
        }
        
        # Export JSON
        json_filename = f"lrfhss_simulation_metrics_{timestamp}.json"
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(structured_report, f, indent=4, default=str)
        
        # Export CSV AVEC SHADOWING
        csv_filename = None
        if self.simulated_packets:
            csv_filename = self._export_enriched_csv(timestamp)
        
        # Export rapport texte
        txt_report = self.generate_report()
        txt_filename = f"lrfhss_simulation_report_{timestamp}.txt"
        with open(txt_filename, 'w', encoding='utf-8') as f:
            f.write(txt_report)
        
        return {
            'json': json_filename,
            'csv': csv_filename,
            'txt': txt_filename
        }
    

    def _export_enriched_csv(self, timestamp):
        """Exporte un CSV enrichi avec shadowing et positions"""
        import pandas as pd
        
        if not self.simulated_packets:
            return None
        
        csv_filename = f"lrfhss_simulation_with_shadowing_{timestamp}.csv"
        
        packet_data = []
        
        for packet in self.simulated_packets:
            # Récupérer les informations du device
            device_id = getattr(packet, 'device_id', '')
            
            # Récupérer position et distance du device
            position = getattr(packet, 'position', (0, 0))
            distance_km = getattr(packet, 'distance_km', 0)
            
            # Si non défini dans le paquet, chercher dans devices_state
            if position == (0, 0) and device_id in self.devices_state:
                state = self.devices_state[device_id]
                position = state.get('position', (0, 0))
                distance_km = state.get('distance_km', 0)
            
            # Calculer shadowing si non déjà calculé
            shadowing_db = getattr(packet, 'shadowing_db', 0)
            path_loss_db = getattr(packet, 'path_loss_db', 0)
            
            # Si shadowing non calculé, le calculer à partir du RSSI
            if shadowing_db == 0 and hasattr(packet, 'rssi_dbm') and hasattr(packet, 'tx_power_dbm'):
                # RSSI = TxPower - PathLoss + Shadowing
                # Donc Shadowing = RSSI - TxPower + PathLoss
                shadowing_db = packet.rssi_dbm - packet.tx_power_dbm + path_loss_db
            
            # Données enrichies
            row = {
                'packet_id': getattr(packet, 'packet_id', ''),
                'device_id': device_id,
                'start_time': getattr(packet, 'start_time', 0),
                'end_time': getattr(packet, 'end_time', 0),
                'toa_ms': getattr(packet, 'toa_ms', 0),
                'frequency_mhz': getattr(packet, 'frequency_mhz', 0),
                'tx_power_dbm': getattr(packet, 'tx_power_dbm', 0),
                'dr': getattr(packet, 'dr', 0),
                'cr': getattr(packet, 'cr', ''),
                'bw_khz': getattr(packet, 'bw_khz', 0),
                'payload_bytes': getattr(packet, 'payload_bytes', 0),
                
                # MÉTRIQUES RF AVEC SHADOWING
                'rssi_with_shadowing_dbm': getattr(packet, 'rssi_dbm', -120),
                'path_loss_db': path_loss_db,
                'shadowing_db': shadowing_db,
                'snr_db': getattr(packet, 'snr_db', 0),
                'ber': getattr(packet, 'ber', 0),
                
                # INFORMATIONS GÉOGRAPHIQUES
                'distance_km': distance_km,
                'position_x_m': position[0],
                'position_y_m': position[1],
                'distance_to_gateway_m': np.sqrt(position[0]**2 + position[1]**2),
                
                # RÉSULTATS
                'success': getattr(packet, 'success', False),
                'collision': getattr(packet, 'collision', False),
                'fec_recovered': getattr(packet, 'fec_recovered', False),
                'failure_reason': getattr(packet, 'failure_reason', ''),
                
                # INFORMATIONS DQN
                'dqn_applied': getattr(packet, 'dqn_applied', False),
                'dqn_dr': getattr(packet, 'dqn_dr', None),
                'dqn_power_dbm': getattr(packet, 'dqn_power', None),
                'dqn_frequency_mhz': getattr(packet, 'dqn_frequency', None),
                
                # INFORMATIONS SCHEDULER
                'scheduler_applied': getattr(packet, 'scheduler_applied', False),
                'scheduler_strategy': getattr(packet, 'scheduler_strategy', ''),
                'scheduler_delay_s': getattr(packet, 'scheduler_delay', 0),
                'scheduler_freq_shift_hz': getattr(packet, 'scheduler_freq_shift', 0),
                'scheduler_power_boost_db': getattr(packet, 'scheduler_power_boost', 0),
                
                # INFORMATIONS ÉNERGIE
                'pa_type_used': getattr(packet, 'pa_type_used', ''),
                'energy_total_j': getattr(packet, 'energy_total_j', 0) if hasattr(packet, 'energy_total_j') else 0,
            }
            
            # Ajouter les détails de collision si disponibles
            collision_details = getattr(packet, 'collision_details', [])
            if collision_details:
                row['collision_count'] = len(collision_details)
                row['header_collisions'] = sum(1 for c in collision_details if 'header' in c.collision_type.lower())
                row['payload_collisions'] = sum(1 for c in collision_details if 'payload' in c.collision_type.lower())
                row['capture_effects'] = sum(1 for c in collision_details if getattr(c, 'capture_effect', False))
            else:
                row['collision_count'] = 0
                row['header_collisions'] = 0
                row['payload_collisions'] = 0
                row['capture_effects'] = 0
            
            packet_data.append(row)
        
        if packet_data:
            df = pd.DataFrame(packet_data)
            
            # Trier par temps de départ
            df = df.sort_values('start_time')
            
            # Sauvegarder
            df.to_csv(csv_filename, index=False, encoding='utf-8')
            
            # Statistiques pour log
            shadowing_mean = df['shadowing_db'].mean()
            shadowing_std = df['shadowing_db'].std()
            shadowing_min = df['shadowing_db'].min()
            shadowing_max = df['shadowing_db'].max()
            
            config_std = self.config.get('shadowing_std_db', 7.0)
            
            self.log_queue.put(f"📊 CSV enrichi exporté: {csv_filename}")
            self.log_queue.put(f"   • {len(packet_data)} paquets avec shadowing")
            self.log_queue.put(f"   • Shadowing: moy={shadowing_mean:.2f}dB, σ={shadowing_std:.2f}dB")
            self.log_queue.put(f"   • Range: [{shadowing_min:.2f}, {shadowing_max:.2f}] dB")
            self.log_queue.put(f"   • Config σ: {config_std:.1f} dB")
            self.log_queue.put(f"   • RSSI moyen: {df['rssi_with_shadowing_dbm'].mean():.2f} dBm")
            
            return csv_filename
        
        return None
    
    
    def _get_shadowing_statistics(self):
        """Calcule les statistiques de shadowing pour le rapport"""
        if not self.simulated_packets:
            return {}
        
        shadowing_values = []
        rssi_values = []
        
        for packet in self.simulated_packets:
            if hasattr(packet, 'shadowing_db'):
                shadowing_values.append(packet.shadowing_db)
            if hasattr(packet, 'rssi_dbm'):
                rssi_values.append(packet.rssi_dbm)
        
        if not shadowing_values:
            return {"error": "Aucune donnée de shadowing disponible"}
        
        # Statistiques de base
        stats = {
            'count': len(shadowing_values),
            'mean_db': float(np.mean(shadowing_values)),
            'std_db': float(np.std(shadowing_values)),
            'min_db': float(np.min(shadowing_values)),
            'max_db': float(np.max(shadowing_values)),
            'median_db': float(np.median(shadowing_values)),
            'q1_db': float(np.percentile(shadowing_values, 25)),
            'q3_db': float(np.percentile(shadowing_values, 75)),
            'config_std_db': self.config.get('shadowing_std_db', 7.0),
            'rssi_mean_db': float(np.mean(rssi_values)) if rssi_values else None,
        }
        
        # Distribution (histogramme)
        hist, bins = np.histogram(shadowing_values, bins=20)
        stats['histogram'] = {
            'counts': hist.tolist(),
            'bin_edges': bins.tolist()
        }
        
        # Vérification de la distribution normale
        from scipy import stats as scipy_stats
        try:
            # Test de Shapiro-Wilk pour normalité
            if len(shadowing_values) > 3 and len(shadowing_values) < 5000:
                shapiro_stat, shapiro_p = scipy_stats.shapiro(shadowing_values)
                stats['normality_test'] = {
                    'shapiro_wilk_statistic': float(shapiro_stat),
                    'shapiro_wilk_p_value': float(shapiro_p),
                    'is_normal': shapiro_p > 0.05  # Au seuil de 5%
                }
        except:
            pass  # Ignorer si scipy n'est pas disponible
        
        # Corrélation avec la distance
        if hasattr(self, 'devices_state'):
            distances = []
            device_shadowings = {}
            
            # Grouper par device pour moyenne
            for device_id, state in self.devices_state.items():
                device_packets = [p for p in self.simulated_packets if getattr(p, 'device_id', '') == device_id]
                if device_packets:
                    device_shadowings_list = [getattr(p, 'shadowing_db', 0) for p in device_packets]
                    if device_shadowings_list:
                        device_shadowings[device_id] = np.mean(device_shadowings_list)
                        distances.append(state.get('distance_km', 0))
            
            if distances and len(device_shadowings) > 1:
                avg_shadowings = list(device_shadowings.values())
                if len(distances) == len(avg_shadowings):
                    try:
                        correlation = np.corrcoef(distances, avg_shadowings)[0, 1]
                        stats['distance_correlation'] = float(correlation)
                    except:
                        pass
        
        return stats