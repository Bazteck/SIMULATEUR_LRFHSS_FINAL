# smart_scheduler.py - VERSION ALIGNÉE AVEC SIMULATEUR
"""
Scheduler intelligent aligné avec la structure du simulateur
- Utilise TransmissionFragment de lr_fhss.py
- Compatible avec la structure actuelle du simulateur
- Maintient la logique d'évitement de collisions
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import heapq
from enum import Enum
import math
import hashlib
from collections import defaultdict, deque
import time

# Importer les structures du simulateur
from lr_fhss import (
    TransmissionFragment,
    SimulatedPacket,
    FragmentCollisionResult,
    generate_lrfhss_fragments,
    HEADER_DURATION_S,
    PAYLOAD_DURATION_S,
    HEADER_COUNT,
    OBW_HZ
)

class CollisionSeverity(Enum):
    NONE = 0
    MINOR = 1      # Payload-payload avec faible overlap
    MODERATE = 2   # Header-payload ou payload-payload modéré
    SEVERE = 3     # Header-payload avec fort overlap
    CRITICAL = 4   # Header-header (FATAL)
    CATASTROPHIC = 5  # Header-header + effet domino

@dataclass
class CollisionPrediction:
    """Prédiction de collision avec scoring"""
    fragment1: TransmissionFragment
    fragment2: TransmissionFragment
    predicted_time: float
    severity: CollisionSeverity
    confidence_score: float  # 0-1
    overlap_ratio: float
    frequency_overlap_khz: float
    recommended_action: str  # 'DELAY', 'FREQ_SHIFT', 'CANCEL', 'ACCEPT'
    avoidance_priority: int
    power_diff_db: float = 0.0
    capture_effect: bool = False

class NetworkState:
    """État complet du réseau pour prise de décision"""
    def __init__(self):
        self.fragment_timeline = []  # Fragments ordonnés par temps
        self.device_history = defaultdict(list)  # Historique par device
        self.collision_history = deque(maxlen=1000)  # Historique collisions
        self.channel_occupancy = defaultdict(float)  # Occupation par canal
        self.time_windows = []  # Fenêtres temporelles occupées
        self.risk_map = {}  # Carte de risque temps-fréquence
        
    def get_risk_at(self, time_s: float, freq_mhz: float) -> float:
        """Retourne le risque de collision à un point donné"""
        risk = 0.0
        lrfhss_bw_hz = OBW_HZ  # 488.28125 Hz
        
        for frag in self.fragment_timeline:
            # Chevaucher temporel
            if not (frag.end_time <= time_s or frag.start_time >= time_s + 0.1):  # Fenêtre de 100ms
                # Chevaucher fréquentiel avec bande LR-FHSS
                freq_diff_hz = abs(frag.frequency_mhz - freq_mhz) * 1e6
                if freq_diff_hz <= lrfhss_bw_hz:
                    time_dist = min(abs(frag.start_time - time_s), abs(frag.end_time - time_s))
                    if time_dist < 0.05:  # Dans les 50ms
                        time_risk = 1.0 - (time_dist / 0.05)
                        risk = max(risk, time_risk)
        
        return min(1.0, risk)

class IntelligentScheduler:
    """Scheduler intelligent avec prédiction et évitement avancé"""
    
    def __init__(self, time_resolution_ms: float = 1.0, prediction_horizon_s: float = 5.0):
        self.time_resolution = time_resolution_ms / 1000.0
        self.prediction_horizon = prediction_horizon_s
        
        # Structures de données optimisées
        self.active_fragments = []  # Fragments actuellement en transmission
        self.scheduled_fragments = []  # Fragments planifiés mais pas encore actifs
        self.network_state = NetworkState()
        
        # Cache pour performances
        self.fragment_cache = {}
        self.collision_cache = {}
        
        # Statistiques
        self.collisions_prevented = 0
        self.delays_applied = 0
        self.freq_shifts_applied = 0
        self.cancellations = 0
        
        # Paramètres LR-FHSS (alignés avec config.py)
        self.lrfhss_params = {
            8: {'bw_khz': 137, 'cr': '1/3', 'grid_khz': 3.9, 'channels': 35},
            9: {'bw_khz': 137, 'cr': '2/3', 'grid_khz': 3.9, 'channels': 35},
            10: {'bw_khz': 336, 'cr': '1/3', 'grid_khz': 3.9, 'channels': 86},
            11: {'bw_khz': 336, 'cr': '2/3', 'grid_khz': 3.9, 'channels': 86},
        }
        
        # Constantes conformes
        self.HEADER_DURATION_S = HEADER_DURATION_S
        self.PAYLOAD_DURATION_S = PAYLOAD_DURATION_S
        self.HEADER_COUNT = HEADER_COUNT
        self.LRFHSS_BW_HZ = OBW_HZ  # 488.28125 Hz
        self.CAPTURE_THRESHOLD_DB = 4.0
        
        # Stratégies d'évitement
        self.avoidance_strategies = {
            'DELAY': {'max_delay_s': 10.0, 'step_ms': 100},
            'FREQ_SHIFT': {'max_shift_khz': 500, 'step_khz': 3.9},
            'POWER_ADJUST': {'max_boost_db': 6, 'step_db': 1},
        }
        
        # Apprentissage par renforcement simple
        self.strategy_success_rates = defaultdict(lambda: 0.8)
        self.device_profiles = {}
        
        # Flag pour indiquer si le scheduler est chargé
        self.is_loaded = True
    
    def schedule_transmission(self, device_id: str, dr,  # Accepte int ou str
                             frequency_mhz: float, payload_bytes: int,
                             tx_power_dbm: float = 14.0,
                             transmission_id: int = None,
                             device_position: tuple = (0, 0)) -> Dict:
        """
        Planifie une transmission avec évitement intelligent
        
        Args:
            device_id: ID du device
            dr: Data Rate (8-11 ou "DR8"-"DR11")
            frequency_mhz: Fréquence de départ
            payload_bytes: Taille du payload
            tx_power_dbm: Puissance TX
            transmission_id: ID unique de transmission
            device_position: Position (x, y) du device
            
        Returns:
            Dict avec start_time, strategy, adjusted_fragments, payload_bytes, etc.
        """
        # Convertir dr en int si c'est une string
        if isinstance(dr, str):
            # Extraire le numéro de "DR8" -> 8
            dr_int = int(dr.replace('DR', '').replace('dr', ''))
        else:
            dr_int = int(dr)
        
        # Préparer les paramètres pour generate_lrfhss_fragments
        params = self.lrfhss_params.get(dr_int, self.lrfhss_params[8])
        params_dict = {
            'dr': dr_int,
            'cr': params['cr'],
            'bw_khz': params['bw_khz'],
            'payload_bytes': payload_bytes,
            'tx_power_dbm': tx_power_dbm
        }
        
        # Générer les fragments via lr_fhss.py
        fragments = generate_lrfhss_fragments(
            start_time=0.0,
            params=params_dict,
            device_id=device_id,
            transmission_id=transmission_id
        )
        
        if not fragments:
            return {
                'start_time': 0.0,
                'strategy': 'IMMEDIATE',
                'adjusted_fragments': [],
                'collision_score': 0.0,
                'delay_applied': 0.0,
                'freq_shift_applied': 0.0,
                'payload_bytes': payload_bytes,  # AJOUTÉ
                'dr': dr_int,  # AJOUTÉ
                'tx_power_dbm': tx_power_dbm  # AJOUTÉ
            }
        
        # Trouver le slot optimal
        slot = self.find_optimal_transmission_slot(
            fragments=fragments,
            device_id=device_id,
            max_search_time_s=10.0
        )
        
        # Ajouter les informations de transmission au résultat
        slot['payload_bytes'] = payload_bytes
        slot['dr'] = dr_int
        slot['tx_power_dbm'] = tx_power_dbm
        
        # Mettre à jour les structures de données
        for frag in slot['adjusted_fragments']:
            self.scheduled_fragments.append(frag)
            
            # Si la transmission commence immédiatement, ajouter aux actifs
            if frag.start_time <= 0.1:  # Dans les 100ms
                self.active_fragments.append(frag)
        
        # Mettre à jour les statistiques
        if slot.get('was_necessary', False):
            self.collisions_prevented += 1
        
        if slot['delay_applied'] > 0:
            self.delays_applied += 1
        
        if slot['freq_shift_applied'] != 0:
            self.freq_shifts_applied += 1
        
        # Mettre à jour le profil du device
        self._update_device_profile(device_id, slot)
        
        return slot
    
    def find_optimal_transmission_slot(self, fragments: List[TransmissionFragment],
                                      device_id: str,
                                      max_search_time_s: float = 10.0) -> Dict:
        """
        Trouve le slot optimal pour une transmission
        
        Args:
            fragments: Liste de TransmissionFragment générés par lr_fhss.py
            device_id: ID du device
            max_search_time_s: Temps max de recherche
            
        Returns:
            Dict avec start_time, strategy, adjusted_fragments, etc.
        """
        # Stratégie 1: Transmission immédiate
        immediate_score = self._evaluate_slot(fragments, 0.0)
        
        if immediate_score < 0.1:  # Risque très faible
            return self._build_slot_result(
                fragments, 0.0, 'IMMEDIATE', immediate_score, 0.0, 0.0
            )
        
        # Stratégie 2: Délai progressif
        best_delay_result = None
        best_delay_score = immediate_score
        
        for delay_ms in range(100, int(max_search_time_s * 1000), 100):
            delay_s = delay_ms / 1000.0
            delayed_frags = [self._adjust_fragment_time(f, delay_s) for f in fragments]
            score = self._evaluate_slot(delayed_frags, delay_s)
            
            if score < best_delay_score * 0.5:  # Amélioration significative
                best_delay_score = score
                best_delay_result = self._build_slot_result(
                    fragments, delay_s, 'DELAY', score, delay_s, 0.0
                )
                
                if score < 0.05:  # Excellent slot trouvé
                    break
        
        # Stratégie 3: Power boost (si délai insuffisant)
        if best_delay_result is None or best_delay_score > 0.3:
            power_boost_result = self._try_power_boost(fragments, device_id)
            if power_boost_result and power_boost_result['collision_score'] < best_delay_score:
                return power_boost_result
        
        # Retourner le meilleur résultat
        if best_delay_result:
            return best_delay_result
        else:
            return self._build_slot_result(
                fragments, 0.0, 'IMMEDIATE', immediate_score, 0.0, 0.0
            )
    
    def _evaluate_slot(self, fragments: List[TransmissionFragment], 
                       start_offset: float) -> float:
        """
        Évalue le score de collision d'un slot
        
        Args:
            fragments: Fragments à évaluer
            start_offset: Décalage temporel
            
        Returns:
            Score de collision (0=aucune, 1=collision sévère)
        """
        if not fragments:
            return 0.0
        
        collision_score = 0.0
        
        # Vérifier contre tous les fragments actifs et planifiés
        all_existing = self.active_fragments + self.scheduled_fragments
        
        for new_frag in fragments:
            for existing_frag in all_existing:
                # Vérifier collision temporelle
                if self._fragments_overlap_time(new_frag, existing_frag):
                    # Vérifier collision fréquentielle
                    if self._fragments_overlap_freq(new_frag, existing_frag):
                        # Calculer sévérité
                        severity = self._calculate_collision_severity(new_frag, existing_frag)
                        collision_score += severity
        
        # Normaliser par nombre de fragments
        if fragments:
            collision_score /= len(fragments)
        
        return min(1.0, collision_score)
    
    def _fragments_overlap_time(self, frag1: TransmissionFragment, 
                               frag2: TransmissionFragment) -> bool:
        """Vérifie si deux fragments se chevauchent temporellement"""
        return not (frag1.end_time <= frag2.start_time or 
                   frag1.start_time >= frag2.end_time)
    
    def _fragments_overlap_freq(self, frag1: TransmissionFragment,
                               frag2: TransmissionFragment) -> bool:
        """Vérifie si deux fragments se chevauchent fréquentiellement"""
        freq_diff_hz = abs(frag1.frequency_mhz - frag2.frequency_mhz) * 1e6
        return freq_diff_hz < self.LRFHSS_BW_HZ
    
    def _calculate_collision_severity(self, frag1: TransmissionFragment,
                                     frag2: TransmissionFragment) -> float:
        """
        Calcule la sévérité d'une collision
        
        Returns:
            Score 0-1 (0=pas grave, 1=critique)
        """
        # Collision header-header = critique
        if frag1.fragment_type == 'header' and frag2.fragment_type == 'header':
            return 1.0
        
        # Collision header-payload = sévère
        if (frag1.fragment_type == 'header' and frag2.fragment_type == 'payload') or \
           (frag1.fragment_type == 'payload' and frag2.fragment_type == 'header'):
            return 0.7
        
        # Collision payload-payload = modéré
        # Calculer overlap ratio
        overlap_start = max(frag1.start_time, frag2.start_time)
        overlap_end = min(frag1.end_time, frag2.end_time)
        overlap_duration = max(0, overlap_end - overlap_start)
        
        total_duration = max(frag1.end_time - frag1.start_time,
                           frag2.end_time - frag2.start_time)
        
        if total_duration > 0:
            overlap_ratio = overlap_duration / total_duration
            return 0.3 * overlap_ratio
        
        return 0.1
    
    def _try_power_boost(self, fragments: List[TransmissionFragment],
                        device_id: str) -> Optional[Dict]:
        """
        Essaie d'améliorer via power boost (effet capture)
        
        Args:
            fragments: Fragments à booster
            device_id: ID du device
            
        Returns:
            Dict avec résultat ou None
        """
        if not fragments:
            return None
        
        # Récupérer puissance actuelle
        current_power = fragments[0].tx_power_dbm
        max_power = 14.0  # EU868 limit
        
        if current_power >= max_power:
            return None  # Déjà au max
        
        # Essayer boost de 3 dB
        boost_db = min(3.0, max_power - current_power)
        new_power = current_power + boost_db
        
        # Créer fragments avec nouvelle puissance
        boosted_frags = []
        for frag in fragments:
            boosted_frag = TransmissionFragment(
                start_time=frag.start_time,
                end_time=frag.end_time,
                frequency_mhz=frag.frequency_mhz,
                fragment_type=frag.fragment_type,
                fragment_index=frag.fragment_index,
                bw_khz=frag.bw_khz,
                channel_offset=frag.channel_offset,
                absolute_channel=frag.absolute_channel,
                grid_id=frag.grid_id,
                cr=frag.cr,
                hop_number=frag.hop_number,
                instantaneous_bw_hz=frag.instantaneous_bw_hz,
                grid_spacing_khz=frag.grid_spacing_khz,
                bits_in_fragment=frag.bits_in_fragment,
                is_last_hop=frag.is_last_hop,
                tx_power_dbm=new_power
            )
            boosted_frags.append(boosted_frag)
        
        # Évaluer avec effet capture (réduction du score)
        score = self._evaluate_slot(boosted_frags, 0.0)
        effective_score = score * 0.5  # 50% de réduction grâce à effet capture
        
        return {
            'start_time': 0.0,
            'strategy': 'POWER_BOOST',
            'adjusted_fragments': boosted_frags,
            'collision_score': effective_score,
            'delay_applied': 0.0,
            'freq_shift_applied': 0.0,
            'power_boost_db': boost_db,
            'was_necessary': True
        }
    
    def _build_slot_result(self, fragments: List[TransmissionFragment], 
                          start_time: float,
                          strategy: str, 
                          collision_score: float,
                          delay: float, 
                          freq_shift: float) -> Dict:
        """Construit le résultat d'un slot"""
        adjusted_frags = [self._adjust_fragment_time(f, start_time) for f in fragments]
        
        return {
            'start_time': start_time,
            'strategy': strategy,
            'adjusted_fragments': adjusted_frags,
            'collision_score': collision_score,
            'delay_applied': delay,
            'freq_shift_applied': freq_shift,
            'was_necessary': strategy != 'IMMEDIATE' and (delay > 0 or abs(freq_shift) > 0)
        }
    
    def _adjust_fragment_time(self, fragment: TransmissionFragment, 
                             time_offset: float) -> TransmissionFragment:
        """Crée une copie d'un fragment avec temps ajusté"""
        return TransmissionFragment(
            start_time=fragment.start_time + time_offset,
            end_time=fragment.end_time + time_offset,
            frequency_mhz=fragment.frequency_mhz,
            fragment_type=fragment.fragment_type,
            fragment_index=fragment.fragment_index,
            bw_khz=fragment.bw_khz,
            channel_offset=fragment.channel_offset,
            absolute_channel=fragment.absolute_channel,
            grid_id=fragment.grid_id,
            cr=fragment.cr,
            hop_number=fragment.hop_number,
            instantaneous_bw_hz=fragment.instantaneous_bw_hz,
            grid_spacing_khz=fragment.grid_spacing_khz,
            bits_in_fragment=fragment.bits_in_fragment,
            is_last_hop=fragment.is_last_hop,
            tx_power_dbm=fragment.tx_power_dbm
        )
    
    def _update_device_profile(self, device_id: str, slot_result: Dict):
        """Met à jour le profil d'apprentissage du device"""
        if device_id not in self.device_profiles:
            self.device_profiles[device_id] = {
                'transmission_count': 0,
                'delayed_count': 0,
                'freq_shifted_count': 0,
                'power_boosted_count': 0,
                'success_rate': 0.5,
                'preferred_strategy': 'IMMEDIATE'
            }
        
        profile = self.device_profiles[device_id]
        profile['transmission_count'] += 1
        
        if slot_result['strategy'] == 'DELAY':
            profile['delayed_count'] += 1
        elif slot_result['strategy'] == 'FREQ_SHIFT':
            profile['freq_shifted_count'] += 1
        elif slot_result['strategy'] == 'POWER_BOOST':
            profile['power_boosted_count'] += 1
        
        # Mettre à jour stratégie préférée
        delay_ratio = profile['delayed_count'] / max(1, profile['transmission_count'])
        if delay_ratio > 0.7:
            profile['preferred_strategy'] = 'PROACTIVE_DELAY'
        elif delay_ratio > 0.3:
            profile['preferred_strategy'] = 'CAUTIOUS'
        else:
            profile['preferred_strategy'] = 'AGGRESSIVE'
    
    def cleanup_old_fragments(self, current_time: float):
        """Nettoie les fragments terminés"""
        # Nettoyer fragments actifs
        self.active_fragments = [
            f for f in self.active_fragments 
            if f.end_time > current_time - 1.0  # Garder 1 seconde de marge
        ]
        
        # Nettoyer fragments planifiés
        self.scheduled_fragments = [
            f for f in self.scheduled_fragments
            if f.start_time > current_time - 5.0  # Garder 5 secondes de planification
        ]
    
    def get_network_metrics(self) -> Dict:
        """Retourne les métriques du réseau"""
        total_fragments = len(self.active_fragments) + len(self.scheduled_fragments)
        
        # Calculer densité temporelle
        if total_fragments > 0:
            time_window = 10.0  # Fenêtre de 10 secondes
            density = total_fragments / (time_window * 10)  # Normalisé
        else:
            density = 0.0
        
        return {
            'active_fragments': len(self.active_fragments),
            'scheduled_fragments': len(self.scheduled_fragments),
            'collisions_prevented': self.collisions_prevented,
            'delays_applied': self.delays_applied,
            'freq_shifts_applied': self.freq_shifts_applied,
            'cancellations': self.cancellations,
            'temporal_density': density,
            'device_profiles': len(self.device_profiles),
            'fragment_cache_size': len(self.fragment_cache),
            'avg_collision_risk': self._calculate_avg_collision_risk()
        }
    
    def _calculate_avg_collision_risk(self) -> float:
        """Calcule le risque de collision moyen dans le réseau"""
        if not self.active_fragments:
            return 0.0
        
        total_risk = 0.0
        count = 0
        
        for frag in self.active_fragments:
            for other_frag in self.active_fragments:
                if id(frag) == id(other_frag):
                    continue
                
                if self._fragments_overlap_time(frag, other_frag) and \
                   self._fragments_overlap_freq(frag, other_frag):
                    total_risk += 1.0
                count += 1
        
        return total_risk / max(1, count)
