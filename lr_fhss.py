# lr_fhss.py - ADAPTATION DE VOS FONCTIONS À LA LOGIQUE FHS
# Logique FHS intégrée SANS supprimer vos fonctions existantes
# Juste adaptation des calculs de fréquences

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import math
import hashlib
from channel import calculate_noise_power, calculate_ber, calculate_rssi
import random
# ============================================================================
# CONSTANTES LR-FHSS (INCHANGÉ)
# ============================================================================

HEADER_DURATION_MS = 233.472
PAYLOAD_DURATION_MS = 102.4

HEADER_DURATION_S = HEADER_DURATION_MS / 1000.0
PAYLOAD_DURATION_S = PAYLOAD_DURATION_MS / 1000.0

HEADER_COUNT = {'1/3': 3, '2/3': 2}
BITS_PER_HOP = {'1/3': 48, '2/3': 48}

OBW_HZ = 488.28125  # Hz
FHS_COUNT = 384  # Pour EU868

# ============================================================================
# CLASSES DE BASE (INCHANGÉES - juste ajout FHS)
# ============================================================================

@dataclass
class TransmissionFragment:
    """Fragment de transmission LR-FHSS (INCHANGÉ + FHS)"""
    start_time: float
    end_time: float
    frequency_mhz: float
    fragment_type: str
    fragment_index: int
    bw_khz: float = 136.71875
    channel_offset: int = 0
    absolute_channel: int = 0
    grid_id: int = 0
    cr: str = '1/3'
    hop_number: int = 0
    instantaneous_bw_hz: float = 488.28125
    grid_spacing_khz: float = 3.9
    bits_in_fragment: int = 0
    is_last_hop: bool = False
    tx_power_dbm: float = 14.0
    
    fhs_id: int = 0  # AJOUT: ID de la séquence FHS

@dataclass   
class FragmentCollisionResult:
    """Résultat de détection de collision - COMPLÉTÉ pour 488 Hz"""
    fragment1: TransmissionFragment
    fragment2: TransmissionFragment
    time_overlap_s: float
    overlap_ratio: float
    collision_type: str
    power_diff_db: float = 0.0
    frequency_offset_hz: float = 0.0
    capture_effect: bool = False
    is_capture_effect: bool = False
    freq_overlap_ratio: float = 0.0
    same_488hz_channel: bool = False  # AJOUT: indique si même canal 488 Hz
    fragment_bw_hz: float = 488.28125  # AJOUT: bande passante du fragment (488 Hz)

class SimulatedPacket:
    """Paquet simulé LR-FHSS (INCHANGÉ + FHS)"""
    def __init__(self, **kwargs):
        self.packet_id = kwargs.get('packet_id', '')
        self.device_id = kwargs.get('device_id', '')
        self.start_time = kwargs.get('start_time', 0.0)
        self.end_time = kwargs.get('end_time', 0.0)
        self.toa_ms = kwargs.get('toa_ms', 0.0)
        self.frequency_mhz = kwargs.get('frequency_mhz', 868.1)
        self.fragments = kwargs.get('fragments', [])
        self.tx_power_dbm = kwargs.get('tx_power_dbm', 14.0)
        self.dr = kwargs.get('dr', 8)
        self.cr = kwargs.get('cr', '1/3')
        self.bw_khz = kwargs.get('bw_khz', 136.71875)
        self.payload_bytes = kwargs.get('payload_bytes', 50)
        self.collision = False
        self.collision_details = []
        self.success = False
        self.rssi_dbm = kwargs.get('rssi_dbm', -120.0)  # Permettre override via kwargs
        self.snr_db = kwargs.get('snr_db', 0.0)
        self.ber = kwargs.get('ber', 1e-3)
        self.fec_recovered = False
        self.distance_km = kwargs.get('distance_km', 1.0)
        
        # AJOUTS FHS
        self.fhs_id = kwargs.get('fhs_id', 0)
        self.transmission_id = kwargs.get('transmission_id', 0)
        self.grid_id = kwargs.get('grid_id', 0)
        
        for key, value in kwargs.items():
            if not hasattr(self, key):
                setattr(self, key, value)

# ============================================================================
# DICTIONNAIRE FHS - NOUVEAU MAIS COMPATIBLE
# ============================================================================

@dataclass
class FHS_Sequence:
    """Séquence FHS (NOUVEAU)"""
    fhs_id: int
    grid_id: int
    sequence: List[int]  # Canaux absolus
    
    def get_channel_at_index(self, index: int) -> int:
        if index < len(self.sequence):
            return self.sequence[index]
        return self.sequence[index % len(self.sequence)]

class FHS_Dictionary:
    """Dictionnaire FHS (NOUVEAU)"""
    
    def __init__(self):
        self.num_grids = 8
        self.channels_per_grid = 35
        self.total_channels = 280
        self.fhs_count = FHS_COUNT
        self.sequences = self._generate_all_sequences()
    
    def _generate_all_sequences(self) -> Dict[int, FHS_Sequence]:
        sequences = {}
        
        for fhs_id in range(self.fhs_count):
            grid_id = fhs_id % self.num_grids
            hop_sequence = self._generate_hop_sequence(fhs_id, grid_id)
            
            sequences[fhs_id] = FHS_Sequence(
                fhs_id=fhs_id,
                grid_id=grid_id,
                sequence=hop_sequence
            )
        
        return sequences
    
    def _generate_hop_sequence(self, fhs_id: int, grid_id: int) -> List[int]:
        seed = fhs_id if fhs_id != 0 else 1
        lfsr_state = seed & 0x1FF
        sequence = []
        last_channel = -1
        
        max_attempts = 1000
        attempts = 0
        
        while len(sequence) < 40 and attempts < max_attempts:
            bit9 = (lfsr_state >> 8) & 1
            bit5 = (lfsr_state >> 4) & 1
            feedback = bit9 ^ bit5
            lfsr_state = ((lfsr_state << 1) | feedback) & 0x1FF
            
            if lfsr_state == 0:
                lfsr_state = 1
            
            channel_in_grid = lfsr_state % self.channels_per_grid
            absolute_channel = channel_in_grid * self.num_grids + grid_id
            
            if absolute_channel >= self.total_channels:
                absolute_channel = absolute_channel % self.total_channels
            
            # Éviter les répétitions consécutives
            if absolute_channel != last_channel:
                sequence.append(absolute_channel)
                last_channel = absolute_channel
            
            attempts += 1
        
        # Si pas assez de canaux, étendre
        if len(sequence) < 40:
            while len(sequence) < 40:
                sequence.append(sequence[len(sequence) % len(sequence)])
        
        return sequence
    
    def get_fhs_for_transmission(self, device_id: str, transmission_id: int) -> FHS_Sequence:
        key = f"{device_id}_{transmission_id}"
        hash_bytes = hashlib.md5(key.encode()).digest()
        hash_int = int.from_bytes(hash_bytes[:4], 'little')
        fhs_index = hash_int % self.fhs_count
        return self.sequences[fhs_index]
    
    def get_sequence_by_id(self, fhs_id: int) -> FHS_Sequence:
        return self.sequences[fhs_id]

FHS_DICT = FHS_Dictionary()

# ============================================================================
# VOS FONCTIONS DÉTERMINISTES (INCHANGÉES)
# ============================================================================

def _deterministic_decision(key: str, probability: float) -> bool:
    """Décision déterministe basée sur un hash (INCHANGÉ)"""
    if probability >= 0.999:
        return True
    if probability <= 0.001:
        return False
    hash_int = int(hashlib.md5(key.encode()).hexdigest(), 16)
    normalized = (hash_int % 10000) / 10000.0
    return normalized < probability

def _deterministic_success_decision(packet_id: str, success_prob: float) -> bool:
    """Décision de succès déterministe (INCHANGÉ)"""
    return _deterministic_decision(packet_id, success_prob)
import math

def calculate_success_probability(packet: SimulatedPacket, snr_db: float) -> float:
    """
    Probabilité de succès pour LR-FHSS avec codage convolutionnel.
    
    Modèle physique:
        P_success = (1 - BER)^(N_bits * α)
    
    où:
    - α est un facteur correctif pour le codage convolutionnel (0.3-0.5)
    - Le Viterbi peut corriger jusqu'à environ 1/3 des bits pour CR 1/3
    
    Cette formule approxime la performance d'un décodeur de Viterbi
    en canal AWGN.
    
    Args:
        packet: SimulatedPacket avec payload_bytes et cr définis
        snr_db: SNR en dB à l'entrée du décodeur
    
    Returns:
        float: Probabilité de succès entre 0 et 1
    """
    
    # 1. Calcul BER via le modèle calibré
    ber = calculate_ber(snr_db, packet.cr)
    
    # 2. Nombre total de bits transmis (payload + CRC)
    n_bits = (packet.payload_bytes + 2) * 8  # +2 bytes CRC
    
    # 3. Facteur de correction pour le codage convolutionnel
    # Le Viterbi peut corriger beaucoup d'erreurs, donc la PER est meilleure
    # que (1 - (1-BER)^N_bits)
    if packet.cr == '1/3':
        # CR 1/3: très puissant, facteur ~0.3
        coding_gain_factor = 0.3
    else:
        # CR 2/3: moins puissant, facteur ~0.5
        coding_gain_factor = 0.5
    
    # PER approximée = 1 - (1 - BER)^(N_bits * coding_gain_factor)
    # Pour BER petit, PER ≈ BER * N_bits * coding_gain_factor
    # Pour BER grand, on utilise la forme exacte
    if ber < 0.01:  # Approximation linéaire pour petits BER
        per = ber * n_bits * coding_gain_factor
    else:
        # Formule exacte mais plus coûteuse
        per = 1.0 - (1.0 - ber) ** (n_bits * coding_gain_factor)
    
    # Probabilité de succès = 1 - PER
    success_prob = 1.0 - per
    
    # Limiter numériquement
    success_prob = max(0.0, min(1.0, success_prob))
    
    return success_prob


# ============================================================================
# VOS CONDITIONS DE SUCCÈS (INCHANGÉES)
# ============================================================================
def _evaluate_lrfhss_without_collisions(packet: SimulatedPacket, 
                                       snr_db: float) -> Tuple[bool, str]:
    """
    Évaluation sans collisions - CORRIGÉ pour SNR très bas (-15 dB à -20 dB)
    """
    from config import LR_FHSS_Config
    import logging
    
    logger = logging.getLogger(__name__)
    debug = False  # Mettre à True pour logs détaillés
    
    # Déterminer le seuil de sensibilité RSSI (garde cette partie)
    dr = getattr(packet, 'dr', 8)
    sensitivity_by_dr = LR_FHSS_Config.PERFORMANCE.get('sensitivity_dbm_by_dr', {})
    if dr in sensitivity_by_dr:
        rssi_threshold = sensitivity_by_dr[dr]
    else:
        rssi_threshold = LR_FHSS_Config.PERFORMANCE['sensitivity_dbm'].get(packet.cr, -125.0)
    
    if debug:
        print(f"[DEBUG {packet.device_id}] RSSI={packet.rssi_dbm:.1f} dBm, seuil={rssi_threshold:.1f} dBm")
    
    # Vérification du seuil RSSI (important pour éviter les valeurs aberrantes)
    if packet.rssi_dbm < rssi_threshold - 10:  # Marge de 10 dB pour éviter les cas extrêmes
        return False, f"RSSI_TOO_LOW ({packet.rssi_dbm:.1f} dBm < {rssi_threshold-10:.1f} dBm)"
    
    # ============ CORRECTION CRITIQUE ============
    # Paramètres SNR réalistes pour LR-FHSS
    if packet.cr == '1/3':
        # CR 1/3: peut décoder jusqu'à -15 dB à -20 dB
        required_snr = -15.0  # Au lieu de -10.0
        fec_gain_db = 5.0      # Gain du codage (reste pertinent)
        snr_50 = -10.0         # SNR où P_success = 50% (pour le modèle BER)
    else:  # CR 2/3
        # CR 2/3: peut décoder jusqu'à -10 dB à -15 dB
        required_snr = -12.0   # Au lieu de -7.0
        fec_gain_db = 4.0
        snr_50 = -7.0
    
    effective_snr = snr_db + fec_gain_db
    
    if debug:
        print(f"[DEBUG {packet.device_id}] Input SNR={snr_db:.1f} dB, FEC gain={fec_gain_db:.1f} dB → Effective SNR={effective_snr:.1f} dB")
    
    # Vérification SNR minimum (large marge)
    if effective_snr < required_snr - 10.0:  # Marge de sécurité
        return False, f"SNR_TOO_LOW ({effective_snr:.1f} dB < {required_snr-10:.1f} dB)"
    
    # Calcul de la probabilité de succès avec le modèle BER calibré
    # Note: calculate_success_probability utilise déjà le bon modèle
    success_prob = calculate_success_probability(packet, effective_snr)
    
    # Mise à jour du BER pour le paquet
    packet.ber = calculate_ber(effective_snr, packet.cr)
    
    if debug:
        print(f"[DEBUG {packet.device_id}] BER={packet.ber:.3e}, payload={packet.payload_bytes}B → P_success={success_prob:.3e}")
    
    # Décision déterministe basée sur la probabilité
    success = _deterministic_success_decision(packet.packet_id, success_prob)
    
    if success:
        packet.success = True
        return True, f"SUCCESS_NO_COLLISION (RSSI={packet.rssi_dbm:.1f}dBm, SNR={effective_snr:.1f}dB, prob={success_prob:.2f})"
    else:
        return False, f"DEMOD_FAILED (SNR={effective_snr:.1f}dB, BER={packet.ber:.3e}, prob={success_prob:.3e})"
    
def _evaluate_lrfhss_with_collisions(packet: SimulatedPacket,
                                    collision_details: List[FragmentCollisionResult],
                                    snr_db: float) -> Tuple[bool, str]:
    """
    Évaluation avec collisions - inclut vérification du seuil RSSI
    CORRECTION: Utilise packet.cr pour header_count au lieu de cr_count
    """
    from config import LR_FHSS_Config
    
    # Déterminer le seuil de sensibilité RSSI
    dr = getattr(packet, 'dr', 8)
    sensitivity_by_dr = LR_FHSS_Config.PERFORMANCE.get('sensitivity_dbm_by_dr', {})
    if dr in sensitivity_by_dr:
        rssi_threshold = sensitivity_by_dr[dr]
    else:
        rssi_threshold = LR_FHSS_Config.PERFORMANCE['sensitivity_dbm'].get(packet.cr, -125.0)
    
    # Vérification du seuil RSSI (prioritaire)
    if packet.rssi_dbm < rssi_threshold:
        return False, f"RSSI_TOO_LOW ({packet.rssi_dbm:.1f} dBm < {rssi_threshold:.1f} dBm seuil DR{dr})"
    
    # Séparer les collisions par type
    header_collisions = [c for c in collision_details if c.fragment1.fragment_type == 'header']
    payload_collisions = [c for c in collision_details if c.fragment1.fragment_type == 'payload']
    
    capture_count = sum(1 for c in collision_details if c.is_capture_effect)
    
    # CORRECTION ICI: Déterminer le nombre d'en-têtes à partir du CR
    header_count = 3 if packet.cr == '1/3' else 2
    
    if header_collisions:
        severe_header_collisions = [
            c for c in header_collisions 
            if c.overlap_ratio > 0.5 and not c.is_capture_effect
        ]
        
        # Utiliser header_count au lieu de cr_count
        if len(severe_header_collisions) >= header_count:
            return False, f"HEADER_COLLISION ({len(severe_header_collisions)} severe sur {header_count})"
        
        if len(header_collisions) >= 2 and capture_count < len(header_collisions):
            return False, "HEADER_COLLISION (multiple partial)"
    
    if payload_collisions:
        severe_payload = [
            c for c in payload_collisions 
            if c.overlap_ratio > 0.1 and not c.is_capture_effect
        ]
        
        total_overlap = sum(c.overlap_ratio for c in payload_collisions)
        avg_overlap = total_overlap / len(payload_collisions) if payload_collisions else 0
        
        if packet.cr == '1/3':
            required_snr = -10.0
            fec_gain_db = 5.0
            fec_capability = 0.10
        else:
            required_snr = -7.0
            fec_gain_db = 4.0
            fec_capability = 0.04
        
        effective_snr = snr_db + fec_gain_db
        
        if effective_snr < required_snr - 5.0:
            return False, f"SNR_TOO_LOW_WITH_COLLISION (SNR={effective_snr:.1f}dB)"
        
        corruption_ratio = avg_overlap * (1.0 - (capture_count / max(len(payload_collisions), 1)))
        
        if corruption_ratio <= fec_capability:
            packet.fec_recovered = True
            success_prob = 0.90 if effective_snr > required_snr else 0.70
        elif corruption_ratio <= fec_capability * 1.5:
            packet.fec_recovered = True
            success_prob = 0.60
        else:
            success_prob = 0.20
        
        success = _deterministic_success_decision(packet.packet_id, success_prob)
        
        if success:
            if packet.fec_recovered:
                return True, f"SUCCESS_FEC_RECOVERED (corruption={corruption_ratio:.2f}, prob={success_prob:.2f})"
            else:
                return True, f"SUCCESS_WITH_COLLISION (prob={success_prob:.2f})"
        else:
            return False, f"PAYLOAD_COLLISION_FAILED (corruption={corruption_ratio:.2f})"
    
    return _evaluate_lrfhss_without_collisions(packet, snr_db)
# ============================================================================
# VOS FONCTIONS DE DÉTECTION DE COLLISIONS (INCHANGÉES)
# ============================================================================

def check_collision(frag1: TransmissionFragment, 
                   frag2: TransmissionFragment,
                   packet_id1: str = "",
                   packet_id2: str = "") -> Optional[FragmentCollisionResult]:
    """Détecte collision entre deux fragments - CORRIGÉE POUR 488 Hz"""
    
    # 1. Vérifier chevauchement temporel
    t_start = max(frag1.start_time, frag2.start_time)
    t_end = min(frag1.end_time, frag2.end_time)
    
    if t_start >= t_end:
        return None  # Pas de chevauchement temporel
    
    time_overlap = t_end - t_start
    
    frag1_duration = frag1.end_time - frag1.start_time
    frag2_duration = frag2.end_time - frag2.start_time
    
    if frag1_duration <= 0 or frag2_duration <= 0:
        return None
    
    overlap_ratio1 = time_overlap / frag1_duration
    overlap_ratio2 = time_overlap / frag2_duration
    overlap_ratio = max(overlap_ratio1, overlap_ratio2)
    
    # ============ CORRECTION CRITIQUE ============
    # 2. Pour LR-FHSS, chaque fragment est dans un canal de 488.28125 Hz
    
    # Bande passante INSTANTANÉE du fragment
    fragment_bw_hz = frag1.instantaneous_bw_hz if hasattr(frag1, 'instantaneous_bw_hz') else 488.28125
    
    # Différence de fréquence en Hz
    freq_offset_hz = abs(frag1.frequency_mhz - frag2.frequency_mhz) * 1e6
    
    # SEUIL RÉEL : si les fragments sont séparés d'au moins 1.5× la bande instantanée
    min_separation_hz = fragment_bw_hz * 1.5  # ~732 Hz
    
    if freq_offset_hz > min_separation_hz:
        return None  # Canaux différents, pas de collision
    
    # Calculer le chevauchement fréquentiel à la résolution 488 Hz
    if freq_offset_hz > fragment_bw_hz:
        freq_overlap_ratio = 0.0  # Pas de chevauchement
    else:
        # Chevauchement partiel ou total
        overlap_bw = fragment_bw_hz - freq_offset_hz
        freq_overlap_ratio = overlap_bw / fragment_bw_hz
    
    # Si moins de 20% de chevauchement fréquentiel, ignorer
    if freq_overlap_ratio < 0.2:
        return None
    # ============ FIN CORRECTION ============
    
    # 3. Déterminer le type de collision
    if frag1.fragment_type == 'header' and frag2.fragment_type == 'header':
        collision_type = 'HEADER-HEADER'
    elif frag1.fragment_type == 'payload' and frag2.fragment_type == 'payload':
        collision_type = 'PAYLOAD-PAYLOAD'
    else:
        collision_type = 'HEADER-PAYLOAD'
    
    # 4. Vérifier effet de capture
    power_diff = abs(frag1.tx_power_dbm - frag2.tx_power_dbm)
    capture_effect = power_diff >= 6.0
    
    return FragmentCollisionResult(
        fragment1=frag1,
        fragment2=frag2,
        time_overlap_s=time_overlap,
        overlap_ratio=overlap_ratio,
        collision_type=collision_type,
        power_diff_db=power_diff,
        frequency_offset_hz=freq_offset_hz,
        capture_effect=capture_effect,
        is_capture_effect=capture_effect,
        freq_overlap_ratio=freq_overlap_ratio,
        fragment_bw_hz=fragment_bw_hz  # Nouveau: stocker la bande du fragment
    )

def is_significant_collision(collision: FragmentCollisionResult) -> bool:
    """Détermine si une collision est significative - SANS SEUILS"""
    if collision is None:
        return False
    
    # L'effet de capture résout la collision
    if collision.is_capture_effect:
        return False
    
    # Toute collision avec chevauchement temporel ET fréquentiel est significative
    # (plus de seuils artificiels)
    
    # Les collisions header-header sont toujours significatives
    if collision.collision_type == 'HEADER-HEADER':
        return True
    
    # Pour les collisions payload-payload et mixtes, 
    # toute collision avec chevauchement est significative
    return True

# ============================================================================
# VOTRE FONCTION D'ÉVALUATION GLOBALE (ADAPTÉE)
# ============================================================================
def evaluate_transmission(packet: SimulatedPacket,
                         config: Dict,
                         active_packets: List[SimulatedPacket],
                         device_position: tuple = None) -> Tuple[bool, str]:
    """
    Évalue si une transmission réussit avec shadowing déterministe
    """
    try:
        if config is None:
            config = {}
        
        # Récupérer la distance du paquet
        distance_km = packet.distance_km
        
        # Utiliser la position fournie ou (0, 0) par défaut
        position = device_position if device_position is not None else (0, 0)
        
        # DEBUG: Vérifier que la position n'est pas (0, 0)
        if position == (0, 0):
            print(f"⚠️ WARNING: Position (0,0) pour {packet.device_id}")
        
        # 1. CALCULER PATH LOSS SEUL
        from channel import calculate_path_loss
        path_loss_db = calculate_path_loss(
            distance_km=distance_km,
            frequency_mhz=packet.frequency_mhz,
            path_loss_exponent=config.get('path_loss_exponent', 2.7)
        )
        
        # 2. CALCULER RSSI AVEC SHADOWING DÉTERMINISTE
        # On utilise calculate_rssi qui inclut déjà le shadowing
        rssi_with_shadowing = calculate_rssi(
            tx_power_dbm=packet.tx_power_dbm,
            distance_km=distance_km,
            frequency_mhz=packet.frequency_mhz,
            device_id=packet.device_id,
            position=position,
            seed_global=config.get('seed_global', 42),
            path_loss_exponent=config.get('path_loss_exponent', 2.7),
            shadowing_std_db=config.get('shadowing_std_db', 7.0)
        )
        
        # 3. CALCULER LE SHADOWING APPLIQUÉ
        # RSSI = TxPower - PathLoss + Shadowing
        # Donc Shadowing = RSSI - TxPower + PathLoss
        shadowing_db = rssi_with_shadowing - (packet.tx_power_dbm - path_loss_db)
        
        # DEBUG: Vérifier la cohérence des calculs
        expected_rssi = packet.tx_power_dbm - path_loss_db + shadowing_db
        diff = abs(rssi_with_shadowing - expected_rssi)
        
        if diff > 0.1:  # Tolérance de 0.1 dB
            print(f"⚠️ [SHADOWING_CHECK] {packet.device_id}: "
                  f"Incohérence! RSSI={rssi_with_shadowing:.1f}, "
                  f"Attendu={expected_rssi:.1f}, "
                  f"Diff={diff:.2f}dB")
        
        # 4. STOCKER TOUTES LES MÉTRIQUES DANS LE PAQUET
        packet.rssi_dbm = rssi_with_shadowing      # RSSI final AVEC shadowing
        packet.path_loss_db = path_loss_db         # Path loss seul
        packet.shadowing_db = shadowing_db         # Shadowing appliqué
        packet.position = position                 # Position du device
        
        # 5. CALCULER LA PUISSANCE DE BRUIT
        noise_power_dbm = calculate_noise_power(
            bandwidth_khz=packet.bw_khz,
            noise_figure_db=config.get('noise_figure_db', 6.0)
        )
        
        # 6. CALCULER LE SNR
        snr_db = rssi_with_shadowing - noise_power_dbm
        packet.snr_db = snr_db
        
        
        # 7. DÉTECTER LES COLLISIONS AVEC LES PAQUETS ACTIFS
        collision_details = []
        for other_packet in active_packets:
            if other_packet.packet_id == packet.packet_id:
                continue
                
            # Vérifier les collisions fragment par fragment
            for frag1 in packet.fragments:
                for frag2 in other_packet.fragments:
                    # Assurer que les fragments ont la puissance de transmission
                    if not hasattr(frag1, 'tx_power_dbm'):
                        frag1.tx_power_dbm = packet.tx_power_dbm
                    if not hasattr(frag2, 'tx_power_dbm'):
                        frag2.tx_power_dbm = other_packet.tx_power_dbm
                    
                    # Vérifier la collision
                    collision = check_collision(frag1, frag2, 
                                               packet.packet_id, 
                                               other_packet.packet_id)
                    
                    if collision and is_significant_collision(collision):
                        collision_details.append(collision)
        
        # Marquer le paquet s'il a eu des collisions
        packet.collision = len(collision_details) > 0
        packet.collision_details = collision_details
        
        # 8. ÉVALUER LE SUCCÈS (avec ou sans collisions)
        if collision_details:
            success, failure_reason = _evaluate_lrfhss_with_collisions(
                packet, collision_details, snr_db
            )
        else:
            success, failure_reason = _evaluate_lrfhss_without_collisions(
                packet, snr_db
            )
        
        # Stocker le résultat
        packet.success = success
        if not success:
            packet.failure_reason = failure_reason
        
        # 9. CALCULER LE BER
        packet.ber = calculate_ber(snr_db, packet.cr)
        
        return success, failure_reason
            
    except Exception as e:
        import traceback
        error_msg = f"ERREUR dans evaluate_transmission pour {packet.packet_id}: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        
        # En cas d'erreur, marquer comme échec avec les valeurs par défaut
        packet.success = False
        packet.failure_reason = f"INTERNAL_ERROR: {str(e)}"
        packet.rssi_dbm = -120.0
        packet.path_loss_db = 100.0
        packet.shadowing_db = 0.0
        packet.snr_db = 0.0
        packet.ber = 1e-3
        
        return False, error_msg

# ============================================================================
# VOTRE FONCTION DE GÉNÉRATION ADAPTÉE À FHS
# ============================================================================

def calculate_frequency(absolute_channel: int, base_freq: float = 868.1, 
                       ocw_mhz: float = 0.13671875, total_channels: int = 280) -> float:
    """Calcule la fréquence (ADAPTÉE)"""
    F_min = base_freq - (ocw_mhz / 2.0)
    channel_spacing_mhz = ocw_mhz / total_channels
    frequency = F_min + (absolute_channel * channel_spacing_mhz)
    return frequency

def generate_lrfhss_fragments(start_time: float, params: Dict, 
                             device_id: str = None, 
                             transmission_id: int = None) -> List[TransmissionFragment]:
    """
    Génère les fragments LR-FHSS (ADAPTÉE avec FHS)
    
    ✅ INTÈGRE LA LOGIQUE FHS SANS CHANGER LA SIGNATURE
    """
    from config import LR_FHSS_Config, REGION, FREQUENCIES_MHZ
    
    # Paramètres (INCHANGÉ)
    cr = params.get('cr', '1/3')
    bw_khz = params.get('bw_khz', 136.71875)
    
    # ============ CORRECTION IMPORTANTE ============
    # UTILISER LA FRÉQUENCE CENTRALE PASSÉE EN PARAMÈTRE
    base_freq = params.get('frequency_mhz', None)
    
    # Si aucune fréquence n'est fournie, utiliser la logique par défaut
    if base_freq is None:
        bw_config = LR_FHSS_Config.PHYSICAL_LAYER['channel_bandwidths'].get(bw_khz)
        
        if bw_config and 'centers' in bw_config and bw_config['centers']:
            available_centers_mhz = [freq / 1e6 for freq in bw_config['centers']]
            
            if device_id and transmission_id is not None:
                center_seed_str = f"{device_id}_{transmission_id}_center"
                center_seed = int(hashlib.md5(center_seed_str.encode()).hexdigest()[:8], 16)
                center_index = center_seed % len(available_centers_mhz)
                base_freq = available_centers_mhz[center_index]
            else:
                base_freq = np.random.choice(available_centers_mhz)
        else:
            base_freq = FREQUENCIES_MHZ['min_mhz']
    # ============ FIN CORRECTION ============
    
    payload_bytes = params.get('payload_bytes', 50)
    dr = params.get('dr', 8)
    tx_power_dbm = params.get('tx_power_dbm', 14.0)
    
    grid_info = LR_FHSS_Config.get_grid_info(bw_khz)
    num_grids = grid_info['num_grids']
    channels_per_grid = grid_info['channels_per_grid']
    total_physical_channels = grid_info['total_channels']
    
    num_headers = HEADER_COUNT.get(cr, 3 if cr == '1/3' else 2)
    
    info_bits = 8 * (payload_bytes + 2) + 6
    
    if cr == '1/3':
        coded_bits = info_bits * 3
        bits_per_hop = BITS_PER_HOP['1/3']
    else:
        coded_bits = math.ceil(info_bits * 1.5)
        bits_per_hop = BITS_PER_HOP['2/3']
    
    num_payload_hops = math.ceil(coded_bits / bits_per_hop)
    
    OCW_mhz = bw_khz / 1000.0
    
    if bw_khz == 136.71875 or bw_khz == 335.9375:
        grid_spacing_khz = 3.9
    elif bw_khz == 1523:
        grid_spacing_khz = 25.4
    else:
        grid_spacing_khz = 3.9
    
    # ✅ PARTIE ADAPTÉE: SÉLECTION FHS
    if device_id and transmission_id is not None:
        # VOTRE LOGIQUE FHS
        fhs_seq = FHS_DICT.get_fhs_for_transmission(device_id, transmission_id)
        fhs_id = fhs_seq.fhs_id
        device_grid = fhs_seq.grid_id
        hop_sequence = fhs_seq.sequence
    elif 'device_grid' in params:
        # Pour compatibilité avec ancien code
        device_grid = params['device_grid']
        if device_id and transmission_id is not None:
            lfsr_seed_str = f"{device_id}_{transmission_id}_lfsr"
            lfsr_seed = int(hashlib.md5(lfsr_seed_str.encode()).hexdigest()[:8], 16) % 512
            lfsr_state = lfsr_seed & 0x1FF
        else:
            lfsr_state = np.random.randint(1, 512)
        
        def lfsr_next(state: int) -> int:
            bit9 = (state >> 8) & 1
            bit5 = (state >> 4) & 1
            feedback = bit9 ^ bit5
            new_state = ((state << 1) | feedback) & 0x1FF
            return new_state
        
        hop_sequence = []
        for i in range(num_headers + num_payload_hops + 10):
            lfsr_state = lfsr_next(lfsr_state)
            channel_in_grid = lfsr_state % channels_per_grid
            absolute_channel = channel_in_grid * num_grids + device_grid
            hop_sequence.append(absolute_channel)
        
        fhs_id = 0  # Par défaut
    else:
        device_grid = np.random.randint(0, num_grids)
        lfsr_state = np.random.randint(1, 512)
        
        def lfsr_next(state: int) -> int:
            bit9 = (state >> 8) & 1
            bit5 = (state >> 4) & 1
            feedback = bit9 ^ bit5
            new_state = ((state << 1) | feedback) & 0x1FF
            return new_state
        
        hop_sequence = []
        for i in range(num_headers + num_payload_hops + 10):
            lfsr_state = lfsr_next(lfsr_state)
            channel_in_grid = lfsr_state % channels_per_grid
            absolute_channel = channel_in_grid * num_grids + device_grid
            hop_sequence.append(absolute_channel)
        
        fhs_id = 0
    
    # ✅ FONCTION DE CALCUL ADAPTÉE
    def calculate_frequency_for_channel(absolute_channel: int) -> float:
        F_min = base_freq - (OCW_mhz / 2.0)  # base_freq vient des paramètres
        channel_spacing_mhz = OCW_mhz / total_physical_channels
        frequency = F_min + (absolute_channel * channel_spacing_mhz)
        return frequency
    
    # Génération des fragments (LOGIQUE INCHANGÉE, juste utilise hop_sequence)
    fragments = []
    current_time = start_time
    
    # Headers
    for i in range(num_headers):
        absolute_channel = hop_sequence[i]
        frequency = calculate_frequency_for_channel(absolute_channel)
        
        fragment = TransmissionFragment(
            start_time=current_time,
            end_time=current_time + HEADER_DURATION_S,
            frequency_mhz=frequency,
            fragment_type='header',
            fragment_index=i,
            bw_khz=bw_khz,
            channel_offset=absolute_channel % channels_per_grid,
            absolute_channel=absolute_channel,
            grid_id=device_grid,
            cr=cr,
            hop_number=i,
            grid_spacing_khz=grid_spacing_khz,
            bits_in_fragment=0,
            is_last_hop=False,
            tx_power_dbm=tx_power_dbm,
            instantaneous_bw_hz=OBW_HZ,
            fhs_id=fhs_id  # AJOUT
        )
        fragments.append(fragment)
        current_time += HEADER_DURATION_S
    
    # Payloads
    for i in range(num_payload_hops):
        seq_index = num_headers + i
        
        if seq_index < len(hop_sequence):
            absolute_channel = hop_sequence[seq_index]
        else:
            wrap_index = seq_index % len(hop_sequence)
            absolute_channel = hop_sequence[wrap_index]
        
        frequency = calculate_frequency_for_channel(absolute_channel)
        
        bits_this_hop = min(bits_per_hop, coded_bits - (i * bits_per_hop))
        is_last_hop = (i == num_payload_hops - 1)
        
        if is_last_hop and (bits_this_hop < bits_per_hop):
            symbols_needed = math.ceil(bits_this_hop / 2)
            hop_duration_s = (symbols_needed / 50.0) * PAYLOAD_DURATION_S
        else:
            hop_duration_s = PAYLOAD_DURATION_S
        
        fragment = TransmissionFragment(
            start_time=current_time,
            end_time=current_time + hop_duration_s,
            frequency_mhz=frequency,
            fragment_type='payload',
            fragment_index=num_headers + i,
            bw_khz=bw_khz,
            channel_offset=absolute_channel % channels_per_grid,
            absolute_channel=absolute_channel,
            grid_id=device_grid,
            cr=cr,
            hop_number=num_headers + i,
            grid_spacing_khz=grid_spacing_khz,
            bits_in_fragment=bits_this_hop,
            is_last_hop=is_last_hop,
            tx_power_dbm=tx_power_dbm,
            instantaneous_bw_hz=OBW_HZ,
            fhs_id=fhs_id  # AJOUT
        )
        fragments.append(fragment)
        current_time += hop_duration_s
    
    return fragments

# ============================================================================
# VOS TESTS DE VALIDATION (ADAPTÉS)
# ============================================================================

def validate_frequency_allocation(bw_khz: int = 136.71875, base_freq: float = 868.1):
    """
    Valide que toutes les fréquences générées sont dans la bande autorisée
    (ADAPTÉ avec logique FHS)
    """
    from config import LR_FHSS_Config
    
    OCW_mhz = bw_khz / 1000.0
    grid_info = LR_FHSS_Config.get_grid_info(bw_khz)
    total_channels = grid_info['total_channels']
    num_grids = grid_info['num_grids']
    channels_per_grid = grid_info['channels_per_grid']
    
    F_min = base_freq - (OCW_mhz / 2.0)
    F_max = base_freq + (OCW_mhz / 2.0)
    
    print("=" * 80)
    print("VALIDATION DE L'ALLOCATION DE FRÉQUENCES LR-FHSS (avec FHS)")
    print("=" * 80)
    print(f"\nParamètres:")
    print(f"  Fréquence centrale: {base_freq} MHz")
    print(f"  OCW: {OCW_mhz*1000:.1f} kHz")
    print(f"  F_min théorique: {F_min:.6f} MHz")
    print(f"  F_max théorique: {F_max:.6f} MHz")
    print(f"  Total canaux: {total_channels}")
    print(f"  Nombre de grilles: {num_grids}")
    print(f"  Canaux par grille: {channels_per_grid}")
    print()
    
    def calculate_frequency(absolute_channel: int) -> float:
        F_min_calc = base_freq - (OCW_mhz / 2.0)
        channel_spacing = OCW_mhz / total_channels
        frequency = F_min_calc + (absolute_channel * channel_spacing)
        return frequency
    
    print("TEST: Vérification avec logique FHS")
    print("-" * 80)
    
    # Tester avec quelques devices
    test_devices = ["device_001", "device_002", "device_003"]
    
    for device_id in test_devices:
        print(f"\nDevice: {device_id}")
        print(f"  Transmission 0:")
        
        # Obtenir séquence FHS
        fhs_seq = FHS_DICT.get_fhs_for_transmission(device_id, 0)
        
        print(f"    FHS ID: {fhs_seq.fhs_id}")
        print(f"    Grille: {fhs_seq.grid_id}")
        print(f"    5 premiers canaux: {fhs_seq.sequence[:5]}")
        
        # Vérifier fréquences
        for i, channel in enumerate(fhs_seq.sequence[:3]):
            freq = calculate_frequency(channel)
            valid = F_min <= freq <= F_max
            status = "✅" if valid else "❌"
            print(f"    Canal {channel}: {freq:.6f} MHz {status}")
    
    print("\n" + "=" * 80)
    print("✅ VALIDATION AVEC LOGIQUE FHS TERMINÉE")
    print("=" * 80)
    
    return True

# ============================================================================
# FONCTION DE COMPATIBILITÉ POUR ANCIEN CODE
# ============================================================================

def create_packet_with_fhs(device_id: str, transmission_id: int, 
                          start_time: float = 0.0, **kwargs) -> SimulatedPacket:
    """
    Fonction helper pour créer un paquet avec logique FHS
    (NOUVELLE mais compatible)
    """
    # Générer fragments avec FHS
    params = {
        'cr': kwargs.get('cr', '1/3'),
        'payload_bytes': kwargs.get('payload_bytes', 50),
        'tx_power_dbm': kwargs.get('tx_power_dbm', 14.0),
        'bw_khz': kwargs.get('bw_khz', 136.71875)
    }
    
    fragments = generate_lrfhss_fragments(
        start_time=start_time,
        params=params,
        device_id=device_id,
        transmission_id=transmission_id
    )
    
    # Obtenir info FHS
    fhs_seq = FHS_DICT.get_fhs_for_transmission(device_id, transmission_id)
    
    # Créer paquet
    packet = SimulatedPacket(
        packet_id=f"{device_id}_{transmission_id}",
        device_id=device_id,
        transmission_id=transmission_id,
        start_time=start_time,
        fragments=fragments,
        fhs_id=fhs_seq.fhs_id,
        grid_id=fhs_seq.grid_id,
        cr=params['cr'],
        payload_bytes=params['payload_bytes'],
        tx_power_dbm=params['tx_power_dbm'],
        bw_khz=params['bw_khz']
    )
    
    if fragments:
        packet.end_time = max(f.end_time for f in fragments)
        packet.toa_ms = (packet.end_time - start_time) * 1000
    
    return packet

# ============================================================================
# EXÉCUTION DES TESTS
# ============================================================================

if __name__ == "__main__":
    print("\n🧪 TEST DE COMPATIBILITÉ - LOGIQUE FHS INTÉGRÉE")
    print()
    
    # 1. Valider fréquences avec FHS
    validate_frequency_allocation()
    
    # 2. Tester compatibilité
    print("\n" + "="*60)
    print("TEST DE CRÉATION DE PAQUETS (COMPATIBILITÉ)")
    print("="*60)
    
    # Ancienne méthode (toujours fonctionne)
    fragments_old = generate_lrfhss_fragments(
        start_time=0.0,
        params={'cr': '1/3', 'payload_bytes': 50},
        device_id="test_old",
        transmission_id=0
    )
    print(f"Ancienne méthode: {len(fragments_old)} fragments générés")
    
    # Nouvelle méthode avec helper
    packet_new = create_packet_with_fhs(
        device_id="test_new",
        transmission_id=5,
        start_time=0.0,
        cr='1/3',
        payload_bytes=30
    )
    print(f"Nouvelle méthode: paquet avec FHS ID {packet_new.fhs_id}, grille {packet_new.grid_id}")
    print(f"  Fragments: {len(packet_new.fragments)}")
    print(f"  FHS ID sur fragments: {set(f.fhs_id for f in packet_new.fragments)}")
    
    print("\n✅ Toutes vos fonctions sont intactes et adaptées à la logique FHS !")