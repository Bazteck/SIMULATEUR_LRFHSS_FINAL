# channel.py - Version avec paramètres librement configurables

import numpy as np
import math
import hashlib
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# SHADOWING MANAGER (simplifié)
# ============================================================================

class ShadowingManager:
    """Gère le shadowing déterministe basé sur position et device_id"""
    
    _instance = None
    _shadowing_cache = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ShadowingManager, cls).__new__(cls)
            cls._instance._local_cache = {}
        return cls._instance
    
    def get_shadowing(self, device_id: str, position: tuple, 
                     seed_global: int = 42, shadowing_std_db: float = 7.0) -> float:
        """
        Génère un shadowing déterministe reproductible.
        
        Args:
            device_id: Identifiant unique du device
            position: Tuple (x, y) en mètres
            seed_global: Graine globale pour reproductibilité
            shadowing_std_db: Écart-type du shadowing (dB)
        
        Returns:
            shadowing_db: Valeur de shadowing (dB)
        """
        # Arrondir position pour éviter les variations infimes
        x_rounded = round(position[0], 1)
        y_rounded = round(position[1], 1)
        
        # Clé unique pour le cache
        cache_key = f"{device_id}_{x_rounded:.1f}_{y_rounded:.1f}_{seed_global}_{shadowing_std_db:.1f}"
        
        # Vérifier le cache
        if cache_key in self._local_cache:
            return self._local_cache[cache_key]
        
        if cache_key in ShadowingManager._shadowing_cache:
            shadowing = ShadowingManager._shadowing_cache[cache_key]
            self._local_cache[cache_key] = shadowing
            return shadowing
        
        # Générer une seed déterministe à partir de device_id et position
        seed_str = f"{device_id}_{x_rounded:.1f}_{y_rounded:.1f}_{seed_global}"
        seed = int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % (2**31)
        rng = np.random.RandomState(seed)
        
        # Shadowing suivant une loi normale N(0, shadowing_std_db)
        shadowing = rng.normal(0, shadowing_std_db)
        
        # Limiter à ±3σ (99.7% des cas)
        max_shadowing = 3 * shadowing_std_db
        shadowing = max(-max_shadowing, min(max_shadowing, shadowing))
        
        # Mettre en cache
        self._local_cache[cache_key] = shadowing
        ShadowingManager._shadowing_cache[cache_key] = shadowing
        
        return shadowing

# Instance globale unique
shadowing_manager = ShadowingManager()

# ============================================================================
# FONCTIONS PRINCIPALES DE PATH LOSS
# ============================================================================

def calculate_path_loss(distance_km: float, 
                        frequency_mhz: float = 868,  # Conservé pour compatibilité
                        path_loss_exponent: float = 3.3,
                        reference_loss_db: float = 125.0) -> float:
    """
    Calcule la perte de trajet avec le modèle log-distance.
    
    Formule: PL(d) = PL_ref + 10 * n * log10(d / d_ref)
    avec d_ref = 1 km
    
    Args:
        distance_km: Distance en kilomètres (>0)
        frequency_mhz: Fréquence en MHz (non utilisé mais gardé)
        path_loss_exponent: Exposant de perte n (typiquement 2.0 à 4.5)
        reference_loss_db: Perte à 1 km (dB) - défaut: 125 dB (EU868)
    
    Returns:
        path_loss_db: Perte de trajet en dB
    """
    if distance_km <= 0:
        return 0.0
    
    # Modèle log-distance: PL(d) = PL(1km) + 10*n*log10(d)
    path_loss = reference_loss_db + 10 * path_loss_exponent * math.log10(distance_km)
    
    return path_loss

def calculate_rssi(tx_power_dbm: float,
                   distance_km: float,
                   frequency_mhz: float = 868,
                   device_id: Optional[str] = None,
                   position: Optional[Tuple[float, float]] = None,
                   seed_global: int = 42,
                   path_loss_exponent: float = 3.5,
                   shadowing_std_db: float = 0.0,
                   reference_loss_db: float = 125.0,
                   apply_shadowing: bool = True) -> float:
    """
    Calcule le RSSI avec modèle configurable.
    
    RSSI = TX_power - path_loss + shadowing
    
    Args:
        tx_power_dbm: Puissance d'émission (dBm)
        distance_km: Distance (km)
        frequency_mhz: Fréquence (MHz) - pour compatibilité
        device_id: ID du device (nécessaire si shadowing activé)
        position: Position (x, y) en mètres (nécessaire si shadowing activé)
        seed_global: Graine globale pour reproductibilité
        path_loss_exponent: Exposant de perte n (2.0 = espace libre, 3.5 = urbain dense)
        shadowing_std_db: Écart-type du shadowing (0 = pas de shadowing)
        reference_loss_db: Perte à 1 km (dB)
        apply_shadowing: Activer/désactiver le shadowing
    
    Returns:
        rssi_dbm: Puissance reçue en dBm
    """
    # 1. Calcul du path loss
    path_loss = calculate_path_loss(
        distance_km=distance_km,
        frequency_mhz=frequency_mhz,
        path_loss_exponent=path_loss_exponent,
        reference_loss_db=reference_loss_db
    )
    
    # 2. Shadowing (optionnel)
    shadowing = 0.0
    if apply_shadowing and shadowing_std_db > 0:
        if device_id is None or position is None:
            raise ValueError("device_id et position requis pour le shadowing")
        
        shadowing = shadowing_manager.get_shadowing(
            device_id=device_id,
            position=position,
            seed_global=seed_global,
            shadowing_std_db=shadowing_std_db
        )
    
    # 3. RSSI final
    rssi = tx_power_dbm - path_loss + shadowing
    
    return rssi

def calculate_rssi_with_details(tx_power_dbm: float,
                               distance_km: float,
                               frequency_mhz: float = 868,
                               device_id: Optional[str] = None,
                               position: Optional[Tuple[float, float]] = None,
                               seed_global: int = 42,
                               path_loss_exponent: float = 3.5,
                               shadowing_std_db: float = 0.0,
                               reference_loss_db: float = 125.0,
                               apply_shadowing: bool = True) -> Tuple[float, float, float]:
    """
    Version détaillée retournant (rssi, path_loss, shadowing)
    """
    path_loss = calculate_path_loss(
        distance_km=distance_km,
        frequency_mhz=frequency_mhz,
        path_loss_exponent=path_loss_exponent,
        reference_loss_db=reference_loss_db
    )
    
    shadowing = 0.0
    if apply_shadowing and shadowing_std_db > 0:
        if device_id is None or position is None:
            raise ValueError("device_id et position requis pour le shadowing")
        
        shadowing = shadowing_manager.get_shadowing(
            device_id=device_id,
            position=position,
            seed_global=seed_global,
            shadowing_std_db=shadowing_std_db
        )
    
    rssi = tx_power_dbm - path_loss + shadowing
    
    return rssi, path_loss, shadowing

# ============================================================================
# FONCTIONS BER ET BRUIT
# ============================================================================

def calculate_ber(snr_db: float, coding_rate: str) -> float:
    """
    Modèle BER exponentiel calibré pour LR-FHSS.
    
    Basé sur la capacité réelle de -20 dB (avec accumulation).
    Modèle: BER = a * exp(b*SNR) où a,b dépendent de CR.
    
    CR 1/3:
    - SNR=-20dB: BER≈15% → PDR~0.1%
    - SNR=-7.3dB: BER≈3% → PDR~25%
    - SNR=0dB: BER≈0.5% → PDR~80%+
    
    CR 2/3:
    - Plus sensible aux erreurs (moins de redondance)
    
    Args:
        snr_db: SNR en dB
        coding_rate: '1/3' ou '2/3'
    
    Returns:
        ber: Taux d'erreur binaire [0, 0.5]
    """
    # Paramètres du modèle exponentiel BER = a * exp(b*SNR)
    if coding_rate == '1/3':
        # CR 1/3: très redondant
        # À SNR=0: BER=0.005 (0.5%)
        # À SNR=-20: BER≈0.15 (15%)
        # Résolution: b ≈ -0.175, a = 0.005
        a = 0.002
        b = -0.17
    else:  # CR 2/3
        # CR 2/3: moins redondant
        # À SNR=0: BER=0.015 (1.5%)
        # À SNR=-20: BER≈0.35 (35%)
        a = 0.003
        b = -0.17
    
    # Clamp SNR
    snr_clamped = max(-30.0, min(30.0, snr_db))
    
    # Calcul BER exponentiel
    try:
        exponent = b * snr_clamped
        exponent = max(-50.0, min(10.0, exponent))
        ber = a * math.exp(exponent)
    except (ValueError, OverflowError):
        ber = 0.5
    
    return max(0.0, min(0.5, ber))

def calculate_noise_power(bandwidth_khz: float, noise_figure_db: float = 6.0) -> float:
    """
    Calcule la puissance de bruit en dBm.
    
    Args:
        bandwidth_khz: Bande passante en kHz
        noise_figure_db: Facteur de bruit du récepteur (dB)
    
    Returns:
        noise_power_dbm: Puissance de bruit en dBm
    """
    k = 1.38e-23  # Constante de Boltzmann
    T = 290.0     # Température en Kelvin
    B = bandwidth_khz * 1000.0  # Bande passante en Hz
    
    noise_power_w = k * T * B
    noise_power_dbm = 10 * math.log10(noise_power_w / 1e-3)
    
    return noise_power_dbm + noise_figure_db

def calculate_snr(rssi_dbm: float, bandwidth_khz: float, noise_figure_db: float = 6.0) -> float:
    """
    Calcule le SNR à partir du RSSI.
    
    SNR = RSSI - Bruit
    
    Args:
        rssi_dbm: RSSI en dBm
        bandwidth_khz: Bande passante en kHz
        noise_figure_db: Facteur de bruit du récepteur (dB)
    
    Returns:
        snr_db: SNR en dB
    """
    noise_power = calculate_noise_power(bandwidth_khz, noise_figure_db)
    return rssi_dbm - noise_power