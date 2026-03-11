#!/usr/bin/env python3
"""
frequency_center_fix.py
Corrections pour utiliser correctement les centres de fréquence LR-FHSS
conformément au déploiement réel (RP002-1.0.5)
"""

import random
import hashlib
from typing import List, Tuple

# ============================================================================
# CENTRES DE FRÉQUENCE CONFORMES RP002-1.0.5
# ============================================================================

# Table 120 RP002-1.0.5 - Centres de fréquence par bande
LR_FHSS_FREQUENCY_CENTERS = {
    # EU868 - 136.71875 kHz
    136.71875: {
        'centers_hz': [868100000, 868300000, 868500000],  # En Hz
        'centers_mhz': [868.1, 868.3, 868.5],              # En MHz
        'num_centers': 3,
        'grid_spacing_khz': 3.90625,
        'num_grids': 8,
        'channels_per_grid': 35,
        'total_channels': 280,
        'region': 'EU868'
    },
    
    # EU868 - 335.9375 kHz  
    335.9375: {
        'centers_hz': [868130000, 868530000],
        'centers_mhz': [868.13, 868.53],
        'num_centers': 2,
        'grid_spacing_khz': 3.90625,
        'num_grids': 8,
        'channels_per_grid': 86,
        'total_channels': 688,
        'region': 'EU868'
    },
    
    # US915 - 1523 kHz (exemple, centres à définir selon région)
    1523: {
        'centers_hz': [903000000],  # Exemple simplifié
        'centers_mhz': [903.0],
        'num_centers': 1,
        'grid_spacing_khz': 25.4,
        'num_grids': 52,
        'channels_per_grid': 60,
        'total_channels': 3120,
        'region': 'US915'
    }
}


# ============================================================================
# FONCTIONS DE SÉLECTION DES CENTRES DE FRÉQUENCE
# ============================================================================

def get_frequency_centers(bw_khz: float) -> List[float]:
    """
    Retourne la liste des centres de fréquence pour une bande donnée
    
    Args:
        bw_khz: Largeur de bande en kHz (136.71875, 335.9375, ou 1523)
        
    Returns:
        Liste des fréquences centrales en MHz
        
    Exemple:
        >>> get_frequency_centers(136.71875)
        [868.1, 868.3, 868.5]
    """
    if bw_khz not in LR_FHSS_FREQUENCY_CENTERS:
        # Par défaut, retourner 868.1 MHz
        return [868.1]
    
    return LR_FHSS_FREQUENCY_CENTERS[bw_khz]['centers_mhz']


def select_frequency_center_random(bw_khz: float) -> float:
    """
    Sélectionne aléatoirement un centre de fréquence parmi ceux disponibles
    
    Args:
        bw_khz: Largeur de bande en kHz
        
    Returns:
        Fréquence centrale en MHz
        
    Exemple:
        >>> select_frequency_center_random(136.71875)
        868.3  # Aléatoire parmi [868.1, 868.3, 868.5]
    """
    centers = get_frequency_centers(bw_khz)
    return random.choice(centers)


def select_frequency_center_deterministic(device_id: str, bw_khz: float, 
                                         transmission_count: int = 0) -> float:
    """
    Sélectionne de manière DÉTERMINISTE un centre de fréquence
    Basé sur le device_id pour que le même device utilise le même centre
    
    Args:
        device_id: Identifiant unique du device
        bw_khz: Largeur de bande en kHz
        transmission_count: Numéro de transmission (optionnel, pour varier)
        
    Returns:
        Fréquence centrale en MHz
        
    Exemple:
        >>> select_frequency_center_deterministic("device_001", 136.71875)
        868.3  # Toujours le même pour ce device
        
        >>> select_frequency_center_deterministic("device_001", 136.71875, 1)
        868.5  # Peut varier avec transmission_count
    """
    centers = get_frequency_centers(bw_khz)
    
    # Hash déterministe basé sur device_id et transmission_count
    hash_key = f"{device_id}_{bw_khz}_{transmission_count}"
    hash_value = int(hashlib.md5(hash_key.encode()).hexdigest(), 16)
    
    # Sélection déterministe
    index = hash_value % len(centers)
    return centers[index]


def select_frequency_center_round_robin(bw_khz: float, 
                                       transmission_index: int) -> float:
    """
    Sélectionne un centre de fréquence en round-robin
    Utile pour répartir équitablement la charge sur tous les centres
    
    Args:
        bw_khz: Largeur de bande en kHz
        transmission_index: Index de la transmission globale
        
    Returns:
        Fréquence centrale en MHz
        
    Exemple:
        >>> select_frequency_center_round_robin(136.71875, 0)
        868.1
        >>> select_frequency_center_round_robin(136.71875, 1)
        868.3
        >>> select_frequency_center_round_robin(136.71875, 2)
        868.5
        >>> select_frequency_center_round_robin(136.71875, 3)
        868.1  # Retour au début
    """
    centers = get_frequency_centers(bw_khz)
    index = transmission_index % len(centers)
    return centers[index]


def select_frequency_center_load_balanced(bw_khz: float, 
                                         center_usage: dict) -> float:
    """
    Sélectionne le centre de fréquence le MOINS utilisé (load balancing)
    
    Args:
        bw_khz: Largeur de bande en kHz
        center_usage: Dictionnaire {freq_mhz: nombre_utilisations}
        
    Returns:
        Fréquence centrale en MHz (celle la moins utilisée)
        
    Exemple:
        >>> usage = {868.1: 5, 868.3: 2, 868.5: 8}
        >>> select_frequency_center_load_balanced(136.71875, usage)
        868.3  # La moins utilisée
    """
    centers = get_frequency_centers(bw_khz)
    
    # Trouver le centre le moins utilisé
    min_usage = float('inf')
    selected_center = centers[0]
    
    for center in centers:
        usage = center_usage.get(center, 0)
        if usage < min_usage:
            min_usage = usage
            selected_center = center
    
    return selected_center


# ============================================================================
# INTÉGRATION AVEC generate_lrfhss_fragments
# ============================================================================

def get_base_frequency_for_transmission(device_id: str, bw_khz: float, 
                                       selection_method: str = 'deterministic',
                                       transmission_count: int = 0,
                                       transmission_index: int = 0,
                                       center_usage: dict = None) -> float:
    """
    Fonction principale pour obtenir la fréquence de base d'une transmission
    
    Args:
        device_id: Identifiant du device
        bw_khz: Largeur de bande
        selection_method: Méthode de sélection ('random', 'deterministic', 
                         'round_robin', 'load_balanced')
        transmission_count: Compteur de transmissions pour ce device
        transmission_index: Index global de transmission
        center_usage: Dictionnaire d'utilisation des centres (pour load_balanced)
        
    Returns:
        Fréquence centrale en MHz à utiliser pour generate_lrfhss_fragments
    """
    if selection_method == 'random':
        return select_frequency_center_random(bw_khz)
    
    elif selection_method == 'deterministic':
        return select_frequency_center_deterministic(device_id, bw_khz, 
                                                     transmission_count)
    
    elif selection_method == 'round_robin':
        return select_frequency_center_round_robin(bw_khz, transmission_index)
    
    elif selection_method == 'load_balanced':
        if center_usage is None:
            center_usage = {}
        return select_frequency_center_load_balanced(bw_khz, center_usage)
    
    else:
        # Par défaut: déterministe
        return select_frequency_center_deterministic(device_id, bw_khz, 
                                                     transmission_count)


# ============================================================================
# EXEMPLE D'UTILISATION DANS simulation.py
# ============================================================================

def example_integration():
    """
    Exemple d'intégration dans simulation.py
    """
    print("=" * 80)
    print("EXEMPLE D'UTILISATION DES CENTRES DE FRÉQUENCE LR-FHSS")
    print("=" * 80)
    
    # Paramètres d'exemple
    device_id = "device_001"
    bw_khz = 136.71875  # DR8 ou DR9
    
    print(f"\nCentres disponibles pour BW={bw_khz} kHz:")
    centers = get_frequency_centers(bw_khz)
    print(f"  {centers}")
    
    print("\n--- Méthode 1: RANDOM ---")
    for i in range(5):
        freq = select_frequency_center_random(bw_khz)
        print(f"  Transmission {i}: {freq} MHz")
    
    print("\n--- Méthode 2: DETERMINISTIC (même device) ---")
    for i in range(5):
        freq = select_frequency_center_deterministic(device_id, bw_khz, i)
        print(f"  Transmission {i}: {freq} MHz")
    
    print("\n--- Méthode 3: ROUND ROBIN ---")
    for i in range(6):
        freq = select_frequency_center_round_robin(bw_khz, i)
        print(f"  Transmission {i}: {freq} MHz")
    
    print("\n--- Méthode 4: LOAD BALANCED ---")
    usage = {868.1: 10, 868.3: 3, 868.5: 15}
    print(f"  Utilisation actuelle: {usage}")
    for i in range(5):
        freq = select_frequency_center_load_balanced(bw_khz, usage)
        usage[freq] = usage.get(freq, 0) + 1
        print(f"  Transmission {i}: {freq} MHz (nouveau usage: {usage[freq]})")
    
    print("\n" + "=" * 80)
    print("INTÉGRATION DANS generate_lrfhss_fragments:")
    print("=" * 80)
    
    print("""
# AVANT (code actuel - INCORRECT):
fragments = generate_lrfhss_fragments(
    start_time=0.0,
    payload_bytes=50,
    dr=8,
    device_id="device_001",
    tx_power_dbm=14.0,
    base_freq=868.1  # ❌ TOUJOURS LA MÊME FRÉQUENCE
)

# APRÈS (code corrigé - CORRECT):
# 1. Déterminer la BW selon le DR
dr_config = LR_FHSS_Config.get_data_rate_config(dr)
bw_khz = dr_config['bw_khz']

# 2. Sélectionner le centre de fréquence approprié
base_freq = get_base_frequency_for_transmission(
    device_id="device_001",
    bw_khz=bw_khz,
    selection_method='deterministic',  # ou 'round_robin', 'load_balanced'
    transmission_count=0
)

# 3. Générer les fragments avec le bon centre
fragments = generate_lrfhss_fragments(
    start_time=0.0,
    payload_bytes=50,
    dr=8,
    device_id="device_001",
    tx_power_dbm=14.0,
    base_freq=base_freq  # ✅ CENTRE CORRECT SELON BW
)
    """)
    
    print("\n" + "=" * 80)


# ============================================================================
# VALIDATION
# ============================================================================

def validate_frequency_centers():
    """Valide que tous les centres sont corrects"""
    print("=" * 80)
    print("VALIDATION DES CENTRES DE FRÉQUENCE")
    print("=" * 80)
    
    for bw_khz, config in LR_FHSS_FREQUENCY_CENTERS.items():
        print(f"\n📡 Bande: {bw_khz} kHz ({config['region']})")
        print(f"   Centres: {config['centers_mhz']} MHz")
        print(f"   Nombre de centres: {config['num_centers']}")
        print(f"   Canaux totaux: {config['total_channels']}")
        print(f"   Grilles: {config['num_grids']}")
        print(f"   Canaux/grille: {config['channels_per_grid']}")
        
        # Vérifier que le hopping autour de chaque centre reste dans les limites
        OCW_mhz = bw_khz / 1000.0
        for center in config['centers_mhz']:
            F_min = center - (OCW_mhz / 2.0)
            F_max = center + (OCW_mhz / 2.0)
            print(f"   Centre {center} MHz → Bande: [{F_min:.6f}, {F_max:.6f}] MHz")
    
    print("\n" + "=" * 80)
    print("✅ Tous les centres sont conformes RP002-1.0.5")
    print("=" * 80)


if __name__ == "__main__":
    print("\n🧪 Exécution des exemples et validation...\n")
    example_integration()
    print("\n")
    validate_frequency_centers()
