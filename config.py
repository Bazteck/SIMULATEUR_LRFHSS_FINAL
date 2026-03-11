# config.py - Configuration LR-FHSS conforme RP002-1.0.5
"""
Configuration LR-FHSS conforme RP002-1.0.5 Table 120-123
Version nettoyée - Seulement les fonctions utilisées
"""

import math
from enum import Enum
from typing import List
import logging

logger = logging.getLogger(__name__)

class Region(Enum):
    EU868 = "EU868"
    US915 = "US915"
    AS923 = "AS923"
    CN470 = "CN470"

# config.py - Configuration LR-FHSS conforme RP002-1.0.5

class LR_FHSS_Config:
    """Configuration complète LR-FHSS conforme RP002-1.0.5"""
    
    # ============ CONSTANTES FONDAMENTALES ============
    SYMBOL_RATE = 488.28125  # Baud (fixe pour LR-FHSS)
    BT_PRODUCT = 1.0         # GMSK BT=1.0
    FREQ_DEVIATION = SYMBOL_RATE / 2  # 244.140625 Hz
    
    # Table 120: LR-FHSS physical layer description - CORRIGÉ
    PHYSICAL_LAYER = {
        'channel_bandwidths': {
            136.71875: {
                'grid_spacing_khz': 3.90625,
                'num_physical_channels': 280,      # 8 grilles × 35 canaux
                'num_grids': 8,
                'channels_per_grid': 35,
                'frequency_tolerance_ppm': 30.0,
                'centers': [868100000, 868300000, 868500000]  # TROIS centres: 868.1, 868.3, 868.5 MHz
            },
            335.9375: {
                'grid_spacing_khz': 3.90625,
                'num_physical_channels': 688,
                'num_grids': 8,
                'channels_per_grid': 86,
                'frequency_tolerance_ppm': 4.3,
                'centers': [868130000, 868530000]  # Deux centres: 868.13 et 868.53 MHz
            },
            1523: {
                'grid_spacing_khz': 25.4,
                'num_physical_channels': 3120,     # 52 grilles × 60 canaux
                'num_grids': 52,
                'channels_per_grid': 60,
                'frequency_tolerance_ppm': 38.9,
                'centers': []  # Pour US915, gamme continue 902-928 MHz
            }
        },
        
        'modulation': {
            'symbol_rate': SYMBOL_RATE,
            'modulation_bandwidth_hz': 488.28125,
            'modulation_type': 'GMSK',
            'bt_product': BT_PRODUCT,
            'freq_deviation_hz': FREQ_DEVIATION,
        },
        
        'timing': {
            'header_hop_duration_ms': 233.472,
            'payload_hop_duration_ms': 102.4,
            'header_symbols': 114,
            'payload_symbols': 50,
            'preamble_bytes': 5,
        },
    }
    
    # Table 121: LR-FHSS packet structure
    PACKET_STRUCTURE = {
        'header': {
            'syncword': 0x2C0F7995,
            'syncword_bits': 32,
            'phdr_bits': 32,
            'phdr_crc_bits': 8,
            'total_header_bits': 40,
        },
        
        'payload': {
            'crc_bits': 16,
            'max_payload_bytes': 230,
            'min_payload_bytes': 1,
        },
    }
    
    # Table 122: LR-FHSS coding rates
    CODING_RATES = {
        '1/3': {
            'code_rate': 1/3,
            'header_repetitions': 3,
            'info_bits_per_payload_hop': 48,
            'coded_bits_per_info_bit': 3,
            'trellis_termination': False,
            'required_snr_db': 6.0,
            'coding_gain_db': 5.0,
        },
        '2/3': {
            'code_rate': 2/3,
            'header_repetitions': 2,
            'info_bits_per_payload_hop': 96,
            'coded_bits_per_info_bit': 1.5,
            'trellis_termination': True,
            'required_snr_db': 8.0,
            'coding_gain_db': 4.0,
        }
    }
    
    # Table 123: LR-FHSS physical layer settings
    PHYSICAL_SETTINGS = {
        'syncword': 0x2C0F7995,
        'preamble': {
            'size_bytes': 5,
            'pattern': 0xAA,
        },
        'crc': {
            'polynomial': 0x1021,
            'initial_value': 0xFFFF,
            'final_xor': 0x0000,
        },
        'whitening': {
            'polynomial': 0x100D,
            'initial_state': 0xFF,
        }
    }
    
    # AN1200.64 Performance parameters
    PERFORMANCE = {
        # Seuils de sensibilité RSSI (dBm) par DR
        'sensitivity_dbm_by_dr': {
            8: -136,   # DR8 (CR 1/3, 136 kHz)
            9: -133,   # DR9 (CR 2/3, 136 kHz)
            10: -134,  # DR10 (CR 1/3, 336 kHz)
            11: -130,  # DR11 (CR 2/3, 336 kHz)
        },
        # Seuils par coding rate (fallback)
        'sensitivity_dbm': {
            '1/3': -135,
            '2/3': -131 ,
        },
        # SNR minimum pour décodage
        'snr_min': {
            '1/3': -15.0,
            '2/3': -12.0,
        },
        # SNR pour P_success ≈ 50%
        'snr_50': {
            '1/3': -10.0,
            '2/3': -7.0,
        },
        # Gain du FEC (dB)
        'fec_gain_db': {
            '1/3': 5.0,
            '2/3': 4.0,
        }
    }
    # Regional parameters (RP002-1.0.5) - CORRIGÉ
    REGIONAL_CONFIGS = {
        Region.EU868: {
            'frequencies_mhz': {
                'min_mhz': 868.1,
                'max_mhz': 868.5,
            },
            'max_tx_power_dbm': 14.0,
            'duty_cycle': 0.01,
            'bandwidths_khz': [136.71875, 335.9375],
            'channel_spacing_khz': 3.90625,
        },
        Region.US915: {
            'frequencies_mhz': {
                'min_mhz': 902.0,
                'max_mhz': 928.0,
            },
            'max_tx_power_dbm': 30.0,
            'duty_cycle': 0.0,
            'dwell_time_ms': 400,
            'bandwidths_khz': [1523],
            'channel_spacing_khz': 25.4,
        }
    }
    # ============ MÉTHODES UTILISÉES ============
    
    @classmethod
    def get_grid_info(cls, bw_khz: int) -> dict:
        """Retourne les informations de grille pour une bande donnée"""
        bw_config = cls.PHYSICAL_LAYER['channel_bandwidths'].get(bw_khz)
        if not bw_config:
            return {'num_grids': 1, 'channels_per_grid': 35, 'total_channels': 35}
        
        return {
            'num_grids': bw_config.get('num_grids', 1),
            'channels_per_grid': bw_config.get('channels_per_grid', 
                                            bw_config['num_physical_channels']),
            'total_channels': bw_config['num_physical_channels']
        }

    @classmethod
    def get_data_rate_config(cls, dr: int) -> dict:
        """Retourne configuration pour un DR spécifique (Table 120)"""
        dr_configs = {
            8: {'bw_khz': 136.71875, 'cr': '1/3', 'nominal_rate_bps': 162, 'info_rate_bps': 488},
            9: {'bw_khz': 136.71875, 'cr': '2/3', 'nominal_rate_bps': 325, 'info_rate_bps': 488},
            10: {'bw_khz': 335.9375, 'cr': '1/3', 'nominal_rate_bps': 162, 'info_rate_bps': 488},
            11: {'bw_khz': 335.9375, 'cr': '2/3', 'nominal_rate_bps': 325, 'info_rate_bps': 488},
        }
        
        config = dr_configs.get(dr, dr_configs[8])
        
        # Ajouter les paramètres de codage
        cr_params = cls.CODING_RATES[config['cr']]
        config.update(cr_params)
        
        return config
    
    @classmethod
    def calculate_toa_ms(cls, dr: int, payload_bytes: int) -> float:
        """
        Calcule le Time-on-Air (ToA) pour LR-FHSS selon RP002-1.0.5.
        """
        # Paramètres fixes
        HEADER_DURATION_MS = 233.472
        PAYLOAD_DURATION_MS = 102.4
        
        # Nombre de headers selon CR
        if dr in [8, 10, 12]:  # CR = 1/3
            N = 3
            cr = '1/3'
        elif dr in [9, 11, 13]:  # CR = 2/3
            N = 2
            cr = '2/3'
        else:
            N = 3
            cr = '1/3'
        
        # ToA des headers
        toa_header_ms = N * HEADER_DURATION_MS
        
        # ToA du payload
        L = payload_bytes
        if cr == '1/3':
            payload_terms = (8 * (L + 2) + 6) * 3
            ceil_term = math.ceil(payload_terms / 48)
            numerator = payload_terms + ceil_term * 2
            hops = numerator / 50.0
        else:  # CR = 2/3
            payload_terms = (8 * (L + 2) + 6) * 3 / 2
            ceil_term = math.ceil(payload_terms / 48)
            numerator = payload_terms + ceil_term * 2
            hops = numerator / 50.0
        
        toa_payload_ms = hops * PAYLOAD_DURATION_MS
        
        return toa_header_ms + toa_payload_ms


# Configuration globale
LR_FHSS_CONFIG = LR_FHSS_Config()
REGION = Region.EU868
FREQUENCIES_MHZ = LR_FHSS_Config.REGIONAL_CONFIGS[REGION]['frequencies_mhz']
MAX_TX_POWER_DBM = LR_FHSS_Config.REGIONAL_CONFIGS[REGION]['max_tx_power_dbm']