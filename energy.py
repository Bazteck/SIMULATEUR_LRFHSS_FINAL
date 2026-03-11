"""
energy.py - Modèle de consommation énergétique LR-FHSS en Joules
Implémente les profils de puissance pour SX1261 (Low Power PA) et SX1262 (High Power PA)
Toutes les valeurs d'énergie sont exprimées en Joules (J)
Conforme aux datasheets Semtech et aux spécifications LR-FHSS
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import math
from config import LR_FHSS_Config

@dataclass
class PowerProfile:
    """Profil de consommation pour un PA spécifique"""
    name: str
    current_ma: Dict[float, float]  # dBm -> mA
    voltage_v: float = 3.3
    efficiency: float = 0.35  # Efficacité typique PA

class EnergyConsumptionModel:
    """Modèle de consommation énergétique LR-FHSS - Toutes valeurs en Joules"""
        
    POWER_PROFILES = {
                        'SX1261_LP': PowerProfile(
                        name="SX1261 Low Power PA",
                        current_ma={
                            -4.0: 8.5, -3.0: 9.2, -2.0: 10.0, -1.0: 10.5,
                            0.0: 11.0, 1.0: 12.5, 2.0: 14.0, 3.0: 15.5,
                            4.0: 17.0, 5.0: 18.5, 6.0: 20.0, 7.0: 21.0,
                            8.0: 22.0, 9.0: 23.0, 10.0: 24.0, 11.0: 26.0,
                            12.0: 28.0, 13.0: 30.0, 14.0: 32.0
                        },
                        voltage_v=3.3,
                        efficiency=0.32
                    ),
                        'SX1262_HP': PowerProfile(
                        name="SX1262 High Power PA",
                        current_ma={
                            0.0: 20.0,  1.0: 22.5,  2.0: 25.0,  3.0: 27.5,
                            4.0: 30.0,  5.0: 32.5,  6.0: 35.0,  7.0: 37.5,
                            8.0: 40.0,  9.0: 42.5,  10.0: 45.0, 11.0: 50.0,
                            12.0: 55.0, 13.0: 60.0, 14.0: 65.0, 15.0: 70.0,
                            16.0: 75.0, 17.0: 82.0, 18.0: 88.0, 19.0: 95.0,
                            20.0: 102.0, 21.0: 110.0, 22.0: 118.0
                        },
                        voltage_v=3.3,
                        efficiency=0.38
                    )
                        }
    
    # Courants de veille et réception (en mA)
    STANDBY_CURRENT_UA = 1.5  # µA en veille profonde
    SLEEP_CURRENT_UA = 0.15   # µA en sleep ultra-basse consommation
    RX_CURRENT_MA = 15.0      # mA en réception
    
    # Consommation des états (en mA)
    STATE_CURRENTS = {
        'SLEEP': SLEEP_CURRENT_UA / 1000.0,  # Converti en mA
        'STANDBY': STANDBY_CURRENT_UA / 1000.0,
        'RX': RX_CURRENT_MA,
        'TX': None  # Dépend de la puissance
    }
    
    # Facteurs de conversion
    HOURS_TO_SECONDS = 3600.0
    MILLIAMP_TO_AMP = 0.001
    MILLIWATT_TO_WATT = 0.001
    MICROJOULE_TO_JOULE = 1e-6
    NANOJOULE_TO_JOULE = 1e-9
    
    @classmethod
    def get_tx_current(cls, tx_power_dbm: float, pa_type: str = 'SX1261_LP') -> float:
        """
        Retourne le courant de transmission pour une puissance donnée (en mA)
        """
        profile = cls.POWER_PROFILES.get(pa_type, cls.POWER_PROFILES['SX1261_LP'])
        
        # Trouver les points de puissance les plus proches
        available_powers = sorted(profile.current_ma.keys())
        
        if not available_powers:
            raise ValueError(f"Profil {pa_type} sans données de courant")
        
        # Puissance exacte
        if tx_power_dbm in profile.current_ma:
            return profile.current_ma[tx_power_dbm]
        
        # Extrapolation linéaire
        if tx_power_dbm < available_powers[0]:
            # Extrapolation vers le bas
            p1, i1 = available_powers[0], profile.current_ma[available_powers[0]]
            if len(available_powers) > 1:
                p2, i2 = available_powers[1], profile.current_ma[available_powers[1]]
                slope = (i2 - i1) / (p2 - p1)
                return max(0, i1 + slope * (tx_power_dbm - p1))
            return i1
        
        if tx_power_dbm > available_powers[-1]:
            # Extrapolation vers le haut
            p2, i2 = available_powers[-1], profile.current_ma[available_powers[-1]]
            p1, i1 = available_powers[-2], profile.current_ma[available_powers[-2]]
            slope = (i2 - i1) / (p2 - p1)
            return i2 + slope * (tx_power_dbm - p2)
        
        # Interpolation entre deux points
        for i in range(len(available_powers) - 1):
            p1, p2 = available_powers[i], available_powers[i + 1]
            if p1 <= tx_power_dbm <= p2:
                i1, i2 = profile.current_ma[p1], profile.current_ma[p2]
                # Interpolation linéaire
                fraction = (tx_power_dbm - p1) / (p2 - p1)
                return i1 + fraction * (i2 - i1)
        
        # Fallback
        return profile.current_ma[available_powers[0]]
    
    @classmethod
    def ma_to_watts(cls, current_ma: float, voltage_v: float = 3.3) -> float:
        """Convertit un courant en mA en puissance en Watts"""
        return (current_ma * cls.MILLIAMP_TO_AMP) * voltage_v
    
    @classmethod
    def calculate_energy_joules(
        cls,
        tx_power_dbm: float,
        toa_ms: float,
        pa_type: str = 'SX1261_LP',
        sleep_duration_s: float = 300.0,
        rx_duration_ms: float = 0.0,
        voltage_v: float = 3.3
    ) -> Dict[str, float]:
        """
        Calcule la consommation énergétique complète pour une transmission LR-FHSS
        TOUTES LES VALEURS D'ÉNERGIE SONT EN JOULES (J)
        
        Args:
            tx_power_dbm: Puissance de transmission (dBm)
            toa_ms: Time-on-Air (ms)
            pa_type: Type d'amplificateur ('SX1261_LP' ou 'SX1262_HP')
            sleep_duration_s: Durée en mode veille après transmission (s)
            rx_duration_ms: Durée en réception (ms) - pour réception ACK
            voltage_v: Tension d'alimentation (V)
        
        Returns:
            Dictionnaire avec les métriques de consommation en Joules
        """
        # 1. Consommation en transmission (en Joules)
        tx_current_ma = cls.get_tx_current(tx_power_dbm, pa_type)
        tx_current_a = tx_current_ma * cls.MILLIAMP_TO_AMP
        tx_power_w = tx_current_a * voltage_v  # P = I * V (Watts)
        
        tx_duration_s = toa_ms / 1000.0  # Secondes
        tx_energy_j = tx_power_w * tx_duration_s  # E = P * t (Joules)
        
        # 2. Consommation en réception (pour ACK) - en Joules
        rx_energy_j = 0.0
        if rx_duration_ms > 0:
            rx_current_a = cls.STATE_CURRENTS['RX'] * cls.MILLIAMP_TO_AMP
            rx_power_w = rx_current_a * voltage_v
            rx_duration_s = rx_duration_ms / 1000.0
            rx_energy_j = rx_power_w * rx_duration_s
        
        # 3. Consommation en veille après transmission - en Joules
        standby_energy_j = 0.0
        if sleep_duration_s > 0:
            # Veille après transmission (1 seconde)
            standby_current_a = cls.STATE_CURRENTS['STANDBY'] * cls.MILLIAMP_TO_AMP
            standby_power_w = standby_current_a * voltage_v
            standby_energy_j = standby_power_w * 1.0  # 1 seconde
            
            # Puis sleep profond
            sleep_current_a = cls.STATE_CURRENTS['SLEEP'] * cls.MILLIAMP_TO_AMP
            sleep_power_w = sleep_current_a * voltage_v
            sleep_duration_s_remaining = sleep_duration_s - 1.0
            sleep_energy_j = sleep_power_w * sleep_duration_s_remaining
            
            standby_energy_j += sleep_energy_j
        
        # 4. Consommation totale en Joules
        total_energy_j = tx_energy_j + rx_energy_j + standby_energy_j
        
        # 5. Énergie par bit (en Joules/bit)
        bits_per_packet = 230 * 8  # Max payload LR-FHSS
        energy_per_bit_j = total_energy_j / bits_per_packet  # J/bit
        
        # 6. Durée de vie batterie
        battery_life_years = cls.calculate_battery_life_joules(
            energy_per_transmission_j=total_energy_j,
            battery_capacity_mah=1000.0,
            voltage_v=voltage_v,
            transmissions_per_day=24
        )
        
        # 7. Énergie RF effective transmise (en Joules)
        # Puissance RF = Puissance électrique * efficacité PA
        tx_power_rf_w = tx_power_w * cls.POWER_PROFILES[pa_type].efficiency
        tx_energy_rf_j = tx_power_rf_w * tx_duration_s
        
        return {
            # Courants et puissances
            'tx_current_ma': tx_current_ma,
            'tx_power_w': tx_power_w,
            'tx_power_rf_w': tx_power_rf_w,
            
            # Énergies en Joules
            'tx_energy_j': tx_energy_j,
            'tx_energy_rf_j': tx_energy_rf_j,
            'rx_energy_j': rx_energy_j,
            'standby_energy_j': standby_energy_j,
            'total_energy_j': total_energy_j,
            
            # Métriques spécifiques
            'energy_per_bit_j': energy_per_bit_j,
            'energy_per_bit_uj': energy_per_bit_j * 1e6,  # µJ/bit
            'battery_life_years': battery_life_years,
            
            # Informations système
            'pa_type': pa_type,
            'pa_efficiency': cls.POWER_PROFILES[pa_type].efficiency,
            'voltage_v': voltage_v,
            'tx_duration_s': tx_duration_s,
            'tx_power_dbm': tx_power_dbm,
            
            # Efficacité énergétique
            'energy_efficiency_bpj': 1.0 / energy_per_bit_j if energy_per_bit_j > 0 else 0,  # bits/Joule
            'energy_per_payload_byte_j': total_energy_j / 230 if 230 > 0 else 0,
        }
    
    @classmethod
    def calculate_battery_life_joules(
        cls,
        energy_per_transmission_j: float,
        battery_capacity_mah: float = 1000.0,
        voltage_v: float = 3.3,
        transmissions_per_day: int = 24
    ) -> float:
        """
        Calcule la durée de vie de la batterie en années
        Basée sur l'énergie par transmission en Joules
        
        Args:
            energy_per_transmission_j: Énergie par transmission (Joules)
            battery_capacity_mah: Capacité batterie (mAh)
            voltage_v: Tension batterie (V)
            transmissions_per_day: Nombre de transmissions par jour
        
        Returns:
            Durée de vie en années
        """
        if energy_per_transmission_j <= 0:
            return float('inf')
        
        # Énergie totale de la batterie en Joules
        # Énergie (J) = Capacité (Ah) * Tension (V) * 3600
        battery_capacity_ah = battery_capacity_mah * cls.MILLIAMP_TO_AMP
        total_battery_energy_j = battery_capacity_ah * voltage_v * cls.HOURS_TO_SECONDS
        
        # Énergie consommée par jour en Joules
        daily_energy_j = energy_per_transmission_j * transmissions_per_day
        
        if daily_energy_j <= 0:
            return float('inf')
        
        # Durée de vie en jours
        battery_life_days = total_battery_energy_j / daily_energy_j
        
        # Convertir en années
        battery_life_years = battery_life_days / 365.25
        
        return battery_life_years
    
    @classmethod
    def optimize_power_for_lifetime_joules(
        cls,
        target_lifetime_years: float,
        toa_ms: float,
        transmissions_per_day: int = 24,
        battery_capacity_mah: float = 1000.0,
        voltage_v: float = 3.3,
        pa_type: str = 'SX1261_LP'
    ) -> Dict[str, float]:
        """
        Optimise la puissance de transmission pour atteindre une durée de vie cible
        Version avec toutes les énergies en Joules
        
        Returns:
            Puissance optimale et métriques associées
        """
        # Énergie maximale disponible par jour (Joules)
        battery_capacity_ah = battery_capacity_mah * cls.MILLIAMP_TO_AMP
        total_battery_energy_j = battery_capacity_ah * voltage_v * cls.HOURS_TO_SECONDS
        daily_energy_budget_j = total_battery_energy_j / (target_lifetime_years * 365.25)
        
        # Énergie maximale par transmission (Joules)
        max_energy_per_tx_j = daily_energy_budget_j / transmissions_per_day
        
        # Essayer différentes puissances
        available_powers = sorted(cls.POWER_PROFILES[pa_type].current_ma.keys())
        optimal_power = available_powers[0]
        optimal_metrics = None
        
        for tx_power in available_powers:
            metrics = cls.calculate_energy_joules(
                tx_power_dbm=tx_power,
                toa_ms=toa_ms,
                pa_type=pa_type,
                voltage_v=voltage_v
            )
            
            if metrics['total_energy_j'] <= max_energy_per_tx_j:
                optimal_power = tx_power
                optimal_metrics = metrics
            else:
                break
        
        if optimal_metrics is None:
            # Utiliser la puissance la plus faible
            optimal_metrics = cls.calculate_energy_joules(
                tx_power_dbm=optimal_power,
                toa_ms=toa_ms,
                pa_type=pa_type,
                voltage_v=voltage_v
            )
        
        # Ajouter informations d'optimisation
        optimal_metrics.update({
            'target_lifetime_years': target_lifetime_years,
            'achievable_lifetime_years': optimal_metrics['battery_life_years'],
            'daily_energy_budget_j': daily_energy_budget_j,
            'max_energy_per_tx_j': max_energy_per_tx_j,
            'transmissions_per_day': transmissions_per_day,
            'battery_capacity_mah': battery_capacity_mah,
            'total_battery_energy_j': total_battery_energy_j,
        })
        
        return optimal_metrics
    
    @classmethod
    def compare_pa_profiles_joules(
        cls,
        tx_power_dbm: float,
        toa_ms: float,
        voltage_v: float = 3.3
    ) -> Dict[str, Dict]:
        """
        Compare les performances des différents profils PA
        Version avec toutes les énergies en Joules
        
        Returns:
            Dictionnaire comparant SX1261_LP et SX1262_HP
        """
        results = {}
        
        for pa_type in ['SX1261_LP', 'SX1262_HP']:
            if tx_power_dbm <= max(cls.POWER_PROFILES[pa_type].current_ma.keys()):
                results[pa_type] = cls.calculate_energy_joules(
                    tx_power_dbm=tx_power_dbm,
                    toa_ms=toa_ms,
                    pa_type=pa_type,
                    voltage_v=voltage_v
                )
        
        # Calculer les économies
        if 'SX1261_LP' in results and 'SX1262_HP' in results:
            lp_energy = results['SX1261_LP']['total_energy_j']
            hp_energy = results['SX1262_HP']['total_energy_j']
            
            if lp_energy > 0:
                energy_saving_pct = ((lp_energy - hp_energy) / lp_energy) * 100
            else:
                energy_saving_pct = 0
            
            # Économie en Joules par transmission
            energy_saving_j = lp_energy - hp_energy
            
            results['comparison'] = {
                'energy_saving_pct': energy_saving_pct,
                'energy_saving_j': energy_saving_j,
                'recommended_pa': 'SX1261_LP' if lp_energy < hp_energy else 'SX1262_HP',
                'pa_efficiency_lp': results['SX1261_LP']['pa_efficiency'],
                'pa_efficiency_hp': results['SX1262_HP']['pa_efficiency'],
            }
        
        return results
    
    @classmethod
    def calculate_daily_energy_consumption(
        cls,
        tx_power_dbm: float,
        toa_ms: float,
        transmissions_per_day: int = 24,
        pa_type: str = 'SX1261_LP',
        voltage_v: float = 3.3
    ) -> Dict[str, float]:
        """
        Calcule la consommation énergétique quotidienne
        
        Returns:
            Dictionnaire avec consommation quotidienne en Joules
        """
        # Énergie par transmission
        per_tx = cls.calculate_energy_joules(
            tx_power_dbm=tx_power_dbm,
            toa_ms=toa_ms,
            pa_type=pa_type,
            voltage_v=voltage_v
        )
        
        # Énergie quotidienne
        daily_energy_j = per_tx['total_energy_j'] * transmissions_per_day
        
        # Pourcentage de la batterie consommé par jour
        battery_capacity_mah = 1000.0  # Exemple: 1000mAh
        battery_capacity_ah = battery_capacity_mah * cls.MILLIAMP_TO_AMP
        total_battery_energy_j = battery_capacity_ah * voltage_v * cls.HOURS_TO_SECONDS
        
        daily_battery_drain_pct = (daily_energy_j / total_battery_energy_j) * 100
        
        result = {
            'energy_per_tx_j': per_tx['total_energy_j'],
            'daily_energy_j': daily_energy_j,
            'daily_energy_wh': daily_energy_j / cls.HOURS_TO_SECONDS,  # Wh/jour
            'transmissions_per_day': transmissions_per_day,
            'daily_battery_drain_pct': daily_battery_drain_pct,
            'battery_life_days': total_battery_energy_j / daily_energy_j if daily_energy_j > 0 else float('inf'),
            'battery_life_years': (total_battery_energy_j / daily_energy_j) / 365.25 if daily_energy_j > 0 else float('inf')
        }
        
        result.update(per_tx)
        
        return result

# ============================================================================
# INTÉGRATION AVEC LA SIMULATION EXISTANTE
# ============================================================================

class LR_FHSS_EnergyAnalyzer:
    """Analyseur de consommation énergétique pour la simulation LR-FHSS (Joules)"""
    
    def __init__(self, dashboard):
        self.dashboard = dashboard
        self.energy_stats = {
            'total_energy_j': 0.0,                    # TOUTES les transmissions
            'energy_successful_j': 0.0,               # Paquets reçus avec succès
            'energy_failed_j': 0.0,                   # Paquets NON reçus (énergie perdue)
            'avg_energy_per_packet_j': 0.0,
            'avg_energy_per_successful_j': 0.0,       # Moyenne par paquet réussi
            'avg_energy_per_failed_j': 0.0,           # Moyenne par paquet échoué
            'avg_current_ma': 0.0,
            'battery_life_years': 0.0,
            'energy_per_bit_j': 0.0,
            'packets_analyzed': 0,
            'packets_successful': 0,                  # Paquets reçus
            'packets_failed': 0,                      # Paquets échoués
            'pa_type': 'SX1261_LP',
            'daily_energy_j': 0.0,
            'efficiency_ratio': 0.0                   # % de l'énergie qui a réussi
        }
    
    def analyze_packet_energy(self, packet) -> Optional[Dict]:
        """
        Analyse la consommation énergétique d'un paquet LR-FHSS
        IMPORTANT: Compte l'énergie pour TOUS les paquets transmis, qu'ils soient reçus ou non
        
        Args:
            packet: SimulatedPacketConforme avec attributs toa_ms, tx_power_dbm, etc.
        
        Returns:
            Métriques de consommation en Joules ou None si impossible à analyser
        """
        try:
            # Vérifier attributs requis
            if not hasattr(packet, 'toa_ms') or not hasattr(packet, 'tx_power_dbm'):
                return None
            
            # Déterminer type PA selon puissance
            if hasattr(packet, 'pa_type'):
                pa_type = packet.pa_type
            else:
                # Auto-sélection selon puissance
                if packet.tx_power_dbm <= 16.0:
                    pa_type = 'SX1261_LP'
                else:
                    pa_type = 'SX1262_HP'
            
            # Calculer consommation en Joules
            energy_metrics = EnergyConsumptionModel.calculate_energy_joules(
                tx_power_dbm=packet.tx_power_dbm,
                toa_ms=packet.toa_ms,
                pa_type=pa_type,
                rx_duration_ms=100.0,  # Réception ACK typique
                voltage_v=3.3
            )
            
            # Ajouter info paquet
            energy_metrics.update({
                'packet_id': getattr(packet, 'packet_id', 'unknown'),
                'device_id': getattr(packet, 'device_id', 'unknown'),
                'dr': getattr(packet, 'dr', 8),
                'success': getattr(packet, 'success', False)
            })
            
            # DEBUG: Log le statut de succès
            if not getattr(packet, 'success', False):
                pass  # Déboguer les échecs
            
            # Mettre à jour les statistiques globales
            # IMPORTANT: Compter l'énergie pour TOUS les paquets, pas seulement les succès
            self._update_stats(energy_metrics)
            
            return energy_metrics
            
        except Exception as e:
            print(f"Erreur analyse énergie paquet: {e}")
            return None
    
    def _update_stats(self, energy_metrics: Dict):
        """Met à jour les statistiques énergétiques globales (en Joules)"""
        self.energy_stats['packets_analyzed'] += 1
        count = self.energy_stats['packets_analyzed']
        
        # Énergie de ce paquet
        packet_energy_j = energy_metrics.get('total_energy_j', 0.0)
        
        # ÉTAPE 1: Cumuler l'énergie TOTALE (tous les paquets)
        self.energy_stats['total_energy_j'] = self.energy_stats.get('total_energy_j', 0.0) + packet_energy_j
        
        # ÉTAPE 2: Séparer succès vs échec
        is_success = energy_metrics.get('success', False)
        
        # DEBUG
        if not is_success and count <= 5:
            pass  # Log débog
        
        if is_success:
            # Paquet reçu avec succès
            self.energy_stats['packets_successful'] = self.energy_stats.get('packets_successful', 0) + 1
            self.energy_stats['energy_successful_j'] = self.energy_stats.get('energy_successful_j', 0.0) + packet_energy_j
        else:
            # Paquet échoué - énergie gaspillée
            self.energy_stats['packets_failed'] = self.energy_stats.get('packets_failed', 0) + 1
            self.energy_stats['energy_failed_j'] = self.energy_stats.get('energy_failed_j', 0.0) + packet_energy_j
        
        # ÉTAPE 3: Énergie moyenne par paquet (tous)
        if count > 0:
            self.energy_stats['avg_energy_per_packet_j'] = self.energy_stats['total_energy_j'] / count
        
        # ÉTAPE 4: Énergie moyenne par paquet réussi
        if self.energy_stats.get('packets_successful', 0) > 0:
            self.energy_stats['avg_energy_per_successful_j'] = self.energy_stats['energy_successful_j'] / self.energy_stats['packets_successful']
        
        # ÉTAPE 5: Énergie moyenne par paquet échoué
        if self.energy_stats.get('packets_failed', 0) > 0:
            self.energy_stats['avg_energy_per_failed_j'] = self.energy_stats['energy_failed_j'] / self.energy_stats['packets_failed']
        
        # ÉTAPE 6: Ratio d'efficacité énergétique
        if self.energy_stats['total_energy_j'] > 0:
            self.energy_stats['efficiency_ratio'] = (self.energy_stats['energy_successful_j'] / self.energy_stats['total_energy_j']) * 100.0
        else:
            self.energy_stats['efficiency_ratio'] = 0.0
        
        # ÉTAPE 7: Courant moyen (moyenne pondérée)
        total_current = self.energy_stats.get('total_current_ma', 0.0) + energy_metrics.get('tx_current_ma', 0.0)
        self.energy_stats['total_current_ma'] = total_current
        if count > 0:
            self.energy_stats['avg_current_ma'] = total_current / count
        
        # ÉTAPE 8: Énergie par bit (basée sur énergie totale)
        if 'energy_per_bit_j' in energy_metrics:
            total_bit_energy = self.energy_stats.get('total_bit_energy_j', 0.0) + energy_metrics['energy_per_bit_j']
            self.energy_stats['total_bit_energy_j'] = total_bit_energy
            self.energy_stats['energy_per_bit_j'] = total_bit_energy / count
    def get_energy_report(self) -> str:
        """Génère un rapport de consommation énergétique en Joules"""
        report = []
        report.append("=" * 70)
        report.append("⚡ RAPPORT DE CONSOMMATION ÉNERGÉTIQUE LR-FHSS (JOULES)")
        report.append("=" * 70)
        
        if self.energy_stats['packets_analyzed'] == 0:
            report.append("\n⚠️ AUCUNE DONNÉE D'ÉNERGIE DISPONIBLE")
            return "\n".join(report)
        
        # Facteurs de conversion
        J_TO_mJ = 1000.0
        J_TO_uJ = 1e6
        
        report.append(f"\n📈 ANALYSE GLOBALE:")
        report.append(f"   • Paquets transmis (total): {self.energy_stats['packets_analyzed']:,}")
        report.append(f"   • Paquets reçus avec succès: {self.energy_stats['packets_successful']:,}")
        report.append(f"   • Paquets échoués (perdus): {self.energy_stats['packets_failed']:,}")
        report.append(f"   • Type PA utilisé: {self.energy_stats['pa_type']}")
        
        report.append(f"\n⚡ CONSOMMATION ÉNERGÉTIQUE TOTALE:")
        report.append(f"   • Énergie TOTALE (toutes transmissions): {self.energy_stats['total_energy_j']:.6f} J")
        report.append(f"   •                                       = {self.energy_stats['total_energy_j'] * J_TO_mJ:.3f} mJ")
        report.append(f"   •                                       = {self.energy_stats['total_energy_j'] * J_TO_uJ:.0f} µJ")
        
        report.append(f"\n✅ ÉNERGIE DES PAQUETS REÇUS:")
        report.append(f"   • Énergie (succès): {self.energy_stats['energy_successful_j']:.6f} J")
        report.append(f"   •                   = {self.energy_stats['energy_successful_j'] * J_TO_mJ:.3f} mJ")
        report.append(f"   •                   = {self.energy_stats['energy_successful_j'] * J_TO_uJ:.0f} µJ")
        if self.energy_stats['packets_successful'] > 0:
            report.append(f"   • Énergie moyenne par paquet réussi: {self.energy_stats['avg_energy_per_successful_j']:.6f} J")
            report.append(f"   •                                    = {self.energy_stats['avg_energy_per_successful_j'] * J_TO_mJ:.3f} mJ")
        
        report.append(f"\n❌ ÉNERGIE PERDUE (PAQUETS NON REÇUS):")
        report.append(f"   • Énergie gaspillée (échecs): {self.energy_stats['energy_failed_j']:.6f} J")
        report.append(f"   •                             = {self.energy_stats['energy_failed_j'] * J_TO_mJ:.3f} mJ")
        report.append(f"   •                             = {self.energy_stats['energy_failed_j'] * J_TO_uJ:.0f} µJ")
        if self.energy_stats['packets_failed'] > 0:
            report.append(f"   • Énergie moyenne par paquet échoué: {self.energy_stats['avg_energy_per_failed_j']:.6f} J")
            report.append(f"   •                                   = {self.energy_stats['avg_energy_per_failed_j'] * J_TO_mJ:.3f} mJ")
        
        report.append(f"\n📊 EFFICACITÉ ÉNERGÉTIQUE:")
        report.append(f"   • Ratio: {self.energy_stats['efficiency_ratio']:.1f}% de l'énergie a réussi")
        report.append(f"   • {100.0 - self.energy_stats['efficiency_ratio']:.1f}% de l'énergie a été perdue")
        
        report.append(f"\n⚡ CONSOMMATION PAR PAQUET (moyenne globale):")
        report.append(f"   • Énergie moyenne: {self.energy_stats['avg_energy_per_packet_j']:.6f} J")
        report.append(f"   •                     = {self.energy_stats['avg_energy_per_packet_j'] * J_TO_mJ:.3f} mJ")
        report.append(f"   •                     = {self.energy_stats['avg_energy_per_packet_j'] * J_TO_uJ:.0f} µJ")
        report.append(f"   • Courant Tx moyen: {self.energy_stats['avg_current_ma']:.1f} mA")
        report.append(f"   • Énergie par bit: {self.energy_stats['energy_per_bit_j']:.2e} J/bit")
        report.append(f"   •                     = {self.energy_stats['energy_per_bit_j'] * 1e9:.1f} nJ/bit")
        
        report.append("=" * 70)
        
        return "\n".join(report)
    
    def get_detailed_metrics(self) -> Dict:
        """Retourne les métriques détaillées en Joules"""
        return {
            'energy_metrics': self.energy_stats,
            'conversion_factors': {
                'joules_to_millijoules': 1000.0,
                'joules_to_microjoules': 1e6,
                'joules_to_watt_hours': 1/3600.0,
                'milliamps_to_amps': 0.001
            },
            'battery_assumptions': {
                'capacity_mah': 1000.0,
                'voltage_v': 3.3,
                'transmissions_per_day': 24
            }
        }
    
    def reset_stats(self):
        """Réinitialise les statistiques énergétiques"""
        self.energy_stats = {
            'total_energy_j': 0.0,                    # TOUTES les transmissions
            'energy_successful_j': 0.0,               # Paquets reçus avec succès
            'energy_failed_j': 0.0,                   # Paquets NON reçus (énergie perdue)
            'avg_energy_per_packet_j': 0.0,
            'avg_energy_per_successful_j': 0.0,       # Moyenne par paquet réussi
            'avg_energy_per_failed_j': 0.0,           # Moyenne par paquet échoué
            'avg_current_ma': 0.0,
            'battery_life_years': 0.0,
            'energy_per_bit_j': 0.0,
            'packets_analyzed': 0,
            'packets_successful': 0,                  # Paquets reçus
            'packets_failed': 0,                      # Paquets échoués
            'pa_type': 'SX1261_LP',
            'daily_energy_j': 0.0,
            'efficiency_ratio': 0.0                   # % de l'énergie qui a réussi
        }

# ============================================================================
# FONCTIONS D'UTILITÉ POUR LE DASHBOARD
# ============================================================================

def add_energy_to_dashboard(dashboard):
    """Ajoute l'analyse énergétique au dashboard existant"""
    
    # Créer l'analyseur
    energy_analyzer = LR_FHSS_EnergyAnalyzer(dashboard)
    
    # Modifier _evaluate_packet_end pour inclure l'analyse énergétique
    original_evaluate_packet_end = dashboard._evaluate_packet_end
    
    def new_evaluate_packet_end(packet):
        # Appeler l'original
        original_evaluate_packet_end(packet)
        
        # Analyser la consommation énergétique (en Joules)
        if hasattr(packet, 'success') and packet.success:
            energy_metrics = energy_analyzer.analyze_packet_energy(packet)
            if energy_metrics:
                packet.energy_metrics = energy_metrics
                
                # Optionnel: log périodique
                if dashboard.total_sent % 50 == 0:
                    energy_j = energy_metrics['total_energy_j']
                    dashboard.add_log(
                        f"⚡ Paquet {packet.packet_id}: "
                        f"{energy_j*1000:.3f} mJ, "
                        f"{energy_metrics['tx_current_ma']:.1f} mA"
                    )
    
    dashboard._evaluate_packet_end = new_evaluate_packet_end
    
    # Ajouter méthode pour obtenir le rapport énergétique
    dashboard.get_energy_report = energy_analyzer.get_energy_report
    dashboard.get_energy_metrics = energy_analyzer.get_detailed_metrics
    
    # Ajouter méthode pour réinitialiser les stats énergétiques
    def reset_energy_stats():
        energy_analyzer.reset_stats()
    
    dashboard.reset_energy_stats = reset_energy_stats
    
    return energy_analyzer

# Exemple d'utilisation
if __name__ == "__main__":
    print("🧪 Test du modèle de consommation énergétique LR-FHSS (Joules)")
    print("-" * 70)
    
    # Exemple 1: Calcul pour une transmission typique
    print("\n📡 Transmission LR-FHSS @ 14 dBm, 1500 ms ToA:")
    metrics = EnergyConsumptionModel.calculate_energy_joules(
        tx_power_dbm=14.0,
        toa_ms=1500.0,
        pa_type='SX1261_LP'
    )
    
    print(f"   • Courant Tx: {metrics['tx_current_ma']:.1f} mA")
    print(f"   • Puissance Tx: {metrics['tx_power_w']:.3f} W")
    print(f"   • Puissance RF: {metrics['tx_power_rf_w']:.3f} W")
    print(f"   • Énergie Tx: {metrics['tx_energy_j']:.6f} J = {metrics['tx_energy_j']*1000:.3f} mJ")
    print(f"   • Énergie totale: {metrics['total_energy_j']:.6f} J = {metrics['total_energy_j']*1000:.3f} mJ")
    print(f"   • Énergie/bit: {metrics['energy_per_bit_j']:.2e} J/bit = {metrics['energy_per_bit_j']*1e9:.1f} nJ/bit")
    print(f"   • Durée vie batterie: {metrics['battery_life_years']:.1f} ans")
    
    # Exemple 2: Comparaison PA
    print("\n🔍 Comparaison SX1261_LP vs SX1262_HP @ 14 dBm:")
    comparison = EnergyConsumptionModel.compare_pa_profiles_joules(
        tx_power_dbm=14.0,
        toa_ms=1500.0
    )
    
    for pa_type, metrics in comparison.items():
        if pa_type != 'comparison':
            energy_j = metrics['total_energy_j']
            print(f"   • {pa_type}: {metrics['tx_current_ma']:.1f} mA, "
                  f"{energy_j*1000:.3f} mJ, "
                  f"Efficacité: {metrics['pa_efficiency']*100:.1f}%")
    
    if 'comparison' in comparison:
        comp = comparison['comparison']
        print(f"   • Économie: {comp['energy_saving_pct']:.1f}% "
              f"({comp['energy_saving_j']*1000:.3f} mJ) avec {comp['recommended_pa']}")
    
    # Exemple 3: Consommation quotidienne
    print("\n📅 Consommation quotidienne (24 transmissions/jour):")
    daily = EnergyConsumptionModel.calculate_daily_energy_consumption(
        tx_power_dbm=14.0,
        toa_ms=1500.0,
        transmissions_per_day=24
    )
    
    print(f"   • Énergie/paquet: {daily['energy_per_tx_j']*1000:.3f} mJ")
    print(f"   • Énergie/jour: {daily['daily_energy_j']*1000:.3f} mJ")
    print(f"   •            : {daily['daily_energy_wh']:.6f} Wh")
    print(f"   • Durée vie: {daily['battery_life_days']:.0f} jours")
    print(f"   •            : {daily['battery_life_years']:.1f} ans")
    
    # Exemple 4: Optimisation pour 10 ans
    print("\n🎯 Optimisation pour 10 ans de durée de vie:")
    optimal = EnergyConsumptionModel.optimize_power_for_lifetime_joules(
        target_lifetime_years=10.0,
        toa_ms=1500.0,
        transmissions_per_day=24,
        battery_capacity_mah=1000.0
    )
    
    print(f"   • Puissance optimale: {optimal['tx_power_dbm']:.1f} dBm")
    print(f"   • Courant nécessaire: {optimal['tx_current_ma']:.1f} mA")
    print(f"   • Énergie/paquet: {optimal['total_energy_j']*1000:.3f} mJ")
    print(f"   • Durée vie atteignable: {optimal['achievable_lifetime_years']:.1f} ans")
    
    print("\n" + "=" * 70)
    print("✅ Tous les calculs sont effectués en unités SI (Joules, Watts, Secondes)")