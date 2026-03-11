#!/usr/bin/env python3
"""
integrated_ddqn.py - Intégration de l'agent DDQN entraîné
Adapté au nouveau modèle de canal et à la distance max 4km
"""

import numpy as np
import torch
import torch.nn as nn
import logging
from typing import Dict, Tuple, Optional, List
import os
import hashlib

logger = logging.getLogger(__name__)

class LightweightDDQN(nn.Module):
    """
    Réseau léger optimisé pour l'inférence rapide
    Architecture identique à ddqn.py
    """
    def __init__(self, state_dim: int = 8, action_dim: int = 56):
        super(LightweightDDQN, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        return self.net(x)


class IntegratedDDQNAgent:
    """
    Agent DDQN intégré pour utilisation en temps réel dans le simulateur
    Adapté au nouveau modèle de canal (distance max 4km)
    """
    def __init__(
        self,
        checkpoint_path: str,
        device: str = None,
        deterministic: bool = True
    ):
        """
        Initialise l'agent depuis un checkpoint
        """
        # Configuration identique à ddqn.py
        self.state_dim = 8
        self.dr_options = [8, 9, 10, 11]
        self.power_options = list(range(-4, 15))  # -4-14 dBm
        self.action_dim = len(self.dr_options) * len(self.power_options)  # 56 actions
        
        # Paramètres adaptés au nouveau modèle
        self.max_distance_km = 4.0  # NOUVELLE distance max
        self.max_payload_bytes = 230
        self.noise_floor_dbm = -174
        self.shadowing_std_db = 7.0
        
        self.deterministic = deterministic
        
        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Charger le modèle
        self.model = LightweightDDQN(self.state_dim, self.action_dim).to(self.device)
        self.model.eval()  # Mode évaluation
        
        # Charger les poids
        self.load_checkpoint(checkpoint_path)
        
        # Statistiques
        self.decision_count = 0
        self.action_distribution = {}
        for dr in self.dr_options:
            for power in self.power_options:
                self.action_distribution[(dr, power)] = 0
        
        logger.info(f"Agent DDQN intégré initialisé sur {self.device}")
        logger.info(f"Mode: {'Déterministe' if deterministic else 'Stochastique'}")
        logger.info(f"Dimension état: {self.state_dim}")
        logger.info(f"Dimension action: {self.action_dim}")
        logger.info(f"Distance max: {self.max_distance_km}km")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Charge le modèle depuis un checkpoint"""
        if checkpoint_path is None:
            raise FileNotFoundError("Checkpoint non fourni")
        
        actual_path = checkpoint_path
        
        # Si le chemin n'existe pas, chercher
        if not os.path.exists(actual_path):
            logger.warning(f"   ⚠️  Chemin principal non trouvé: {actual_path}")
            base = checkpoint_path
            candidate_dirs = ["", "ddqn_checkpoints", "BEST/dqn_models", "dqn_models"]
            found = False
            
            logger.info(f"   🔍 Recherche du fichier dans les répertoires candidats...")
            for d in candidate_dirs:
                for ext in [".pt", ".pth", ""]:
                    candidate = os.path.join(d, base + ext) if d else base + ext
                    logger.debug(f"      Tentative: {candidate}")
                    if os.path.exists(candidate):
                        actual_path = candidate
                        found = True
                        logger.info(f"      ✓ Trouvé: {candidate}")
                        break
                if found:
                    break
            if not found:
                raise FileNotFoundError(f"Checkpoint non trouvé: {checkpoint_path}")
        else:
            logger.info(f"   ✓ Chemin trouvé directement: {actual_path}")
        
        # Charger le checkpoint
        logger.info(f"   📥 Chargement du checkpoint...")
        checkpoint = torch.load(actual_path, map_location=self.device, weights_only=False)
        
        # Extraire le state_dict
        if 'policy_net_state_dict' in checkpoint:
            state_dict = checkpoint['policy_net_state_dict']
            logger.info(f"      Clé trouvée: policy_net_state_dict")
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            logger.info(f"      Clé trouvée: model_state_dict")
        else:
            state_dict = checkpoint
            logger.info(f"      Utilisation du checkpoint complet")
        
        # Vérification des dimensions
        if 'net.0.weight' in state_dict:
            expected_input_shape = (128, self.state_dim)
            actual_input_shape = state_dict['net.0.weight'].shape
            
            if actual_input_shape != expected_input_shape:
                error_msg = (
                    f"ERREUR: Dimension état incompatible!\n"
                    f"  Modèle chargé: {actual_input_shape[1]} dimensions\n"
                    f"  Modèle attendu: {expected_input_shape[1]} dimensions"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)
        
        # Charger
        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            logger.warning(f"   ⚠️  Clés manquantes: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"   ⚠️  Clés inattendues: {unexpected_keys}")
        
        logger.info(f"✅ Modèle chargé avec succès depuis: {actual_path}")
    
    def get_state_from_device(self, device_state: Dict) -> np.ndarray:
        """
        Construit l'état depuis les informations du device
        """
        # Extraire les informations
        distance_km = device_state.get('distance_km', 0.0)
        snr_history = device_state.get('snr_history', [0.0, 0.0, 0.0])
        success_history = device_state.get('success_history', [1.0, 1.0, 1.0, 1.0, 1.0])
        retry_count = device_state.get('retry_count', 0)
        step_count = device_state.get('tx_count', 0)
        payload_bytes = device_state.get('payload_bytes', 50)
        
        # 1. Distance normalisée [0,1]
        distance_norm = min(distance_km / self.max_distance_km, 1.0)
        
        # 2. SNR moyen normalisé [-1,1]
        avg_snr = np.mean(list(snr_history)) if snr_history else 0.0
        snr_norm = max(-1.0, min(1.0, avg_snr / 20.0))
        
        # 3. Taux de succès récent [0,1]
        recent_success_rate = np.mean(list(success_history)) if success_history else 1.0
        
        # 4. Compteur de retry normalisé [0,1]
        retry_norm = min(retry_count / 5.0, 1.0)
        
        # 5. Log distance [0,1]
        log_distance = np.log1p(distance_km) / np.log1p(self.max_distance_km)
        
        # 6. Variation diurne simulée [0,1]
        time_of_day = (step_count % 24) / 24.0
        
        # 7. Stabilité du signal [0,1]
        if len(snr_history) > 1:
            snr_stability = 1.0 / (1.0 + np.std(list(snr_history)))
        else:
            snr_stability = 0.5
            
        # 8. Taille du payload normalisée [0,1]
        payload_norm = payload_bytes / self.max_payload_bytes
        
        state = np.array([
            distance_norm,
            snr_norm,
            recent_success_rate,
            retry_norm,
            log_distance,
            time_of_day,
            snr_stability,
            payload_norm
        ], dtype=np.float32)
        
        return state
    
    def action_to_params(self, action: int) -> Tuple[int, float]:
        """
        Convertit un index d'action en (DR, tx_power)
        """
        if action < 0 or action >= self.action_dim:
            logger.error(f"Action {action} hors limites")
            return self.dr_options[0], self.power_options[0]
        
        dr_idx = action // len(self.power_options)
        power_idx = action % len(self.power_options)
        
        dr_idx = min(dr_idx, len(self.dr_options) - 1)
        power_idx = min(power_idx, len(self.power_options) - 1)
        
        return self.dr_options[dr_idx], self.power_options[power_idx]
    
    def predict(self, device_state: Dict) -> Dict[str, float]:
        """
        Prédit les paramètres optimaux pour un device donné
        """
        # Construire l'état
        state = self.get_state_from_device(device_state)
        
        # Vérifier les dimensions
        if len(state) != 8:
            if len(state) < 8:
                state = np.pad(state, (0, 8 - len(state)), mode='constant')
            else:
                state = state[:8]
        
        # Prédiction
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.model(state_tensor)
            
            if self.deterministic:
                action = q_values.argmax(dim=1).item()
                confidence = q_values.max().item()
            else:
                probs = torch.softmax(q_values / 0.1, dim=1)
                action = torch.multinomial(probs, 1).item()
                confidence = q_values[0, action].item()
        
        # Convertir en paramètres
        dr, tx_power = self.action_to_params(action)
        
        # Déterminer le coding rate et la bande passante
        if dr in [8, 9]:
            coding_rate = '1/3' if dr == 8 else '2/3'
            bw_khz = 136.71875
            # Fréquence centrale selon la bande passante
            center_index = hash(device_state.get('device_id', '')) % 3
            frequency_mhz = [868.1, 868.3, 868.5][center_index]
        else:  # DR 10, 11
            coding_rate = '1/3' if dr == 10 else '2/3'
            bw_khz = 335.9375
            center_index = hash(device_state.get('device_id', '')) % 2
            frequency_mhz = [868.13, 868.53][center_index]
        
        # Enregistrer la décision
        self.decision_count += 1
        self.action_distribution[(dr, tx_power)] += 1
        
        return {
            'dr': dr,
            'tx_power_dbm': float(tx_power),
            'confidence': float(confidence),
            'action_index': action,
            'coding_rate': coding_rate,
            'bw_khz': bw_khz,
            'frequency_mhz': frequency_mhz
        }
    
    def get_statistics(self) -> Dict:
        """Retourne les statistiques d'utilisation de l'agent"""
        if self.decision_count == 0:
            return {'total_decisions': 0}
        
        distribution_pct = {}
        for (dr, power), count in self.action_distribution.items():
            if count > 0:
                pct = (count / self.decision_count) * 100
                distribution_pct[f"DR{dr}_P{power}dBm"] = f"{pct:.1f}%"
        
        most_used = max(self.action_distribution.items(), key=lambda x: x[1])
        
        return {
            'total_decisions': self.decision_count,
            'action_distribution': distribution_pct,
            'most_used_action': {
                'dr': most_used[0][0],
                'power': most_used[0][1],
                'count': most_used[1],
                'percentage': f"{(most_used[1] / self.decision_count) * 100:.1f}%"
            }
        }
    
    def reset_statistics(self):
        """Réinitialise les statistiques"""
        self.decision_count = 0
        for key in self.action_distribution:
            self.action_distribution[key] = 0


class DQNManager:
    """
    Gestionnaire DQN pour intégration dans le simulateur
    """
    def __init__(self, simulation):
        self.simulation = simulation
        self.enabled = False
        self.agent = None
        self.auto_save_enabled = True
        
        self.stats = {
            'decisions': 0,
            'successes': 0,
            'failures': 0,
            'avg_confidence': 0.0,
            'dr_distribution': {8: 0, 9: 0, 10: 0, 11: 0},
            'power_distribution': {i: 0 for i in range(-4, 15)}
        }
        
        logger.info("DQNManager initialisé")
    
    def initialize(self, checkpoint_path: str, deterministic: bool = True) -> bool:
        """Initialise l'agent DDQN depuis un checkpoint"""
        try:
            if checkpoint_path is None or checkpoint_path == '':
                logger.error("❌ Erreur initialisation DQN: Chemin checkpoint vide ou None!")
                self.enabled = False
                return False
            
            logger.info(f"📥 Tentative initialisation DQN avec checkpoint: {checkpoint_path}")
            
            # Vérifier si le fichier existe
            import os
            if not os.path.exists(checkpoint_path):
                logger.warning(f"   ⚠️  Fichier introuvable: {checkpoint_path}")
                logger.info(f"   🔍 Tentative recherche du fichier...")
            else:
                logger.info(f"   ✓ Fichier trouvé: {checkpoint_path}")
            
            self.agent = IntegratedDDQNAgent(
                checkpoint_path=checkpoint_path,
                deterministic=deterministic
            )
            self.enabled = True
            logger.info("✅ Agent DDQN initialisé et activé avec succès!")
            logger.info(f"   • Modèle: LightweightDDQN")
            logger.info(f"   • Dimension état: {self.agent.state_dim}")
            logger.info(f"   • Dimension action: {self.agent.action_dim}")
            logger.info(f"   • Device: {self.agent.device}")
            logger.info(f"   • Mode: {'Déterministe' if deterministic else 'Stochastique'}")
            return True
        except FileNotFoundError as e:
            logger.error(f"❌ ERREUR: Checkpoint non trouvé!")
            logger.error(f"   Chemin demandé: {checkpoint_path}")
            logger.error(f"   Message: {e}")
            self.enabled = False
            self.agent = None
            return False
        except Exception as e:
            logger.error(f"❌ ERREUR initialisation DQN: {type(e).__name__}")
            logger.error(f"   Message: {e}")
            import traceback
            logger.error(f"   Traceback: {traceback.format_exc()}")
            self.enabled = False
            self.agent = None
            return False
    
    def get_recommendation(self, device_id: str) -> Optional[Dict]:
        """Obtient une recommandation de l'agent pour un device"""
        if not self.enabled or self.agent is None:
            return None
        
        device_state = self.simulation.devices_state.get(device_id)
        if device_state is None:
            logger.warning(f"Device {device_id} non trouvé")
            return None
        
        # Ajouter les champs manquants
        if 'success_history' not in device_state:
            device_state['success_history'] = [1.0, 1.0, 1.0, 1.0, 1.0]
        if 'snr_history' not in device_state:
            device_state['snr_history'] = [0.0, 0.0, 0.0]
        if 'retry_count' not in device_state:
            device_state['retry_count'] = 0
        if 'payload_bytes' not in device_state:
            device_state['payload_bytes'] = 50
        
        # Obtenir la prédiction
        recommendation = self.agent.predict(device_state)
        
        # Mettre à jour les statistiques
        self.stats['decisions'] += 1
        self.stats['dr_distribution'][recommendation['dr']] += 1
        self.stats['power_distribution'][int(recommendation['tx_power_dbm'])] += 1
        
        prev_avg = self.stats['avg_confidence']
        n = self.stats['decisions']
        self.stats['avg_confidence'] = ((n - 1) * prev_avg + recommendation['confidence']) / n
        
        return recommendation
    
    def record_feedback(self, device_id: str, success: bool, 
                       rssi_dbm: float, snr_db: float, 
                       failure_reason: str = None):
        """Enregistre le feedback d'une transmission"""
        if not self.enabled:
            return
        
        if success:
            self.stats['successes'] += 1
        else:
            self.stats['failures'] += 1
        
        # Mettre à jour l'historique du device
        if device_id in self.simulation.devices_state:
            device_state = self.simulation.devices_state[device_id]
            
            # Historique SNR
            if 'snr_history' not in device_state:
                device_state['snr_history'] = [0.0, 0.0, 0.0]
            device_state['snr_history'].append(snr_db)
            if len(device_state['snr_history']) > 3:
                device_state['snr_history'] = device_state['snr_history'][-3:]
            
            # Historique succès
            if 'success_history' not in device_state:
                device_state['success_history'] = []
            device_state['success_history'].append(1.0 if success else 0.0)
            if len(device_state['success_history']) > 5:
                device_state['success_history'] = device_state['success_history'][-5:]
            
            # Compteur retry
            if not success:
                device_state['retry_count'] = device_state.get('retry_count', 0) + 1
            else:
                device_state['retry_count'] = 0
    
    def get_stats(self) -> Dict:
        """Retourne les statistiques globales du DQN Manager"""
        if not self.enabled or self.agent is None:
            return {'enabled': False}
        
        agent_stats = self.agent.get_statistics()
        
        total_outcomes = self.stats['successes'] + self.stats['failures']
        success_rate = (self.stats['successes'] / total_outcomes) if total_outcomes > 0 else 0.0
        
        total_dr = sum(self.stats['dr_distribution'].values())
        dr_dist_pct = {}
        if total_dr > 0:
            for dr, count in self.stats['dr_distribution'].items():
                dr_dist_pct[f"DR{dr}"] = f"{(count / total_dr) * 100:.1f}%"
        
        return {
            'enabled': True,
            'total_decisions': self.stats['decisions'],
            'success_rate': f"{success_rate * 100:.1f}%",
            'successes': self.stats['successes'],
            'failures': self.stats['failures'],
            'avg_confidence': f"{self.stats['avg_confidence']:.2f}",
            'dr_distribution': dr_dist_pct,
            'agent_stats': agent_stats
        }
    
    def save_model(self, filepath: str = None, suffix: str = "") -> Optional[str]:
        """Sauvegarde le modèle actuel"""
        if not self.enabled or self.agent is None:
            return None
        
        try:
            import time
            if filepath is None:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                if suffix:
                    suffix = f"_{suffix}"
                filepath = f"BEST/dqn_models/dqn_model_{timestamp}{suffix}.pth"
            
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            checkpoint = {
                'policy_net_state_dict': self.agent.model.state_dict(),
                'stats': self.stats,
                'agent_stats': self.agent.get_statistics(),
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            }
            torch.save(checkpoint, filepath)
            logger.info(f"Modèle sauvegardé: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Erreur sauvegarde: {e}")
            return None
    
    def reset_stats(self):
        """Réinitialise les statistiques"""
        self.stats = {
            'decisions': 0,
            'successes': 0,
            'failures': 0,
            'avg_confidence': 0.0,
            'dr_distribution': {8: 0, 9: 0, 10: 0, 11: 0},
            'power_distribution': {i: 0 for i in range(-4, 15)}
        }
        
        if self.agent:
            self.agent.reset_statistics()
        
        logger.info("Statistiques DQN réinitialisées")