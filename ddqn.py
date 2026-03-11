#!/usr/bin/env python3
"""
ddqn.py - Entraînement DDQN pour politique générale LR-FHSS
Adapté au nouveau modèle de canal et aux conditions d'évaluation physiques
Distance max: 4km, Payload: 1-230 bytes
"""

from channel import calculate_path_loss, calculate_rssi, calculate_noise_power, calculate_snr, calculate_ber
from lr_fhss import evaluate_transmission, calculate_success_probability, _deterministic_success_decision

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
import os
import logging
from typing import Dict, Tuple, List, Optional
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
import time
import math
import hashlib

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import des fonctions de configuration
try:
    from config import LR_FHSS_CONFIG, Region
    logger.info("Configuration LR-FHSS importée avec succès")
except ImportError:
    logger.warning("config.py non trouvé, utilisation des valeurs par défaut")
    # Définition minimale si config.py n'est pas disponible
    class LR_FHSS_CONFIG:
        @staticmethod
        def calculate_toa_ms(dr: int, payload_bytes: int) -> float:
            # Paramètres fixes
            HEADER_DURATION_MS = 233.472
            PAYLOAD_DURATION_MS = 102.4
            
            # Nombre de headers selon CR
            if dr in [8, 10]:  # CR = 1/3
                N = 3
                cr = '1/3'
            elif dr in [9, 11]:  # CR = 2/3
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

# Transition pour la mémoire de replay
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class LightweightDDQN(nn.Module):
    """
    Réseau léger optimisé pour l'inférence rapide
    """
    def __init__(self, state_dim: int = 8, action_dim: int = 56):  # 4 DR × 14 puissances
        super(LightweightDDQN, self).__init__()
        
        # Architecture minimaliste mais efficace
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        
        # Initialisation
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    """
    Mémoire de replay standard
    """
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, *args):
        self.buffer.append(Transition(*args))
    
    def sample(self, batch_size: int):
        return random.sample(self.buffer, min(len(self.buffer), batch_size))
    
    def __len__(self):
        return len(self.buffer)

class TrainingEnvironment:
    """
    Environnement d'entraînement adapté au nouveau modèle de canal
    Distance max: 4km, Payload: 1-230 bytes
    """
    
    def __init__(self):
        # Espace d'actions: 4 DR × 14 puissances = 56 actions
        self.dr_options = [8, 9, 10, 11]
        self.power_options = list(range(-4, 15))  # -4-14 dBm
        self.action_dim = len(self.dr_options) * len(self.power_options)
        
        # Paramètres réalistes (adaptés au nouveau canal)
        self.max_distance_km = 4.0  # NOUVELLE distance max: 4km
        self.max_payload_bytes = 230  # Max LR-FHSS
        self.min_payload_bytes = 1    # Min LR-FHSS
        
        # Paramètres canal (identiques à simulation.py)
        self.noise_figure_db = 6.0
        self.path_loss_exponent = 3.3
        self.shadowing_std_db = 7.0
        self.reference_loss_db = 125.0  # EU868
        self.seed_global = 42
        
        # Paramètres démodulation
        self.dr_to_bw = {
            8: 136.71875,   # DR8: BW=136.71875 kHz, CR=1/3
            9: 136.71875,   # DR9: BW=136.71875 kHz, CR=2/3
            10: 335.9375,   # DR10: BW=335.9375 kHz, CR=1/3
            11: 335.9375    # DR11: BW=335.9375 kHz, CR=2/3
        }
        
        # Seuils de sensibilité RSSI (config.PERFORMANCE.sensitivity_dbm_by_dr)
        self.rssi_thresholds = {
            8: -136.0,   # DR8
            9: -133.0,   # DR9  
            10: -133.0,  # DR10
            11: -130.0   # DR11
        }
        
        # État interne pour l'épisode
        self.current_distance = None
        self.current_snr_history = deque(maxlen=3)  # Derniers 3 SNR
        self.success_history = deque(maxlen=5)      # Derniers 5 succès (1=succès, 0=échec)
        self.retry_count = 0
        self.step_count = 0
        self.max_steps = 100
        self.current_payload_bytes = 1 # Payload par défaut
        self.current_position = (0, 0)   # Position en mètres
        
        # Pour génération déterministe du shadowing
        self.shadowing_cache = {}
        
        logger.info(f"Environnement initialisé: {self.action_dim} actions, distance max={self.max_distance_km}km")
    
    def _calculate_shadowing(self, device_id: str, position: tuple) -> float:
        """Calcule le shadowing déterministe (identique à channel.py)"""
        cache_key = f"{device_id}_{position[0]:.1f}_{position[1]:.1f}"
        
        if cache_key in self.shadowing_cache:
            return self.shadowing_cache[cache_key]
        
        # Générer une seed déterministe
        seed_str = f"{device_id}_{position[0]:.1f}_{position[1]:.1f}_{self.seed_global}"
        seed = int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % (2**31)
        rng = np.random.RandomState(seed)
        
        # Shadowing suivant N(0, shadowing_std_db)
        shadowing = rng.normal(0, self.shadowing_std_db)
        
        # Limiter à ±3σ
        max_shadowing = 3 * self.shadowing_std_db
        shadowing = max(-max_shadowing, min(max_shadowing, shadowing))
        
        self.shadowing_cache[cache_key] = shadowing
        return shadowing
    
    def reset(self):
        """Réinitialise l'environnement pour un nouvel épisode"""
        # Distance aléatoire (distribution réaliste entre 0.1 et 4.0 km)
        u = np.random.random()
        if u < 0.4:  # 40% des nœuds proches
            self.current_distance = np.random.uniform(0.1, 1.0)
        elif u < 0.7:  # 30% des nœuds moyens
            self.current_distance = np.random.uniform(1.0, 2.5)
        else:  # 30% des nœuds éloignés
            self.current_distance = np.random.uniform(2.5, self.max_distance_km)
        
        # Position en mètres (pour le shadowing)
        angle = np.random.uniform(0, 2 * np.pi)
        self.current_position = (
            self.current_distance * 1000 * np.cos(angle),
            self.current_distance * 1000 * np.sin(angle)
        )
        
        # Payload aléatoire (1-230 bytes)
        self.current_payload_bytes = 1
        
        # Réinitialiser l'historique
        self.current_snr_history.clear()
        self.success_history.clear()
        for _ in range(3):
            self.current_snr_history.append(0.0)
        for _ in range(5):
            self.success_history.append(1.0)  # On commence optimiste (1 = succès)
        
        self.retry_count = 0
        self.step_count = 0
        
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """
        État pour la décision (8 dimensions)
        """
        # 1. Distance normalisée [0,1]
        distance_norm = self.current_distance / self.max_distance_km
        
        # 2. Moyenne des derniers SNR (normalisée [-1,1])
        avg_snr = np.mean(list(self.current_snr_history)) if self.current_snr_history else 0.0
        snr_norm = max(-1.0, min(1.0, avg_snr / 20.0))
        
        # 3. Taux de succès récent [0,1]
        recent_success_rate = np.mean(list(self.success_history)) if self.success_history else 1.0
        
        # 4. Compteur de retry normalisé [0,1]
        retry_norm = min(self.retry_count / 5.0, 1.0)
        
        # 5. Log distance (capture non-linéarité du path loss) [0,1]
        log_distance = np.log1p(self.current_distance) / np.log1p(self.max_distance_km)
        
        # 6. Variation diurne simulée [0,1]
        time_of_day = (self.step_count % 24) / 24.0
        
        # 7. Stabilité du signal [0,1]
        if len(self.current_snr_history) > 1:
            snr_stability = 1.0 / (1.0 + np.std(list(self.current_snr_history)))
        else:
            snr_stability = 0.5
            
        # 8. Taille du payload normalisée [0,1]
        payload_norm = self.current_payload_bytes / self.max_payload_bytes
        
        state = np.array([
            distance_norm,           # 0: Distance normalisée
            snr_norm,                # 1: SNR normalisé
            recent_success_rate,      # 2: Taux de succès récent
            retry_norm,               # 3: Compteur de retry
            log_distance,             # 4: Log distance
            time_of_day,              # 5: Temps simulé
            snr_stability,            # 6: Stabilité du signal
            payload_norm              # 7: Payload normalisé
        ], dtype=np.float32)
        
        return state
    
    def action_to_params(self, action: int) -> Tuple[int, float]:
        """Convertit l'action en (DR, puissance)"""
        dr_idx = action // len(self.power_options)
        power_idx = action % len(self.power_options)
        
        dr_idx = min(dr_idx, len(self.dr_options) - 1)
        power_idx = min(power_idx, len(self.power_options) - 1)
        
        return self.dr_options[dr_idx], self.power_options[power_idx]
    
    def _calculate_physical_metrics(self, tx_power: float, dr: int) -> Dict:
        """
        Calcule les métriques physiques (RSSI, SNR, BER) avec le nouveau modèle de canal
        """
        # Device ID temporaire pour l'entraînement
        device_id = f"train_device_{self.step_count}"
        
        # 1. Calculer le shadowing déterministe
        shadowing_db = self._calculate_shadowing(device_id, self.current_position)
        
        # 2. Calculer le path loss
        path_loss_db = calculate_path_loss(
            distance_km=self.current_distance,
            frequency_mhz=868.0,  # Fréquence centrale typique
            path_loss_exponent=self.path_loss_exponent,
            reference_loss_db=self.reference_loss_db
        )
        
        # 3. Calculer le RSSI
        rssi_dbm = tx_power - path_loss_db + shadowing_db
        
        # 4. Calculer le SNR
        bw_khz = self.dr_to_bw.get(dr, 136.71875)
        noise_power_dbm = calculate_noise_power(
            bandwidth_khz=bw_khz,
            noise_figure_db=self.noise_figure_db
        )
        snr_db = rssi_dbm - noise_power_dbm
        
        # 5. Déterminer le coding rate
        cr = '1/3' if dr in [8, 10] else '2/3'
        
        # 6. Calculer le BER
        ber = calculate_ber(snr_db, cr)
        
        return {
            'rssi_dbm': rssi_dbm,
            'snr_db': snr_db,
            'ber': ber,
            'path_loss_db': path_loss_db,
            'shadowing_db': shadowing_db,
            'cr': cr,
            'bw_khz': bw_khz
        }
    
    def _transmission_success(self, tx_power: float, dr: int) -> Tuple[bool, str]:
        """
        Détermine si la transmission réussit en utilisant le modèle physique
        de lr_fhss.py
        """
        # Calculer les métriques physiques
        metrics = self._calculate_physical_metrics(tx_power, dr)
        
        # 1. Vérification du seuil RSSI
        rssi_threshold = self.rssi_thresholds.get(dr, -135.0)
        if metrics['rssi_dbm'] < rssi_threshold - 10:  # Marge de 10 dB
            return False, f"RSSI_TOO_LOW ({metrics['rssi_dbm']:.1f} dBm < {rssi_threshold-10:.1f} dBm)"
        
        # 2. Calculer la probabilité de succès avec le modèle BER
        # Créer un objet paquet simulé pour utiliser calculate_success_probability
        class TempPacket:
            def __init__(self, cr, payload_bytes):
                self.cr = cr
                self.payload_bytes = payload_bytes
        
        temp_packet = TempPacket(metrics['cr'], self.current_payload_bytes)
        success_prob = calculate_success_probability(temp_packet, metrics['snr_db'])
        
        # 3. Décision déterministe basée sur la probabilité
        packet_id = f"train_{self.step_count}_{self.retry_count}"
        success = _deterministic_success_decision(packet_id, success_prob)
        
        if success:
            return True, f"SUCCESS (SNR={metrics['snr_db']:.1f}dB, prob={success_prob:.2f})"
        else:
            return False, f"DEMOD_FAILED (SNR={metrics['snr_db']:.1f}dB, BER={metrics['ber']:.3e})"
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Exécute une action (une transmission)
        """
        self.step_count += 1
        
        # Convertir l'action en paramètres
        dr, tx_power = self.action_to_params(action)
        
        # Calculer le ToA
        toa_ms = LR_FHSS_CONFIG.calculate_toa_ms(dr, self.current_payload_bytes)
        
        # Déterminer le succès de la transmission
        success, reason = self._transmission_success(tx_power, dr)
        
        # Calculer les métriques physiques pour le log
        metrics = self._calculate_physical_metrics(tx_power, dr)
        
        # Mettre à jour l'historique
        self.current_snr_history.append(metrics['snr_db'])
        self.success_history.append(1.0 if success else 0.0)
        
        if success:
            self.retry_count = 0
        else:
            self.retry_count += 1
        
        # Calculer la récompense
        reward = self._calculate_reward(success, tx_power, dr, metrics['snr_db'], toa_ms)
        
        # Informations de débogage
        info = {
            'distance_km': self.current_distance,
            'dr': dr,
            'tx_power_dbm': tx_power,
            'snr_db': metrics['snr_db'],
            'rssi_dbm': metrics['rssi_dbm'],
            'ber': metrics['ber'],
            'shadowing_db': metrics['shadowing_db'],
            'path_loss_db': metrics['path_loss_db'],
            'toa_ms': toa_ms,
            'success': success,
            'reason': reason,
            'retry_count': self.retry_count,
            'step': self.step_count,
            'payload_bytes': self.current_payload_bytes,
            'success_probability': 1.0 if success else 0.0  # Simplifié
        }
        
        # Vérifier si l'épisode est terminé
        done = self.step_count >= self.max_steps
        
        # Pour le prochain état, petit mouvement aléatoire
        if not done:
            # Petit changement de distance (±50m)
            distance_delta = np.random.uniform(-0.05, 0.05)
            self.current_distance = max(0.1, min(self.max_distance_km, 
                                                self.current_distance + distance_delta))
            
            # Mettre à jour la position en conséquence
            angle = np.arctan2(self.current_position[1], self.current_position[0])
            new_distance_m = self.current_distance * 1000
            self.current_position = (
                new_distance_m * np.cos(angle),
                new_distance_m * np.sin(angle)
            )
            
            # Changer occasionnellement le payload
            if np.random.random() < 0.1:
                self.current_payload_bytes = 1
        
        next_state = self._get_state() if not done else np.zeros(8, dtype=np.float32)
        
        return next_state, reward, done, info
    
    def _calculate_reward(self, success: bool, tx_power: float, 
                        dr: int, snr_db: float, toa_ms: float) -> float:
        """
        Calcul de la récompense adapté au nouveau modèle
        """
        # 1. ÉCHEC : Pénalité forte
        if not success:
            return -200.0  # Pénalité légèrement réduite car les échecs sont plus réalistes

        # 2. BASE DE SUCCÈS
        reward = 100.0
        
        # 3. PÉNALITÉ D'OCCUPATION DU CANAL (ToA)
        toa_norm = min(toa_ms / 4000.0, 1.0)
        reward -= toa_norm * 70.0  # Pénalité jusqu'à -70 points
        
        # 4. PÉNALITÉ DE PUISSANCE TX (Énergie)
        power_norm = (tx_power + 4.0) / 18.0
        reward -= power_norm * 40.0  # Pénalité jusqu'à -40 points
        
        # 5. BONUS POUR SNR OPTIMAL (évite le surdimensionnement)
        # SNR idéal: entre -5 et +5 dB
        if -5.0 <= snr_db <= 5.0:
            reward += 20.0
        elif snr_db > 15.0:
            # Pénalité pour SNR trop élevé (gaspillage)
            overkill = snr_db - 15.0
            reward -= overkill * 2.0
        elif snr_db < -10.0:
            # Pénalité pour SNR trop faible (risque d'échec)
            reward -= 20.0
        
        # 6. PÉNALITÉ DE RETRANSMISSION
        reward -= (self.retry_count * 25.0)
        
        # 7. BONUS POUR DR ÉLEVÉ (efficacité spectrale)
        dr_bonus = {8: 0, 9: 10, 10: 5, 11: 15}
        reward += dr_bonus.get(dr, 0)
        
        return reward
    
    def get_state_dim(self):
        return 8
    
    def get_action_dim(self):
        return self.action_dim

class DDQNAgent:
    """
    Agent DDQN pour apprentissage de politique générale
    """
    
    def __init__(self, state_dim: int, action_dim: int, 
                 learning_rate: float = 1e-4, gamma: float = 0.99):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Utilisation du device: {self.device}")
        
        # Réseaux
        self.policy_net = LightweightDDQN(state_dim, action_dim).to(self.device)
        self.target_net = LightweightDDQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Optimiseur
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Mémoire
        self.memory = ReplayBuffer(capacity=100000)
        
        # Hyperparamètres
        self.batch_size = 256
        self.target_update = 1000
        self.epsilon_start = 1.0
        self.epsilon_end = 0.05
        self.epsilon_decay = 0.995
        self.epsilon = self.epsilon_start
        
        # Statistiques
        self.steps_done = 0
        self.episodes_done = 0
        self.stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'losses': [],
            'success_rates': [],
            'avg_powers': [],
            'avg_drs': [],
            'avg_toa': [],
            'avg_snr': [],
            'epsilon_values': []
        }
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Sélection d'action epsilon-greedy"""
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()
    
    def store_transition(self, state, action, next_state, reward, done):
        self.memory.push(state, action, next_state, reward, done)
    
    def optimize_model(self):
        """Optimise le modèle sur un batch de la mémoire"""
        if len(self.memory) < self.batch_size:
            return None
        
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        
        state_batch = torch.FloatTensor(np.array(batch.state)).to(self.device)
        action_batch = torch.LongTensor(batch.action).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        done_batch = torch.BoolTensor(batch.done).to(self.device)
        
        # Calculer Q(s_t, a)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # Calculer V(s_{t+1}) avec le target network (Double DQN)
        with torch.no_grad():
            # Sélectionner l'action avec policy_net
            next_actions = self.policy_net(next_state_batch).argmax(1, keepdim=True)
            # Évaluer avec target_net
            next_state_values = self.target_net(next_state_batch).gather(1, next_actions).squeeze()
            next_state_values[done_batch] = 0.0
            
            expected_state_action_values = reward_batch + self.gamma * next_state_values
        
        # Calculer la loss
        loss = F.smooth_l1_loss(state_action_values.squeeze(), expected_state_action_values)
        
        # Optimiser
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Mettre à jour le target network périodiquement
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss.item()
    
    def update_epsilon(self):
        """Décroissance exponentielle de epsilon"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def save_model(self, path: str):
        """Sauvegarde le modèle"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
            'episodes_done': self.episodes_done,
            'stats': self.stats
        }, path)
        logger.info(f"Modèle sauvegardé: {path}")
    
    def load_model(self, path: str):
        """Charge le modèle"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint['steps_done']
        self.episodes_done = checkpoint['episodes_done']
        self.stats = checkpoint['stats']
        logger.info(f"Modèle chargé: {path}")

def train_ddqn(
    num_episodes: int = 5000,
    save_dir: str = "./ddqn_checkpoints",
    resume_from: str = None,
    save_interval: int = 500
):
    """
    Fonction principale d'entraînement
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Environnement
    env = TrainingEnvironment()
    
    # Agent
    agent = DDQNAgent(
        state_dim=env.get_state_dim(),
        action_dim=env.get_action_dim(),
        learning_rate=1e-4,
        gamma=0.99
    )
    
    # Reprise si spécifié
    start_episode = 0
    if resume_from and os.path.exists(resume_from):
        agent.load_model(resume_from)
        start_episode = agent.episodes_done
        logger.info(f"Reprise depuis l'épisode {start_episode}")
    
    logger.info("=" * 80)
    logger.info("DÉBUT DE L'ENTRAÎNEMENT DDQN - MODÈLE CANAL PHYSIQUE")
    logger.info(f"Distance max: {env.max_distance_km}km, Payload: 1-230 bytes")
    logger.info(f"Épisodes: {num_episodes}")
    logger.info(f"Dimension état: {env.get_state_dim()}")
    logger.info(f"Dimension action: {env.get_action_dim()}")
    logger.info("=" * 80)
    
    # Boucle d'entraînement
    for episode in range(start_episode, num_episodes):
        state = env.reset()
        episode_reward = 0.0
        episode_loss = 0.0
        episode_steps = 0
        
        # Statistiques de l'épisode
        episode_successes = 0
        episode_powers = []
        episode_drs = []
        episode_toa = []
        episode_snr = []
        
        done = False
        
        while not done:
            # Sélectionner l'action
            action = agent.select_action(state, training=True)
            
            # Exécuter l'action
            next_state, reward, done, info = env.step(action)
            
            # Stocker dans la mémoire
            agent.store_transition(state, action, next_state, reward, done)
            
            # Entraîner
            loss = agent.optimize_model()
            if loss is not None:
                episode_loss += loss
            
            # Mettre à jour les compteurs
            agent.steps_done += 1
            episode_steps += 1
            episode_reward += reward
            
            # Collecter les statistiques
            if info['success']:
                episode_successes += 1
            episode_powers.append(info['tx_power_dbm'])
            episode_drs.append(info['dr'])
            episode_toa.append(info['toa_ms'])
            episode_snr.append(info['snr_db'])
            
            # Passer à l'état suivant
            state = next_state
        
        # Mettre à jour epsilon
        agent.update_epsilon()
        agent.episodes_done += 1
        
        # Calculer les moyennes
        success_rate = episode_successes / episode_steps if episode_steps > 0 else 0.0
        avg_power = np.mean(episode_powers) if episode_powers else 0.0
        avg_dr = np.mean(episode_drs) if episode_drs else 8.0
        avg_toa = np.mean(episode_toa) if episode_toa else 0.0
        avg_snr = np.mean(episode_snr) if episode_snr else 0.0
        avg_loss = episode_loss / episode_steps if episode_steps > 0 else 0.0
        
        # Enregistrer les statistiques
        agent.stats['episode_rewards'].append(episode_reward)
        agent.stats['episode_lengths'].append(episode_steps)
        agent.stats['losses'].append(avg_loss)
        agent.stats['success_rates'].append(success_rate)
        agent.stats['avg_powers'].append(avg_power)
        agent.stats['avg_drs'].append(avg_dr)
        agent.stats['avg_toa'].append(avg_toa)
        agent.stats['avg_snr'].append(avg_snr)
        agent.stats['epsilon_values'].append(agent.epsilon)
        
        # Affichage périodique
        if (episode + 1) % 10 == 0:
            window = min(100, len(agent.stats['episode_rewards']))
            recent_rewards = agent.stats['episode_rewards'][-window:]
            recent_success = agent.stats['success_rates'][-window:]
            recent_snr = agent.stats['avg_snr'][-window:]
            
            logger.info(
                f"Épisode {episode + 1}/{num_episodes} | "
                f"Reward: {episode_reward:.1f} (moy: {np.mean(recent_rewards):.1f}) | "
                f"Success: {success_rate:.1%} (moy: {np.mean(recent_success):.1%}) | "
                f"Power: {avg_power:.1f} dBm | "
                f"DR: {avg_dr:.1f} | "
                f"SNR: {avg_snr:.1f} dB | "
                f"ToA: {avg_toa:.0f} ms | "
                f"Loss: {avg_loss:.4f} | "
                f"ε: {agent.epsilon:.3f}"
            )
        
        # Sauvegarde périodique
        if (episode + 1) % save_interval == 0:
            checkpoint_path = os.path.join(save_dir, f"ddqn_ep{episode+1}.pth")
            agent.save_model(checkpoint_path)
    
    # Sauvegarde finale
    final_path = os.path.join(save_dir, "ddqn_final.pth")
    agent.save_model(final_path)
    
    # Sauvegarde format léger pour déploiement
    deploy_path = os.path.join(save_dir, "ddqn_deploy.pth")
    torch.save(agent.policy_net.state_dict(), deploy_path)
    logger.info(f"Modèle de déploiement sauvegardé: {deploy_path}")
    
    # Générer les graphiques
    plot_training_results(agent.stats, save_dir)
    
    logger.info("=" * 80)
    logger.info("ENTRAÎNEMENT TERMINÉ")
    logger.info(f"Modèle final: {final_path}")
    logger.info("=" * 80)
    
    return agent

def plot_training_results(stats: Dict, save_dir: str):
    """Génère les graphiques de l'entraînement"""
    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    fig.suptitle('Résultats de l\'entraînement DDQN - Modèle Physique', fontsize=16)
    
    # Fonction moyenne mobile
    def moving_average(data, window=100):
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    # 1. Récompenses par épisode
    ax = axes[0, 0]
    rewards = stats['episode_rewards']
    ax.plot(rewards, alpha=0.3, label='Brut')
    ax.plot(moving_average(rewards), label='Moyenne mobile (100)', linewidth=2)
    ax.set_xlabel('Épisode')
    ax.set_ylabel('Récompense')
    ax.set_title('Récompenses par épisode')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Taux de succès
    ax = axes[0, 1]
    success = stats['success_rates']
    ax.plot(success, alpha=0.3, label='Brut')
    ax.plot(moving_average(success), label='Moyenne mobile (100)', linewidth=2)
    ax.set_xlabel('Épisode')
    ax.set_ylabel('Taux de succès')
    ax.set_title('Taux de succès')
    ax.set_ylim([0, 1])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Perte d'entraînement
    ax = axes[0, 2]
    losses = stats['losses']
    ax.plot(losses, alpha=0.3, label='Brut')
    ax.plot(moving_average(losses), label='Moyenne mobile (100)', linewidth=2)
    ax.set_xlabel('Épisode')
    ax.set_ylabel('Perte')
    ax.set_title('Perte d\'entraînement')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. SNR moyen
    ax = axes[0, 3]
    snr_values = stats['avg_snr']
    ax.plot(snr_values, alpha=0.3, label='Brut')
    ax.plot(moving_average(snr_values), label='Moyenne mobile (100)', linewidth=2)
    ax.set_xlabel('Épisode')
    ax.set_ylabel('SNR (dB)')
    ax.set_title('SNR moyen')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Epsilon
    ax = axes[1, 0]
    ax.plot(stats['epsilon_values'])
    ax.set_xlabel('Épisode')
    ax.set_ylabel('Epsilon')
    ax.set_title('Exploration (ε)')
    ax.grid(True, alpha=0.3)
    
    # 6. Puissance moyenne
    ax = axes[1, 1]
    powers = stats['avg_powers']
    ax.plot(powers, alpha=0.3, label='Brut')
    ax.plot(moving_average(powers), label='Moyenne mobile (100)', linewidth=2)
    ax.set_xlabel('Épisode')
    ax.set_ylabel('Puissance (dBm)')
    ax.set_title('Puissance moyenne')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 7. DR moyen
    ax = axes[1, 2]
    drs = stats['avg_drs']
    ax.plot(drs, alpha=0.3, label='Brut')
    ax.plot(moving_average(drs), label='Moyenne mobile (100)', linewidth=2)
    ax.set_xlabel('Épisode')
    ax.set_ylabel('DR')
    ax.set_title('DR moyen')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 8. ToA moyen
    ax = axes[1, 3]
    toa_values = stats['avg_toa']
    ax.plot(toa_values, alpha=0.3, label='Brut')
    ax.plot(moving_average(toa_values), label='Moyenne mobile (100)', linewidth=2)
    ax.set_xlabel('Épisode')
    ax.set_ylabel('ToA (ms)')
    ax.set_title('Time-on-Air moyen')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Sauvegarder
    plot_path = os.path.join(save_dir, 'training_results.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Graphiques sauvegardés: {plot_path}")
    plt.close()

def test_trained_policy(model_path: str, num_tests: int = 100):
    """
    Teste la politique apprise sur des scénarios variés
    """
    logger.info("=" * 80)
    logger.info("TEST DE LA POLITIQUE APPRISE")
    logger.info("=" * 80)
    
    # Créer un agent en mode test
    env = TrainingEnvironment()
    agent = DDQNAgent(
        state_dim=env.get_state_dim(),
        action_dim=env.get_action_dim()
    )
    agent.load_model(model_path)
    agent.epsilon = 0.0  # Pas d'exploration
    
    # Scénarios de test
    test_distances = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    test_payloads = [1, 50, 100, 150, 200, 230]
    
    results = []
    
    for dist in test_distances:
        for payload in test_payloads:
            # Simuler un nœud avec ces paramètres
            env.current_distance = dist
            env.current_payload_bytes = payload
            env.current_snr_history = deque([0.0, 0.0, 0.0], maxlen=3)
            env.success_history = deque([1.0, 1.0, 1.0, 1.0, 1.0], maxlen=5)
            
            # Position en mètres
            angle = np.random.uniform(0, 2*np.pi)
            env.current_position = (
                dist * 1000 * np.cos(angle),
                dist * 1000 * np.sin(angle)
            )
            
            # Obtenir plusieurs décisions
            decisions = []
            toa_values = []
            snr_values = []
            for _ in range(5):
                state = env._get_state()
                action = agent.select_action(state, training=False)
                dr, power = env.action_to_params(action)
                toa_ms = LR_FHSS_CONFIG.calculate_toa_ms(dr, payload)
                
                # Calculer le SNR attendu
                metrics = env._calculate_physical_metrics(power, dr)
                
                decisions.append((dr, power, toa_ms))
                toa_values.append(toa_ms)
                snr_values.append(metrics['snr_db'])
            
            # Moyenne des décisions
            avg_dr = np.mean([d[0] for d in decisions])
            avg_power = np.mean([d[1] for d in decisions])
            avg_toa = np.mean(toa_values)
            avg_snr = np.mean(snr_values)
            
            # Simuler le résultat avec le modèle physique
            success_prob = 1.0 / (1.0 + np.exp(-(avg_snr - (-5.0)) / 3.0))
            
            results.append({
                'distance_km': dist,
                'payload_bytes': payload,
                'recommended_dr': round(avg_dr, 1),
                'recommended_power': round(avg_power, 1),
                'estimated_toa_ms': round(avg_toa, 0),
                'estimated_snr': round(avg_snr, 1),
                'success_probability': round(success_prob, 2)
            })
            
            logger.info(
                f"Dist: {dist:.1f}km, Payload: {payload}B → "
                f"DR: {round(avg_dr, 1)}, "
                f"Power: {round(avg_power, 1)} dBm, "
                f"ToA: {avg_toa:.0f} ms, "
                f"SNR: {avg_snr:.1f} dB, "
                f"Succès: {success_prob:.1%}"
            )
    
    # Afficher un résumé
    logger.info("\n" + "=" * 80)
    logger.info("RÉSUMÉ DE LA POLITIQUE APPRISE:")
    logger.info("-" * 80)
    
    for r in results:
        logger.info(f"{r['distance_km']:.1f}km/{r['payload_bytes']}B: "
                   f"DR={r['recommended_dr']}, "
                   f"Pwr={r['recommended_power']}dBm, "
                   f"ToA={r['estimated_toa_ms']}ms "
                   f"({r['success_probability']:.0%} succès)")
    
    logger.info("=" * 80)

def create_deployment_model(trained_model_path: str, output_path: str = "lr_fhss_policy.pth"):
    """
    Crée un modèle léger optimisé pour le déploiement
    """
    # Charger le modèle entraîné
    checkpoint = torch.load(trained_model_path, map_location='cpu')
    
    # Créer un modèle minimal pour déploiement
    class DeploymentModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(8, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 56)  # 56 actions
            )
        
        def forward(self, x):
            return self.layers(x)
    
    # Initialiser et charger les poids
    model = DeploymentModel()
    
    if 'policy_net_state_dict' in checkpoint:
        trained_state_dict = checkpoint['policy_net_state_dict']
        
        # Mapping des couches
        new_state_dict = {}
        for key, value in trained_state_dict.items():
            new_key = key.replace('net.', 'layers.')
            if new_key in model.state_dict():
                new_state_dict[new_key] = value
        
        model.load_state_dict(new_state_dict, strict=False)
    else:
        model.load_state_dict(checkpoint)
    
    # Sauvegarder
    torch.save(model.state_dict(), output_path)
    
    # Afficher la taille
    size_bytes = os.path.getsize(output_path)
    logger.info(f"Modèle de déploiement créé: {output_path}")
    logger.info(f"Taille: {size_bytes/1024:.1f} KB")
    
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Entraînement DDQN pour LR-FHSS')
    parser.add_argument('--episodes', type=int, default=1000, help='Nombre d\'épisodes')
    parser.add_argument('--save-dir', type=str, default='./ddqn_checkpoints', 
                       help='Répertoire de sauvegarde')
    parser.add_argument('--resume', type=str, default=None, 
                       help='Reprendre depuis un checkpoint')
    parser.add_argument('--test', type=str, default=None, 
                       help='Tester un modèle entraîné')
    parser.add_argument('--deploy', type=str, default=None,
                       help='Créer un modèle optimisé pour déploiement')
    
    args = parser.parse_args()
    
    if args.test:
        test_trained_policy(args.test)
    elif args.deploy:
        create_deployment_model(args.deploy, "lr_fhss_policy_deploy.pth")
    else:
        train_ddqn(
            num_episodes=args.episodes,
            save_dir=args.save_dir,
            resume_from=args.resume,
            save_interval=100
        )