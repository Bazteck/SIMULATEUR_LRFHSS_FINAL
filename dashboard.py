"""
dashboard.py
Dashboard Simulation LR-FHSS - Interface graphique seulement
"""

import panel as pn
import param
import numpy as np
import pandas as pd
import threading
import time
import json
import os
from queue import Queue, Empty
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Liberation Sans', 'Arial']
matplotlib.rcParams['font.family'] = 'sans-serif'
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.figure import Figure
from matplotlib.colors import Normalize
import logging

# Importer la simulation
from simulation import LR_FHSS_Simulation, DQN_AVAILABLE, INTELLIGENT_SCHEDULER_AVAILABLE, ENERGY_MODULE_AVAILABLE

# Configuration logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Initialisation Panel
pn.extension(sizing_mode='stretch_width', notifications=True)


class SimulationDashboard(param.Parameterized):
    """Dashboard Simulation LR-FHSS - Interface graphique seulement"""
    
    # === PARAMÈTRES CONFORMES ===
    simulation_duration = param.Number(86400, bounds=(60, 86400), label="Durée simulation (s)")
    num_devices = param.Integer(100, bounds=(1, 40000), label="Nombre d'EndDevices")
    distance_gtw = param.Number(15000, bounds=(100, 500000), label="Distance Max Noeud - Passerelle (m)")
    tx_power = param.Number(14.0, bounds=(-4, 30), label="Puissance transmission (dBm)")
    
    data_rate = param.ObjectSelector(default='DR8', objects=['DR8', 'DR9', 'DR10', 'DR11'], label="Data Rate (DR)")
    coding_rate = param.ObjectSelector(default='1/3', objects=['1/3', '2/3'], label="Coding Rate (CR)")
    bandwidth_khz = param.ObjectSelector(default=136.71875, objects=[136.71875, 335.9375, 1523], label="Bande passante (kHz)")
    region = param.ObjectSelector(default='EU868', objects=['EU868'], label="Région")
    
    # Paramètres de transmission
    payload_min = param.Integer(1, bounds=(1, 230), label="Payload min (bytes)")
    payload_max = param.Integer(1, bounds=(1, 230), label="Payload max (bytes)")
    tx_interval_min = param.Number(3600, bounds=(10, 86400), label="Interval min (s)")
    tx_interval_max = param.Number(3600, bounds=(10, 86400), label="Interval max (s)")
    
    # Paramètres canal RF
    shadowing_std_db = param.Number(7.0, bounds=(0, 20), label="Shadowing std (dB)")
    path_loss_exponent = param.Number(2.7, bounds=(2.0, 4.0), label="Exposant perte trajet")
    doppler_hz = param.Number(0.0, bounds=(0, 100), label="Doppler (Hz)")
    multipath_enabled = param.Boolean(True, label="Multi-trajets")
    
    # Paramètres démodulation
    # sample_rate_khz = param.Integer(2000, bounds=(500, 10000), label="Sample rate (kHz)")
    noise_figure_db = param.Number(6.0, bounds=(0, 10), label="Facteur bruit Rx (dB)")
    
    # === PARAMÈTRES ÉNERGIE ===
    pa_type = param.ObjectSelector(default='SX1261_LP', 
                                  objects=['SX1261_LP', 'SX1262_HP'], 
                                  label="Type d'amplificateur")
    
    battery_capacity_mah = param.Number(1000.0, bounds=(100, 10000), 
                                       label="Capacité batterie (mAh)")
    
    transmissions_per_day = param.Integer(24, bounds=(1, 1440), 
                                         label="Transmissions par jour")
    
    # === PARAMÈTRES SCHEDULER INTELLIGENT ===
    enable_intelligent_scheduler = param.Boolean(False, label="Activer le Scheduler Intelligent")
    scheduler_max_delay = param.Number(10.0, bounds=(0, 60), label="Délai max scheduler (s)")
    scheduler_allow_freq_shift = param.Boolean(True, label="Autoriser shift fréquentiel")
    scheduler_allow_power_boost = param.Boolean(False, label="Autoriser boost puissance")
    
    # === PARAMÈTRES DQN ===
    enable_dqn = param.Boolean(False, label="Activer DQN")
    dqn_model_name = param.Selector(default=None, objects=[])
    dqn_exploration = param.Number(0, bounds=(0.0, 0.3), label="Exploration DQN")
    use_dqn_for_dr = param.Boolean(True, label="DQN pour DR")
    use_dqn_for_power = param.Boolean(True, label="DQN pour Puissance")
    # use_dqn_for_frequency = param.Boolean(False, label="DQN pour Fréquence")
    
    # === NOUVEAU PARAMÈTRE POUR SHADOWING ===
    show_shadowing_map = param.Boolean(False, label="Afficher la carte de Shadowing")
    
    # === ÉTAT SIMULATION ===
    is_running = param.Boolean(False, precedence=-1)
    progress = param.Number(0, bounds=(0, 100), precedence=-1)
    status = param.String("Prêt", precedence=-1)
    results_text = param.String("", precedence=-1)
    
    def __init__(self, **params):
        super().__init__(**params)
        self.log_buffer = []
        self.latest_metrics = {}
        self.simulation = None
        
        self.config = {}
        
        # Cache pour les valeurs de shadowing par position
        self.shadowing_cache = {}
        
        # Mettre à jour la liste des modèles
        self.update_dqn_model_list()
        
        # État simulation
        self.device_positions = []
        self.gateway_position = (0, 0)
        self.simulation_thread = None
        self.update_timer = None
        self.metric_queue = Queue()
        
        # Seed FIXE pour positions reproductibles
        self.initial_position_seed = 42  # <-- CHANGÉ de position_seed à initial_position_seed
        
        # Créer interface
        self.create_interface()
        self.generate_device_positions()
        if self.config.get('enable_dqn', False):
            from integrated_ddqn import integrate_dqn_with_simulation
            integrate_dqn_with_simulation(self.simulation)
        
        # Démarrer la surveillance des modèles DQN pour mise à jour dynamique
        self._start_model_watcher()
            
        # Ajouter des watchers
        self.param.watch(self._on_num_devices_changed, 'num_devices')
        self.param.watch(self._on_distance_changed, 'distance_gtw')
        self.param.watch(self._on_data_rate_changed, 'data_rate')
        self.param.watch(self._on_bandwidth_changed, 'bandwidth_khz')
        self.param.watch(self._on_enable_scheduler_changed, 'enable_intelligent_scheduler')
        self.param.watch(self._on_scheduler_params_changed, ['scheduler_max_delay', 'scheduler_allow_freq_shift', 'scheduler_allow_power_boost'])
        self.param.watch(self._on_enable_dqn_changed, 'enable_dqn')
        self.param.watch(self._on_dqn_model_changed, 'dqn_model_name')
        self.param.watch(self._on_pa_type_changed, 'pa_type')
        self.param.watch(self._on_battery_changed, 'battery_capacity_mah')
        self.param.watch(self._on_shadowing_changed, 'shadowing_std_db')  # Nouveau watcher
        self.param.watch(self._on_show_shadowing_changed, 'show_shadowing_map')  # Nouveau watcher
        
        logger.info("Dashboard initialisé")
    
    def update_dqn_model_list(self):
        """Met à jour la liste des modèles DQN disponibles"""
        # Chercher dans plusieurs répertoires
        model_dirs = ["ddqn_models"]
        
        # Récupérer tous les fichiers .pt et .pth
        files = []
        for model_path in model_dirs:
            if not os.path.exists(model_path):
                continue
            
            try:
                for f in os.listdir(model_path):
                    if f.endswith('.pt') or f.endswith('.pth'):
                        # Supprimer l'extension
                        filename = f[:-3] if f.endswith('.pt') else f[:-4]
                        if filename not in files:
                            files.append(filename)
            except Exception as e:
                logger.warning(f"Erreur lors de la lecture du répertoire {model_path}: {e}")
        
        # Séparer les modèles fixed nodes des autres
        fixed_nodes_models = [f for f in files if 'fixed_nodes' in f.lower()]
        regular_models = [f for f in files if 'fixed_nodes' not in f.lower()]
        
        # Organiser la liste avec séparation visuelle
        all_models = ['Nouveau modèle']
        
        if regular_models:
            all_models.extend(sorted(regular_models))
        
        if fixed_nodes_models:
            all_models.append('--- Modèles Noeuds Fixes ---')
            all_models.extend(sorted(fixed_nodes_models))
        
        if all_models:
            self.param.dqn_model_name.objects = all_models
            if self.dqn_model_name is None or self.dqn_model_name not in self.param.dqn_model_name.objects:
                self.dqn_model_name = 'Nouveau modèle'
        else:
            self.param.dqn_model_name.objects = ['Nouveau modèle']
            self.dqn_model_name = 'Nouveau modèle'
        
        logger.info(f" DQN Model List: {len(regular_models)} réguliers + {len(fixed_nodes_models)} noeuds fixes")
    
    def _refresh_dqn_models_periodic(self):
        """Rafraîchit périodiquement la liste des modèles DQN pour détecter les nouveaux fichiers"""
        if not hasattr(self, '_last_model_count'):
            self._last_model_count = 0
        
        try:
            # Chercher dans plusieurs répertoires
            model_dirs = ["BEST/dqn_models", "ddqn_models", "ddqn_checkpoints"]
            current_files = []
            
            for model_path in model_dirs:
                if os.path.exists(model_path):
                    for f in os.listdir(model_path):
                        if f.endswith('.pt') or f.endswith('.pth'):
                            current_files.append(f)
            
            # Si le nombre de fichiers a changé, mettre à jour la liste
            if len(current_files) != self._last_model_count:
                self._last_model_count = len(current_files)
                self.update_dqn_model_list()
                if len(current_files) > 0:
                    logger.info(f" Modèles DQN mises à jour: {len(current_files)} fichiers détectés")
        except Exception as e:
            logger.debug(f"Erreur rafraîchissement modèles: {e}")
    
    def _start_model_watcher(self):
        """Démarre la surveillance des nouveaux modèles DQN sans redémarrage"""
        def watch_models():
            import time
            while getattr(self, '_watching_models', True):
                try:
                    self._refresh_dqn_models_periodic()
                except Exception as e:
                    logger.debug(f"Erreur watcher modèles: {e}")
                # Vérifier chaque 5 secondes
                time.sleep(5)
        
        # Démarrer le thread de surveillance
        if not hasattr(self, '_model_watcher_thread'):
            import threading
            self._watching_models = True
            self._model_watcher_thread = threading.Thread(target=watch_models, daemon=True)
            self._model_watcher_thread.start()
            logger.info("📡 Surveillance des modèles DQN activée (mise à jour dynamique)")
    
    def _on_data_rate_changed(self, event):
        """Met à jour CR et BW en fonction du DR"""
        dr_map = {
            'DR8': ('1/3', 136.71875),
            'DR9': ('2/3', 136.71875),
            'DR10': ('1/3', 335.9375),
            'DR11': ('2/3', 335.9375)
        }
        if event.new in dr_map:
            self.coding_rate, self.bandwidth_khz = dr_map[event.new]
    
    def _on_bandwidth_changed(self, event):
        """Quand la bande passante change"""
        self.add_log(f" Bande passante changée: BW={event.new} kHz")
    
    def _on_enable_scheduler_changed(self, event):
        """Quand le scheduler est activé/désactivé"""
        if event.new:
            self.add_log(" Scheduler intelligent activé")
        else:
            self.add_log(" Scheduler intelligent désactivé")
    
    def _on_scheduler_params_changed(self, event):
        """Quand les paramètres du scheduler changent"""
        self.add_log(f" Paramètres scheduler mis à jour")
    
    def _on_enable_dqn_changed(self, event):
        """Quand DQN est activé/désactivé"""
        if event.new:
            self.add_log(" DQN activé")
        else:
            self.add_log("DQN désactivé")
    
    def _on_dqn_model_changed(self, event):
        """Quand le modèle DQN change"""
        if self.enable_dqn:
            self.add_log(f" Modèle DQN changé: {event.new}")
    
    def _on_pa_type_changed(self, event):
        """Quand le type d'amplificateur change"""
        self.add_log(f" Type PA changé: {event.new}")
    
    def _on_battery_changed(self, event):
        """Quand la capacité batterie change"""
        self.add_log(f" Capacité batterie changée: {event.new} mAh")
    
    def _on_shadowing_changed(self, event):
        """Quand l'écart-type de shadowing change"""
        self.add_log(f" Écart-type shadowing changé: {event.new} dB")
        # Vider le cache quand le shadowing change
        self.shadowing_cache = {}
        # Régénérer la carte si elle est affichée
        if self.show_shadowing_map:
            self.update_deployment_map()
    
    def _on_show_shadowing_changed(self, event):
        """Quand l'affichage de la carte de shadowing change"""
        if event.new:
            self.add_log(" Affichage carte de shadowing activé")
        else:
            self.add_log(" Affichage carte de shadowing désactivé")
        self.update_deployment_map()
    
    def _on_num_devices_changed(self, event):
        """Quand le nombre de devices change"""
        self.add_log(f" Nombre de devices changé: {event.new}")
        self.generate_device_positions()

    def _on_distance_changed(self, event):
        """Quand la distance change"""
        self.add_log(f" Distance changée: {event.new}m")
        self.generate_device_positions()
    
    def calculate_shadowing_for_position(self, x, y, device_id=""):
        """Calcule le shadowing pour une position donnée (déterministe)"""
        cache_key = f"{x:.1f}_{y:.1f}"
        
        if cache_key in self.shadowing_cache:
            return self.shadowing_cache[cache_key]
        
        # Calcul déterministe basé sur la position (similaire à channel.py)
        import hashlib
        
        # Créer une seed basée sur la position
        x_rounded = round(x, 1)
        y_rounded = round(y, 1)
        seed_str = f"shadowing_{x_rounded:.1f}_{y_rounded:.1f}_{self.initial_position_seed}"
        seed = int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % (2**31)
        
        # Utiliser RandomState pour isolation (évite d'affecter le générateur global)
        rng = np.random.RandomState(seed)
        
        # Générer une valeur de base (distribution normale standard)
        base_value = rng.normal(0, 1.0)
        
        # Ajouter une composante spatiale (dépend de la position)
        # Créer un motif sinusoïdal pour simuler la corrélation spatiale
        pos_factor = hash(f"{x_rounded:.0f}_{y_rounded:.0f}") % 1000 / 1000.0
        spatial_component = np.sin(pos_factor * 2 * np.pi) * 0.5
        
        # Combiner les deux composantes
        raw_shadowing = base_value + spatial_component
        
        # Appliquer directement l'écart-type souhaité
        # Note: base_value suit déjà N(0,1), donc multiplier par std donne N(0, std²)
        shadowing_value = raw_shadowing * self.shadowing_std_db
        
        # Limiter à ±3 sigma (99.7% des cas)
        max_value = 3 * self.shadowing_std_db
        shadowing_value = np.clip(shadowing_value, -max_value, max_value)
        
        # Mettre en cache
        self.shadowing_cache[cache_key] = shadowing_value
        
        return shadowing_value
    

    def show_shadowing_by_node(self, event=None):
        """Affiche le shadowing moyen par nœud"""
        try:
            if not self.simulation or not hasattr(self.simulation, 'simulated_packets'):
                self.add_log("Aucune donnée de simulation disponible")
                
                # Afficher un message dans le panneau de résultats
                self.results_pane.clear()
                self.results_pane.append(pn.pane.Markdown(
                    "##  Shadowing par Nœud\n\n"
                    "Aucune donnée de simulation disponible.\n"
                    "Veuillez d'abord exécuter une simulation."
                ))
                return
            
            packets = self.simulation.simulated_packets
            if not packets:
                self.add_log("Aucun paquet simulé")
                return
            
            # Collecter les données par device
            device_stats = {}
            
            for packet in packets:
                device_id = getattr(packet, 'device_id', 'unknown')
                
                if device_id not in device_stats:
                    device_stats[device_id] = {
                        'packet_count': 0,
                        'total_shadowing': 0.0,
                        'total_rssi': 0.0,
                        'total_path_loss': 0.0,
                        'success_count': 0,
                        'collision_count': 0,
                        'positions': set(),
                        'distances_km': []
                    }
                
                stats = device_stats[device_id]
                stats['packet_count'] += 1
                
                # Récupérer shadowing (déjà calculé dans le paquet)
                shadowing = getattr(packet, 'shadowing_db', 0)
                stats['total_shadowing'] += shadowing
                
                # RSSI avec shadowing
                rssi = getattr(packet, 'rssi_dbm', -120)
                stats['total_rssi'] += rssi
                
                # Path loss
                path_loss = getattr(packet, 'path_loss_db', 0)
                stats['total_path_loss'] += path_loss
                
                # Résultats
                if getattr(packet, 'success', False):
                    stats['success_count'] += 1
                if getattr(packet, 'collision', False):
                    stats['collision_count'] += 1
                
                # Position et distance
                position = getattr(packet, 'position', (0, 0))
                stats['positions'].add(tuple(position))  # Utiliser tuple pour set
                
                distance_km = getattr(packet, 'distance_km', 0)
                if distance_km > 0:
                    stats['distances_km'].append(distance_km)
            
            # Générer le rapport
            report_lines = ["##  Shadowing par Nœud\n"]
            report_lines.append("*Statistiques moyennes par device*")
            report_lines.append("")
            
            # Tableau principal
            report_lines.append("| Device | Position (x,y) m | Distance km | Paquets | Shadowing (dB) | RSSI (dBm) | PL (dB) | Taux Succès |")
            report_lines.append("|--------|------------------|-------------|---------|----------------|------------|---------|-------------|")
            
            for device_id, stats in sorted(device_stats.items(), 
                                        key=lambda x: x[0]):  # Trier par ID de device
                
                if stats['packet_count'] > 0:
                    # Calculer les moyennes
                    avg_shadowing = stats['total_shadowing'] / stats['packet_count']
                    avg_rssi = stats['total_rssi'] / stats['packet_count']
                    avg_path_loss = stats['total_path_loss'] / stats['packet_count']
                    success_rate = (stats['success_count'] / stats['packet_count'] * 100) if stats['packet_count'] > 0 else 0
                    
                    # Position (prendre la première si plusieurs)
                    position_str = "N/A"
                    if stats['positions']:
                        pos = next(iter(stats['positions']))  # Premier élément
                        position_str = f"({pos[0]:.0f}, {pos[1]:.0f})"
                    
                    # Distance moyenne
                    avg_distance = np.mean(stats['distances_km']) if stats['distances_km'] else 0
                    
                    # Ligne du tableau
                    report_lines.append(
                        f"| {device_id} | {position_str} | {avg_distance:.2f} | "
                        f"{stats['packet_count']} | {avg_shadowing:.2f} | "
                        f"{avg_rssi:.2f} | {avg_path_loss:.1f} | {success_rate:.1f}% |"
                    )
            
            # Statistiques globales
            report_lines.append("\n### 📈 Statistiques Globales du Shadowing")
            
            if device_stats:
                all_shadowings = []
                all_rssi = []
                
                for stats in device_stats.values():
                    if stats['packet_count'] > 0:
                        all_shadowings.append(stats['total_shadowing'] / stats['packet_count'])
                        all_rssi.append(stats['total_rssi'] / stats['packet_count'])
                
                if all_shadowings:
                    report_lines.append(f"- **Nombre de nœuds:** {len(device_stats)}")
                    report_lines.append(f"- **Nombre total de paquets:** {sum(s['packet_count'] for s in device_stats.values())}")
                    report_lines.append(f"- **Shadowing moyen:** {np.mean(all_shadowings):.2f} dB")
                    report_lines.append(f"- **Écart-type shadowing:** {np.std(all_shadowings):.2f} dB")
                    report_lines.append(f"- **Shadowing min:** {min(all_shadowings):.2f} dB")
                    report_lines.append(f"- **Shadowing max:** {max(all_shadowings):.2f} dB")
                    report_lines.append(f"- **RSSI moyen:** {np.mean(all_rssi):.2f} dBm")
                    
                    # Paramètres de configuration
                    config_std = self.shadowing_std_db
                    report_lines.append(f"- **σ configuré:** {config_std:.1f} dB")
                    
                    # Vérification de la distribution
                    if np.std(all_shadowings) > 0:
                        diff_ratio = abs(np.std(all_shadowings) - config_std) / config_std
                        if diff_ratio < 0.2:
                            report_lines.append(f"-  Écart-type proche de la configuration ({diff_ratio*100:.1f}% d'écart)")
                        else:
                            report_lines.append(f"- Écart-type différent de la configuration ({diff_ratio*100:.1f}% d'écart)")
            
            # Ajouter un graphique des shadowings par distance
            report_lines.append("\n###  Distribution du Shadowing")
            report_lines.append("```")
            
            # Histogramme textuel simple
            if all_shadowings and len(all_shadowings) > 1:
                hist, bins = np.histogram(all_shadowings, bins=10)
                bin_centers = (bins[:-1] + bins[1:]) / 2
                
                for i, (center, count) in enumerate(zip(bin_centers, hist)):
                    bar = '█' * int(count / max(hist) * 20) if max(hist) > 0 else ''
                    report_lines.append(f"{center:6.1f} dB: {bar} ({count})")
            
            report_lines.append("```")
            
            # Afficher le rapport
            report_md = "\n".join(report_lines)
            
            self.results_pane.clear()
            self.results_pane.append(pn.pane.Markdown(report_md))
            
            self.add_log(" Shadowing par nœud affiché")
            pn.state.notifications.info(
                f"Shadowing par nœud - {len(device_stats)} devices analysés", 
                duration=5000
            )
            
        except Exception as e:
            logger.error(f"Erreur affichage shadowing par nœud: {e}")
            self.add_log(f" Erreur affichage shadowing: {str(e)}")
            
            # Afficher l'erreur
            self.results_pane.clear()
            self.results_pane.append(pn.pane.Markdown(
                f"##  Erreur d'affichage\n\n"
                f"```\n{str(e)}\n```"
            ))

    def calculate_shadowing_grid(self):
        """Calcule une grille de valeurs de shadowing pour la visualisation"""
        if not self.device_positions:
            return None
        
        # Créer une grille de points pour visualisation
        positions = np.array(self.device_positions)
        x_coords = positions[:, 0]
        y_coords = positions[:, 1]
        
        # Déterminer les limites
        max_distance = max(np.sqrt(x_coords**2 + y_coords**2)) if len(x_coords) > 0 else self.distance_gtw
        margin = max_distance * 0.1
        
        # Créer une grille régulière
        grid_size = 100  # Résolution de la grille
        x_grid = np.linspace(-max_distance - margin, max_distance + margin, grid_size)
        y_grid = np.linspace(-max_distance - margin, max_distance + margin, grid_size)
        
        # Calculer le shadowing pour chaque point de la grille
        shadowing_grid = np.zeros((len(y_grid), len(x_grid)))
        
        for i, y in enumerate(y_grid):
            for j, x in enumerate(x_grid):
                shadowing_grid[i, j] = self.calculate_shadowing_for_position(x, y)
        
        return {
            'x_grid': x_grid,
            'y_grid': y_grid,
            'shadowing': shadowing_grid,
            'x_coords': x_coords,
            'y_coords': y_coords
        }
    
    def show_energy_report(self, event=None):
        """Affiche le rapport énergétique complet"""
        try:
            if self.simulation and hasattr(self.simulation, 'energy_analyzer'):
                report = self.simulation.energy_analyzer.get_energy_report()
                
                # Afficher dans le panneau de résultats
                self.results_pane.clear()
                self.results_pane.append(pn.pane.Markdown(report))
                
                # Afficher une notification
                pn.state.notifications.success(
                    " Rapport énergétique généré", 
                    duration=3000
                )
                
            else:
                self.add_log("Simulation non disponible ou analyseur d'énergie non initialisé")
                
        except Exception as e:
            logger.error(f"Erreur génération rapport énergie: {e}")
            self.add_log(f" Erreur rapport énergie: {str(e)}")
    
    def reset_energy_stats(self, event=None):
        """Réinitialise les statistiques énergétiques"""
        try:
            if self.simulation and hasattr(self.simulation, 'energy_analyzer'):
                self.simulation.energy_analyzer.reset_stats()
                self.add_log(" Statistiques énergétiques réinitialisées")
                
                # Mettre à jour l'interface
                if hasattr(self, 'rt_total_energy'):
                    self.rt_total_energy.value = 0.0
                    self.rt_avg_energy_per_packet.value = 0.0
                    self.rt_battery_life.value = 0.0
                    self.rt_energy_per_bit.value = 0.0
                    self.rt_current_tx.value = 0.0
                
                pn.state.notifications.info(
                    "Statistiques énergétiques réinitialisées", 
                    duration=3000
                )
            else:
                self.add_log("Simulation non disponible")
                
        except Exception as e:
            logger.error(f"Erreur réinitialisation énergie: {e}")
            self.add_log(f" Erreur réinit énergie: {str(e)}")
    
    def create_interface(self):
        """Crée l'interface utilisateur"""
        self.create_widgets()
        self.create_control_pane()
        self.create_metrics_panel()
        self.create_log_panel()
        self.create_deployment_map_panel()
    
    def create_widgets(self):
        """Crée tous les widgets"""
        # Paramètres de base
        basic_panel = pn.Column(
            pn.pane.Markdown("**Paramètres de base**"),
            pn.Param(self.param.simulation_duration, widgets={'simulation_duration': pn.widgets.IntInput}),
            pn.Param(self.param.num_devices, widgets={'num_devices': pn.widgets.IntInput}),
            pn.Param(self.param.distance_gtw, widgets={'distance_gtw': pn.widgets.IntInput}),
            pn.Param(self.param.tx_power, widgets={'tx_power': pn.widgets.FloatInput}),
            pn.Spacer(height=6)
        )

        # Paramètres LR-FHSS
        lrfhss_panel = pn.Column(
            pn.pane.Markdown("**LR-FHSS**"),
            pn.Param(self.param.region, widgets={'region': pn.widgets.Select}),
            pn.Param(self.param.data_rate, widgets={'data_rate': pn.widgets.Select}),
            pn.Param(self.param.coding_rate, widgets={'coding_rate': {'type': pn.widgets.Select, 'disabled': True}}),
            pn.Param(self.param.bandwidth_khz, widgets={'bandwidth_khz': {'type': pn.widgets.Select, 'disabled': True}}),
            pn.Spacer(height=6)
        )

        # Transmission
        tx_panel = pn.Column(
            pn.pane.Markdown("**Transmission**"),
            pn.Param(self.param.payload_min, widgets={'payload_min': pn.widgets.IntInput}),
            pn.Param(self.param.payload_max, widgets={'payload_max': pn.widgets.IntInput}),
            pn.Param(self.param.tx_interval_min, widgets={'tx_interval_min': pn.widgets.FloatInput}),
            pn.Param(self.param.tx_interval_max, widgets={'tx_interval_max': pn.widgets.FloatInput}),
            pn.Spacer(height=6),
            pn.pane.Markdown("**Optimisations**"),
        )

        # Canal RF
        rf_panel = pn.Column(
            pn.pane.Markdown("**Canal RF**"),
            pn.Param(self.param.shadowing_std_db, widgets={'shadowing_std_db': pn.widgets.FloatInput}),
            pn.Param(self.param.path_loss_exponent, widgets={'path_loss_exponent': pn.widgets.FloatInput}),
            pn.Param(self.param.doppler_hz, widgets={'doppler_hz': pn.widgets.FloatInput}),
            pn.Param(self.param.multipath_enabled, widgets={'multipath_enabled': pn.widgets.Checkbox}),
        )

        # Réception
        rx_panel = pn.Column(
            pn.pane.Markdown("**Réception**"),
            pn.Param(self.param.noise_figure_db, widgets={'noise_figure_db': pn.widgets.FloatInput}),
        )
        
        # Panel Énergie
        if ENERGY_MODULE_AVAILABLE:
            energy_panel = pn.Column(
                pn.pane.Markdown("** Énergie & Batterie**"),
                pn.Param(self.param.pa_type, widgets={'pa_type': pn.widgets.Select}),
                pn.Param(self.param.battery_capacity_mah, 
                        widgets={'battery_capacity_mah': pn.widgets.FloatInput}),
                pn.Param(self.param.transmissions_per_day, 
                        widgets={'transmissions_per_day': pn.widgets.IntInput}),
                pn.Spacer(height=5),
                pn.pane.Markdown("*SX1261_LP: ≤14 dBm, SX1262_HP: ≤22 dBm*", 
                               styles={'font-size': '10px', 'color': '#666'}),
            )
        else:
            energy_panel = pn.Column(
                pn.pane.Markdown("** Énergie & Batterie**", styles={'color': '#666'}),
                pn.pane.Markdown("Module énergie non disponible", 
                               styles={'color': 'orange', 'font-size': '12px'}),
                pn.Spacer(height=5),
            )
        
        # Paramètres Scheduler Intelligent
        scheduler_panel = pn.Column(
            pn.pane.Markdown("** Scheduler Intelligent**"),
            pn.Param(self.param.enable_intelligent_scheduler, widgets={'enable_intelligent_scheduler': pn.widgets.Checkbox}),
            pn.Param(self.param.scheduler_max_delay, widgets={'scheduler_max_delay': pn.widgets.FloatInput}),
            pn.Param(self.param.scheduler_allow_freq_shift, widgets={'scheduler_allow_freq_shift': pn.widgets.Checkbox}),
            pn.Param(self.param.scheduler_allow_power_boost, widgets={'scheduler_allow_power_boost': pn.widgets.Checkbox}),
            pn.Spacer(height=5),
            pn.pane.Markdown("*Analyse fragment par fragment pour éviter les collisions*", 
                        styles={'font-size': '10px', 'color': '#666'}),
        )
        
        # Paramètres DQN
        dqn_panel = pn.Column(
            pn.pane.Markdown("** DQN**"),
            pn.Param(self.param.enable_dqn, widgets={'enable_dqn': pn.widgets.Checkbox}),
            pn.Param(self.param.dqn_model_name, widgets={
                'dqn_model_name': {'type': pn.widgets.Select, 'name': 'Modèle DQN'}
            }),
            pn.Param(self.param.dqn_exploration, widgets={'dqn_exploration': pn.widgets.FloatInput}),
            pn.Spacer(height=5),
            pn.pane.Markdown("**Optimisations DQN:**", margin=(0, 0, 5, 0), styles={'font-size': '11px'}),
            pn.Param(self.param.use_dqn_for_dr, widgets={'use_dqn_for_dr': pn.widgets.Checkbox}),
            pn.Param(self.param.use_dqn_for_power, widgets={'use_dqn_for_power': pn.widgets.Checkbox}),
            # pn.Param(self.param.use_dqn_for_frequency, widgets={'use_dqn_for_frequency': pn.widgets.Checkbox}),
            pn.Spacer(height=5),
        )
        
        # Nouvelle section pour la carte de shadowing
        shadowing_panel = pn.Column(
            pn.pane.Markdown("** Carte de Shadowing**"),
            pn.Param(self.param.show_shadowing_map, widgets={'show_shadowing_map': pn.widgets.Checkbox}),
            pn.Spacer(height=5),
            pn.pane.Markdown("*Affiche la carte de shadowing déterministe basée sur la position*", 
                           styles={'font-size': '10px', 'color': '#666'}),
        )
        
        # Créer les onglets
        tabs_list = [
            ('Basics', basic_panel),
            ('LR-FHSS', lrfhss_panel),
            ('Tx', tx_panel),
            ('RF', rf_panel),
            ('Rx', rx_panel),
            ('Énergie', energy_panel),
            ('DQN', dqn_panel),
        ]
        
        self.param_column = pn.Card(
            pn.Tabs(*tabs_list, active=0, sizing_mode='stretch_width'),
            title="Paramètres",
            width=400,
        )
    
    def create_control_pane(self):
        """Crée le panneau de contrôle"""
        self.start_btn = pn.widgets.Button(
            name="▶ Démarrer Simulation",
            button_type="primary",
            width=300
        )
        self.stop_btn = pn.widgets.Button(
            name="⏹ Arrêter",
            button_type="danger",
            width=300,
            disabled=True
        )
        self.reset_btn = pn.widgets.Button(
            name=" Réinitialiser",
            button_type="warning",
            width=300
        )
        self.export_btn = pn.widgets.Button(
            name=" Exporter Rapport",
            button_type="success",
            width=300,
            disabled=False
        )
        self.regenerate_positions_btn = pn.widgets.Button(
            name=" Regénérer positions",
            button_type="default",
            width=300
        )
        self.save_dqn_btn = pn.widgets.Button(
            name=" Sauvegarder DQN",
            button_type="success",
            width=300,
            disabled=False
        )
        
        # Boutons Énergie
        self.energy_report_btn = pn.widgets.Button(
            name="⚡ Rapport Énergie",
            button_type="primary",
            width=300
        )
        self.energy_report_btn.on_click(self.show_energy_report)
        
        self.reset_energy_btn = pn.widgets.Button(
            name=" Réinit. Énergie",
            button_type="warning",
            width=300
        )
        self.reset_energy_btn.on_click(self.reset_energy_stats)
        
        # Bouton pour afficher les détails DQN
        self.show_dqn_btn = pn.widgets.Button(
            name='Afficher détails DQN',
            button_type='primary',
            width=300
        )
        self.show_dqn_btn.on_click(self.show_dqn_details)
        
        # NOUVEAU BOUTON POUR CARTE DE SHADOWING
        self.toggle_shadowing_btn = pn.widgets.Toggle(
            name=' Carte Shadowing',
            value=self.show_shadowing_map,
            button_type='primary',
            width=300
        )
        self.toggle_shadowing_btn.param.watch(self._on_toggle_shadowing, 'value')
        
        # Configurer les événements
        self.start_btn.on_click(self.start_simulation)
        self.stop_btn.on_click(self.stop_simulation)
        self.reset_btn.on_click(self.reset_simulation)
        self.export_btn.on_click(self.export_report)
        self.regenerate_positions_btn.on_click(self.regenerate_positions)
        self.save_dqn_btn.on_click(self.save_dqn_model)
        
        self.progress_bar = pn.indicators.Progress(
            name="Progression",
            value=self.progress,
            width=320,
            bar_color='primary'
        )
        self.shadowing_by_node_btn = pn.widgets.Button(
            name=" Shadowing par Nœud",
            button_type="primary",
            width=300
        )
        self.shadowing_by_node_btn.on_click(self.show_shadowing_by_node)
        
        # AJOUTER: Bouton pour exporter CSV enrichi
        self.export_enriched_csv_btn = pn.widgets.Button(
            name=" Exporter CSV Enrichi",
            button_type="success",
            width=300
        )
        self.export_enriched_csv_btn.on_click(self.export_enriched_csv)
        
        buttons_column = pn.Column(
            self.start_btn,
            pn.Spacer(height=8),
            self.stop_btn,
            pn.Spacer(height=8),
            self.reset_btn,
            pn.Spacer(height=8),
            self.export_btn,
            pn.Spacer(height=8),
            self.save_dqn_btn,
            pn.Spacer(height=8),
            self.regenerate_positions_btn,
            pn.Spacer(height=8),
            self.show_dqn_btn,
            pn.Spacer(height=8),
            self.toggle_shadowing_btn,  # Nouveau bouton
            pn.Spacer(height=8),
            self.energy_report_btn,
            pn.Spacer(height=8),
            self.reset_energy_btn,
            self.shadowing_by_node_btn,
            pn.Spacer(height=8),
            self.export_enriched_csv_btn,
            align='center',
            sizing_mode='stretch_width'
        )

        self.control_pane = pn.Card(
            pn.Column(
                pn.pane.Markdown("**Contrôles**", margin=(0, 0, 8, 0)),
                buttons_column,
                pn.Spacer(height=10),
                self.progress_bar,
                sizing_mode='stretch_width'
            ),
            width=380,
            styles={'border': '1px solid #ddd', 'border-radius': '6px', 'padding': '12px', 'background': '#fafafa'}
        )
    
    def _on_toggle_shadowing(self, event):
        """Gère le toggle de la carte de shadowing"""
        self.show_shadowing_map = event.new
        self.update_deployment_map()
        
    def save_dqn_model(self, event=None):
        """Sauvegarde le modèle DQN"""
        try:
            if self.simulation and self.enable_dqn:
                saved_path = self.simulation._save_dqn_model(suffix="_manual")
                if saved_path:
                    self.add_log(f" Modèle DQN sauvegardé: {saved_path}")
                    pn.state.notifications.success(f" Modèle DQN sauvegardé !", duration=5000)
                else:
                    self.add_log("Échec sauvegarde DQN")
            else:
                self.add_log("DQN non activé ou simulation non initialisée")
        except Exception as e:
            logger.error(f"Erreur sauvegarde DQN: {e}")
            self.add_log(f"Erreur sauvegarde DQN: {str(e)}")
    
    def _show_csv_preview(self, csv_filename, n_rows=20):
        """Affiche un aperçu du CSV exporté"""
        try:
            import pandas as pd
            
            # Lire les premières lignes
            df = pd.read_csv(csv_filename)
            
            # Générer l'aperçu
            preview_lines = [f"##  Aperçu du CSV : {csv_filename}\n"]
            preview_lines.append(f"**Shape:** {df.shape[0]} lignes × {df.shape[1]} colonnes\n")
            
            # Aperçu des données
            preview_lines.append("### Premières lignes :")
            preview_lines.append("```")
            
            # Afficher les colonnes importantes
            important_cols = [
                'device_id', 'position_x_m', 'position_y_m', 'distance_km',
                'tx_power_dbm', 'path_loss_db', 'shadowing_db', 'rssi_with_shadowing_dbm',
                'success', 'collision'
            ]
            
            # Filtrer les colonnes existantes
            existing_cols = [col for col in important_cols if col in df.columns]
            
            if existing_cols:
                preview_df = df[existing_cols].head(n_rows)
                preview_lines.append(preview_df.to_string())
            else:
                preview_lines.append(df.head(n_rows).to_string())
            
            preview_lines.append("```")
            
            # Statistiques rapides
            preview_lines.append("\n###  Statistiques rapides :")
            
            if 'shadowing_db' in df.columns:
                preview_lines.append(f"- **Shadowing moyen:** {df['shadowing_db'].mean():.2f} dB")
                preview_lines.append(f"- **Écart-type shadowing:** {df['shadowing_db'].std():.2f} dB")
                preview_lines.append(f"- **Shadowing min/max:** [{df['shadowing_db'].min():.2f}, {df['shadowing_db'].max():.2f}] dB")
            
            if 'rssi_with_shadowing_dbm' in df.columns:
                preview_lines.append(f"- **RSSI moyen:** {df['rssi_with_shadowing_dbm'].mean():.2f} dBm")
            
            if 'success' in df.columns:
                success_rate = df['success'].mean() * 100
                preview_lines.append(f"- **Taux de succès:** {success_rate:.1f}%")
            
            preview_md = "\n".join(preview_lines)
            
            # Afficher dans le panneau de résultats
            self.results_pane.clear()
            self.results_pane.append(pn.pane.Markdown(preview_md))
            
            self.add_log(f" Aperçu CSV affiché : {csv_filename}")
            
        except Exception as e:
            logger.error(f"Erreur affichage aperçu CSV: {e}")
            self.add_log(f" Erreur aperçu CSV: {str(e)}")


    def export_enriched_csv(self, event=None):
        """Exporte un CSV enrichi directement depuis le dashboard"""
        try:
            if not self.simulation or not hasattr(self.simulation, 'simulated_packets'):
                self.add_log("Aucune donnée de simulation à exporter")
                pn.state.notifications.warning(
                    "Aucune donnée de simulation disponible", 
                    duration=3000
                )
                return
            
            # Utiliser la méthode d'export enrichi de la simulation
            import time as time_module
            timestamp = time_module.strftime("%Y%m%d_%H%M%S")
            
            csv_filename = self.simulation._export_enriched_csv(timestamp)
            
            if csv_filename:
                success_msg = (
                    f" CSV enrichi exporté avec succès !\n"
                    f"Fichier : {csv_filename}\n"
                    f"Contient :\n"
                    f"• Shadowing par paquet\n"
                    f"• Positions des nœuds\n"
                    f"• RSSI avec shadowing\n"
                    f"• Path loss détaillé"
                )
                
                pn.state.notifications.success(success_msg, duration=10000)
                
                # Afficher un aperçu dans le panneau de résultats
                self._show_csv_preview(csv_filename)
            else:
                self.add_log("Échec de l'export CSV")
                pn.state.notifications.error(
                    "Échec de l'export CSV", 
                    duration=5000
                )
                
        except Exception as e:
            logger.error(f"Erreur export CSV enrichi: {e}")
            self.add_log(f" Erreur export CSV: {str(e)}")
            pn.state.notifications.error(
                f"Erreur : {str(e)[:100]}...", 
                duration=5000
            )
    def show_dqn_details(self, event=None):
        """Affiche les détails des décisions DQN"""
        try:
            if not self.simulation:
                self.add_log("Simulation non initialisée")
                return
            
            hist = getattr(self.simulation, 'dqn_decision_history', []) or []
            hist = hist[-500:]
            metrics = self.latest_metrics or {}

            total = metrics.get('dqn_decisions', len(hist))
            avg_dist = metrics.get('dqn_avg_distance_km', 0.0)
            avg_power = metrics.get('dqn_avg_power_dbm', 0.0)
            
            # Récupérer les vraies distances depuis devices_state
            real_distances = {}
            if hasattr(self.simulation, 'devices_state'):
                for device_id, state in self.simulation.devices_state.items():
                    if 'distance_km' in state:
                        real_distances[device_id] = state['distance_km']
            
            md_lines = ["##  DQN - Détails des décisions\n"]
            md_lines.append(f"- **Décisions affichées:** {len(hist)}")
            md_lines.append(f"- **Total décisions:** {total}")
            md_lines.append(f"- **Distance moyenne (km):** {avg_dist:.3f}")
            md_lines.append(f"- **Puissance moyenne (dBm):** {avg_power:.2f}")
            
            # Afficher les distances réelles pour quelques devices
            md_lines.append("\n### Distances réelles des devices (km):")
            for i, (device_id, dist_km) in enumerate(list(real_distances.items())[:10]):  # Premier 10
                md_lines.append(f"- {device_id}: {dist_km:.3f} km")
            if len(real_distances) > 10:
                md_lines.append(f"- ... et {len(real_distances) - 10} autres devices")
            
            # Table des dernières décisions
            md_lines.append("\n### Dernières décisions DQN")
            md_lines.append("| time | device | dist_km | dr | power_dBm | freq_MHz | payload |")
            md_lines.append("|---:|:---:|---:|---:|---:|---:|---:|")
            for entry in hist[-50:]:
                t = entry.get('time', 0.0)
                dev = entry.get('device_id', '')
                dist = entry.get('distance_km', 0.0)
                dr = entry.get('dqn_dr', '')
                pw = entry.get('dqn_power_dbm', 0.0)
                fq = entry.get('frequency_mhz', 0.0)
                pl = entry.get('payload_bytes', 0)
                md_lines.append(f"| {t:.2f} | {dev} | {dist:.3f} | {dr} | {pw:.1f} | {fq:.3f} | {pl} |")

            md = "\n".join(md_lines)

            # Afficher dans le panneau de résultats
            try:
                self.results_pane.clear()
                self.results_pane.append(pn.pane.Markdown(md))
                pn.state.notifications.info("Détails DQN affichés", duration=3000)
            except Exception:
                # Fallback: remplacer l'objet complet
                self.results_pane.objects = [pn.pane.Markdown(md)]

        except Exception as e:
            logger.error(f"Erreur show_dqn_details: {e}")
            self.add_log(f"Erreur affichage détails DQN: {str(e)}")
    
    def create_metrics_panel(self):
        """Crée le panneau de métriques"""
        # Indicateurs principaux
        self.rt_total_pkts = pn.indicators.Number(name='Transmissions', value=0, format='{value:,.0f}', 
                                                title_size='16px', font_size='20px', styles={'font-size': '10px'})
        self.rt_success = pn.indicators.Number(name='Paquets Transmis', value=0, format='{value:,.0f}', 
                                            title_size='16px', font_size='20px', styles={'font-size': '10px'})
        self.rt_collisions = pn.indicators.Number(name='Collisions', value=0, format='{value:,.0f}', 
                                                title_size='16px', font_size='20px', styles={'font-size': '10px'})
        self.rt_success_rate_pct = pn.indicators.Number(name='Taux Transmission', value=0.0, format='{value:.1f}%', 
                                                    title_size='20px', font_size='20px', styles={'font-size': '10px'})
        self.rt_total_echecs = pn.indicators.Number(name='Échecs Total', value=0, format='{value:,.0f}', 
                                                title_size='16px', font_size='20px', styles={'font-size': '10px'})
        self.rt_simulated_time = pn.indicators.Number(name='Temps simulé', value=0.0, format='{value:.1f}s', 
                                                    title_size='16px', font_size='20px', styles={'font-size': '10px'})
        
        # WIDGETS POUR TOA
        self.rt_toa_brut = pn.indicators.Number(name='ToA Brut', value=0.0, format='{value:.2f}s', 
                                            title_size='16px', font_size='20px', styles={'font-size': '10px'})
        self.rt_toa = pn.indicators.Number(name='ToA Net', value=0.0, format='{value:.2f}s', 
                                        title_size='16px', font_size='20px', styles={'font-size': '10px'})
        
        # Indicateurs scheduler
        self.rt_scheduler_delays = pn.indicators.Number(name='Délais appliqués', value=0, format='{value:,.0f}', 
                                                    title_size='16px', font_size='20px', styles={'font-size': '10px'})
        self.rt_scheduler_delays_sum = pn.indicators.Number(name='Somme des délais (s)', value=0.0, format='{value:.2f}', 
                                                    title_size='16px', font_size='20px', styles={'font-size': '10px'})
        
        # Indicateurs Énergie
        if ENERGY_MODULE_AVAILABLE:
            self.rt_total_energy = pn.indicators.Number(
                name='Énergie totale', 
                value=0.0, 
                format='{value:.3f} J', 
                title_size='16px', 
                font_size='20px', 
                styles={'font-size': '10px'}
            )
            
            energy_section = pn.Column(
                pn.Spacer(height=5),
                pn.pane.Markdown("**Énergie**", margin=(0, 0, 5, 0), styles={'font-size': '11px'}),
                self.rt_total_energy,
            )
        else:
            energy_section = pn.Column(
                pn.Spacer(height=5),
                pn.pane.Markdown("**Énergie**", margin=(0, 0, 5, 0), 
                            styles={'font-size': '11px', 'color': '#666'}),
                pn.pane.Markdown("Module non disponible", 
                            styles={'font-size': '9px', 'color': 'orange'}),
            )
        
        # Organisation des widgets
        metrics_column = pn.Column(
            # Section Temps
            pn.Spacer(height=5),
            pn.pane.Markdown("**Temps**", margin=(0, 0, 5, 0), styles={'font-size': '11px'}),
            self.rt_simulated_time,
            self.rt_toa_brut,
            self.rt_toa,

            # Section Transmissions
            pn.Spacer(height=5),
            pn.pane.Markdown("**📡 Transmissions**", margin=(0, 0, 5, 0), styles={'font-size': '11px'}),
            self.rt_total_pkts,
            self.rt_success,
            self.rt_collisions,
            self.rt_success_rate_pct,
            self.rt_total_echecs,
            
            # Section Scheduler
            pn.Spacer(height=5),
            pn.pane.Markdown("**Scheduler**", margin=(0, 0, 5, 0), styles={'font-size': '11px'}),
            self.rt_scheduler_delays,
            self.rt_scheduler_delays_sum,
            
            # Section Énergie
            energy_section,
            
            sizing_mode='stretch_width'
        )

        self.metrics_panel = pn.Column(
            pn.pane.Markdown("### Métriques Temps Réel", margin=(0, 0, 5, 0), styles={'font-size': '11px'}),
            metrics_column,
            sizing_mode='stretch_width',
            styles={'border': '1px solid #ddd', 'border-radius': '5px', 'padding': '8px 10px'}
        )
    
    def update_ui(self):
        """Met à jour l'interface utilisateur"""
        try:
            metrics = self.latest_metrics
            
            if not metrics:
                return  # Pas de métriques disponibles
            
            if not hasattr(self, 'metrics_panel'):
                return
            
            try:
                # 1. MÉTRIQUES GÉNÉRALES
                self.rt_total_pkts.value = metrics.get('total_sent', 0)
                self.rt_success.value = metrics.get('successful_rx', 0)
                self.rt_collisions.value = metrics.get('collisions', 0)
                self.rt_success_rate_pct.value = metrics.get('success_rate', 0.0)
                self.rt_total_echecs.value = metrics.get('failed_packets', 0)
                self.rt_simulated_time.value = metrics.get('simulated_time', 0.0)
                
                # 2. MÉTRIQUES TOA
                if hasattr(self, 'rt_toa_brut'):
                    self.rt_toa_brut.value = metrics.get('toa_brut_total', 0.0)
                
                if hasattr(self, 'rt_toa'):
                    self.rt_toa.value = metrics.get('toa_net_total', 0.0)
                
                # 3. MÉTRIQUES SCHEDULER
                if hasattr(self, 'rt_scheduler_delays'):
                    self.rt_scheduler_delays.value = metrics.get('scheduler_delays', 0)
                if hasattr(self, 'rt_scheduler_delays_sum'):
                    self.rt_scheduler_delays_sum.value = metrics.get('scheduler_delays_sum', 0.0)
                
                # 4. MÉTRIQUES ÉNERGIE
                if ENERGY_MODULE_AVAILABLE:
                    if hasattr(self, 'rt_total_energy'):
                        self.rt_total_energy.value = metrics.get('total_energy_j', 0.0)
                
            except Exception as e:
                logger.error(f"Erreur dans update_ui widgets: {e}")

            # 8. Mettre à jour logs
            if self.log_buffer and hasattr(self, 'log_panel'):
                try:
                    new_logs = "\n".join(self.log_buffer[-30:])
                    self.log_panel.value = new_logs
                except Exception as e:
                    logger.debug(f"Erreur update logs: {e}")
            
            # 9. Mettre à jour barre de progression
            if hasattr(self, 'progress_bar'):
                self.progress_bar.value = int(metrics.get('progress', 0))
                    
        except Exception as e:
            logger.error(f"Erreur dans update_ui: {e}")
    
    def create_log_panel(self):
        """Crée le panneau de logs"""
        self.log_panel = pn.widgets.TextAreaInput(
            name="Logs",
            value="Simulateur LR-FHSS avec DQN, scheduler intelligent et énergie initialisé\n",
            height=600,
            rows=3,
            disabled=True,
            sizing_mode='stretch_width'
        )
        
        self.results_pane = pn.Column(
            pn.pane.Markdown("## Résultats détaillés", margin=(0, 0, 10, 0)),
            pn.pane.Str(self.param.results_text),
            sizing_mode='stretch_width',
            styles={'border': '1px solid #ddd', 'border-radius': '5px', 'padding': '15px', 'background': '#ffffff'}
        )
    
    def create_deployment_map_panel(self):
        """Crée le panneau avec la carte de déploiement"""
        try:
            map_fig = self.create_deployment_map()
            self.deployment_map = pn.pane.Matplotlib(
                map_fig,
                sizing_mode='scale_width',
                height=900
            )
            
            self.deployment_panel = pn.Card(
                self.deployment_map,
                title="Carte de Déploiement des Nœuds",
                sizing_mode='stretch_width',
                styles={'border': '1px solid #ddd', 'border-radius': '5px', 'padding': '15px', 'background': '#ffffff'}
            )
        except Exception as e:
            logger.error(f"Erreur création panneau carte: {e}")
            self.deployment_panel = pn.Card(
                pn.pane.Markdown(f"Erreur: {str(e)}"),
                title="Carte de Déploiement",
                sizing_mode='stretch_width'
            )
    
    def create_deployment_map(self):
        """Crée la carte de déploiement des nœuds"""
        try:
            fig = Figure(figsize=(10, 10))
            ax = fig.add_subplot(111)
            
            if not self.device_positions or len(self.device_positions) == 0:
                ax.text(0.5, 0.5, 'Pas de nœuds à afficher\nCliquez sur "Générer positions"', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_xlim(-1000, 1000)
                ax.set_ylim(-1000, 1000)
                ax.set_aspect('equal')
                ax.set_title('Carte de déploiement (vide)', fontsize=12)
                fig.tight_layout()
                return fig
            
            positions = np.array(self.device_positions)
            x_coords = positions[:, 0]
            y_coords = positions[:, 1]
            
            distances = np.sqrt(x_coords**2 + y_coords**2)
            
            if self.show_shadowing_map:
                # ==============================================
                # AFFICHAGE DE LA CARTE DE SHADOWING
                # ==============================================
                # Calculer la grille de shadowing
                shadowing_data = self.calculate_shadowing_grid()
                
                if shadowing_data is not None:
                    # Afficher la carte de shadowing en arrière-plan
                    X, Y = np.meshgrid(shadowing_data['x_grid'], shadowing_data['y_grid'])
                    shadowing_values = shadowing_data['shadowing']
                    
                    # Utiliser une colormap divergente pour shadowing (négatif vs positif)
                    cmap = plt.cm.RdBu_r
                    
                    # Afficher la carte de shadowing
                    contourf = ax.contourf(X, Y, shadowing_values, 
                                          cmap=cmap, 
                                          alpha=0.6,
                                          levels=20)
                    
                    # Ajouter une colorbar
                    cbar = fig.colorbar(contourf, ax=ax, shrink=0.8)
                    cbar.set_label(f'Shadowing (dB) - σ={self.shadowing_std_db:.1f} dB', fontsize=10)
                    
                    # Ajouter les contours
                    contour = ax.contour(X, Y, shadowing_values, 
                                        colors='black', 
                                        linewidths=0.5, 
                                        alpha=0.3,
                                        levels=10)
                    
                    ax.clabel(contour, inline=True, fontsize=8, fmt='%.1f dB')
                
                # Afficher les nœuds avec couleur selon shadowing
                node_shadowings = []
                for x, y in self.device_positions:
                    shadowing = self.calculate_shadowing_for_position(x, y)
                    node_shadowings.append(shadowing)
                
                node_shadowings = np.array(node_shadowings)
                
                # Taille des points selon shadowing
                point_size = 50 + np.abs(node_shadowings) * 2
                
                # Scatter plot avec couleur selon shadowing
                scatter = ax.scatter(x_coords, y_coords, 
                                   c=node_shadowings, 
                                   s=point_size,
                                   cmap=cmap,
                                   edgecolors='black', 
                                   linewidth=1.0,
                                   alpha=0.8,
                                   vmin=-3*self.shadowing_std_db, 
                                   vmax=3*self.shadowing_std_db)
                
                # Ajouter texte pour chaque nœud (shadowing)
                for i, (x, y) in enumerate(self.device_positions):
                    if len(self.device_positions) < 50:  # Limiter pour lisibilité
                        shadowing_val = node_shadowings[i]
                        ax.text(x, y, f'{shadowing_val:.1f}', 
                               fontsize=7, ha='center', va='center',
                               bbox=dict(boxstyle='round,pad=0.2', 
                                       facecolor='white', 
                                       alpha=0.7, 
                                       edgecolor='none'))
                
                title_suffix = " - Carte de Shadowing"
                
            else:
                # ==============================================
                # AFFICHAGE NORMAL (SANS SHADOWING)
                # ==============================================
                if len(distances) > 0 and distances.max() > distances.min():
                    norm_distances = (distances - distances.min()) / (distances.max() - distances.min())
                else:
                    norm_distances = np.zeros_like(distances)
                
                if len(x_coords) < 100:
                    point_size = 50
                elif len(x_coords) < 1000:
                    point_size = 20
                else:
                    point_size = 10
                
                scatter = ax.scatter(x_coords, y_coords, c=norm_distances, s=point_size, 
                                   cmap='viridis', alpha=0.7, edgecolors='black', linewidth=0.5)
                
                title_suffix = ""
                
                # Ajouter une colorbar pour la distance
                if len(distances) > 0:
                    cbar = fig.colorbar(scatter, ax=ax, shrink=0.8)
                    cbar.set_label('Distance relative à la passerelle', fontsize=9)
            
            # Afficher la passerelle (toujours visible)
            ax.plot(0, 0, marker='*', markersize=35, color='red', label='Passerelle (Gateway)', 
                   markeredgecolor='darkred', markeredgewidth=2.5)
            
            max_distance = max(distances) if len(distances) > 0 else self.distance_gtw
            distances_to_show = np.linspace(0, max_distance, 5)[1:]
            
            for i, dist in enumerate(distances_to_show):
                color = plt.cm.Blues(0.3 + 0.2 * i)
                circle = patches.Circle((0, 0), dist, fill=False, 
                                       edgecolor=color, linestyle='--', alpha=0.5, linewidth=1.2)
                ax.add_patch(circle)
                
                angle = np.pi/4
                x_text = dist * np.cos(angle)
                y_text = dist * np.sin(angle)
                ax.text(x_text, y_text, f'{dist/1000:.1f} km', 
                       fontsize=8, color=color, ha='center')
            
            margin = max_distance * 0.1
            ax.set_xlim(-max_distance - margin, max_distance + margin)
            ax.set_ylim(-max_distance - margin, max_distance + margin)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_xlabel('Distance Est-Ouest (m)', fontsize=10)
            ax.set_ylabel('Distance Nord-Sud (m)', fontsize=10)
            
            title = f'Carte de déploiement - {len(self.device_positions)} nœuds{title_suffix}'
            if self.show_shadowing_map:
                title += f'\n(Shadowing σ={self.shadowing_std_db:.1f} dB)'
            
            ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
            
            stats_text = f'Nombre de nœuds: {len(self.device_positions)}\n'
            if len(distances) > 0:
                stats_text += f'Distance max: {distances.max():.0f} m\n'
                stats_text += f'Distance moyenne: {distances.mean():.0f} m\n'
                stats_text += f'Distance min: {distances.min():.0f} m'
            
            if self.show_shadowing_map and len(node_shadowings) > 0:
                stats_text += f'\nShadowing moyen: {node_shadowings.mean():.1f} dB\n'
                stats_text += f'Shadowing min: {node_shadowings.min():.1f} dB\n'
                stats_text += f'Shadowing max: {node_shadowings.max():.1f} dB'
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   fontsize=8, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='orange'))
            
            ax.legend(loc='upper right', fontsize=9)
            
            fig.tight_layout()
            return fig
            
        except Exception as e:
            logger.error(f"Erreur création carte de déploiement: {e}")
            self.add_log(f"Erreur création carte: {str(e)}")
            
            fig = Figure(figsize=(8, 8))
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, f'Erreur création carte:\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10, color='red')
            ax.set_xlim(-1000, 1000)
            ax.set_ylim(-1000, 1000)
            ax.set_title('Erreur', fontsize=12, color='red')
            fig.tight_layout()
            return fig
    
    def update_deployment_map(self):
        """Met à jour la carte de déploiement"""
        try:
            if hasattr(self, 'deployment_map'):
                map_fig = self.create_deployment_map()
                self.deployment_map.object = map_fig
                self.deployment_map.param.trigger('object')
                
                if self.show_shadowing_map:
                    self.add_log(" Carte de shadowing mise à jour")
                else:
                    self.add_log("Carte de déploiement mise à jour")
                    
        except Exception as e:
            logger.error(f"Erreur mise à jour carte: {e}")
            self.add_log(f"Erreur mise à jour carte: {str(e)}")
    
    def generate_device_positions(self):
        """Génère les positions des devices avec seed FIXE"""
        try:
            self.gateway_position = (0, 0)
            
            original_state = np.random.get_state()
            
            # UTILISER initial_position_seed POUR TOUJOURS AVOIR LES MÊMES POSITIONS
            np.random.seed(self.initial_position_seed)
            
            self.device_positions = []
            
            # Génération aléatoire des positions avec distribution uniforme
            angles = np.random.uniform(0, 2 * np.pi, self.num_devices)
            
            # Utiliser le paramètre distance_gtw comme rayon MAXIMUM
            # sqrt(uniform) pour distribution uniforme dans l'aire du cercle
            radii = self.distance_gtw * np.sqrt(np.random.uniform(0, 1, self.num_devices))
            
            x_coords = radii * np.cos(angles)
            y_coords = radii * np.sin(angles)
            self.device_positions = list(zip(x_coords, y_coords))
            
            np.random.set_state(original_state)
            
            # Vider le cache de shadowing quand les positions changent
            self.shadowing_cache = {}
            
            # DEBUG: Calculer et afficher les distances réelles
            if self.device_positions:
                distances = [np.sqrt(x**2 + y**2) for x, y in self.device_positions]
                min_dist = min(distances)
                max_dist = max(distances)
                avg_dist = np.mean(distances)
                
                self.add_log(f"📐 {self.num_devices} positions générées (seed: {self.initial_position_seed})")
                self.add_log(f"   Distance min: {min_dist:.1f}m, max: {max_dist:.1f}m, moy: {avg_dist:.1f}m")
            
            self.update_deployment_map()
            
        except Exception as e:
            logger.error(f"Erreur génération positions: {e}")
            self.device_positions = []
    
    def regenerate_positions(self, event=None):
        """Regénère les positions des nœuds avec un nouveau seed"""
        try:
            # INCREMENTER initial_position_seed pour changer les positions
            self.initial_position_seed = (self.initial_position_seed + 1) % 1000
            self.add_log(f" Regénération des positions (nouveau seed: {self.initial_position_seed})")
            self.generate_device_positions()
        except Exception as e:
            logger.error(f"Erreur regénération positions: {e}")
            self.add_log(f" Erreur: {str(e)}")
        
    def add_log(self, message):
        """Ajoute un message au log"""
        timestamp = time.strftime("%H:%M:%S")
        self.log_buffer.append(f"[{timestamp}] {message}")
        if len(self.log_buffer) > 30:
            self.log_buffer = self.log_buffer[-30:]
    
    def _start_ui_updates(self):
        """Démarre les mises à jour de l'interface"""
        if self.update_timer:
            self.update_timer.cancel()
        
        def update():
            try:
                # Récupérer les métriques de la simulation
                if self.simulation:
                    metrics_list = self.simulation.get_metrics()
                    logs = self.simulation.get_logs()
                    
                    # Traiter les logs
                    for log in logs:
                        self.add_log(log)
                    
                    # Traiter les métriques
                    for msg_type, data in metrics_list:
                        if msg_type in ['METRICS', 'FINAL_METRICS']:
                            self.latest_metrics = data
                            self._update_ui_indicators(data)
                            
                            if msg_type == 'FINAL_METRICS':
                                self.is_running = False
                                self.status = "Simulation terminée"
                                self._enable_buttons_after_simulation()
                
                # Récupérer les métriques de la queue locale
                while not self.metric_queue.empty():
                    try:
                        msg_type, data = self.metric_queue.get_nowait()
                        if msg_type == 'METRICS':
                            self.latest_metrics = data
                            self._update_ui_indicators(data)
                    except Empty:
                        break
                
                # Planifier la prochaine mise à jour
                if self.is_running:
                    self.update_timer = threading.Timer(0.2, update)
                    self.update_timer.daemon = True
                    self.update_timer.start()
                
            except Exception as e:
                logger.error(f"Erreur dans update thread: {e}")
                if self.is_running:
                    self.update_timer = threading.Timer(0.5, update)
                    self.update_timer.daemon = True
                    self.update_timer.start()
        
        # Démarrer le timer
        self.update_timer = threading.Timer(0.2, update)
        self.update_timer.daemon = True
        self.update_timer.start()
    
    def _update_ui_indicators(self, metrics):
        """Met à jour les indicateurs UI"""
        self.update_ui()
    
    def _enable_buttons_after_simulation(self):
        """Réactive les boutons après la simulation"""
        try:
            self.start_btn.disabled = False
            self.stop_btn.disabled = True
            self.reset_btn.disabled = False
            self.export_btn.disabled = False
            self.save_dqn_btn.disabled = False
            if hasattr(self, 'energy_report_btn'):
                self.energy_report_btn.disabled = False
                self.reset_energy_btn.disabled = False
            if hasattr(self, 'toggle_shadowing_btn'):
                self.toggle_shadowing_btn.disabled = False
        except Exception as e:
            logger.error(f"Erreur réactivation boutons: {e}")
    
    def start_simulation(self, event=None):
        """Démarre la simulation"""
        if self.is_running:
            return
        
        # RÉGÉNÉRER LES POSITIONS AVEC LE SEED COURANT (pour cohérence)
        self.add_log(f" Génération des positions pour la simulation (seed: {self.initial_position_seed})")
        self.generate_device_positions()
        
        # Créer la configuration avec les positions actuelles
        config = {
            'simulation_duration': self.simulation_duration,
            'num_devices': self.num_devices,
            'distance_gtw': self.distance_gtw,
            'tx_power': self.tx_power,
            'region': self.region,
            'coding_rate': self.coding_rate,
            'bandwidth_khz': self.bandwidth_khz,
            'payload_min': self.payload_min,
            'payload_max': self.payload_max,
            'tx_interval_min': self.tx_interval_min,
            'tx_interval_max': self.tx_interval_max,
            'shadowing_std_db': self.shadowing_std_db,
            'path_loss_exponent': self.path_loss_exponent,
            'doppler_hz': self.doppler_hz,
            'multipath_enabled': self.multipath_enabled,
            'noise_figure_db': self.noise_figure_db,
            'pa_type': self.pa_type,
            'battery_capacity_mah': self.battery_capacity_mah,
            'transmissions_per_day': self.transmissions_per_day,
            'enable_intelligent_scheduler': self.enable_intelligent_scheduler,
            'scheduler_max_delay': self.scheduler_max_delay,
            'scheduler_allow_freq_shift': self.scheduler_allow_freq_shift,
            'scheduler_allow_power_boost': self.scheduler_allow_power_boost,
            'enable_dqn': self.enable_dqn,
            'dqn_model_name': self.dqn_model_name,
            'dqn_exploration': self.dqn_exploration,
            'use_dqn_for_dr': self.use_dqn_for_dr,
            'use_dqn_for_power': self.use_dqn_for_power,
            # 'use_dqn_for_frequency': self.use_dqn_for_frequency,
            # AJOUTER LES POSITIONS DES DEVICES À LA CONFIGURATION
            'device_positions': self.device_positions,
            'position_seed': self.initial_position_seed,  # Inclure aussi le seed pour débogage
            'show_shadowing_map': self.show_shadowing_map  # Ajouter l'état de la carte
        }
        
        # Créer la simulation
        self.simulation = LR_FHSS_Simulation(config)
        
        # Passer la référence du dashboard à la simulation
        self.simulation.dashboard = self
        
        # Log configuration
        self.add_log("=" * 50)
        self.add_log(" Démarrage de la simulation LR-FHSS")
        self.add_log("=" * 50)
        self.add_log(f"📐 Positions générées avec seed: {self.initial_position_seed}")
        
        if self.enable_dqn:
            self.add_log(f" DQN activé - Modèle: {self.dqn_model_name}")
        
        if self.enable_intelligent_scheduler:
            self.add_log(" Scheduler intelligent activé")
        
        if ENERGY_MODULE_AVAILABLE:
            self.add_log(f" Module énergie activé - PA: {self.pa_type}, Batterie: {self.battery_capacity_mah:.0f} mAh")
        
        if self.show_shadowing_map:
            self.add_log(f" Carte de shadowing activée (σ={self.shadowing_std_db:.1f} dB)")
        
        # Mettre à jour interface
        self.is_running = True
        self.status = "Simulation en cours..."
        self.progress = 0
        
        self.start_btn.disabled = True
        self.stop_btn.disabled = False
        self.reset_btn.disabled = True
        self.export_btn.disabled = True
        self.save_dqn_btn.disabled = True
        self.energy_report_btn.disabled = True
        self.reset_energy_btn.disabled = True
        self.toggle_shadowing_btn.disabled = True
        
        # Démarrer la simulation
        self.simulation.start()
        
        # Démarrer mises à jour UI
        self._start_ui_updates()
    
    def stop_simulation(self, event=None):
        """Arrête la simulation"""
        if self.simulation:
            self.simulation.stop()
        
        self.is_running = False
        self.status = "Simulation arrêtée"
        self.add_log("⏹ Simulation arrêtée")
        
        try:
            self.start_btn.disabled = False
            self.stop_btn.disabled = True
            self.reset_btn.disabled = False
            self.export_btn.disabled = False
            self.save_dqn_btn.disabled = False
            self.energy_report_btn.disabled = False
            self.reset_energy_btn.disabled = False
            self.toggle_shadowing_btn.disabled = False
        except Exception:
            pass
    
    def reset_simulation(self, event=None):
        """Réinitialise la simulation"""
        if self.simulation:
            self.simulation.reset()
        
        self.is_running = False
        self.status = "Prêt"
        self.progress = 0
        self.latest_metrics = {}
        self.simulation = None
        
        # RÉINITIALISER LE SEED À LA VALEUR ORIGINALE
        self.initial_position_seed = 42
        self.add_log(" Réinitialisation avec seed original (42)")
        
        # Vider le cache de shadowing
        self.shadowing_cache = {}
        
        try:
            self.start_btn.disabled = False
            self.stop_btn.disabled = True
            self.reset_btn.disabled = False
            self.export_btn.disabled = False
            self.save_dqn_btn.disabled = False
            self.energy_report_btn.disabled = False
            self.reset_energy_btn.disabled = False
            self.toggle_shadowing_btn.disabled = False
        except Exception:
            pass
        
        # Réinitialiser tous les indicateurs à zéro
        try:
            self.rt_total_pkts.value = 0
            self.rt_success.value = 0
            self.rt_collisions.value = 0
            self.rt_success_rate_pct.value = 0.0
            self.rt_total_echecs.value = 0
            self.rt_simulated_time.value = 0.0
            self.rt_toa_brut.value = 0.0
            self.rt_toa.value = 0.0
            self.rt_scheduler_delays.value = 0
            
            if ENERGY_MODULE_AVAILABLE and hasattr(self, 'rt_total_energy'):
                self.rt_total_energy.value = 0.0
            
            # Réinitialiser les logs
            self.log_buffer = []
            if hasattr(self, 'log_panel'):
                self.log_panel.value = "Simulateur LR-FHSS initialisé\n"
            
            # Réinitialiser la barre de progression
            if hasattr(self, 'progress_bar'):
                self.progress_bar.value = 0
            
            # Réinitialiser le panneau de résultats
            if hasattr(self, 'results_pane'):
                self.results_pane.clear()
                self.results_pane.append(pn.pane.Markdown("## Résultats détaillés", margin=(0, 0, 10, 0)))
            
            self.add_log(" Réinitialisation complète - Application prête")
            
        except Exception as e:
            logger.error(f"Erreur réinitialisation widgets: {e}")
        
        # Régénérer les positions
        self.generate_device_positions()
        self.update_ui()
    
    def export_report(self, event=None):
        """Exporte le rapport complet"""
        try:
            if self.simulation:
                files = self.simulation.export_report()
                
                success_msg = (
                    f" Rapport exporté avec succès !\n"
                    f"Fichiers créés :\n"
                    f"• {files['txt']} (Rapport détaillé)\n"
                    f"• {files['json']} (Métriques JSON)"
                )
                
                if files.get('csv'):
                    success_msg += f"\n• {files['csv']} (Données paquets)"
                
                pn.state.notifications.success(success_msg, duration=15000)
                self.add_log(f" Rapport exporté : {files['txt']}")
                
                # Afficher le rapport dans l'interface
                report = self.simulation.generate_report()
                self.results_pane.clear()
                self.results_pane.append(pn.pane.Markdown(report))
                
            else:
                self.add_log("Aucune donnée à exporter")
                pn.state.notifications.warning("Aucune donnée de simulation à exporter", duration=5000)
                
        except Exception as e:
            logger.error(f"Erreur lors de l'export: {e}")
            self.add_log(f" Erreur export: {str(e)}")
    
    def get_dashboard(self):
        """Retourne le dashboard complet"""
        
        # Colonne de gauche (Paramètres)
        left_col = pn.Column(
            self.param_column,
            self.control_pane,
            width=400,
            scroll=True,
        )
        
        # Colonne centrale (Carte)
        center_col = pn.Column(
            self.deployment_panel,
            sizing_mode='stretch_width',
            scroll=True,
        )
        
        # Colonne de droite (Métriques rapides)
        right_col = pn.Column(
            self.metrics_panel,
            width=300,
            scroll=True,
        )
        
        top_section = pn.Row(
            left_col, center_col, right_col,
            sizing_mode='stretch_width',
        )
        
        # Section basse : Logs + Résultats
        bottom_section = pn.Row(
            pn.Column(
                pn.pane.Markdown("##  Logs"),
                self.log_panel,
                sizing_mode='stretch_width',
            ),
            pn.Column(
                pn.pane.Markdown("##  Résultats"),
                self.results_pane,
                sizing_mode='stretch_width',
            ),
            sizing_mode='stretch_width',
        )
        
        # Assemblage final
        dashboard = pn.Column(
            top_section,
            bottom_section,
            sizing_mode='stretch_width',
        )
        
        return dashboard


# === FONCTION DE LANCEMENT ===

def launch_dashboard(port=5006, show=True):
    """Lance le dashboard"""
    try:
        dashboard_app = SimulationDashboard()
        
        if show:
            pn.serve(
                dashboard_app.get_dashboard(), 
                title="Simulateur LR-FHSS avec DDQN", 
                port=port, 
                show=True,
                threaded=True,
                autoreload=True
            )
        else:
            return dashboard_app.get_dashboard()
    except Exception as e:
        print(f" Impossible de lancer le simulateur: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    # Afficher le statut des modules
    print("=" * 70)
    print(" Initialisation du Simulateur LR-FHSS")
    print("=" * 70)
    
    print(f" Panel: Version {pn.__version__}")
    print(f" Module DQN: {' Disponible' if DQN_AVAILABLE else 'Non disponible'}")
    print(f" Scheduler Intelligent: {' Disponible' if INTELLIGENT_SCHEDULER_AVAILABLE else 'Non disponible'}")
    print(f" Module Énergie: {' Disponible' if ENERGY_MODULE_AVAILABLE else 'Non disponible'}")
    print("=" * 70)
    
    # Lancer le dashboard
    launch_dashboard(port=5008, show=True)