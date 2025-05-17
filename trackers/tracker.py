from ultralytics import YOLO
import supervision as sv
import numpy as np
import cv2
from scipy.spatial.distance import cdist
from collections import defaultdict
from ultralytics import YOLO
import supervision as sv
import pickle
import os
import numpy as np
import pandas as pd
import cv2
import math
import csv
from team_assigners import TeamAssigner


from scipy.spatial.distance import cdist


import numpy as np
import json  # Pour une belle sortie formatée

import sys
sys.path.append("C:/Users/HP GAMING/Desktop/AI/computer vision/match")
from utils import get_center_of_bbox,get_bbox_width,get_foot_position
from team_assigners import TeamAssigner




from ultralytics import YOLO
import supervision as sv
import numpy as np
import cv2
from scipy.spatial.distance import cdist
from collections import defaultdict

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path) 
        self.tracker = sv.ByteTrack()
        
        # Gestion des IDs joueurs (1-22)
        self.player_ids = list(range(1, 23))
        self.available_ids = set(self.player_ids)
        self.id_mapping = {}  # Mappe track_id vers player_id
        self.player_history = defaultdict(list)
        self.team_assigner = TeamAssigner()





    def detect_frames(self, frames):
        batch_size=20 
        detections = [] 
        for i in range(0,len(frames),batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size],conf=0.1)
            detections += detections_batch
        return detections













        
          











    def _assign_player_id(self, track_id):
        """Assignation d'un ID joueur disponible (1-22)"""
        if self.available_ids:
            player_id = min(self.available_ids)
            self.available_ids.remove(player_id)
        else:
            # Recyclage de l'ID le moins utilisé
            player_id = min(self.player_history.items(), key=lambda x: len(x[1]))[0]
        
        self.id_mapping[track_id] = player_id
        return player_id

    def assign_teams(self, tracks, frame):
        """Assignation des équipes aux joueurs"""
        for frame_num, frame_tracks in enumerate(tracks["players"]):
            for player_id, player in frame_tracks.items():
                bbox = player["bbox"]
                player["team_color"] = self.team_assigner.assign_team_color(frame, bbox)







    def draw_annotations(self, video_frames, tracks):
            """Dessin des annotations sur les frames"""
            output_frames = []
            
            for frame_num, frame in enumerate(video_frames):
                frame = frame.copy()
                player_dict = tracks["players"][f"frame{frame_num}"]
                ball_dict = tracks["ball"][f"frame{frame_num}"]
                referee_dict = tracks["referees"][f"frame{frame_num}"]

                # Dessin des joueurs
                for player_id, player in player_dict.items():
                    color = player.get("team_color", (0, 0, 255))
                    frame = self._draw_player_marker(frame, player["bbox"], color, player_id)
                    
                    # Marqueur si le joueur a le ballon
                    if self._player_has_ball(player, ball_dict):
                        frame = self._draw_ball_possession(frame, player["bbox"])

                # Dessin des arbitres
                for _, referee in referee_dict.items():
                    frame = self._draw_player_marker(frame, referee["bbox"], (0, 255, 255), "R")

                # Dessin du ballon
                for _, ball in ball_dict.items():
                    frame = self._draw_ball_marker(frame, ball["bbox"])

                output_frames.append(frame)
            
            return output_frames



    def _draw_player_marker(self, frame, bbox, color, player_id):
        x1, y1, x2, y2 = map(int, bbox)




        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center,y2),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color = color,
            thickness=2,
            lineType=cv2.LINE_4
        )
        
        # Position pour l'ID (sous la boîte)
        id_pos = ((x1 + x2) // 2 - 10, y2 + 20)
        
        # Rectangle de fond pour l'ID
        text_size = cv2.getTextSize(str(player_id), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]


        
        cv2.rectangle(frame, 
                    (id_pos[0] - 5, id_pos[1] - text_size[1] - 5),
                    (id_pos[0] + text_size[0] + 5, id_pos[1] + 5),
                    (255, 255, 255), -1)
        
        # Texte de l'ID
        cv2.putText(frame, str(player_id), id_pos,
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        return frame














    def _draw_ball_marker(self, frame, bbox):
        """Dessin du ballon"""
        x_center = int((bbox[0] + bbox[2]) / 2)
        y_center = int((bbox[1] + bbox[3]) / 2)
        
        triangle_points = np.array([
            [x_center, y_center-10],
            [x_center-10, y_center+10],
            [x_center+10, y_center+10]
        ])
        cv2.drawContours(frame, [triangle_points], 0, (0,255,0), cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0,0,0), 2)
        return frame

    def _draw_ball_possession(self, frame, bbox):
        """Indicateur de possession du ballon"""
        x1, y1, x2, y2 = map(int, bbox)
        cv2.circle(frame, (x1, y1-15), 5, (0,255,255), -1)
        return frame







    def _player_has_ball(self, player, ball_dict):
        """Vérifie si un joueur a le ballon"""
        if not ball_dict:
            return False
            
        ball_pos = ball_dict[1]["position"]
        player_pos = player["position"]
        distance = np.sqrt((ball_pos[0]-player_pos[0])**2 + (ball_pos[1]-player_pos[1])**2)
        return distance < 25
    






























    def joeurs(self,all_frames_data,seuil=50, max_joueurs=24):

        joueurs = {} 
        prochain_id = 0
        positions_precedentes = {}

        for frame in all_frames_data:
            frame_id = frame["frame_number"]
            detections = self.extraire_centres(frame["detections"])

            correspondances = {}
            utilises = set()

            for idx, detection in enumerate(detections):
                meilleur_id = None
                distance_min = seuil
                for joueur_id, ancienne_pos in positions_precedentes.items():
                    distance = self.calcul_distance(detection, ancienne_pos)
                    if distance < distance_min and joueur_id not in utilises:
                        distance_min = distance
                        meilleur_id = joueur_id

                if meilleur_id is not None:
                    correspondances[idx] = meilleur_id
                    utilises.add(meilleur_id)
                elif prochain_id < max_joueurs: 
                    correspondances[idx] = prochain_id
                    prochain_id += 1


            positions_precedentes = {}
            for idx, detection in enumerate(detections):
                joueur_id = correspondances.get(idx)
                if joueur_id is not None:
                    positions_precedentes[joueur_id] = detection
                    if joueur_id not in joueurs:
                        joueurs[joueur_id] = []
                    joueurs[joueur_id].append((frame_id, detection))

        print("Trajectoires des joueurs :")
        for joueur_id, trajectoire in joueurs.items():
            print(f"Joueur {joueur_id} : {trajectoire}")

        return joueurs
    












        
    def save_to_csv(self, alldata, output_path="C:/Users/HP GAMING/Desktop/AI/computer vision/match/player_dataytes55454545454.csv"):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        headers = [
            "player_id", "frame_num", 
            "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
            "position_x", "position_y",
            "team_color", "team", "confidence",
            "ball_pos_x", "ball_pos_y"
        ]

        try:
            with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=headers)
                writer.writeheader()

                for frame_name, frame_data in alldata["players"].items():
                    frame_num = int(frame_name.replace("frame", ""))

                    ball_frame_data = alldata["ball"].get(frame_name, {})
                    if ball_frame_data:
                        first_ball = next(iter(ball_frame_data.values()))
                        ball_pos = first_ball.get("position", [None, None])
                    else:
                        ball_pos = [None, None]

                    for player_id, player_data in frame_data.items():
                        position = player_data.get("position", [None, None])

                        writer.writerow({
                            "player_id": player_id,
                            "frame_num": frame_num,
                            "bbox_x1": player_data["bbox"][0],
                            "bbox_y1": player_data["bbox"][1],
                            "bbox_x2": player_data["bbox"][2],
                            "bbox_y2": player_data["bbox"][3],
                            "position_x": position[0],
                            "position_y": position[1],
                            "team_color": player_data.get("team_color", None),
                            "team": player_data.get("team", None),
                            "confidence": player_data.get("confidence", None),
                            "ball_pos_x": ball_pos[0],
                            "ball_pos_y": ball_pos[1]
                        })

            print(f"Data saved to {output_path}")

        except Exception as e:
            print(f"Error saving CSV: {e}")












        
    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
            
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
                with open(stub_path,'rb') as f:
                    tracks = pickle.load(f)
                return tracks
        """Suivi principal des objets avec gestion des IDs"""
        detections = self.detect_frames(frames)
        alldata = {
            "players": defaultdict(dict),
            "referees": [],
            "ball": []
        }

        # tracks devient un dictionnaire de frames
        tracks = {
            "players": {},
            "referees": {},
            "ball": {}
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            detection_sv = sv.Detections.from_ultralytics(detection)
            detection_sv = detection_sv[detection_sv.confidence > 0.3]

            for i, class_id in enumerate(detection_sv.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_sv.class_id[i] = cls_names_inv["player"]

            tracked_detections = self.tracker.update_with_detections(detection_sv)
            print(f"frame{frame_num}",detection_sv)

            # Initialiser les tracks pour cette frame
            tracks["players"][f"frame{frame_num}"] = {}
            tracks["referees"][f"frame{frame_num}"] = {}
            tracks["ball"][f"frame{frame_num}"] = {}

            player_detections = [d for d in tracked_detections if d[3] == cls_names_inv['player']]
            player_detections = sorted(player_detections, key=lambda x: x[2], reverse=True)[:22]

            for detection in player_detections:
                bbox = detection[0].tolist()
                track_id = detection[4]

                if track_id in self.id_mapping:
                    player_id = self.id_mapping[track_id]
                else:
                    player_id = self._assign_player_id(track_id)

                tracks["players"][f"frame{frame_num}"][player_id] = {
                    "bbox": bbox,
                    "position": get_foot_position(bbox)
                }

                alldata["players"][f"player number {player_id}"][f"frame{frame_num}"] = {
                    "bbox": bbox,
                    "position": get_foot_position(bbox),
                    "confidence": float(detection[2]),
                }

                self.player_history[player_id].append(bbox)

            for detection in tracked_detections:
                if detection[3] == cls_names_inv['referee']:
                    bbox = detection[0].tolist()
                    tracks["referees"][f"frame{frame_num}"][detection[4]] = {
                        "bbox": bbox,
                        "position": get_foot_position(bbox)
                    }

            ball_detections = [d for d in detection_sv if d[3] == cls_names_inv['ball']]
            if ball_detections:
                bbox = ball_detections[0][0].tolist()
                tracks["ball"][f"frame{frame_num}"][1] = {
                    "bbox": bbox,
                    "position": get_center_of_bbox(bbox)
                }

        return tracks





        
    def team(self, tracks, video_frames):
        team_assigner = TeamAssigner()

        # Assigner les couleurs d'équipe pour les joueurs dans la première frame
        team_assigner.assign_team_color(video_frames[0], tracks['players']['frame0'])

        # Assignation des couleurs d'équipe pour chaque joueur dans chaque frame
        for frame_key, player_track in tracks['players'].items():  # Utilisation de .items() pour parcourir le dictionnaire
            # Extraction du numéro de la frame à partir de la clé, par exemple 'frame0' -> 0
            frame_num = int(frame_key.replace('frame', ''))

            for player_id, track in player_track.items():
                # Récupérer l'équipe pour chaque joueur
                team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)

                # Assignation de l'équipe et de la couleur dans les données de suivi
                tracks['players'][frame_key][player_id]['team'] = team
                tracks['players'][frame_key][player_id]['team_color'] = team_assigner.team_colors.get(team, "default_color")

        return tracks





    def detect_possession_percentage(self,tracks):
        possession = {
            "teams": defaultdict(int),  
            "total_possession_frames": 0
        }

        for frame_key, players in tracks['players'].items():
            ball_data = tracks['ball'].get(frame_key)
            if not ball_data:
                continue 

            ball_pos = ball_data[1]['position']

            min_dist = float('inf')
            team_possessor = None

            for player_id, player_data in players.items():
                player_pos = player_data['position']
                dist =((ball_pos[0]-player_pos[0])**2+  (ball_pos[1]-player_pos[1])**2)**0.5

                if dist < min_dist:
                    min_dist = dist
                    team_possessor = player_data.get('team', 'unknown')

            # Seuil pour considérer la possession (par exemple 50 pixels)
            if min_dist < 50:
                possession["teams"][team_possessor] += 1
                possession["total_possession_frames"] += 1

        # Calcul des pourcentages
        possession_percent = {}
        total = possession["total_possession_frames"]
        if total > 0:
            for team, frames in possession["teams"].items():
                possession_percent[team] = round((frames / total) * 100, 2)
        else:
            for team in possession["teams"]:
                possession_percent[team] = 0.0

        return possession_percent










        
    def get_object_trackss(self, frames, read_from_stub=False, stub_path=None):
            
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
                with open(stub_path,'rb') as f:
                    tracks = pickle.load(f)
                return tracks
        """Suivi principal des objets avec gestion des IDs"""
        detections = self.detect_frames(frames)
        alldata = {
            "players": defaultdict(dict),
            "referees": [],
            "ball": []
        }

        # tracks devient un dictionnaire de frames
        tracks = {
            "players": {},
            "referees": {},
            "ball": {}
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            detection_sv = sv.Detections.from_ultralytics(detection)
            detection_sv = detection_sv[detection_sv.confidence > 0.3]

            for i, class_id in enumerate(detection_sv.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_sv.class_id[i] = cls_names_inv["player"]

            tracked_detections = self.tracker.update_with_detections(detection_sv)
            print(f"frame{frame_num}",detection_sv)

            # Initialiser les tracks pour cette frame
            tracks["players"][f"frame{frame_num}"] = {}
            tracks["referees"][f"frame{frame_num}"] = {}
            tracks["ball"][f"frame{frame_num}"] = {}

            player_detections = [d for d in tracked_detections if d[3] == cls_names_inv['player']]
            player_detections = sorted(player_detections, key=lambda x: x[2], reverse=True)[:22]

            for detection in player_detections:
                bbox = detection[0].tolist()
                track_id = detection[4]

                if track_id in self.id_mapping:
                    player_id = self.id_mapping[track_id]
                else:
                    player_id = self._assign_player_id(track_id)

                tracks["players"][f"frame{frame_num}"][player_id] = {
                    "bbox": bbox,
                    "position": get_foot_position(bbox)
                }

                alldata["players"][f"player number {player_id}"][f"frame{frame_num}"] = {
                    "bbox": bbox,
                    "position": get_foot_position(bbox),
                    "confidence": float(detection[2]),
                }

                self.player_history[player_id].append(bbox)

            for detection in tracked_detections:
                if detection[3] == cls_names_inv['referee']:
                    bbox = detection[0].tolist()
                    tracks["referees"][f"frame{frame_num}"][detection[4]] = {
                        "bbox": bbox,
                        "position": get_foot_position(bbox)
                    }

            ball_detections = [d for d in detection_sv if d[3] == cls_names_inv['ball']]
            if ball_detections:
                bbox = ball_detections[0][0].tolist()
                tracks["ball"][f"frame{frame_num}"][1] = {
                    "bbox": bbox,
                    "position": get_center_of_bbox(bbox)
                }

        return alldata
    
    def calculate_speed(self, track_id, current_position):
        if track_id in self.previous_positions:
            x1, y1 = self.previous_positions[track_id]
            x2, y2 = current_position

            distance_px = math.hypot(x2 - x1, y2 - y1)
            distance_m = distance_px * 0.05  # facteur à ajuster selon la vidéo

            speed_m_per_s = distance_m * self.fps
            speed_kmh = speed_m_per_s * 3.6
        else:
            speed_kmh = 0.0

        self.previous_positions[track_id] = current_position
        return speed_kmh