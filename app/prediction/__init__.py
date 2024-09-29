"""
Module de prétraitement et de prédiction pour les caractéristiques des waypoints.

Ce module contient deux classes principales : `PreTreatment` et `Prediction`.

- `PreTreatment` : Cette classe est responsable du calcul des caractéristiques géométriques à partir des waypoints,
  y compris les angles et les distances.
- `Prediction` : Cette classe utilise les caractéristiques calculées par `PreTreatment` pour prédire une classe
  à l'aide d'un modèle de machine learning.

Modules nécessaires :
    - os
    - pandas
    - joblib
    - numpy

Variables globales :
    - model_path (str) : Chemin vers le fichier du modèle de machine learning.
    - model : Modèle de machine learning chargé à partir de `model_path`.
    - class_names (list) : Liste des noms de classes possibles.

Classes :
    - PreTreatment : Classe pour le calcul des caractéristiques des waypoints.
    - Prediction : Classe pour la prédiction de la classe à partir des caractéristiques.

Fonctions :
    - Aucune fonction globale définie dans ce module.
"""

import os

import pandas as pd
import joblib
import numpy as np

model_path = os.path.join(os.path.dirname(__file__), "../static/model/model_xgb_v2.pkl")
model = joblib.load(model_path)
class_names = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
]


class PreTreatment:
    """
    Une classe pour gérer le calcul des caractéristiques des waypoints et prédire une classe à partir de ces caractéristiques.

    Attributs:
        waypoints (list of dict): Liste de waypoints où chaque waypoint contient les coordonnées `x` et `y`.

    Méthodes:
        calculate_features_from_wrist():
            Calcule les angles et distances géométriques à partir des coordonnées du poignet (le premier waypoint) et des
            waypoints suivants. Extrait également les coordonnées d'un sous-ensemble de waypoints.

        predict_class_from_features(features):
            Prend en entrée un dictionnaire de caractéristiques comprenant les waypoints, angles et distances,
            et prédit une classe à l'aide d'un modèle de machine learning.
    """

    def __init__(self, waypoints):
        """
        Initialise la classe Prediction avec une liste de waypoints.

        Args:
            waypoints (list of dict): Une liste de dictionnaires représentant les coordonnées des waypoints,
                                      chaque dict contient les clés "x" et "y" représentant les coordonnées.
        """
        self.waypoints = waypoints

    def calculate_features_from_wrist(self):
        """
        Calcule les caractéristiques géométriques à partir du poignet et des waypoints.

        Cette méthode extrait les coordonnées du poignet (le premier waypoint) et calcule :
        - Les angles (en degrés) entre le vecteur formé par le poignet et chaque waypoint suivant.
        - Les distances euclidiennes entre le poignet et chaque waypoint.
        - Les distances spécifiques entre certaines paires de waypoints prédéfinies.
        - Les coordonnées `x` et `y` d'un sous-ensemble de waypoints spécifiés.

        Les caractéristiques calculées incluent :
        - Les angles des vecteurs formés par le poignet et chaque waypoint.
        - Les distances entre le poignet et chaque waypoint, ainsi que certaines distances entre waypoints spécifiques.
        - Les coordonnées `x` et `y` d'un sous-ensemble de waypoints (indices 0, 4, 8, 12, 16, 20).

        Returns:
            dict: Un dictionnaire contenant trois clés :
            - "angles" (list of float): Liste des angles en degrés entre le poignet et chaque waypoint.
            - "distances" (list of float): Liste des distances euclidiennes depuis le poignet et entre certaines paires de waypoints.
            - "waypoints" (dict): Dictionnaire des coordonnées `x` et `y` des waypoints d'intérêt.
        """

        wrist = np.array([self.waypoints[0]["x"], self.waypoints[0]["y"]])
        angles = []
        distances = []

        for i in range(1, len(self.waypoints)):
            waypoint = np.array([self.waypoints[i]["x"], self.waypoints[i]["y"]])

            vector_2d = waypoint[:2] - wrist[:2]
            angle_rad = np.arctan2(vector_2d[1], vector_2d[0])
            angle_deg = np.degrees(angle_rad)
            angles.append(angle_deg)

            distance = np.linalg.norm(waypoint - wrist)
            distances.append(distance)

        specific_waypoint_pairs = [
            (4, 8),
            (8, 12),
            (12, 16),
            (16, 20),
            (4, 17),
        ]

        for pair in specific_waypoint_pairs:
            point_a = np.array(
                [self.waypoints[pair[0]]["x"], self.waypoints[pair[0]]["y"]]
            )
            point_b = np.array(
                [self.waypoints[pair[1]]["x"], self.waypoints[pair[1]]["y"]]
            )
            specific_distance = np.linalg.norm(point_a - point_b)
            distances.append(specific_distance)

        waypoint_indices = [0, 4, 8, 12, 16, 20]
        waypoint_features = {}
        for index, hand in enumerate(self.waypoints):
            if index in waypoint_indices:
                waypoint_features[f"x_{index}"] = hand["x"]
                waypoint_features[f"y_{index}"] = hand["y"]
        waypoints = waypoint_features

        return {"angles": angles, "distances": distances, "waypoints": waypoints}


class Prediction:
    """
    Classe Prediction pour le prétraitement des données et la prédiction de la classe.

    Attributes:
        features (dict): Un dictionnaire contenant les caractéristiques des waypoints, des angles et des distances.

    Methods:
        __init__(self, features): Initialise l'objet Prediction avec les caractéristiques données.
        predict_class_from_features(self): Prédit la classe à partir des caractéristiques des waypoints, des angles et des distances.
    """

    def __init__(self, features) -> None:
        """
        Initialise l'objet Prediction avec les caractéristiques données.

        Args:
            features (dict): Un dictionnaire contenant les caractéristiques des waypoints, des angles et des distances.
        """
        self.features = features

    def predict_class_from_features(self):
        """
        Prédire la classe à partir des caractéristiques des waypoints, des angles et des distances.

        Cette méthode extrait les données des waypoints,
        des angles et des distances à partir de l'attribut `waypoints` de l'objet.
        Elle reformate ensuite ces données en un dictionnaire, les convertit en un DataFrame pandas,
        et utilise un modèle de machine learning
        (contenu dans `self.model`) pour prédire la classe correspondante.
        La classe prédite est retournée sous forme de nom,
        basé sur un tableau de classes (`self.class_names`).

        Returns:
            str: Le nom de la classe prédite par le modèle,
            correspondant à l'index de la prédiction.
        """

        waypoints = self.features["waypoints"]
        angles = self.features["angles"]
        distances = self.features["distances"]

        angles_dict = {f"angle_{i}": value for i, value in enumerate(angles)}
        distances_dict = {f"dist_{i}": value for i, value in enumerate(distances)}
        data = {**waypoints, **angles_dict, **distances_dict}

        s = pd.DataFrame([data])

        prediction = model.predict(s)
        result = class_names[prediction[0]]

        return result
