"""
Module Flask pour gérer les routes et les prédictions.

Ce module utilise Flask pour créer des routes et gérer les requêtes HTTP.
Il utilise les classes `PreTreatment` et `Prediction` du module `prediction`
pour calculer les caractéristiques des waypoints et prédire une classe.

Modules nécessaires :
    - flask
    - ast
    - .prediction (module local)

Variables globales :
    - routes (Blueprint) : Blueprint pour les routes de l'application.

Fonctions :
    - index() : Route pour la page d'accueil.
    - prediction_with_right() : Route pour la prédiction de la classe à partir des waypoints.
"""

import ast

from flask import render_template, request, jsonify, Blueprint, make_response

from .prediction import PreTreatment, Prediction

routes = Blueprint("routes", __name__)


@routes.route("/")
def index():
    """
    Route pour la page d'accueil.

    Returns:
        str: Rendu du template HTML pour la page d'accueil.
    """
    return render_template("index.html")


@routes.route("/predict", methods=["POST"])
def prediction_with_right():
    """
    Route pour la prédiction de la classe à partir des waypoints.

    Cette route accepte une requête POST contenant les waypoints et utilise les classes
    `PreTreatment` et `Prediction` pour calculer les caractéristiques et prédire la classe.

    Returns:
        json: Un objet JSON contenant la lettre prédite ou un message d'erreur en cas d'exception.
    """
    try:
        waypoints = request.form["waypoints"]
        waypoints = ast.literal_eval(waypoints)

        features = PreTreatment(waypoints).calculate_features_from_wrist()
        prediction = Prediction(features).predict_class_from_features()

        return jsonify({"letter": prediction})

    except Exception as e:
        return make_response(
            jsonify({"error": e}),
            500,
        )
