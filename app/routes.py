from flask import render_template, request, jsonify, Blueprint, make_response
import ast

from .services import Prediction

routes = Blueprint("routes", __name__)


@routes.route("/")
def index():
    return render_template("index.html")


@routes.route("/predict", methods=["POST"])
def prediction_with_right():
    try:
        waypoints = request.form["waypoints"]
        waypoints = ast.literal_eval(waypoints)
        waypoints = Prediction(waypoints)

        points_prepared = waypoints.calculate_features_from_wrist()
        prediction = waypoints.predict_class_from_features(points_prepared)

        return jsonify({"letter": prediction})

    except Exception as e:
        return make_response(
            jsonify(
                {
                    "error": "Une erreur s'est produite. Veuillez contacter l'administrateur."
                }
            ),
            500,
        )
