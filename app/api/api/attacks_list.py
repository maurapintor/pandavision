from flask import jsonify
from flask_restful import Resource

from adv.evaluation_manager import ATTACK_CHOICES


class AttackList(Resource):

    def get(self, pert_type):
        attacks = ATTACK_CHOICES[pert_type]
        return jsonify({"attacks": attacks})


class Attacks(Resource):

    def get(self):
        return list(ATTACK_CHOICES.keys())
