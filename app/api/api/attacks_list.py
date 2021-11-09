from flask import jsonify
from flask_restful import Resource

from adv.evaluation_manager import ATTACK_CHOICES, PERT_SIZES


class AttackList(Resource):

    def get(self, pert_type):
        attacks = ATTACK_CHOICES[pert_type]
        return jsonify({"attacks": attacks})


class Attacks(Resource):

    def get(self):
        return list(ATTACK_CHOICES.keys())

class PertSizes(Resource):
    def get(self, pert_type):
        pert_sizes = PERT_SIZES[pert_type]
        return jsonify({'pert_sizes': pert_sizes})