from flask import render_template
from flask_restful import Resource


class Home(Resource):

    def get(self):
        return render_template('home.html')
