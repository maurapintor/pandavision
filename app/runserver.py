from flask import Flask
from optparse import OptionParser

import api
import config


def create_app():
    app = Flask(__name__, static_folder='static')
    app.register_blueprint(
        api.bp,
        url_prefix='')
    app.config['SECRET_KEY'] = config.SECRET_KEY

    return app


def new_app(*args, **kwargs):
    my_app = create_app()

    return my_app(*args, **kwargs)


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-p", "--port", dest="port",
                      help="Flask port", metavar="", default=8080)

    (options, args) = parser.parse_args()

    option_dict = vars(options)
    port = int(option_dict['port'])

    app = create_app()
    app.run(debug=True, port=port, host='0.0.0.0', threaded=True)
