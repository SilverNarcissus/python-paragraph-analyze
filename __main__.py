# from flask import Flask, render_template, request, jsonify
#
# app = Flask(__name__)
#
# @app.route('/')
# @app.route('/index.html')
# def index():
#     return render_template('Hello_world.html')
#
# @app.route('/api/result')
# def result():
#     # a = request.args.get('a', 0, type=float)
#     # b = request.args.get('b', 0, type=float)
#     return jsonify(12)
from wordNLTK.interface import process

if __name__ == "__main__":
    process()
