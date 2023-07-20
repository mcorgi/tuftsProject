import flask
import tensorflow as tf


# model = tf.keras.models.load_model(r'C:\Users\User\Desktop\github\tuftsProject')


app = flask.Flask(__name__, template_folder='main.html')
@app.route("/")
def main():
    return(flask.render_template('main.html'))


if __name__== '__main__':
    app.run()


# app = flask.Flask(__name__, template_folder='main.html')
# @app.route('/')
# def main():
#      if flask.request.method == 'GET':
#         return(flask.render_template('main.html'))
#      if flask.request.method == 'POST':
#         return flask.render_template('main.html')
# if __name__ == '__main__':
#     app.run()

