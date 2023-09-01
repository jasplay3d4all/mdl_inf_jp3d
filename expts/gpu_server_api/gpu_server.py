import eventlet
import socketio

# from flask import Flask, render_template
# from flask_socketio import SocketIO
# from flask import send_from_directory

# app = Flask(__name__)
# app.config['SECRET_KEY'] = 'secret!'
# sio = SocketIO(app, engineio_logger=True, logger=True)

# @app.route('/reports/<path:path>')
# def send_report(path):
#     return send_from_directory('data_io', path)


static_files={
    # '/': {'content_type': 'text/html', 'filename': './data_io/index.html'},
    '/static': './data_io/',
}
sio = socketio.Server()
app = socketio.WSGIApp(sio, static_files=static_files)

# # create a Socket.IO server
# sio = socketio.AsyncServer()
# # wrap with ASGI application
# app = socketio.ASGIApp(sio, static_files=static_files)

@sio.event
def connect(sid, environ):
    print('connect ', sid, environ)

@sio.event
def connect_error(data):
    print("The connection failed!", data)


@sio.event
def my_message(sid, data):
    print('message ', data)

# on - generateImage getLoraModels getTextualInversionTriggers requestSystemConfig requestModelChange cancel 
# emit - generationResult postprocessingResult systemConfig modelChanged progressUpdate intermediateResult
# foundLoras foundTextualInversionTriggers error 
@sio.on('generateImage')
def my_message(sid, data):
    print('message ', data)

@sio.event
def disconnect(sid):
    print('disconnect ', sid)

if __name__ == '__main__':
    eventlet.wsgi.server(eventlet.listen(('', 8000)), app)
# if __name__ == '__main__':
#     sio.run(app)    