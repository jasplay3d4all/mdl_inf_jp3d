import socketio

# standard Python
sio = socketio.Client()

# asyncio
# sio = socketio.AsyncClient()

sio.connect('http://127.0.0.1:8000')
sio.emit('my_message', {'foo': 'bar'})
