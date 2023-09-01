

import eventlet
import socketio
from types import SimpleNamespace
import json
# from dict2obj import Dict2Obj
# from bunch import bunchify
# from sklearn.utils import Bunch

from lib.utils_plg.dict_obj import dict2obj, obj2dict
from pipeline.list import get_pipeline_list
#from pipeline_dev.list import get_pipeline_list



config = dict2obj({
    'data_io_path' : './share_vol/data_io/',
    'model_path' : './per_shr_vol/models/',
    'namespace': '/gpu'
    # shared_volume_path = "../per_shr_vol"
    # app_volume_path = "../per_shr_vol"
})

static_files = {
    # '/': {'content_type': 'text/html', 'filename': './data_io/index.html'},
    '/static' : config.data_io_path,
}

sio = socketio.Server(ping_timeout=60000, ping_interval=60000)

class SocketIOApi(socketio.Namespace):
    def on_connect(self, sid, environ):
        print('connect ', sid, environ)
        self.active_pipe = None;
        self.pipe_list = get_pipeline_list(config)
        return

    def on_disconnect(self, sid):
        print('disconnect ', sid)
        return
    def on_connect_error(data):
        print("The connection failed!", data)

    def on_gpu_job_start(self, sid, pipe_state):
        # Convert JSON object to python object
        # pipe_state = Dict2Obj(pipe_state)#json.loads(pipe_state, object_hook=lambda d: SimpleNamespace(**d))
        pipe_state = dict2obj(pipe_state)
        print('gpu_job_start id ', sid)

        # Check whether the current pipe is active but it is not the current job to be executed
        if(self.active_pipe and self.active_pipe.name != pipe_state.name):
            self.active_pipe.unload()
            self.active_pipe = None
        if(self.active_pipe == None):
            self.active_pipe = self.pipe_list[pipe_state.name].load(pipe_state, self)
        self.active_pipe.process(pipe_state)

        # sio.of(config.namespace).emit('gpu_job_complete', pipe_state)
        self.emit('gpu_job_complete', obj2dict(pipe_state))
        print('gpu_job_start complete id ', sid)

        # sio.emit('gpu_job_progress', pipe_state)
        # sio.emit('gpu_job_error', pipe_state)
        return

    def progress_cb(self, pipe_state):
        # sio.of(config.namespace).emit('gpu_job_progress', pipe_state)
        self.emit('gpu_job_progress', pipe_state)
        return

    def error_cb(self, pipe_state):
        # sio.of(config.namespace).emit('gpu_job_error', pipe_state)
        self.emit('gpu_job_error', pipe_state)
        return

sio.register_namespace(SocketIOApi(config.namespace))

app = socketio.WSGIApp(sio, static_files=static_files)

# # on - generateImage getLoraModels getTextualInversionTriggers requestSystemConfig requestModelChange cancel 
# # emit - generationResult postprocessingResult systemConfig modelChanged progressUpdate intermediateResult
# # foundLoras foundTextualInversionTriggers error 
# @sio.on('generate_image')
# def generate_image(sid, runtime_cfg):
#     print('message ', data)

#     sio.emit('generated_image', {'status': 'image generated'})

if __name__ == '__main__':
    eventlet.wsgi.server(eventlet.listen(('', 80)), app)
