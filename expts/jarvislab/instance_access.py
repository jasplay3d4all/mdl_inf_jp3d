from jlclient import jarvisclient
from jlclient.jarvisclient import *
import os


jarvisclient.token = os.environ['JARVIS_TOK']
jarvisclient.user_id = os.environ['JARVIS_USR_ID'] 

print("Jarvis client ", jarvisclient)

instances_list = User.get_instances(status=None)
# User.add_script(script_path='install_fastai.sh', script_name='myscript')
# User.delete_script()
# User.get_script()

# print("Instance details ", instances_list)

def get_instance_by_name(name):
    instances_list = User.get_instances(status=None)
    machine_id = None
    for instance in instances_list:
        if(instance.name == name):
            return instance
    # if(machine_id):
    #     return User.get_instance(machine_id)
    # else:
    print("The specified name instance does not exist ", name)
    return None

instance = get_instance_by_name('jp3d_gpu')
print("jp3d_gpu details ", instance)
print("Instance url ", instance.tboard_url)
# instance = Instance.create(gpu_type='A100',
#                             num_gpus=1,
#                             hdd=20,
#                             framework_id=0,
#                             name='IamAI',
#                             script_id=1)

# instance = User.get_instance(34063)
# instance.resume()

# #Example 2:
# instance.resume(num_gpus=1,
#                 gpu_type='RTX5000',
#                 hdd=100)