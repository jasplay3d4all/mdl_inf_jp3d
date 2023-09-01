import runpod
# https://github.com/runpod/runpod-python/blob/main/runpod/api_wrapper/ctl_commands.py
# https://github.com/runpod/runpod-python/blob/main/examples/graphql_wrapper.py
# https://www.tutorialspoint.com/run-a-script-on-startup-in-linux
# https://github.com/runpod/runpodctl/blob/main/doc/runpodctl_get_pod.md
# https://docs.runpod.io/docs/templates

runpod.api_key = "TWK62N53DHYKACTNDZCHNPSS4SMELCJIOZKAEJAH"

# Create a pod
# pod = runpod.create_pod("test", "runpod/stack", "NVIDIA GeForce RTX 3070")

runpod_id = "q00z52fjcxtmx8"
# resume_pod(pod_id: str, gpu_count: int)
print (runpod.resume_pod(runpod_id, 1))

# print(runpod.get_gpu(runpod_id))

print(runpod.get_gpus())



# Stop the pod
runpod.stop_pod(runpod_id)

# Start the pod
# runpod.resume_pod(runpod_id)

# Terminate the pod
# runpod.terminate_pod(runpod_id)