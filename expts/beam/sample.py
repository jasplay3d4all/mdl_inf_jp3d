import beam

# The environment your code will run on
app = beam.App(
    name="stable-diffusion-app",
    cpu=8,
    memory="32Gi",
    gpu="A10G",
    python_version="python3.8",
    python_packages=[
        "diffusers[torch]>=0.10",
        "transformers",
        "torch",
        "pillow",
        "accelerate",
        "safetensors",
        "xformers",
    ],
)

# Deploys function as async task queue
app.Trigger.TaskQueue(
    inputs={"prompt": beam.Types.String()},
    handler="run.py:generate_image",
)

# File to store image outputs
app.Output.File(path="output.png", name="myimage")

# Persistent volume to store cached model
app.Mount.PersistentVolume(path="./models", name="models")



pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        revision="fp16",
        torch_dtype=torch.float16,
        cache_dir=cache_path,
        # Add your own auth token from Huggingface
        use_auth_token=os.environ["HUGGINGFACE_API_KEY"],
    ).to("cuda")