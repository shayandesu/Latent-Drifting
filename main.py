import os, torch

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

from diffusion import StableDiffusionLDPipeline
from scheduler import ShiftedDDIMScheduler, ShiftedDPMSolverMultistepScheduler

device = "cuda"
model_id = "stabilityai/stable-diffusion-2-1"
SHIFT = torch.tensor([0.01, 0.01, 0.01, 0.0], dtype=torch.float16, device=device)

torch.cuda.manual_seed(0)
pipe = StableDiffusionLDPipeline.from_pretrained(model_id)
pipe.scheduler = ShiftedDPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.scheduler.set_channel_shifts(SHIFT)

# pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
# pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(device)


prompt = "Elon Musk on a mountain"
image = pipe(prompt, num_inference_steps=20, channel_shifts=SHIFT).images[0]
image.save(f"Elon Musk on a mountain.png")