import time
import runpod
import requests
from requests.adapters import HTTPAdapter, Retry

LOCAL_URL = "http://127.0.0.1:3000/sdapi/v1"

automatic_session = requests.Session()
retries = Retry(total=10, backoff_factor=0.1, status_forcelist=[502, 503, 504])
automatic_session.mount('http://', HTTPAdapter(max_retries=retries))

# ---------------------------------------------------------------------------- #
#                              Automatic Functions                             #
# ---------------------------------------------------------------------------- #
def wait_for_service(url):
    '''
    Check if the service is ready to receive requests.
    '''
    while True:
        try:
            requests.get(url, timeout=120)
            return
        except requests.exceptions.RequestException:
            print("Service not ready yet. Retrying...")
        except Exception as err:
            print("Error: ", err)
        time.sleep(0.2)

def setup_default_settings():
    '''
    Setup default model settings
    '''
    try:
        options = {
            "sd_model_checkpoint": "wai-nsfw-illustrious-sdxl.safetensors",  # This matches the downloaded model filename
            "CLIP_stop_at_last_layers": 2,
            "sd_vae": "None",  # Use model's built-in VAE
            "sampling_method": "DPM++ 2M Karras",  # Good sampler for quality
            "sampling_steps": 30,
            "batch_size": 1
        }
        automatic_session.post(f'{LOCAL_URL}/options', json=options)
        print("Default settings applied successfully")
    except Exception as err:
        print(f"Error setting defaults: {err}")

def run_inference(inference_request):
    '''
    Run inference on a request with optimized defaults
    '''
    # Default parameters optimized for your model
    default_params = {
        "prompt": inference_request.get("prompt", ""),
        "negative_prompt": inference_request.get("negative_prompt", 
            "low quality, bad anatomy, worst quality, low resolution, blurry, ugly, bad proportions, bad composition"),
        "steps": inference_request.get("steps", 30),
        "cfg_scale": inference_request.get("cfg_scale", 7),
        "width": inference_request.get("width", 512),
        "height": inference_request.get("height", 768),
        "sampler_name": inference_request.get("sampler_name", "DPM++ 2M Karras"),
        "enable_hr": inference_request.get("enable_hr", False),
        "denoising_strength": inference_request.get("denoising_strength", 0.7),
        "batch_size": inference_request.get("batch_size", 1)
    }

    # Merge with user request, keeping user values if provided
    inference_request = {**default_params, **inference_request}

    try:
        response = automatic_session.post(
            url=f'{LOCAL_URL}/txt2img',
            json=inference_request,
            timeout=600
        )
        return response.json()
    except Exception as err:
        print(f"Inference error: {err}")
        return {"error": str(err)}

# ---------------------------------------------------------------------------- #
#                                RunPod Handler                                #
# ---------------------------------------------------------------------------- #
def handler(event):
    '''
    Handler function for serverless
    '''
    try:
        return run_inference(event["input"])
    except Exception as err:
        return {"error": str(err)}

if __name__ == "__main__":
    wait_for_service(url=f'{LOCAL_URL}/txt2img')
    setup_default_settings()  # Apply optimal settings on startup
    print("WebUI API Service is ready. Starting RunPod...")
    runpod.serverless.start({"handler": handler})