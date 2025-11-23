import json
from backend.models.vae_copy import VAEModel

# instantiate once, reused across warm Lambda calls
vae = VAEModel()

def handler(request, context):
    grid = vae.generate_grid(level="easy", shape=(3, 5))
    return {
        "statusCode": 200,
        "headers": { "Content-Type": "application/json" },
        "body": json.dumps({"grid": grid.tolist()})
    }