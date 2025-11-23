import json
from backend.models.vae_copy import VAEModel

vae = VAEModel()

def handler(request, context):
    grid = vae.generate_grid(level="hard", shape=(6, 8))
    return {
        "statusCode": 200,
        "headers": { "Content-Type": "application/json" },
        "body": json.dumps({"grid": grid.tolist()})
    }