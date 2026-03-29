import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY")
)

def list_nvidia_models():
    try:
        models = client.models.list()
        for m in models:
            print(f"- {m.id}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    list_nvidia_models()
