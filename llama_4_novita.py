import requests
import json

def describe_image_with_url(image_url, api_key):
    url = "https://api.novita.ai/v3/openai/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe this image in full detail. Include the objects, environment, mood, purpose, country and architecture style."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        }
                    }
                ]
            }
        ],
        "max_tokens": 1000
    }

    response = requests.post(url, headers=headers, json=payload)
    return response.json()["choices"][0]["message"]["content"]

# Example usage:
api_key = "sk_f04mvdKLeyY-xBWCIWuWrH-paZqhgEZ2T9fISibzOuE"
image_url = "https://drive.google.com/uc?export=view&id=1xnoIkpxSYkrruEyu2madSv3xDrwafPJd"  # or any uploaded image
print(describe_image_with_url(image_url, api_key))


