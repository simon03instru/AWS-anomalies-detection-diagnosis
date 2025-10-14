from ollama import Client

client = Client(host="http://10.33.205.34:11112")

models = client.list()

print("Available models on server:")
print("=" * 80)

for model in models['models']:
    print(f"\nModel: {model.model}")
    print(f"  Size: {model.details.parameter_size}")
    print(f"  Family: {model.details.family}")
    print(f"  Quantization: {model.details.quantization_level}")
    print(f"  Modified: {model.modified_at}")