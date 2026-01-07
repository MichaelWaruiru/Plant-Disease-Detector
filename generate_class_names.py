import os
import json
from disease_data import DISEASE_INFO

os.makedirs("models", exist_ok=True)

with open("models/class_names.json", "w") as f:
    json.dump(list(DISEASE_INFO.keys()), f, indent=2)

print("class_names.json generated successfully")
