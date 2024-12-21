import os

# Ensure the environment variable is set
blackhole_path = os.getenv('BLACKHOLE')
if not blackhole_path:
    raise EnvironmentError("The environment variable $BLACKHOLE is not set.")

# Join the path and list files
data_path = os.path.join(blackhole_path, "EARS-WHAM", "train")
print(f"Data path: {data_path}")

# Check if path exists and list files
if os.path.exists(data_path):
    print(f"Files in path: {os.listdir(data_path)}")
else:
    print("Path does not exist.")

