import urllib.request
import os

# Better street scene images (dashcam perspective)
street_images = {
    "urban_street.jpg": "https://images.unsplash.com/photo-1502920514313-52581002a659?w=800",
    "highway.jpg": "https://images.unsplash.com/photo-1606836591695-4d58a73eba1e?w=800",
    "city_traffic.jpg": "https://images.unsplash.com/photo-1519003722824-194d4455a60c?w=800",
    "pedestrian_crossing.jpg": "https://images.unsplash.com/photo-1572375992501-4b0892d50c69?w=800",
}

os.makedirs("data/test_images", exist_ok=True)

print("Downloading better street-level test images...")
for filename, url in street_images.items():
    filepath = f"data/test_images/{filename}"
    try:
        print(f"  Downloading {filename}...")
        urllib.request.urlretrieve(url, filepath)
        print(f"  ✓ Saved to {filepath}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")

print("\n✓ Done! Run test_inference.py again.")