import io
from PIL import Image

def test_predict_returns_two_objects(client):
    # Generate dummy image in memory
    image = Image.new("RGB", (100, 100), "white")
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="JPEG")
    img_byte_arr.seek(0)

    resp = client.post(
        "/predict", 
        files={"file": ("sample.jpg", img_byte_arr, "image/jpeg")}
    )

    assert resp.status_code == 200
    data = resp.json()
    assert len(data["predictions"]) == 2
    assert data["predictions"][0]["class_name"] == "person"
    assert data["inference_time_s"] >= 0

def test_predict_rejects_text_file(client):
    resp = client.post("/predict", files={"file": ("bad.txt", b"notimage", "text/plain")})
    assert resp.status_code == 400