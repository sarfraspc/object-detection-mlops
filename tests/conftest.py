import os
import pytest
from fastapi.testclient import TestClient
from src.main import app

# DUMMY CLASSES 
class DummyBox:
    def __init__(self, x1, y1, x2, y2, conf, cls):
        self.xyxy = type('obj', (), {'tolist': lambda *args: [x1, y1, x2, y2]})()
        self.xyxy = [type('tensor', (), {'tolist': lambda *args: [x1, y1, x2, y2]})()]
        self.conf = [conf]
        self.cls = [cls]

class DummyResult:
    def __init__(self):
        self.boxes = [
            DummyBox(50, 100, 300, 500, 0.94, 0),
            DummyBox(400, 150, 600, 400, 0.87, 1),
        ]
        self.names = {0: "person", 1: "bicycle"}

class DummyYOLO:
    def __init__(self, *args, **kwargs):
        pass
    def __call__(self, img, **kwargs):
        return [DummyResult()]


@pytest.fixture
def mock_yolo(monkeypatch):
    """
    Patches the YOLO class *inside* src.main.
    """
    if os.getenv("REAL_YOLO") == "1":
        return
    
    monkeypatch.setattr("src.main.YOLO", DummyYOLO)

@pytest.fixture
def client(mock_yolo): 
    """
    The client fixture MUST request 'mock_yolo' to ensure the patch 
    is applied BEFORE the app starts up.
    """
    with TestClient(app) as c:
        yield c