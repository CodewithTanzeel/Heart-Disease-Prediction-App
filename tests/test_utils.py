from src.utils import save_pickle, load_pickle
import os

def test_save_and_load_pickle(tmp_path):
    obj = {"a": 1, "b": 2}
    file_path = tmp_path / "test.pkl"

    save_pickle(obj, str(file_path))
    assert os.path.exists(file_path)

    loaded_obj = load_pickle(str(file_path))
    assert loaded_obj == obj
