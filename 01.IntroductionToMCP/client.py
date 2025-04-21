import requests


def test_multiply(a, b):
    res = requests.post(
        "http://127.0.0.1:8000/multiply",
        json={"a": a, "b": b}
    )

    if res.status_code == 200:
        print(f"Result : {res.json()}")
    else:
        print("Failed")


if __name__ == "__main__":
    test_multiply(10, 5)
