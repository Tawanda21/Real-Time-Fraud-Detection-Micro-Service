from locust import HttpUser, task


class FraudServiceUser(HttpUser):
    @task
    def predict(self):
        self.client.post(
            "/predict",
            json={"features": {"amount": 123.0, "distance": 10.0, "night": 0}},
        )
