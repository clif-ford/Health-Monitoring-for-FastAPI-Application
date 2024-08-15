# Adding Health Monitoring to the FastAPI Application for MNIST digit prediction and Dockerize it
We add prometheus monitoring hooks into the application to
track the API usage and to monitor the App’s health. Further added Grafana visualization to work with
Prometheus metrics. We create multiple instances (cluster) of the docker image and monitor the cluster’s health.
