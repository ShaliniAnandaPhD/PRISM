import time
from fastapi import FastAPI, Request
from prometheus_client import Counter, Histogram, make_asgi_app
from pydantic import BaseModel

# Configuration for monitoring
class MonitoringConfig(BaseModel):
    prometheus_endpoint: str = "/metrics"
    latency_buckets: list = [0.1, 0.5, 1.0, 2.0, 5.0]  # Histogram buckets for latency


# Initialize Prometheus Metrics
class PrometheusMetrics:
    def __init__(self, config: MonitoringConfig):
        self.REQUEST_COUNT = Counter(
            "request_count", "Total number of requests", ["method", "endpoint"]
        )
        self.REQUEST_LATENCY = Histogram(
            "request_latency_seconds",
            "Request latency in seconds",
            ["endpoint"],
            buckets=config.latency_buckets,
        )
        self.ERROR_COUNT = Counter(
            "error_count", "Total number of errors", ["method", "endpoint", "status_code"]
        )


# Initialize FastAPI and Prometheus
config = MonitoringConfig()
app = FastAPI(title="Hybrid Search Monitoring Tool", version="1.0.0")
metrics = PrometheusMetrics(config)

# Expose Prometheus metrics endpoint
prometheus_app = make_asgi_app()
app.mount(config.prometheus_endpoint, prometheus_app)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """
    Middleware to track request metrics including count, latency, and errors.
    """
    start_time = time.time()
    method = request.method
    endpoint = request.url.path

    try:
        response = await call_next(request)
        status_code = response.status_code
        # Increment counters and histograms
        metrics.REQUEST_COUNT.labels(method=method, endpoint=endpoint).inc()
        metrics.REQUEST_LATENCY.labels(endpoint=endpoint).observe(time.time() - start_time)
    except Exception as e:
        # Increment error count for 500 status codes
        metrics.ERROR_COUNT.labels(method=method, endpoint=endpoint, status_code=500).inc()
        raise e

    return response


@app.get("/")
async def root():
    """
    Root endpoint for health check.
    """
    return {"message": "Hybrid Search Monitoring Tool is running."}


@app.get("/analytics/usage")
async def get_usage_metrics():
    """
    Endpoint to retrieve simulated API usage metrics (example implementation).
    """
    return {
        "total_requests": metrics.REQUEST_COUNT._value.get(),
        "error_count": metrics.ERROR_COUNT._value.get(),
        "endpoints_tracked": list(metrics.REQUEST_COUNT._metrics.keys()),
    }

# ---------------------------------------------------------------
# What We Did:
# ---------------------------------------------------------------
# - Modularized Prometheus metric definitions using a class.
# - Added configurable histogram buckets for request latency.
# - Improved middleware to match a reusable, extensible format.
# ---------------------------------------------------------------
# What's Next:
# ---------------------------------------------------------------
# - Introduce a logging and alerting service for real-time error tracking.
# - Build a Prometheus PushGateway integration for dynamic environments.
# - Develop a Slack or PagerDuty alerting system for critical thresholds.
# ---------------------------------------------------------------

