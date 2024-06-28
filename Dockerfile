FROM nvcr.io/nvidia/nemo:24.05 AS base

COPY requirements.txt .
RUN pip install -r requirements.txt

WORKDIR /app

COPY . .

FROM base AS test
#RUN pip install -r test-requirements.txt
CMD ["pytest", "-vv", "/app/tests/test_all.py"]

FROM base AS production
CMD ["/bin/sh", "-c", "bash"]
