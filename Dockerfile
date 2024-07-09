FROM nvcr.io/nvidia/nemo:24.05 AS base

COPY requirements.txt .
RUN pip install -r requirements.txt

WORKDIR /app

COPY . .

FROM base AS test

#RUN pip install -r test-requirements.txt

CMD ["/bin/sh", "-c", "bash"]
#CMD ["pytest", "-vv", "/app/tests/test_all.py"]

FROM base AS eval
#CMD ["python", "eval/run_eval.py"]
CMD ["python", "tools/convert_to_onxx.py"]

FROM base AS production
CMD ["python", "-m", "src.main"]
