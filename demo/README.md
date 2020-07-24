# Demo pipeline for BEAM-2717 w/tfx-bsl

To run the pipeline, first use `make_model.py` to save a demo inference model to a GCP bucket:

```
python make_model.py gs://MY_BUCKET/model
```

Then launch the pipeline with `inference_pipeline.py`:

```
python inference_pipeline.py \
    gs://MY_BUCKET/model \
    ../dist/tfx_bsl-0.23.0.dev0-cp37-cp37m-manylinux2010_x86_64.whl \
    --project MY_PROJECT \
    --runner DataflowRunner \
    --experiments=use_runner_v2 \
    --temp_location gs://MY_BUCKET/temp \
    --output gs://MY_BUCKET/out \
    --job_name beam2717 \
    --region us-central1
```
