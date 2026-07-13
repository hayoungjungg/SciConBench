"""scicon CLI — SciConBench-Track command-line interface."""

import sys
from pathlib import Path

# Make scicon-track/ sub-packages (config, db, huggingface) importable,
# and scripts/ sub-packages (data_preprocessing, data_labeling, data_collection).
_track_dir = str(Path(__file__).resolve().parent)
_scripts_dir = str(Path(__file__).resolve().parent.parent / "scripts")
for _p in (_track_dir, _scripts_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import click


@click.group()
def main():
    """SciConBench-Track — longitudinal benchmark pipeline CLI."""


@main.command()
@click.option("--once", is_flag=True,
              help="Run immediately once instead of starting the 30-day scheduler.")
@click.option("--max-dois", type=int, default=None, metavar="N",
              help="Limit to N DOIs per run (useful for smoke-testing).")
@click.option("--batch-size", type=int, default=500, show_default=True,
              help="Atomic-fact batch size.")
def workflow(once, max_dois, batch_size):
    """Run the monthly SciConBench-Track pipeline.

    Pipeline stages (all idempotent):

    \b
      1.  Initialize DB
      2.  Register core set from HuggingFace
      3.  Discover new rolling reviews (diff against HuggingFace benchmark)
      4.  Download rolling PDFs via Wiley TDM
      5.  Extract reference text from PDFs
      6.  Generate clinical questions
      7.  Generate Cochrane atomic facts
      8.  Query all models
      9.  Generate model-response atomic facts
      10. Run precision & recall analysis
      11. Upload to HuggingFace
    """
    from datetime import timedelta
    from run_workflow import sciconbench_track_pipeline

    if once:
        sciconbench_track_pipeline(batch_size=batch_size, max_dois=max_dois)
    else:
        sciconbench_track_pipeline.serve(
            name="sciconbench-track-monthly",
            interval=timedelta(days=30),
        )


@main.command()
@click.option("--force", is_flag=True, help="Drop and recreate all tables.")
def init_db(force):
    """Initialize (or re-create) the SQLite database."""
    from db import init_db as _init_db
    _init_db(force=force)
    click.echo("Database initialized.")


@main.command()
@click.option("--cohort-month", default=None, metavar="YYYY-MM",
              help="Cohort month label (default: current month).")
@click.option("--limit", type=int, default=None, metavar="N",
              help="Maximum number of new reviews to register.")
def discover(cohort_month, limit):
    """Discover new Cochrane reviews and register them in the DB."""
    from datetime import date
    from data_collection.collector import DataCollector

    cohort_month = cohort_month or date.today().strftime("%Y-%m")
    works = DataCollector().discover_new_dois(cohort_month=cohort_month, limit=limit)
    click.echo(f"Registered {len(works)} new DOIs for cohort {cohort_month}.")


@main.command()
def download_pdfs():
    """Download PDFs for all rolling reviews that are awaiting download."""
    from data_collection.collector import DataCollector
    paths = DataCollector().download_pdfs()
    click.echo(f"Downloaded {len(paths)} PDF(s).")


@main.command()
def extract_text():
    """Extract reference text from downloaded PDFs and store in the database."""
    from data_collection.collector import DataCollector
    n = DataCollector().extract_and_store_text()
    click.echo(f"Extracted text for {n} review(s).")


@main.command()
def upload():
    """Upload the current benchmark state to HuggingFace."""
    from huggingface.uploader import SciConBenchUploader
    uploader = SciConBenchUploader()
    uploader.save_to_parquet()
    url = uploader.upload()
    click.echo(f"Uploaded: {url}")
