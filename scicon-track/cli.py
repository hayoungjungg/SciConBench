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
              help="Run immediately once instead of starting the calendar-month scheduler.")
@click.option("--max-dois", type=int, default=None, metavar="N",
              help="Limit to N DOIs per run (useful for smoke-testing).")
@click.option("--batch-size", type=int, default=500, show_default=True,
              help="Atomic-fact batch size.")
@click.option("--rolling-month", default=None, metavar="YYYY-MM",
              help="Latest closed month for evals (default: previous calendar month). "
                   "New reviews from any month are still ingested; rolling DOIs "
                   "published after this month are not queried until that month ends.")
@click.option("--interval", type=click.Choice(["monthly", "bimonthly"]), default="monthly",
              show_default=True,
              help="Scheduler cadence (ignored with --once): 1st of each month, "
                   "or 1st of odd months.")
def workflow(once, max_dois, batch_size, rolling_month, interval):
    """Run the monthly SciConBench-Track pipeline.

    Requires a core set created once via `scicon-track init-core-set`.

    Pipeline stages (all idempotent):

    \b
      1.  Initialize DB
      2.  Load the registered core set (read-only)
      3.  Discover new reviews not already on HuggingFace + prune stale DOIs
      4.  Download PDFs via Wiley TDM
      5.  Extract reference text from PDFs (assigns rolling cohort from publication month)
      6.  Generate clinical questions
      7.  Generate Cochrane atomic facts
      8.  Upload to HuggingFace (hayoungjung/SciConBench, test split)
      9.  Query models (core + closed rolling months only)
      10. Generate model-response atomic facts
      11. Run precision & recall analysis
    """
    from prefect.schedules import Cron
    from run_workflow import sciconbench_track_pipeline

    if once:
        sciconbench_track_pipeline(
            batch_size=batch_size,
            max_dois=max_dois,
            rolling_month=rolling_month,
        )
    else:
        cron = "0 0 1 * *" if interval == "monthly" else "0 0 1 1,3,5,7,9,11 *"
        sciconbench_track_pipeline.serve(
            name=f"sciconbench-track-{interval}",
            schedules=[Cron(cron, timezone="America/New_York")],
        )


@main.command("init-core-set")
@click.option("--per-month", "n", type=int, default=None, metavar="N",
              help="Max reviews to sample per calendar month "
                   "(default: core_per_month in config.yaml).")
@click.option("--force", is_flag=True,
              help="Drop the existing CORE panel and redraw from HuggingFace.")
def init_core_set(n, force):
    """Draw the one-time core set from the curated HuggingFace benchmark.

    Samples up to 10 already-curated reviews per calendar month
    (Jul 2025–Jun 2026) from hayoungjung/SciConBench — not raw Crossref
    (which includes protocols). Idempotent unless --force is passed.
    The monthly pipeline never redraws this set.
    """
    from db import init_db as _init_db
    from data_collection.collector import DataCollector

    _init_db(force=False)
    dois = DataCollector().register_core_set(n=n, force=force)
    click.echo(f"Core set: {len(dois)} DOI(s).")
    for doi in dois:
        click.echo(f"  {doi}")


@main.command()
@click.option("--force", is_flag=True, help="Drop and recreate all tables.")
@click.option("--force-drop-core-set", is_flag=True,
              help="Required together with --force if a finalized core set exists. "
                   "Does not delete data_track/core_set.json.")
def init_db(force, force_drop_core_set):
    """Initialize (or re-create) the SQLite database."""
    from db import init_db as _init_db
    _init_db(force=force, force_drop_core_set=force_drop_core_set)
    click.echo("Database initialized.")


@main.command()
@click.option("--cohort-month", default=None, metavar="YYYY-MM",
              help="Unused for panel assignment (cohorts come from publication "
                   "date). Kept as the prune-event timestamp "
                   "(default: previous calendar month).")
@click.option("--limit", type=int, default=None, metavar="N",
              help="Maximum number of new reviews to register.")
def discover(cohort_month, limit):
    """Discover new Cochrane reviews not already on HuggingFace and prune stale DOIs."""
    from data_collection.collector import DataCollector
    from data_collection.utils import previous_year_month

    month = cohort_month or previous_year_month()
    collector = DataCollector()
    new_dois = collector.discover_rolling_for_month(month, limit=limit)
    events = collector.prune_stale_dois(month)
    click.echo(f"Registered {len(new_dois)} new rolling DOI(s) "
               "(cohort_month set from publication date after extract).")
    click.echo(f"Stale events: {len(events)}.")


@main.command()
def download_pdfs():
    """Download PDFs for all reviews that are awaiting download."""
    from data_collection.collector import DataCollector
    paths = DataCollector().download_pdfs()
    click.echo(f"Downloaded {len(paths)} PDF(s).")


@main.command()
def extract_text():
    """Extract reference text from downloaded PDFs and persist it to the database."""
    from data_collection.collector import DataCollector
    n = DataCollector().extract_and_store_text()
    click.echo(f"Extracted text for {n} review(s).")


@main.command()
def upload():
    """Upload the current benchmark state to HuggingFace (hayoungjung/SciConBench test)."""
    from huggingface.uploader import SciConBenchUploader
    uploader = SciConBenchUploader()
    uploader.save_to_parquet()
    uploader.refresh_filter_caches()
    url = uploader.upload()
    click.echo(f"Uploaded: {url}")
