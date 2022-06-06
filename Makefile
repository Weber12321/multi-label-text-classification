run_celery:
	@eval "celery -A celery_app worker -n worker1@%n -l INFO -P solo -Q queue1 --logfile=logs/celery_w.log"

run_api:
	@eval "python api.py"
