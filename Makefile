.PHONY: build dev test tests up-dev bash prod up-dev up-prod
train:
	python trainer.py  --config configs/config.yaml


fit:
	/home/lab09/.conda/envs/gpcr/bin/python trainer.py  --config configs/cli_config.yaml fit


fit-faraday:
	/home/lab09/.conda/envs/gpcr/bin/python trainer.py  --config configs/cli_config-faraday.yaml fit

fit-faraday-gpcr:
	/home/lab09/.conda/envs/gpcr/bin/python trainer.py  --config configs/cli_config-faraday-gpcr_cc.yaml fit

fit-faraday-gpcr-local:
	/home/lab09/.conda/envs/gpcr/bin/python trainer.py  --config configs/cli_config-faraday-gpcr-local.yaml fit



# fit-faraday-gpcr-local-cc:
# 	/home/lab09/.conda/envs/gpcr/bin/python trainer.py  --config configs/cli_config-faraday-gpcr_cc-local.yaml fit

fit-faraday-gpcr-local-cc:
	/home/lab09/.conda/envs/gpcr/bin/python trainer.py  --config configs/cli_config-faraday-gpcr_cc-local.yaml fit


fit_log:
	python trainer.py  --config configs/cli_config.yaml fit --trainer.logger.class_path=lightning.pytorch.loggers.WandbLogger

fit_log1:
	python trainer.py  --config configs/cli_config.yaml fit --trainer.logger.class_path=WandbLogger


fit-faraday-gpcr_ec:
	/home/lab09/.conda/envs/gpcr/bin/python trainer.py  --config configs/cli_config-faraday-gpcr_ec.yaml fit


fit-faraday-gpcr_ic:
	/home/lab09/.conda/envs/gpcr/bin/python trainer.py  --config configs/cli_config-faraday-gpcr_ic.yaml fit


fit-faraday-gpcr_cc:
	/home/lab09/.conda/envs/gpcr/bin/python trainer.py  --config configs/cli_config-faraday-gpcr_cc.yaml fit


test:

	python trainer.py  --config configs/cli_config.yaml test	

clean:
	rm -rf logs/* models/*



hello:
	echo "hello world"

help:
	@echo "make train"
	@echo "make fit"
	@echo "make fit_log"
	@echo "make fit_log1"
	@echo "make test"
	@echo "make clean"
	@echo "make hello"
	@echo "make help"



build: dev prod test

dev:
	docker compose build dev

prod:
	docker compose build prod

test:
	docker compose build test

tests:
	docker compose run test

up-dev:
	docker compose up dev

up-prod:
	docker run -p 8097:8097 aigpro/aigpro:latest-prod

bash:
	docker compose run --entrypoint=bash dev

up:
	docker compose up -d 